"""
Punto de entrada: captura de clips del diccionario visual (Direccion 2).

Graba cada palabra (o letra) como un CLIP de landmarks crudos para el muneco de
capsulas: manos (21 x 2) + torso superior (9 puntos de Pose), en coordenadas
[0, 1] de la imagen, un JSON por toma (contrato en comun/clips.py). No guarda
ningun pixel de video ni escribe al CSV del dataset.

FLUJO PENSADO PARA EL DICCIONARIO (una toma BUENA por palabra):

  1. Cuenta regresiva y grabacion de la toma (como el dataset).
  2. Validacion de encuadre: manos, hombros, CODOS y CADERAS visibles (el
     muneco dibuja brazos y torso completos; hay que estar un poco mas lejos
     de la camara que en la captura del dataset).
  3. REVISION EN VIVO: la toma se reproduce de inmediato SOBRE EL MUNECO real,
     en bucle. ENTER la guarda; R la repite. Asi se sale de la sesion con
     clips ya validados, sin pasadas posteriores.
  4. Al final se imprime un resumen de calidad (fotogramas, fps, huecos).

Uso (desde la carpeta training/):

    python capture/captura_diccionario.py
        MODO LIBRE (por defecto): se escribe la palabra en la ventana, ENTER
        la graba, se revisa en el muneco, y al guardar vuelve a pedir palabra.
        Ideal para grabar el diccionario de corrido, sin lista previa.

    python capture/captura_diccionario.py --archivo capture/palabras/piloto.txt
    python capture/captura_diccionario.py --palabras "HOLA,BUENOS DIAS,AGUA"
        MODO LISTA: recorre las palabras dadas en orden.

El visor (python demo/visor_clips.py) sigue disponible para re-revisar clips
ya guardados con camara lenta, avance por fotograma y espejo.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# Permite ejecutar el archivo directamente: agrega training/ al path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from comun.clips import crear_clip, fotograma_clip  # noqa: E402
from comun.landmarks import DetectorLandmarks  # noqa: E402
from comun.pose import DetectorPose  # noqa: E402
from capture import dibujo  # noqa: E402
from capture.escritor_clips import EscritorClips, fps_de_secuencia  # noqa: E402
from capture.escritor_dataset import MODO_DINAMICO  # noqa: E402
from capture.interfaz_captura import InterfazCaptura, PREPARACION  # noqa: E402
from capture.sesion_captura import ParametrosCaptura, SesionCaptura  # noqa: E402
from demo.visor_clips import ClipPreparado, MunecoCapsulas  # noqa: E402

# Estado propio del MODO LIBRE: se escribe la palabra en pantalla y se graba.
ESCRIBIENDO = "escribiendo"

# Carpeta raiz de los clips grabados (de aqui salen hacia los assets de la app).
RUTA_CLIPS = config.RAIZ_TRAINING / "clips"

# Limites de una toma de clip. El minimo es mas laxo que el del dataset porque
# en una computadora lenta MediaPipe puede bajar de 10 fps; el maximo es alto
# porque un clip no se trunca (la sena completa es lo que se reproduce).
MIN_FRAMES_CLIP = 20
MAX_FRAMES_CLIP = 300


class SesionCapturaClips(SesionCaptura):
    """Sesion de captura de clips con revision inmediata sobre el muneco.

    Reemplaza la pantalla de "guardado" del flujo del dataset por una revision
    interactiva: la toma recien grabada se reproduce en bucle como muneco de
    capsulas (el mismo renderizador del visor y de la app) y la persona decide
    con una tecla si la guarda o la repite. Todo lo demas (cuenta regresiva,
    pausa, reanudacion, escritor) es el flujo de captura de siempre.
    """

    def __init__(self, *args, revisar=True, modo_libre=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.revisar = revisar
        # MODO LIBRE: en vez de una lista fija, la palabra se escribe en la
        # misma ventana antes de cada grabacion. El ciclo es: escribir palabra
        # -> cuenta de 3 s -> grabar -> revision en el muneco -> ENTER guarda y
        # vuelve a pedir palabra (R repite la toma con otra cuenta de 3 s).
        self.modo_libre = modo_libre
        self._palabra_actual = ""
        self._grabadas = 0
        if modo_libre:
            self.estado = ESCRIBIENDO
        self._muneco = MunecoCapsulas(self.interfaz.ancho, self.interfaz.alto)

    # -- Modo libre: escribir la palabra en pantalla y grabarla ---------------

    def ejecutar(self) -> None:
        """Bucle de camara. En modo libre agrega el estado ESCRIBIENDO.

        Es el mismo bucle de la sesion base, con una diferencia: cuando se
        esta escribiendo la palabra, en vez del overlay normal se dibuja el
        campo de texto. En modo lista se usa el bucle base tal cual.
        """
        if not self.modo_libre:
            super().ejecutar()
            return

        cap = self._abrir_camara()
        try:
            fallos_lectura = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    fallos_lectura += 1
                    if fallos_lectura > 30:
                        raise RuntimeError("La camara dejo de entregar imagen.")
                    continue
                fallos_lectura = 0

                frame = cv2.flip(frame, 1)  # vista espejo (selfie)
                ahora = time.time()

                ctx = self._procesar_estado(frame, ahora)
                if self.estado == ESCRIBIENDO:
                    self._overlay_escribiendo(frame)
                else:
                    self.interfaz.componer(frame, ctx)
                self.interfaz.mostrar(frame)

                if not self._manejar_teclas():
                    break
                if self.interfaz.ventana_cerrada():
                    break
        finally:
            self._confirmar_pendiente()
            cap.release()

    def _manejar_teclas(self) -> bool:
        if self.modo_libre and self.estado == ESCRIBIENDO:
            tecla = self.interfaz.leer_tecla(1)
            if tecla == "esc":
                return False  # terminar la sesion
            if tecla:
                self._procesar_tecla_texto(tecla)
            return True
        return super()._manejar_teclas()

    def _procesar_tecla_texto(self, tecla: str) -> None:
        """Edita la palabra en pantalla; ENTER arranca la grabacion.

        Las letras llegan del teclado (la enie incluida; en un teclado sin
        enie, la tecla ; la escribe). El ESPACIO separa las palabras de una
        sena compuesta (BUENOS DIAS), que se guarda como un solo clip.
        """
        if tecla in ("\r", "\n"):
            palabra = _identificador(self._palabra_actual)
            if palabra:
                self.clases = [palabra]
                self.clase_idx = 0
                self._tomas_en_clase = 0
                self._cambiar_estado(PREPARACION, time.time())
            return
        if tecla == "espacio":
            if self._palabra_actual and not self._palabra_actual.endswith(" "):
                self._palabra_actual += " "
        elif tecla == "\x08":  # retroceso
            self._palabra_actual = self._palabra_actual[:-1]
        elif tecla == ";":  # respaldo para teclados sin enie
            self._palabra_actual += "ñ"
        elif len(tecla) == 1 and tecla.isalpha():
            self._palabra_actual += tecla

    def _overlay_escribiendo(self, frame) -> None:
        """Campo de texto de la palabra, sobre la vista de la camara."""
        cx = frame.shape[1] // 2
        velo = frame.copy()
        velo[:] = (18, 17, 16)
        cv2.addWeighted(velo, 0.45, frame, 0.55, 0, frame)

        dibujo.texto_centrado(frame, "Escriba la palabra o frase a grabar",
                              cx, 140, 1.0, dibujo.BLANCO, 2,
                              dibujo.FUENTE_TITULO)
        cursor = "_" if int(time.time() * 2) % 2 == 0 else " "
        dibujo.texto_centrado(frame, self._palabra_actual.upper() + cursor,
                              cx, 300, 2.2, dibujo.AMBAR, 3,
                              dibujo.FUENTE_TITULO)
        lineas = [
            "ENTER grabar   ·   ESPACIO para frases compuestas (BUENOS DIAS)",
            "RETROCESO corregir   ·   ESC terminar la sesion",
            "Encuadre: cara, hombros, codos y caderas visibles",
        ]
        y = 400
        for linea in lineas:
            dibujo.texto_centrado(frame, linea, cx, y, 0.65,
                                  dibujo.GRIS_CLARO, 1)
            y += 38
        if self._grabadas:
            dibujo.texto_centrado(
                frame, f"Clips guardados en esta sesion: {self._grabadas}",
                cx, y + 24, 0.65, dibujo.VERDE, 1)

    def _cerrar_toma(self) -> None:
        """En modo libre, tras guardar se vuelve a pedir la palabra."""
        if not self.modo_libre:
            super()._cerrar_toma()
            return
        if self._pendiente is None:
            # Toma invalida (encuadre o manos): otra toma de la misma palabra,
            # con la cuenta regresiva completa para prepararse.
            self._cambiar_estado(PREPARACION, time.time())
            return
        self._confirmar_pendiente()
        self._grabadas += 1
        self._palabra_actual = ""
        self._cambiar_estado(ESCRIBIENDO, time.time())

    def _paso_guardado(self, ahora, ctx):
        # Toma invalida (encuadre o manos): comportamiento normal, mensaje y
        # nueva toma. Con revision apagada, tambien.
        if self._pendiente is None or not self.revisar:
            super()._paso_guardado(ahora, ctx)
            return
        if self._revisar_pendiente():
            self._cerrar_toma()
        else:
            self._pendiente = None
            self._buffer = []
            self._cambiar_estado(PREPARACION, time.time())

    # -- Revision sobre el muneco --------------------------------------------

    def _revisar_pendiente(self) -> bool:
        """Reproduce la toma pendiente como muneco. True = guardar, False = repetir."""
        clip = ClipPreparado(self._clip_de_pendiente())
        self._muneco.preparar_marco(clip, vista_espejo=False)

        indice = 0.0
        espera_ms = max(1, int(1000.0 / clip.fps))
        while True:
            cuerpo, mi, md = clip.fotograma(indice)
            img = self._muneco.dibujar(cuerpo, mi, md)
            self._hud_revision(img, clip)
            self.interfaz.mostrar(img)

            indice += 1.0
            if indice >= clip.num_frames:
                indice = 0.0  # bucle

            tecla = self.interfaz.leer_tecla(espera_ms)
            if tecla in ("\r", "\n", "g"):
                return True
            if tecla == "r":
                return False
            if tecla in ("q", "esc"):
                # No perder la toma por salir: se guarda y la sesion decide.
                return True
            if self.interfaz.ventana_cerrada():
                return True

    def _clip_de_pendiente(self) -> dict:
        """Arma el clip en memoria (mismo contrato del JSON) para revisarlo."""
        secuencia = self._pendiente["datos"]
        frames = [
            fotograma_clip(f["cuerpo"], f["mano_izq"], f["mano_der"])
            for f in secuencia
        ]
        fps = fps_de_secuencia(secuencia, config.FPS_OBJETIVO)
        aspecto = self.params.ancho_camara / self.params.alto_camara
        return crear_clip(self._clase_actual, fps, aspecto, frames)

    def _hud_revision(self, img, clip):
        """Panel de la revision: palabra, toma y controles."""
        h, w = img.shape[:2]
        alto_panel = 96
        y = h - alto_panel - 20
        margen = 24
        dibujo.panel(img, margen, y, w - 2 * margen, alto_panel, alpha=0.68)
        dibujo.texto(img, f"Revision: {clip.palabra}", margen + 26, y + 40,
                     1.0, dibujo.BLANCO, 2, dibujo.FUENTE_TITULO)
        existentes = self.escritor.muestras_existentes(
            self._clase_actual, self.persona)
        detalle = (f"toma {existentes + 1}   ·   {clip.num_frames} fotogramas "
                   f"a {clip.fps:.1f} fps")
        dibujo.texto(img, detalle, margen + 26, y + 74, 0.6,
                     dibujo.GRIS_CLARO, 1)
        controles = "ENTER guardar    R repetir"
        ancho_txt, _ = dibujo.medir(controles, 0.75, 2)
        dibujo.texto(img, controles, w - margen - ancho_txt - 26, y + 58,
                     0.75, dibujo.VERDE, 2)


def _identificador(texto: str) -> str:
    """Convierte una palabra o frase a identificador de clip.

    Convencion del proyecto (la misma de las 50 clases): MAYUSCULAS y espacios
    a guion bajo. Una sena COMPUESTA se escribe con espacios en la lista
    ("BUENOS DIAS") y se convierte en un solo identificador (BUENOS_DIAS),
    porque en LESHO es UNA sena, no dos.
    """
    return "_".join(texto.upper().split())


def _cargar_palabras(texto_palabras, ruta_archivo) -> list:
    """Arma la lista de palabras desde --palabras y/o --archivo."""
    palabras: list[str] = []
    if ruta_archivo:
        ruta = Path(ruta_archivo)
        if not ruta.exists():
            print(f"No existe el archivo de palabras: {ruta}")
            sys.exit(1)
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            linea = linea.split("#", 1)[0].strip()
            if linea:
                palabras.append(_identificador(linea))
    if texto_palabras:
        palabras.extend(
            _identificador(p) for p in texto_palabras.split(",") if p.strip()
        )
    # Sin duplicados, conservando el orden.
    return list(dict.fromkeys(palabras))


def main(palabras: list, tomas: int, persona: str, carpeta: str,
         revisar: bool) -> None:
    # Sin lista de palabras, la sesion arranca en MODO LIBRE: la palabra se
    # escribe en la misma ventana antes de cada grabacion.
    modo_libre = not palabras
    escritor = EscritorClips(
        RUTA_CLIPS / carpeta,
        aspecto=config.ANCHO_CAMARA / config.ALTO_CAMARA,
        fps_respaldo=config.FPS_OBJETIVO,
    )
    interfaz = InterfazCaptura(
        config.NOMBRE_VENTANA, config.ANCHO_CAMARA, config.ALTO_CAMARA,
        config.RUTA_REFERENCIAS,
    )
    detector = DetectorLandmarks(
        max_manos=config.MAX_MANOS,
        confianza_deteccion=config.CONFIANZA_DETECCION_MANO,
    )
    detector_pose = DetectorPose(
        confianza_deteccion=config.CONFIANZA_DETECCION_MANO,
    )
    params = ParametrosCaptura(
        indice_camara=config.INDICE_CAMARA,
        ancho_camara=config.ANCHO_CAMARA,
        alto_camara=config.ALTO_CAMARA,
        segundos_cuenta=config.SEGUNDOS_CUENTA_REGRESIVA,
        segundos_grabacion=config.SEGUNDOS_GRABACION,
        # Al repetir una toma tambien se dan 3 segundos para prepararse (en el
        # dataset la cuenta corta de 1 s funciona porque son 25 repeticiones
        # seguidas; aca cada toma es deliberada).
        segundos_entre_reps=3.0,
        segundos_guardado=config.SEGUNDOS_MOSTRAR_GUARDADO,
        muestras_por_clase=1 if modo_libre else tomas,
        min_frames=MIN_FRAMES_CLIP,
        max_frames=MAX_FRAMES_CLIP,
    )

    # Sesion en modo dinamico (formato de secuencia) con captura CRUDA: el
    # buffer acumula fotogramas completos para el clip, no vectores del modelo.
    sesion = SesionCapturaClips(
        MODO_DINAMICO, palabras or [""], persona, escritor, interfaz, detector,
        params, detector_pose=detector_pose, captura_cruda=True,
        revisar=revisar, modo_libre=modo_libre,
    )
    try:
        sesion.ejecutar()
    finally:
        detector.cerrar()
        detector_pose.cerrar()
        escritor.cerrar()
        interfaz.cerrar()
    print()
    print(escritor.resumen())
    print()
    print(f"Clips en: {RUTA_CLIPS / carpeta}")
    print("Para re-revisarlos con camara lenta y espejo: "
          "python demo/visor_clips.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Captura de clips del diccionario visual LESHO (muneco)."
    )
    parser.add_argument(
        "--palabras",
        help="Palabras o letras a grabar, separadas por coma "
             "(por ejemplo HOLA,GRACIAS,AGUA o A,B,CH).",
    )
    parser.add_argument(
        "--archivo",
        help="Archivo de texto con una palabra por linea (admite comentarios "
             "con #). Por ejemplo capture/palabras/piloto.txt.",
    )
    parser.add_argument(
        "--tomas", type=int, default=3,
        help="Tomas por palabra (por defecto 3; con la revision en vivo suele "
             "bastar 1 buena, las demas quedan de respaldo).",
    )
    parser.add_argument(
        "--persona", default="autor",
        help="Quien graba (metadato de control de calidad, por defecto 'autor').",
    )
    parser.add_argument(
        "--carpeta", default="piloto",
        help="Subcarpeta de salida dentro de training/clips/ (por defecto 'piloto').",
    )
    parser.add_argument(
        "--sin-revision", action="store_true",
        help="No mostrar el muneco tras cada toma (grabar de corrido).",
    )
    args = parser.parse_args()
    lista = _cargar_palabras(args.palabras, args.archivo)
    if not lista:
        print("Modo libre: escriba cada palabra en la ventana y presione "
              "ENTER para grabarla (ESC termina).")
    main(lista, args.tomas, args.persona, args.carpeta,
         revisar=not args.sin_revision)
