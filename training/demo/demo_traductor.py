"""
Demo TRADUCTOR: deletreo continuo + palabras con ventana explicita (INICIO).

Diseno final de la Direccion 1 (es el diseno original de la tesis, mejorado):

  - DELETREO: continuo, sin gestos. El Modelo A corre con su ventana rodante y
    escribe las letras confirmadas (incluidas J, LL, RR, Ñ, Z). Dos punos
    (FIN) quietos = espacio.
  - PALABRA: la ventana se ABRE de forma explicita, con cualquiera de:
      - la sena INICIO (dos palmas abiertas, quietas),
      - un CLIC en la pantalla (simula el boton que tendra la app movil),
      - la barra ESPACIADORA.
    Se hace UNA sena y la ventana se CIERRA con cualquiera de:
      - bajar las manos (lo mas natural),
      - la sena FIN (dos punos quietos),
      - clic o ESPACIO,
      - o sola, por tiempo maximo.
    Al cerrar, el Modelo B clasifica la sena y escribe la palabra.

Como la ventana se abre a proposito, el sistema SABE que viene una palabra: no
necesita clase de rechazo ni umbrales estrictos (se probaron y degradaban el
reconocimiento). Se escribe la mejor candidata con su confianza; solo si es
francamente baja se pide repetir.

INICIO y FIN solo cuentan QUIETOS (son poses estaticas): asi una palabra que use
punos en movimiento (TRABAJO) no puede cerrar la ventana por accidente.

Uso (desde la carpeta training/):

    python demo/demo_traductor.py

Teclas:  ESPACIO abrir/cerrar palabra   clic igual   C limpiar   retroceso borrar   Q salir
"""

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from comun.definiciones import (  # noqa: E402
    CLASES_DINAMICAS,
    CLASES_ESTATICAS,
    LATERALIDAD_DERECHA,
    LATERALIDAD_IZQUIERDA,
    LETRAS_CON_MOVIMIENTO,
    TAMANO_VECTOR,
    TAMANO_VENTANA_A,
)
from comun.landmarks import DetectorLandmarks  # noqa: E402
from comun.pose import DetectorPose  # noqa: E402
from comun.marco import marco_desde_puntos  # noqa: E402
from comun.normalizacion import (  # noqa: E402
    componer_vector_dos_manos,
    escalar_lote,
    escalar_vector,
)
from comun.suavizado import FiltroUnEuro  # noqa: E402
from comun.vector_modelo_b import componer_vector_modelo_b  # noqa: E402
from preprocessing.secuencias_b import procesar_secuencia  # noqa: E402
from preprocessing.ventanas import velocidad_media  # noqa: E402
from capture import dibujo  # noqa: E402

# --- Deletreo (Modelo A), COPIADO de demo_deletreo.py (funcionaba bien asi) --
PERSISTENCIA = 5            # ventanas consecutivas para confirmar una clase
UMBRAL_CONFIANZA = 0.60     # confianza minima para contar hacia la persistencia
COOLDOWN_MS = 1200          # pausa tras escribir una letra antes de volver a
                            # escanear. Da tiempo a ver la letra escrita, reaccionar
                            # y formar la siguiente. Permite repetir la misma letra.
TIEMPO_MUERTO_PALABRA_MS = 1000  # tiempo muerto tras escribir una palabra
UMBRAL_MOVIMIENTO_GATE = config.UMBRAL_MOVIMIENTO_ABS
UMBRAL_MOVIMIENTO_MOVIENDO = config.UMBRAL_MOVIMIENTO_MOVIENDO

# --- Gestos INICIO / FIN (poses quietas de dos manos) ------------------------
PERSISTENCIA_GESTO = 6      # ventanas consecutivas para confirmar INICIO o FIN
UMBRAL_GESTO = 0.80         # confianza minima del gesto

# --- Ventana de palabra (Modelo B) -------------------------------------------
RECORTE_CIERRE_GESTO = 14   # cola descartada si cerro la sena FIN (formacion de punos)
RECORTE_CIERRE_MANOS = 10   # cola descartada si cerro bajando las manos (descenso)
RECORTE_CIERRE_BOTON = 2    # cola descartada si cerro con clic o ESPACIO
FRAMES_SIN_MANOS = 8        # fotogramas sin manos para cerrar por bajada
MAX_PALABRA_MS = 8000       # cierre automatico de la ventana
ESPERA_SOLTAR_MS = 1500     # tope de espera a que se suelten las palmas de INICIO
MIN_FRAMES_B = 12           # minimo, tras recortes, para clasificar
UMBRAL_CONF_B = 0.60        # confianza (renormalizada) minima para escribir la
                            # palabra. Medido con tomas reales: los aciertos
                            # promedian 99% y los errores rondan 57%; este umbral
                            # convierte los errores en "repita" sin costar aciertos.
MAX_FRAMES_EMISION = 300
MS_FLASH = 1500
_MARGEN = 24

MODO_LETRAS = "letras"
MODO_PALABRA = "palabra"


# ---------------------------------------------------------------------------
# Texto traducido: mezcla letras deletreadas y palabras del Modelo B
# ---------------------------------------------------------------------------

class TextoTraducido:
    """Acumula unidades (letras, palabras, espacios) y las muestra como frase."""

    def __init__(self):
        self.unidades: list[tuple[str, str]] = []

    def agregar_letra(self, letra: str) -> None:
        self.unidades.append(("letra", letra))

    def agregar_palabra(self, palabra: str) -> None:
        self.unidades.append(("palabra", palabra))

    def agregar_espacio(self) -> None:
        if self.unidades and self.unidades[-1][0] != "espacio":
            self.unidades.append(("espacio", " "))

    def borrar_ultima(self) -> None:
        if self.unidades:
            self.unidades.pop()

    def limpiar(self) -> None:
        self.unidades.clear()

    @property
    def cadena(self) -> str:
        partes: list[str] = []
        for tipo, txt in self.unidades:
            if tipo == "palabra":
                if partes and not partes[-1].endswith(" "):
                    partes.append(" ")
                partes.append(txt)
                partes.append(" ")
            elif tipo == "espacio":
                if partes and not partes[-1].endswith(" "):
                    partes.append(" ")
            else:  # letra
                partes.append(txt)
        return "".join(partes).strip()


# ---------------------------------------------------------------------------
# Confirmador de clases del Modelo A (persistencia + cooldown)
# ---------------------------------------------------------------------------

class ConfirmadorA:
    """Confirma clases del Modelo A. Devuelve la clase confirmada o None.

    Tras cada confirmacion aplica su TIEMPO MUERTO (cooldown_ms): durante ese
    periodo no se acumula ni confirma nada, para dar tiempo a cambiar de sena.
    """

    def __init__(self, persistencia: int, umbral: float, cooldown_ms: float = 0.0):
        self.persistencia = persistencia
        self.umbral = umbral
        self.cooldown_ms = cooldown_ms
        self._clase = None
        self._contador = 0
        self._fin_cooldown = 0.0

    def bloquear(self, hasta_ms: float) -> None:
        self._fin_cooldown = max(self._fin_cooldown, hasta_ms)

    def reiniciar(self) -> None:
        self._clase = None
        self._contador = 0

    def registrar(self, clase: str, confianza: float, ahora_ms: float):
        """Devuelve (clase_confirmada | None, progreso_actual)."""
        if ahora_ms < self._fin_cooldown or confianza < self.umbral:
            self.reiniciar()
            return None, 0
        if clase == self._clase:
            self._contador += 1
        else:
            self._clase = clase
            self._contador = 1
        if self._contador >= self.persistencia:
            self.reiniciar()
            self._fin_cooldown = ahora_ms + self.cooldown_ms  # tiempo muerto
            return clase, self.persistencia
        return None, self._contador


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------

def _asignar_manos(manos):
    izquierda = derecha = None
    sin_asignar = []
    for mano in manos:
        if mano.lateralidad == LATERALIDAD_IZQUIERDA and izquierda is None:
            izquierda = mano.landmarks
        elif mano.lateralidad == LATERALIDAD_DERECHA and derecha is None:
            derecha = mano.landmarks
        else:
            sin_asignar.append(mano.landmarks)
    for landmarks in sin_asignar:
        if izquierda is None:
            izquierda = landmarks
        elif derecha is None:
            derecha = landmarks
    return izquierda, derecha


def _clases_presentes():
    """Indices de clase con datos en el dataset B (mini prueba), o None."""
    if not config.RUTA_DATASET_B.exists():
        return None
    datos = np.load(config.RUTA_DATASET_B, allow_pickle=True)
    return sorted(set(int(v) for v in datos["y"]))


def clasificar_palabra(modelo_b, buffer_b, presentes, recorte_cola):
    """Clasifica la ventana de palabra. Devuelve (palabra, confianza) o None.

    Primero se descarta la cola del cierre (bajada de manos o formacion de FIN,
    segun como se cerro), y luego se clasifica. El recorte de los tramos quietos
    (nucleo activo) lo hace procesar_secuencia, igual que en el entrenamiento. La
    confianza se renormaliza entre las clases presentes.
    """
    seq = list(buffer_b)
    if recorte_cola and len(seq) > recorte_cola + MIN_FRAMES_B:
        seq = seq[:len(seq) - recorte_cola]
    if len(seq) < MIN_FRAMES_B:
        return None
    # El recorte de tramos quietos (nucleo activo) lo hace procesar_secuencia,
    # igual que en el entrenamiento, para que ambos vean la misma representacion.
    proc = procesar_secuencia(
        np.asarray(seq, dtype=np.float32), config.LONGITUD_FIJA_SECUENCIA,
        suavizar=config.SUAVIZADO_ACTIVO, fps=config.FPS_OBJETIVO,
        min_cutoff=config.SUAVIZADO_MIN_CUTOFF, beta=config.SUAVIZADO_BETA,
        d_cutoff=config.SUAVIZADO_D_CUTOFF,
    )
    pred = modelo_b.predict(proc[None, ...], verbose=0)[0]
    if presentes is not None:
        indices = [i for i in presentes if i < len(pred)]
        total = float(pred[indices].sum())
        if total <= 0:
            return None
        mascara = np.zeros(len(pred))
        mascara[indices] = pred[indices] / total
        pred = mascara
    idx = int(np.argmax(pred))
    return CLASES_DINAMICAS[idx], float(pred[idx])


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------

def main():
    for ruta, cual in [(config.RUTA_MODELO_A_KERAS, "A (alfabeto)"),
                       (config.RUTA_MODELO_B_KERAS, "B (palabras)")]:
        if not ruta.exists():
            print(f"Falta el modelo {cual}: {ruta}")
            sys.exit(1)

    import tensorflow as tf
    modelo_a = tf.keras.models.load_model(config.RUTA_MODELO_A_KERAS)
    modelo_b = tf.keras.models.load_model(config.RUTA_MODELO_B_KERAS)
    presentes = _clases_presentes()

    idx_movimiento = np.array([CLASES_ESTATICAS.index(l) for l in LETRAS_CON_MOVIMIENTO])
    idx_gemelas = np.array(
        [CLASES_ESTATICAS.index(l) for l in config.ESTATICAS_GEMELAS_DE_MOVIMIENTO]
    )

    detector = DetectorLandmarks(max_manos=config.MAX_MANOS,
                                 confianza_deteccion=config.CONFIANZA_DETECCION_MANO)
    detector_pose = DetectorPose(confianza_deteccion=config.CONFIANZA_DETECCION_MANO)
    cap = cv2.VideoCapture(config.INDICE_CAMARA)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.ANCHO_CAMARA)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.ALTO_CAMARA)
    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        sys.exit(1)

    ventana = "LESHO - Traductor"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana, config.ANCHO_CAMARA, config.ALTO_CAMARA)

    # El clic simula el boton en pantalla que tendra la app movil.
    clics: list = []
    cv2.setMouseCallback(ventana, lambda ev, x, y, fl, ud: clics.append(1)
                         if ev == cv2.EVENT_LBUTTONDOWN else None)

    texto = TextoTraducido()
    letras = ConfirmadorA(PERSISTENCIA, UMBRAL_CONFIANZA, COOLDOWN_MS)
    gesto = ConfirmadorA(PERSISTENCIA_GESTO, UMBRAL_GESTO, 800.0)
    buffer_a = deque(maxlen=TAMANO_VENTANA_A)
    buffer_b = deque(maxlen=MAX_FRAMES_EMISION)
    filtro_a = FiltroUnEuro(config.FPS_OBJETIVO, config.SUAVIZADO_MIN_CUTOFF,
                            config.SUAVIZADO_BETA, config.SUAVIZADO_D_CUTOFF)
    marco_actual = None
    cuerpo_dibujo = None    # ultimo esqueleto de Pose, para dibujar sin parpadeo
    modo = MODO_LETRAS
    sin_manos = 0
    inicio_palabra_ms = 0.0
    esperando_soltar = False   # abierto con INICIO: aun no se sueltan las palmas
    aviso = None
    aviso_hasta = 0.0
    frame_idx = 0

    def abrir_palabra(ahora, por_gesto=False):
        nonlocal modo, inicio_palabra_ms, esperando_soltar
        modo = MODO_PALABRA
        inicio_palabra_ms = ahora
        esperando_soltar = por_gesto   # con INICIO: esperar a soltar las palmas
        buffer_b.clear()
        gesto.reiniciar()

    def cerrar_palabra(ahora, recorte_cola):
        nonlocal modo, aviso, aviso_hasta
        modo = MODO_LETRAS
        letras.bloquear(ahora + TIEMPO_MUERTO_PALABRA_MS)  # pausa tras la palabra
        gesto.reiniciar()
        resultado = clasificar_palabra(modelo_b, buffer_b, presentes, recorte_cola)
        buffer_b.clear()
        if resultado is None:
            aviso = ("sena muy corta, repita", dibujo.AMBAR)
        elif resultado[1] >= UMBRAL_CONF_B:
            texto.agregar_palabra(resultado[0])
            aviso = (f"{resultado[0]}  {resultado[1]*100:.0f}%", dibujo.VERDE)
        else:
            aviso = ("sena no reconocida, repita", dibujo.AMBAR)
        aviso_hasta = ahora + MS_FLASH

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            ahora = time.time() * 1000.0
            frame_idx += 1

            manos = detector.procesar(frame)

            # Pose: por fotograma en modo palabra; 1 de cada 5 en deletreo
            # (simula el ahorro del movil). El DIBUJO usa el ultimo esqueleto
            # conocido en todos los fotogramas, para que no parpadee; el
            # reconocimiento no se ve afectado (el marco usa carry-forward).
            if modo == MODO_PALABRA or frame_idx % 5 == 0:
                puntos = detector_pose.procesar(frame)
                if puntos is not None:
                    cuerpo_dibujo = puntos
                    m = marco_desde_puntos(puntos)
                    if m is not None:
                        marco_actual = m
            if cuerpo_dibujo is not None:
                dibujo.dibujar_cuerpo(frame, cuerpo_dibujo)

            candidato = None
            progreso = 0
            flash_letra = None

            if manos:
                sin_manos = 0
                for mano in manos:
                    dibujo.dibujar_mano(frame, mano.landmarks)
                izq, der = _asignar_manos(manos)

                # Ventana rodante del Modelo A (siempre: letras o gestos).
                trasladado = componer_vector_dos_manos(izq, der)
                if config.SUAVIZADO_ACTIVO:
                    trasladado = filtro_a.filtrar(trasladado).tolist()
                buffer_a.append(escalar_vector(trasladado))

                pred_a = None
                mov = 0.0
                if len(buffer_a) == TAMANO_VENTANA_A:
                    entrada = np.asarray(buffer_a, dtype=np.float32)[None, ...]
                    pred_a = modelo_a(entrada, training=False).numpy()[0]
                    mov = velocidad_media(entrada[0])

                if modo == MODO_LETRAS and pred_a is not None:
                    # Compuertas EXACTAS de demo_deletreo (Modelo A original).
                    p = pred_a.copy()
                    if mov < UMBRAL_MOVIMIENTO_GATE:
                        p[idx_movimiento] = 0.0
                    elif mov > UMBRAL_MOVIMIENTO_MOVIENDO:
                        p[idx_gemelas] = 0.0
                    idx = int(np.argmax(p))
                    candidato = CLASES_ESTATICAS[idx]
                    conf = float(p[idx])
                    if candidato == "INICIO":
                        # Unica diferencia con demo_deletreo: INICIO abre la
                        # ventana de palabra (alli borraba; borrar queda en la
                        # tecla de retroceso). Solo cuenta quieto.
                        if mov < UMBRAL_MOVIMIENTO_GATE:
                            g, progreso = gesto.registrar(candidato, conf, ahora)
                            if g == "INICIO":
                                abrir_palabra(ahora, por_gesto=True)
                    else:
                        gesto.reiniciar()
                        confirmada, progreso = letras.registrar(candidato, conf, ahora)
                        if confirmada == "FIN":
                            texto.agregar_espacio()
                            flash_letra = "espacio"
                        elif confirmada == "REPOSO":
                            pass
                        elif confirmada is not None:
                            texto.agregar_letra(confirmada)
                            flash_letra = confirmada

                elif modo == MODO_PALABRA:
                    # Si se abrio con INICIO, la sena empieza cuando se SUELTAN
                    # las palmas: mientras el Modelo A siga viendo INICIO no se
                    # acumula nada (las palmas no son parte de la sena).
                    if esperando_soltar:
                        sigue_inicio = (
                            pred_a is not None
                            and CLASES_ESTATICAS[int(np.argmax(pred_a))] == "INICIO"
                        )
                        if (not sigue_inicio
                                or ahora - inicio_palabra_ms > ESPERA_SOLTAR_MS):
                            esperando_soltar = False
                    if not esperando_soltar:
                        buffer_b.append(
                            componer_vector_modelo_b(izq, der, marco_actual)
                        )
                    if pred_a is not None and mov < UMBRAL_MOVIMIENTO_GATE:
                        idx = int(np.argmax(pred_a))
                        if CLASES_ESTATICAS[idx] == "FIN":
                            g, _ = gesto.registrar("FIN", float(pred_a[idx]), ahora)
                            if g == "FIN":
                                cerrar_palabra(ahora, RECORTE_CIERRE_GESTO)
                        else:
                            gesto.reiniciar()
                    if modo == MODO_PALABRA and ahora - inicio_palabra_ms > MAX_PALABRA_MS:
                        cerrar_palabra(ahora, 0)
            else:
                sin_manos += 1
                buffer_a.clear()
                filtro_a.reiniciar()
                letras.reiniciar()
                gesto.reiniciar()
                if modo == MODO_PALABRA and sin_manos >= FRAMES_SIN_MANOS:
                    cerrar_palabra(ahora, RECORTE_CIERRE_MANOS)

            # Boton (clic) y ESPACIO: abrir o cerrar la ventana de palabra.
            tecla = cv2.waitKey(1) & 0xFF
            accion_boton = bool(clics) or tecla == 32
            clics.clear()
            if accion_boton:
                if modo == MODO_LETRAS:
                    abrir_palabra(ahora)
                else:
                    cerrar_palabra(ahora, RECORTE_CIERRE_BOTON)

            _ui(frame, modo, texto.cadena, candidato, progreso, len(buffer_b),
                esperando_soltar, flash_letra,
                aviso if ahora < aviso_hasta else None)
            cv2.imshow(ventana, frame)

            if tecla in (ord("q"), 27):
                break
            if tecla == ord("c"):
                texto.limpiar()
            if tecla == 8:
                texto.borrar_ultima()
            if cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        detector.cerrar()
        detector_pose.cerrar()
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------

def _ui(frame, modo, cadena, candidato, progreso, n_palabra, esperando,
        flash_letra, aviso):
    alto, ancho = frame.shape[:2]
    dibujo.panel(frame, _MARGEN, 18, 330, 46, alpha=0.55, radio=12)
    dibujo.texto(frame, "LESHO", _MARGEN + 18, 50, 0.82, dibujo.BLANCO, 2,
                 dibujo.FUENTE_TITULO)
    dibujo.texto(frame, "Traductor", _MARGEN + 120, 48, 0.56, dibujo.GRIS_CLARO, 1)

    x_der = ancho - _MARGEN
    if modo == MODO_PALABRA:
        if esperando:
            dibujo.chip_estado(frame, x_der, 36,
                               "Suelte las palmas y haga la sena", dibujo.AMBAR)
        else:
            dibujo.chip_estado(frame, x_der, 36,
                               f"GRABANDO SENA  {n_palabra} frames", dibujo.TEAL)
        dibujo.texto(frame, "Termine: baje las manos, FIN (dos punos), clic o ESPACIO",
                     _MARGEN + 18, 86, 0.55, dibujo.TEAL, 1)
    else:
        activo = progreso > 0 and candidato not in (None, "REPOSO")
        if activo:
            legibles = {"INICIO": "Abriendo palabra...", "FIN": "Espacio"}
            etiqueta = legibles.get(candidato, f"Detectando  {candidato}")
            dibujo.chip_estado(frame, x_der, 36, etiqueta, dibujo.AMBAR)
            tope = PERSISTENCIA_GESTO if candidato == "INICIO" else PERSISTENCIA
            dibujo.barra_progreso(frame, x_der - 200, 78, 200, 6,
                                  min(1.0, progreso / tope), color=dibujo.AMBAR)
        else:
            dibujo.chip_estado(frame, x_der, 36,
                               "Deletree, o abra una palabra: dos palmas, clic o ESPACIO",
                               dibujo.VERDE)

    # Panel del texto.
    h = 96
    y = alto - h - _MARGEN
    dibujo.panel(frame, _MARGEN, y, ancho - 2 * _MARGEN, h, alpha=0.62, radio=14)
    dibujo.texto(frame, "TEXTO", _MARGEN + 22, y + 30, 0.5, dibujo.GRIS_TENUE, 1)
    mostrar = cadena if cadena else "Firme letras o palabras"
    if len(mostrar) > 38:
        mostrar = "..." + mostrar[-38:]
    dibujo.texto(frame, mostrar, _MARGEN + 22, y + 72, 1.2,
                 dibujo.BLANCO if cadena else dibujo.GRIS_TENUE, 2,
                 dibujo.FUENTE_TITULO)
    dibujo.texto(frame,
                 "Dos punos: espacio   C limpiar   retroceso borrar   Q salir",
                 _MARGEN + 4, alto - 6, 0.5, dibujo.GRIS_TENUE, 1)

    # Avisos grandes al centro.
    if flash_letra is not None:
        color = dibujo.TEAL if flash_letra == "espacio" else dibujo.VERDE
        dibujo.texto_centrado(frame, flash_letra, ancho // 2, alto // 2 - 40,
                              3.0, color, 3, dibujo.FUENTE_TITULO)
    elif aviso is not None:
        txt, color = aviso
        escala = 2.4 if color == dibujo.VERDE else 1.1
        dibujo.texto_centrado(frame, txt, ancho // 2, alto // 2 - 40,
                              escala, color, 3 if color == dibujo.VERDE else 2,
                              dibujo.FUENTE_TITULO)


if __name__ == "__main__":
    main()
