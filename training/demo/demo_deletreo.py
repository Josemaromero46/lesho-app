"""
Demo en tiempo real del deletreo LESHO (Modelo A).

Abre la camara, detecta las manos con MediaPipe, arma la ventana rodante de
fotogramas y clasifica las letras del alfabeto en vivo, formando texto en
pantalla. Reproduce la misma logica que tendra la app: ventana temporal, filtro
de persistencia y periodo de enfriamiento.

Controles por gesto (dos manos, no chocan con las letras de una mano):
  - FIN   (dos puños cerrados)  -> inserta un espacio
  - INICIO (dos palmas abiertas) -> borra la ultima letra
  - REPOSO / sin mano            -> no hace nada (descanso entre letras)

Controles de teclado (operador):
  - C  limpia todo el texto
  - retroceso  borra la ultima letra
  - Q o ESC  salir

Uso (desde la carpeta training/):

    python demo/demo_deletreo.py
"""

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# Permite ejecutar el archivo directamente: agrega training/ al path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from comun.definiciones import (  # noqa: E402
    CLASES_ESTATICAS,
    LATERALIDAD_DERECHA,
    LATERALIDAD_IZQUIERDA,
    LETRAS_CON_MOVIMIENTO,
    SENAS_CONTROL,
    TAMANO_VENTANA_A,
)
from comun.landmarks import DetectorLandmarks  # noqa: E402
from comun.normalizacion import componer_vector_dos_manos, escalar_vector  # noqa: E402
from comun.suavizado import FiltroUnEuro  # noqa: E402
from preprocessing.ventanas import velocidad_media  # noqa: E402
from capture import dibujo  # noqa: E402


# --- Parametros del reconocimiento (afinables para la demo) -----------------
PERSISTENCIA = 5           # ventanas consecutivas para confirmar una clase
UMBRAL_CONFIANZA = 0.60    # confianza minima para contar hacia la persistencia
COOLDOWN_MS = 1200         # pausa tras escribir una letra antes de volver a
                           # escanear. Da tiempo a ver la letra escrita, reaccionar
                           # y formar la siguiente. Permite repetir la misma letra.
MS_FLASH = 550             # duracion del aviso grande al confirmar

# Una letra de movimiento (J, Ñ, Z, LL, RR) no se puede confirmar si la ventana
# esta quieta. Evita que una L quieta se lea como LL, o una N como Ñ. Si al hacer
# LL o Ñ cuesta que salgan, bajar un poco este valor; si una L quieta todavia
# salta a LL, subirlo.
UMBRAL_MOVIMIENTO_GATE = config.UMBRAL_MOVIMIENTO_ABS

# Sentido inverso: con la mano en movimiento claro, una letra estatica gemela de
# una de movimiento (N/Ñ, L/LL, R/RR) no puede ganar. Evita que en medio del
# vaiven de una Ñ se cuele una N. Si al mover cuesta que salga Ñ/LL/RR y aparece
# la estatica, bajar este valor; si una estatica sostenida se bloquea, subirlo.
UMBRAL_MOVIMIENTO_MOVIENDO = config.UMBRAL_MOVIMIENTO_MOVIENDO


def _asignar_manos(manos):
    """Separa las manos detectadas en (izquierda, derecha) por lateralidad.

    Las manos con lateralidad desconocida o repetida caen en el primer slot
    libre, de modo que una sena a una mano siempre produce datos.
    """
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


def _landmarks_suaves(filtros: dict, clave: str, landmarks: list) -> list:
    """Suaviza los landmarks CRUDOS de una mano para dibujarla sin temblor.

    Es solo para lo que se ve en pantalla; se filtra cada mano por su slot
    (izquierda/derecha) con un filtro propio que guarda su estado.
    """
    filtro = filtros.get(clave)
    if filtro is None:
        filtro = FiltroUnEuro(
            config.FPS_OBJETIVO, config.SUAVIZADO_MIN_CUTOFF,
            config.SUAVIZADO_BETA, config.SUAVIZADO_D_CUTOFF,
        )
        filtros[clave] = filtro
    plano = np.asarray(landmarks, dtype=np.float64).reshape(-1)
    suave = filtro.filtrar(plano, preservar_ceros=False)
    return [(float(x), float(y), float(z)) for x, y, z in suave.reshape(-1, 3)]


class Deletreador:
    """Estado del texto y confirmacion de clases (persistencia + cooldown)."""

    def __init__(self):
        self.texto: list[str] = []
        self._clase = None
        self._contador = 0
        self._fin_cooldown = 0.0
        self.flash = None
        self._flash_hasta = 0.0

    def registrar(self, clase: str, confianza: float, ahora_ms: float) -> int:
        """Actualiza la persistencia. Devuelve el conteo actual del candidato."""
        if ahora_ms < self._fin_cooldown or confianza < UMBRAL_CONFIANZA:
            self._clase = None
            self._contador = 0
            return 0
        if clase == self._clase:
            self._contador += 1
        else:
            self._clase = clase
            self._contador = 1
        if self._contador >= PERSISTENCIA:
            self._confirmar(clase, ahora_ms)
            self._clase = None
            self._contador = 0
            self._fin_cooldown = ahora_ms + COOLDOWN_MS
        return self._contador

    def reiniciar_candidato(self):
        self._clase = None
        self._contador = 0

    def _confirmar(self, clase: str, ahora_ms: float):
        if clase == "REPOSO":
            return
        if clase == "FIN":
            if self.texto and self.texto[-1] != " ":
                self.texto.append(" ")
            self.flash = "espacio"
        elif clase == "INICIO":
            if self.texto:
                self.texto.pop()
            self.flash = "borrar"
        else:
            self.texto.append(clase)
            self.flash = clase
        self._flash_hasta = ahora_ms + MS_FLASH

    def borrar_ultima(self):
        if self.texto:
            self.texto.pop()

    def limpiar(self):
        self.texto.clear()

    def flash_activo(self, ahora_ms: float):
        return self.flash if ahora_ms < self._flash_hasta else None

    @property
    def cadena(self) -> str:
        return "".join(self.texto)


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------

_MARGEN = 24


def _barra_superior(frame, ancho):
    dibujo.panel(frame, _MARGEN, 18, 320, 46, alpha=0.55, radio=12)
    dibujo.texto(frame, "LESHO", _MARGEN + 18, 50, 0.82, dibujo.BLANCO, 2,
                 dibujo.FUENTE_TITULO)
    dibujo.texto(frame, "Deletreo en tiempo real", _MARGEN + 120, 48, 0.56,
                 dibujo.GRIS_CLARO, 1)


def _chip_deteccion(frame, ancho, candidato, progreso, num_manos):
    x_der = ancho - _MARGEN
    # Solo se muestra "Detectando" cuando de verdad avanza hacia la confirmacion.
    activo = progreso > 0 and candidato is not None and candidato != "REPOSO"
    es_control = candidato in SENAS_CONTROL
    if activo and es_control:
        legibles = {"INICIO": "Borrar", "FIN": "Espacio"}
        dibujo.chip_estado(frame, x_der, 22, legibles.get(candidato, candidato),
                           dibujo.TEAL)
    elif activo:
        dibujo.chip_estado(frame, x_der, 22, f"Detectando  {candidato}",
                           dibujo.AMBAR)
    elif num_manos == 0:
        dibujo.chip_estado(frame, x_der, 22, "Acerque la mano", dibujo.GRIS_TENUE)
    else:
        dibujo.chip_estado(frame, x_der, 22, "Listo", dibujo.VERDE)

    # Barra fina de avance de la persistencia, bajo el chip.
    if activo:
        w = 200
        color = dibujo.TEAL if es_control else dibujo.AMBAR
        dibujo.barra_progreso(frame, x_der - w, 66, w, 6,
                              progreso / PERSISTENCIA, color=color)


def _panel_texto(frame, ancho, alto, cadena):
    h = 150
    y = alto - h - _MARGEN
    w = ancho - 2 * _MARGEN
    dibujo.panel(frame, _MARGEN, y, w, h, alpha=0.62, radio=16)

    dibujo.texto(frame, "TEXTO", _MARGEN + 28, y + 36, 0.5, dibujo.GRIS_TENUE, 1)

    # Muestra la cola del texto si es muy largo, para que siempre quepa.
    mostrar = cadena if len(cadena) <= 26 else "..." + cadena[-26:]
    if not mostrar:
        dibujo.texto(frame, "Comience a deletrear", _MARGEN + 28, y + 100, 1.1,
                     dibujo.GRIS_TENUE, 1, dibujo.FUENTE_TITULO)
    else:
        dibujo.texto(frame, mostrar, _MARGEN + 28, y + 104, 1.7, dibujo.BLANCO, 2,
                     dibujo.FUENTE_TITULO)

    pista = ("Dos puños  espacio      Dos palmas  borrar      "
             "C  limpiar      Q  salir")
    dibujo.texto(frame, pista, _MARGEN + 28, y + h - 16, 0.48, dibujo.GRIS_TENUE, 1)


def _flash(frame, ancho, alto, flash):
    if flash is None:
        return
    legibles = {"espacio": "espacio", "borrar": "borrar"}
    etiqueta = legibles.get(flash, flash)
    color = dibujo.TEAL if flash in legibles else dibujo.VERDE
    cx, cy = ancho // 2, alto // 2 - 40
    dibujo.texto_centrado(frame, etiqueta, cx, cy, 3.4, color, 3,
                          dibujo.FUENTE_TITULO)


def _lectura_movimiento(frame, movimiento):
    """Lector de diagnostico: muestra el movimiento de la ventana y si la
    compuerta lo considera quieto (bloquea letras de movimiento) o en movimiento.
    """
    quieto = movimiento < UMBRAL_MOVIMIENTO_GATE
    color = dibujo.GRIS_CLARO if quieto else dibujo.AMBAR
    etiqueta = "quieto" if quieto else "movimiento"
    dibujo.texto(frame, f"mov {movimiento:.4f}   {etiqueta}", _MARGEN + 18, 88,
                 0.5, color, 1)


def componer_ui(frame, cadena, candidato, progreso, num_manos, flash,
                manos_dibujo, movimiento):
    alto, ancho = frame.shape[:2]
    for mano in manos_dibujo:
        dibujo.dibujar_mano(frame, mano)
    _barra_superior(frame, ancho)
    _lectura_movimiento(frame, movimiento)
    _chip_deteccion(frame, ancho, candidato, progreso, num_manos)
    _panel_texto(frame, ancho, alto, cadena)
    _flash(frame, ancho, alto, flash)
    return frame


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------

def main():
    if not config.RUTA_MODELO_A_KERAS.exists():
        print(f"No se encontro el modelo: {config.RUTA_MODELO_A_KERAS}")
        print("Entrene primero con: python models/entrenar_a.py")
        sys.exit(1)

    import tensorflow as tf
    modelo = tf.keras.models.load_model(config.RUTA_MODELO_A_KERAS)

    detector = DetectorLandmarks(
        max_manos=config.MAX_MANOS,
        confianza_deteccion=config.CONFIANZA_DETECCION_MANO,
    )
    cap = cv2.VideoCapture(config.INDICE_CAMARA)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.ANCHO_CAMARA)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.ALTO_CAMARA)
    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        sys.exit(1)

    ventana_nombre = "LESHO - Deletreo"
    cv2.namedWindow(ventana_nombre, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana_nombre, config.ANCHO_CAMARA, config.ALTO_CAMARA)

    idx_movimiento = np.array(
        [CLASES_ESTATICAS.index(letra) for letra in LETRAS_CON_MOVIMIENTO]
    )
    idx_gemelas = np.array(
        [CLASES_ESTATICAS.index(letra)
         for letra in config.ESTATICAS_GEMELAS_DE_MOVIMIENTO]
    )
    buffer = deque(maxlen=TAMANO_VENTANA_A)
    estado = Deletreador()

    # Filtro One Euro para lo que ve el MODELO (vector trasladado), igual que en
    # el preprocesamiento. Los filtros de display suavizan cada mano dibujada.
    filtro_vector = FiltroUnEuro(
        config.FPS_OBJETIVO, config.SUAVIZADO_MIN_CUTOFF,
        config.SUAVIZADO_BETA, config.SUAVIZADO_D_CUTOFF,
    )
    filtros_display: dict = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)  # vista espejo (selfie), igual que la captura
            ahora = time.time() * 1000.0

            manos = detector.procesar(frame)
            candidato = None
            progreso = 0
            manos_dibujo = []
            num_manos = 0
            movimiento = 0.0

            if manos:
                num_manos = len(manos)
                izquierda, derecha = _asignar_manos(manos)

                # Traslacion (respecto a la muneca).
                trasladado = componer_vector_dos_manos(izquierda, derecha)
                if config.SUAVIZADO_ACTIVO:
                    # Suavizado del vector del MODELO, igual que en el preproceso.
                    trasladado = filtro_vector.filtrar(trasladado).tolist()
                    # Suavizado de cada mano para DIBUJARLA sin temblor.
                    if izquierda is not None:
                        izquierda = _landmarks_suaves(filtros_display, "izq", izquierda)
                    if derecha is not None:
                        derecha = _landmarks_suaves(filtros_display, "der", derecha)
                manos_dibujo = [m for m in (izquierda, derecha) if m is not None]

                # Escala (invariante a la distancia) y a la ventana rodante.
                buffer.append(escalar_vector(trasladado))
                if len(buffer) == TAMANO_VENTANA_A:
                    entrada = np.asarray(buffer, dtype=np.float32)[None, ...]
                    pred = modelo(entrada, training=False).numpy()[0]
                    movimiento = velocidad_media(entrada[0])
                    # Compuerta de movimiento (dos sentidos):
                    #  - Ventana quieta: se bloquean las letras de movimiento
                    #    (una N quieta no es Ñ).
                    #  - Movimiento claro: se bloquean las estaticas gemelas de
                    #    una de movimiento (en pleno vaiven de una Ñ no puede
                    #    ganar la N). Entre ambos umbrales decide el modelo.
                    if movimiento < UMBRAL_MOVIMIENTO_GATE:
                        pred[idx_movimiento] = 0.0
                    elif movimiento > UMBRAL_MOVIMIENTO_MOVIENDO:
                        pred[idx_gemelas] = 0.0
                    idx = int(np.argmax(pred))
                    candidato = CLASES_ESTATICAS[idx]
                    progreso = estado.registrar(candidato, float(pred[idx]), ahora)
            else:
                buffer.clear()
                estado.reiniciar_candidato()
                # La mano desaparecio: los filtros arrancan limpios al volver,
                # para no interpolar un salto entre la ultima pose y la nueva.
                filtro_vector.reiniciar()
                filtros_display.clear()

            componer_ui(frame, estado.cadena, candidato, progreso, num_manos,
                        estado.flash_activo(ahora), manos_dibujo, movimiento)
            cv2.imshow(ventana_nombre, frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == ord("c"):
                estado.limpiar()
            if tecla == 8:  # retroceso
                estado.borrar_ultima()
            if cv2.getWindowProperty(ventana_nombre, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        detector.cerrar()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
