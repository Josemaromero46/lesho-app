"""
Demo de frases con AUTO-SEGMENTACION por movimiento (senas dinamicas).

En vez de marcar cada palabra con INICIO/FIN, el sistema detecta solo cuando
empezas y cuando terminas una sena, por el movimiento de las manos:

  - Manos quietas o abajo        -> reposo (esperando).
  - Empiezan a moverse           -> empieza a acumular la sena.
  - Pausa o bajas las manos      -> termina, el Modelo B clasifica y ESCRIBE la
                                    palabra en la frase.
  - Repetis: otra palabra, otra pausa, se agrega la siguiente.

Es mas fluido que INICIO/FIN (no hay gestos extra) y reusa el mismo Modelo B.
A cambio, la deteccion de los limites es por movimiento, asi que conviene hacer
una PAUSA CLARA (o bajar las manos) entre palabra y palabra.

Respaldo con teclado: ESPACIO fuerza el fin de la sena actual y la clasifica.

Los umbrales de arriba (ONSET, OFFSET, ...) se pueden afinar en vivo.

Uso (desde la carpeta training/):

    python demo/demo_frases_auto.py

Teclas:  ESPACIO cerrar sena ahora   C limpiar frase   Q salir
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
    LATERALIDAD_DERECHA,
    LATERALIDAD_IZQUIERDA,
)
from comun.landmarks import DetectorLandmarks  # noqa: E402
from comun.pose import DetectorPose  # noqa: E402
from comun.marco import marco_desde_puntos  # noqa: E402
from comun.normalizacion import componer_vector_dos_manos, escalar_vector  # noqa: E402
from comun.suavizado import FiltroUnEuro  # noqa: E402
from comun.vector_modelo_b import componer_vector_modelo_b  # noqa: E402
from preprocessing.secuencias_b import procesar_secuencia  # noqa: E402
from preprocessing.ventanas import velocidad_media  # noqa: E402
from capture import dibujo  # noqa: E402

# --- Umbrales de auto-segmentacion (afinables en vivo) ----------------------
# Movimiento medido sobre la mano suavizada y escalada. Reposo real ~0.001-0.002;
# una palabra en plena ejecucion ~0.009. Con histeresis: ONSET > OFFSET.
ONSET = 0.0050            # empieza a acumular cuando el movimiento sube de aqui
OFFSET = 0.0028           # se considera pausa cuando baja de aqui
ONSET_FRAMES = 2          # fotogramas seguidos por encima de ONSET para arrancar
QUIET_FRAMES = 11         # fotogramas seguidos en pausa para cerrar (~0.5 s)
SIN_MANOS_FIN = 4         # si desaparecen las manos, tambien cierra la sena
MIN_FRAMES_B = 15         # minimo de fotogramas para clasificar
COOLDOWN_MS = 600         # pausa tras escribir una palabra
MS_FLASH = 1500
_MARGEN = 24

REPOSO = "reposo"
ACTIVO = "activo"


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
    if not config.RUTA_DATASET_B.exists():
        return None
    datos = np.load(config.RUTA_DATASET_B, allow_pickle=True)
    return sorted(set(int(v) for v in datos["y"]))


def main():
    if not config.RUTA_MODELO_B_KERAS.exists():
        print(f"No se encontro el modelo: {config.RUTA_MODELO_B_KERAS}")
        print("Entrene primero con: python models/entrenar_b.py")
        sys.exit(1)

    import tensorflow as tf
    modelo = tf.keras.models.load_model(config.RUTA_MODELO_B_KERAS)
    presentes = _clases_presentes()

    detector = DetectorLandmarks(max_manos=config.MAX_MANOS,
                                 confianza_deteccion=config.CONFIANZA_DETECCION_MANO)
    detector_pose = DetectorPose(confianza_deteccion=config.CONFIANZA_DETECCION_MANO)
    cap = cv2.VideoCapture(config.INDICE_CAMARA)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.ANCHO_CAMARA)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.ALTO_CAMARA)
    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        sys.exit(1)

    ventana = "LESHO - Frases (auto)"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana, config.ANCHO_CAMARA, config.ALTO_CAMARA)

    filtro_mov = FiltroUnEuro(config.FPS_OBJETIVO, config.SUAVIZADO_MIN_CUTOFF,
                              config.SUAVIZADO_BETA, config.SUAVIZADO_D_CUTOFF)
    mov_buffer = deque(maxlen=5)          # mano escalada, para medir movimiento
    pre_roll = deque(maxlen=6)            # ultimos 140-vectores (arranque de la sena)
    buffer_b: list = []
    marco_actual = None
    estado = REPOSO
    onset_cont = 0
    quiet_cont = 0
    sin_manos = 0
    cooldown = 0.0
    frase: list = []
    flash = None
    flash_hasta = 0.0

    def clasificar(recorte_final):
        seq = buffer_b[:-recorte_final] if recorte_final and len(buffer_b) > recorte_final else buffer_b
        if len(seq) < MIN_FRAMES_B:
            return None
        proc = procesar_secuencia(
            np.asarray(seq, dtype=np.float32), config.LONGITUD_FIJA_SECUENCIA,
            suavizar=config.SUAVIZADO_ACTIVO, fps=config.FPS_OBJETIVO,
            min_cutoff=config.SUAVIZADO_MIN_CUTOFF, beta=config.SUAVIZADO_BETA,
            d_cutoff=config.SUAVIZADO_D_CUTOFF,
        )
        pred = modelo.predict(proc[None, ...], verbose=0)[0]
        if presentes is not None:
            mm = np.full(len(pred), -1.0)
            mm[presentes] = pred[presentes]
            pred = mm
        return CLASES_DINAMICAS[int(np.argmax(pred))]

    def cerrar_sena(recorte, ahora):
        nonlocal estado, flash, flash_hasta, cooldown, buffer_b, quiet_cont
        palabra = clasificar(recorte)
        estado = REPOSO
        quiet_cont = 0
        buffer_b = []
        cooldown = ahora + COOLDOWN_MS
        if palabra:
            frase.append(palabra)
            flash = palabra
            flash_hasta = ahora + MS_FLASH

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            ahora = time.time() * 1000.0

            manos = detector.procesar(frame)
            puntos = detector_pose.procesar(frame)
            if puntos is not None:
                dibujo.dibujar_cuerpo(frame, puntos)
                m = marco_desde_puntos(puntos)
                if m is not None:
                    marco_actual = m

            movimiento = 0.0
            if manos:
                sin_manos = 0
                for mano in manos:
                    dibujo.dibujar_mano(frame, mano.landmarks)
                izq, der = _asignar_manos(manos)

                # Movimiento de la mano (suavizado + escalado).
                v_mano = escalar_vector(
                    filtro_mov.filtrar(componer_vector_dos_manos(izq, der)).tolist()
                )
                mov_buffer.append(v_mano)
                if len(mov_buffer) >= 2:
                    movimiento = velocidad_media(np.asarray(mov_buffer, dtype=np.float32))

                # Vector del Modelo B (config trasladada + ubicacion).
                vec_b = componer_vector_modelo_b(izq, der, marco_actual)
                pre_roll.append(vec_b)
                if estado == ACTIVO:
                    buffer_b.append(vec_b)

                if estado == REPOSO and ahora >= cooldown:
                    onset_cont = onset_cont + 1 if movimiento > ONSET else 0
                    if onset_cont >= ONSET_FRAMES:
                        estado = ACTIVO
                        buffer_b = list(pre_roll)   # incluye el arranque de la sena
                        onset_cont = 0
                        quiet_cont = 0
                elif estado == ACTIVO:
                    quiet_cont = quiet_cont + 1 if movimiento < OFFSET else 0
                    if quiet_cont >= QUIET_FRAMES:
                        cerrar_sena(QUIET_FRAMES, ahora)   # descarta la pausa final
            else:
                sin_manos += 1
                mov_buffer.clear()
                filtro_mov.reiniciar()
                onset_cont = 0
                if estado == ACTIVO and sin_manos >= SIN_MANOS_FIN:
                    cerrar_sena(0, ahora)   # bajaste las manos: fin de la sena

            _ui(frame, estado, len(buffer_b), movimiento, frase,
                flash if ahora < flash_hasta else None)
            cv2.imshow(ventana, frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == ord("c"):
                frase.clear()
            if tecla == 32 and estado == ACTIVO:   # ESPACIO: cerrar ahora
                cerrar_sena(0, ahora)
            if cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        detector.cerrar()
        detector_pose.cerrar()
        cap.release()
        cv2.destroyAllWindows()


def _ui(frame, estado, n_b, movimiento, frase, flash):
    alto, ancho = frame.shape[:2]
    dibujo.panel(frame, _MARGEN, 18, 360, 46, alpha=0.55, radio=12)
    dibujo.texto(frame, "LESHO", _MARGEN + 18, 50, 0.82, dibujo.BLANCO, 2,
                 dibujo.FUENTE_TITULO)
    dibujo.texto(frame, "Frases (auto)", _MARGEN + 120, 48, 0.56, dibujo.GRIS_CLARO, 1)
    dibujo.texto(frame, f"mov {movimiento:.4f}", _MARGEN + 18, 86, 0.5,
                 dibujo.GRIS_CLARO, 1)

    x_der = ancho - _MARGEN
    if estado == ACTIVO:
        dibujo.chip_estado(frame, x_der, 36, f"Sena en curso... {n_b} frames",
                           dibujo.TEAL)
    else:
        dibujo.chip_estado(frame, x_der, 36, "Reposo: empiece a senar",
                           dibujo.VERDE)

    h = 92
    y = alto - h - _MARGEN
    dibujo.panel(frame, _MARGEN, y, ancho - 2 * _MARGEN, h, alpha=0.62, radio=14)
    dibujo.texto(frame, "FRASE", _MARGEN + 22, y + 30, 0.5, dibujo.GRIS_TENUE, 1)
    texto = " ".join(frase) if frase else "(vacia)"
    if len(texto) > 40:
        texto = "..." + texto[-40:]
    dibujo.texto(frame, texto, _MARGEN + 22, y + 70, 1.1,
                 dibujo.BLANCO if frase else dibujo.GRIS_TENUE, 2, dibujo.FUENTE_TITULO)
    dibujo.texto(frame, "Pausa o baje las manos entre palabras   ESPACIO cerrar   C limpiar   Q salir",
                 _MARGEN + 4, alto - 6, 0.5, dibujo.GRIS_TENUE, 1)

    if flash is not None:
        dibujo.texto_centrado(frame, flash, ancho // 2, alto // 2 - 30, 3.0,
                              dibujo.VERDE, 3, dibujo.FUENTE_TITULO)


if __name__ == "__main__":
    main()
