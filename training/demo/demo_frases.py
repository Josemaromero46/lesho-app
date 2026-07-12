"""
Demo integrado: senas dinamicas encadenadas con INICIO / FIN.

Reproduce el flujo real de la Direccion 1 para palabras completas:

  1. Se hace la sena de INICIO (dos palmas abiertas)  -> abre la ventana de
     grabacion dinamica.
  2. Se hace UNA sena dinamica (una palabra).
  3. Se hace la sena de FIN (dos punos cerrados)      -> cierra la ventana, el
     Modelo B clasifica la sena y la palabra se escribe en la frase.
  4. Se repite: INICIO -> otra sena -> FIN agrega la siguiente palabra.

INICIO y FIN se detectan con el modelo del alfabeto (Modelo A, que tiene esas dos
clases). La palabra se reconoce con el Modelo B. Reconocer varias senas seguidas
SIN marcadores es reconocimiento continuo, que esta fuera de alcance: por eso va
una sena por ventana INICIO/FIN.

Respaldo con teclado: ESPACIO empieza/termina la ventana igual que INICIO/FIN.

Uso (desde la carpeta training/):

    python demo/demo_frases.py

Teclas:  ESPACIO abrir/cerrar ventana   C limpiar frase   Q salir
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
    TAMANO_VENTANA_A,
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

# --- Parametros -------------------------------------------------------------
PERSISTENCIA_GESTO = 6      # ventanas seguidas para confirmar INICIO o FIN
UMBRAL_GESTO = 0.85         # confianza minima de INICIO/FIN (alta: evita falsos)
COOLDOWN_MS = 800           # pausa tras confirmar un gesto (bajar/subir manos)
MIN_FRAMES_B = 15           # minimo de fotogramas para clasificar una sena
FRAMES_FIN = 12             # fotogramas del final que se descartan (pose de FIN)
MS_FLASH = 1500

UMBRAL_MOVIMIENTO_GATE = config.UMBRAL_MOVIMIENTO_ABS
UMBRAL_MOVIMIENTO_MOVIENDO = config.UMBRAL_MOVIMIENTO_MOVIENDO
_MARGEN = 24

ESPERANDO = "esperando"
GRABANDO = "grabando"


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
    """Indices de clase con datos en el dataset (para no predecir clases vacias)."""
    if not config.RUTA_DATASET_B.exists():
        return None
    datos = np.load(config.RUTA_DATASET_B, allow_pickle=True)
    return sorted(set(int(v) for v in datos["y"]))


def main():
    for ruta, cual in [(config.RUTA_MODELO_A_KERAS, "A (alfabeto/INICIO-FIN)"),
                       (config.RUTA_MODELO_B_KERAS, "B (senas)")]:
        if not ruta.exists():
            print(f"Falta el modelo {cual}: {ruta}")
            sys.exit(1)

    import tensorflow as tf
    modelo_a = tf.keras.models.load_model(config.RUTA_MODELO_A_KERAS)
    modelo_b = tf.keras.models.load_model(config.RUTA_MODELO_B_KERAS)
    presentes = _clases_presentes()

    idx_inicio = CLASES_ESTATICAS.index("INICIO")
    idx_fin = CLASES_ESTATICAS.index("FIN")
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

    ventana = "LESHO - Frases (INICIO / FIN)"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana, config.ANCHO_CAMARA, config.ALTO_CAMARA)

    buffer_a = deque(maxlen=TAMANO_VENTANA_A)   # ventana rodante del Modelo A
    buffer_b: list = []                          # secuencia acumulada del Modelo B
    filtro_a = FiltroUnEuro(config.FPS_OBJETIVO, config.SUAVIZADO_MIN_CUTOFF,
                            config.SUAVIZADO_BETA, config.SUAVIZADO_D_CUTOFF)
    marco_actual = None
    estado = ESPERANDO
    gesto_cont = 0
    fin_cooldown = 0.0
    frase: list = []
    flash = None
    flash_hasta = 0.0

    def clasificar_sena():
        seq = buffer_b[:-FRAMES_FIN] if len(buffer_b) > FRAMES_FIN + MIN_FRAMES_B else buffer_b
        if len(seq) < MIN_FRAMES_B:
            return None
        proc = procesar_secuencia(
            np.asarray(seq, dtype=np.float32), config.LONGITUD_FIJA_SECUENCIA,
            suavizar=config.SUAVIZADO_ACTIVO, fps=config.FPS_OBJETIVO,
            min_cutoff=config.SUAVIZADO_MIN_CUTOFF, beta=config.SUAVIZADO_BETA,
            d_cutoff=config.SUAVIZADO_D_CUTOFF,
        )
        pred = modelo_b.predict(proc[None, ...], verbose=0)[0]
        if presentes is not None:
            m = np.full(len(pred), -1.0)
            m[presentes] = pred[presentes]
            pred = m
        return CLASES_DINAMICAS[int(np.argmax(pred))]

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

            candidato = None
            if manos:
                for mano in manos:
                    dibujo.dibujar_mano(frame, mano.landmarks)
                izq, der = _asignar_manos(manos)

                # Modelo A: vector trasladado + suavizado + escala -> ventana rodante.
                trasladado = componer_vector_dos_manos(izq, der)
                if config.SUAVIZADO_ACTIVO:
                    trasladado = filtro_a.filtrar(trasladado).tolist()
                buffer_a.append(escalar_vector(trasladado))

                # Modelo B: acumular la sena mientras la ventana esta abierta.
                if estado == GRABANDO:
                    buffer_b.append(componer_vector_modelo_b(izq, der, marco_actual))

                # Deteccion de INICIO / FIN con el Modelo A.
                if len(buffer_a) == TAMANO_VENTANA_A and ahora >= fin_cooldown:
                    entrada = np.asarray(buffer_a, dtype=np.float32)[None, ...]
                    pred = modelo_a(entrada, training=False).numpy()[0]
                    mov = velocidad_media(entrada[0])
                    if mov < UMBRAL_MOVIMIENTO_GATE:
                        pred[idx_movimiento] = 0.0
                    elif mov > UMBRAL_MOVIMIENTO_MOVIENDO:
                        pred[idx_gemelas] = 0.0
                    idx = int(np.argmax(pred))
                    candidato = CLASES_ESTATICAS[idx]
                    objetivo = idx_inicio if estado == ESPERANDO else idx_fin
                    if idx == objetivo and pred[idx] >= UMBRAL_GESTO:
                        gesto_cont += 1
                    else:
                        gesto_cont = 0
                    if gesto_cont >= PERSISTENCIA_GESTO:
                        gesto_cont = 0
                        fin_cooldown = ahora + COOLDOWN_MS
                        if estado == ESPERANDO:
                            estado = GRABANDO
                            buffer_b = []
                        else:
                            palabra = clasificar_sena()
                            estado = ESPERANDO
                            if palabra:
                                frase.append(palabra)
                                flash = palabra
                                flash_hasta = ahora + MS_FLASH
            else:
                buffer_a.clear()
                filtro_a.reiniciar()
                gesto_cont = 0

            _ui(frame, estado, len(buffer_b), frase,
                flash if ahora < flash_hasta else None, candidato)
            cv2.imshow(ventana, frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == ord("c"):
                frase.clear()
            if tecla == 32:  # ESPACIO: respaldo manual de INICIO/FIN
                if estado == ESPERANDO:
                    estado = GRABANDO
                    buffer_b = []
                else:
                    palabra = clasificar_sena()
                    estado = ESPERANDO
                    if palabra:
                        frase.append(palabra)
                        flash = palabra
                        flash_hasta = ahora + MS_FLASH
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

def _ui(frame, estado, n_b, frase, flash, candidato):
    alto, ancho = frame.shape[:2]
    dibujo.panel(frame, _MARGEN, 18, 360, 46, alpha=0.55, radio=12)
    dibujo.texto(frame, "LESHO", _MARGEN + 18, 50, 0.82, dibujo.BLANCO, 2,
                 dibujo.FUENTE_TITULO)
    dibujo.texto(frame, "Frases  INICIO / FIN", _MARGEN + 120, 48, 0.56,
                 dibujo.GRIS_CLARO, 1)

    x_der = ancho - _MARGEN
    if estado == GRABANDO:
        dibujo.chip_estado(frame, x_der, 36,
                           f"Grabando sena... {n_b} frames  (FIN o ESPACIO)",
                           dibujo.TEAL)
    else:
        dibujo.chip_estado(frame, x_der, 36,
                           "Haga INICIO (dos palmas) o ESPACIO", dibujo.VERDE)

    # Panel de la frase, abajo.
    h = 92
    y = alto - h - _MARGEN
    dibujo.panel(frame, _MARGEN, y, ancho - 2 * _MARGEN, h, alpha=0.62, radio=14)
    dibujo.texto(frame, "FRASE", _MARGEN + 22, y + 30, 0.5, dibujo.GRIS_TENUE, 1)
    texto = " ".join(frase) if frase else "(vacia)"
    if len(texto) > 40:
        texto = "..." + texto[-40:]
    dibujo.texto(frame, texto, _MARGEN + 22, y + 70, 1.1,
                 dibujo.BLANCO if frase else dibujo.GRIS_TENUE, 2, dibujo.FUENTE_TITULO)
    dibujo.texto(frame, "ESPACIO abrir/cerrar   C limpiar   Q salir",
                 _MARGEN + 4, alto - 6, 0.5, dibujo.GRIS_TENUE, 1)

    if flash is not None:
        dibujo.texto_centrado(frame, flash, ancho // 2, alto // 2 - 30, 3.0,
                              dibujo.VERDE, 3, dibujo.FUENTE_TITULO)


if __name__ == "__main__":
    main()
