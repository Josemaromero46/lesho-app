"""
Demo en vivo del Modelo B (senas dinamicas).

Abre la camara, detecta manos (Hands) y cuerpo (Pose), y clasifica una sena
dinamica completa. El usuario controla cuando empieza y termina la sena con la
barra espaciadora:

  ESPACIO  ->  empezar a grabar la sena / terminar y clasificar
  Q o ESC  ->  salir

Reproduce el mismo procesamiento que el entrenamiento (suavizado, escala solo de
las manos, remuestreo a longitud fija), asi que lo que ve el modelo en vivo es lo
mismo con lo que se entreno. Si solo se grabaron algunas de las 50 senas (mini
prueba), la prediccion se restringe a las clases que de verdad se entrenaron
(leidas del dataset), para que una clase sin datos no pueda ganar.

Uso (desde la carpeta training/):

    python demo/demo_senas.py
"""

import sys
import time
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
from comun.vector_modelo_b import componer_vector_modelo_b  # noqa: E402
from preprocessing.secuencias_b import procesar_secuencia  # noqa: E402
from capture import dibujo  # noqa: E402

MIN_FRAMES = 15          # minimo de fotogramas para intentar clasificar
MS_RESULTADO = 4000      # cuanto se muestra el resultado
_MARGEN = 24


def _asignar_manos(manos):
    """Separa las manos detectadas en (izquierda, derecha) por lateralidad."""
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
    """Indices de clase que de verdad tienen datos en el dataset (o None = todas).

    En una mini prueba solo se entrenan algunas de las 50; asi la prediccion no
    puede caer en una clase sin datos.
    """
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

    detector = DetectorLandmarks(
        max_manos=config.MAX_MANOS,
        confianza_deteccion=config.CONFIANZA_DETECCION_MANO,
    )
    detector_pose = DetectorPose(confianza_deteccion=config.CONFIANZA_DETECCION_MANO)

    cap = cv2.VideoCapture(config.INDICE_CAMARA)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.ANCHO_CAMARA)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.ALTO_CAMARA)
    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        sys.exit(1)

    ventana = "LESHO - Senas dinamicas"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana, config.ANCHO_CAMARA, config.ALTO_CAMARA)

    grabando = False
    buffer: list = []
    marco_actual = None
    resultado = None       # (etiqueta, confianza, top3)
    resultado_hasta = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)  # vista espejo (selfie), igual que la captura
            ahora = time.time() * 1000.0
            alto, ancho = frame.shape[:2]

            manos = detector.procesar(frame)
            puntos = detector_pose.procesar(frame)
            if puntos is not None:
                dibujo.dibujar_cuerpo(frame, puntos)
                m = marco_desde_puntos(puntos)
                if m is not None:
                    marco_actual = m
            if manos:
                for mano in manos:
                    dibujo.dibujar_mano(frame, mano.landmarks)

            if grabando and manos:
                izq, der = _asignar_manos(manos)
                buffer.append(componer_vector_modelo_b(izq, der, marco_actual))

            _componer_ui(frame, grabando, len(buffer), resultado,
                         resultado if ahora < resultado_hasta else None)
            cv2.imshow(ventana, frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == 32:  # ESPACIO: alterna grabar / clasificar
                if not grabando:
                    grabando = True
                    buffer = []
                    resultado = None
                else:
                    grabando = False
                    resultado = _clasificar(modelo, buffer, presentes)
                    resultado_hasta = ahora + MS_RESULTADO
            if cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        detector.cerrar()
        detector_pose.cerrar()
        cap.release()
        cv2.destroyAllWindows()


def _clasificar(modelo, buffer, presentes):
    """Procesa la secuencia grabada y devuelve (etiqueta, confianza, top3)."""
    if len(buffer) < MIN_FRAMES:
        return ("sena muy corta", 0.0, [])
    seq = procesar_secuencia(
        np.asarray(buffer, dtype=np.float32),
        config.LONGITUD_FIJA_SECUENCIA,
        suavizar=config.SUAVIZADO_ACTIVO, fps=config.FPS_OBJETIVO,
        min_cutoff=config.SUAVIZADO_MIN_CUTOFF, beta=config.SUAVIZADO_BETA,
        d_cutoff=config.SUAVIZADO_D_CUTOFF,
    )
    pred = modelo.predict(seq[None, ...], verbose=0)[0]
    # Restringir a las clases con datos (mini prueba): las demas no pueden ganar.
    if presentes is not None:
        mascara = np.full(len(pred), -1.0)
        mascara[presentes] = pred[presentes]
        pred = mascara
    orden = np.argsort(pred)[::-1][:3]
    top3 = [(CLASES_DINAMICAS[i], float(pred[i])) for i in orden]
    return (top3[0][0], top3[0][1], top3)


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------

def _componer_ui(frame, grabando, n_frames, resultado, resultado_visible):
    alto, ancho = frame.shape[:2]
    dibujo.panel(frame, _MARGEN, 18, 330, 46, alpha=0.55, radio=12)
    dibujo.texto(frame, "LESHO", _MARGEN + 18, 50, 0.82, dibujo.BLANCO, 2,
                 dibujo.FUENTE_TITULO)
    dibujo.texto(frame, "Senas dinamicas", _MARGEN + 120, 48, 0.56,
                 dibujo.GRIS_CLARO, 1)

    x_der = ancho - _MARGEN
    if grabando:
        dibujo.chip_estado(frame, x_der, 36, f"Grabando... {n_frames} frames",
                           dibujo.TEAL)
    else:
        dibujo.chip_estado(frame, x_der, 36, "ESPACIO para grabar una sena",
                           dibujo.VERDE)

    if resultado_visible is not None:
        etiqueta, conf, top3 = resultado_visible
        cy = alto // 2
        dibujo.texto_centrado(frame, etiqueta, ancho // 2, cy, 2.6, dibujo.VERDE,
                              3, dibujo.FUENTE_TITULO)
        if top3:
            dibujo.texto_centrado(frame, f"{conf*100:.0f}%", ancho // 2, cy + 46,
                                  0.9, dibujo.GRIS_CLARO, 2)
            y = cy + 90
            for nombre, c in top3[1:]:
                dibujo.texto_centrado(frame, f"{nombre}   {c*100:.0f}%",
                                      ancho // 2, y, 0.6, dibujo.GRIS_TENUE, 1)
                y += 28

    pista = "ESPACIO grabar / clasificar      Q salir"
    dibujo.texto(frame, pista, _MARGEN + 4, alto - 24, 0.55, dibujo.GRIS_TENUE, 1)


if __name__ == "__main__":
    main()
