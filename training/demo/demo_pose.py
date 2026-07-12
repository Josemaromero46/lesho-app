"""
Demo del marco de referencia del cuerpo (MediaPipe Pose) para el Modelo B.

Abre la camara, detecta el cuerpo con Pose y las manos con Hands, y muestra en
vivo lo que el Modelo B usara para ubicar cada sena: el centro de los hombros
(origen), el ancho de los hombros (escala) y la posicion de cada muneca expresada
en "anchos de hombro" respecto al centro del cuerpo. Tambien muestra una zona
aproximada (cabeza, cara, pecho, abdomen), a modo ilustrativo: las zonas exactas
se definiran con las 50 senas y la asesoria LESHO.

Sirve para verificar que Pose corre y que el marco del cuerpo es estable e
invariante a la distancia a la camara.

Uso (desde la carpeta training/):

    python demo/demo_pose.py

Teclas:  Q o ESC  salir
"""

import sys
from pathlib import Path

import cv2

# Permite ejecutar el archivo directamente: agrega training/ al path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from comun.definiciones import (  # noqa: E402
    POSE_CADERA_DER,
    POSE_CADERA_IZQ,
    POSE_HOMBRO_DER,
    POSE_HOMBRO_IZQ,
    POSE_NARIZ,
    INDICE_MUNECA,
)
from comun.landmarks import DetectorLandmarks  # noqa: E402
from comun.pose import DetectorPose, marco_desde_puntos  # noqa: E402
from capture import dibujo  # noqa: E402

# Codos, para dibujar los brazos.
POSE_CODO_IZQ = 13
POSE_CODO_DER = 14

_MARGEN = 24


def _zona_ilustrativa(ry: float) -> str:
    """Zona vertical aproximada segun la altura de la mano en anchos de hombro.

    ry es la posicion vertical respecto al centro de los hombros: negativo arriba
    (cara/cabeza), positivo abajo (abdomen). Es solo ilustrativo.
    """
    if ry < -1.5:
        return "cabeza"
    if ry < -0.6:
        return "cara / cuello"
    if ry < 0.5:
        return "pecho"
    if ry < 1.4:
        return "abdomen"
    return "espacio bajo"


def _px(punto, ancho, alto):
    """Pasa un landmark normalizado a pixeles enteros."""
    return int(punto.x * ancho), int(punto.y * alto)


def _dibujar_cuerpo(frame, puntos):
    """Dibuja el esqueleto minimo del cuerpo (hombros, brazos, tronco)."""
    alto, ancho = frame.shape[:2]
    hi = _px(puntos[POSE_HOMBRO_IZQ], ancho, alto)
    hd = _px(puntos[POSE_HOMBRO_DER], ancho, alto)
    ci = _px(puntos[POSE_CADERA_IZQ], ancho, alto)
    cd = _px(puntos[POSE_CADERA_DER], ancho, alto)
    codo_i = _px(puntos[POSE_CODO_IZQ], ancho, alto)
    codo_d = _px(puntos[POSE_CODO_DER], ancho, alto)
    nariz = _px(puntos[POSE_NARIZ], ancho, alto)

    # Tronco
    for a, b in [(hi, hd), (hd, cd), (cd, ci), (ci, hi), (hi, codo_i), (hd, codo_d)]:
        cv2.line(frame, a, b, dibujo.GRIS_CLARO, 2, cv2.LINE_AA)
    for p in (hi, hd, ci, cd, codo_i, codo_d, nariz):
        cv2.circle(frame, p, 5, dibujo.TEAL, -1, cv2.LINE_AA)

    # Centro de los hombros (origen del marco)
    centro = ((hi[0] + hd[0]) // 2, (hi[1] + hd[1]) // 2)
    cv2.circle(frame, centro, 7, dibujo.AMBAR, -1, cv2.LINE_AA)
    cv2.circle(frame, centro, 12, dibujo.AMBAR, 1, cv2.LINE_AA)
    return centro


def _muneca_px(landmarks_mano, ancho, alto):
    """Muneca (landmark 0) de una mano de Hands, en pixeles."""
    m = landmarks_mano[INDICE_MUNECA]
    # Los landmarks de Hands vienen como tuplas (x, y, z) normalizadas.
    return int(m[0] * ancho), int(m[1] * alto), m[0], m[1]


def main():
    detector_pose = DetectorPose(
        confianza_deteccion=config.CONFIANZA_DETECCION_MANO,
    )
    detector_manos = DetectorLandmarks(
        max_manos=config.MAX_MANOS,
        confianza_deteccion=config.CONFIANZA_DETECCION_MANO,
    )
    cap = cv2.VideoCapture(config.INDICE_CAMARA)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.ANCHO_CAMARA)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.ALTO_CAMARA)
    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        sys.exit(1)

    ventana = "LESHO - Marco del cuerpo (Pose)"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana, config.ANCHO_CAMARA, config.ALTO_CAMARA)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)  # vista espejo (selfie), igual que la captura
            alto, ancho = frame.shape[:2]

            puntos = detector_pose.procesar(frame)
            manos = detector_manos.procesar(frame)

            dibujo.panel(frame, _MARGEN, 18, 360, 46, alpha=0.55, radio=12)
            dibujo.texto(frame, "LESHO", _MARGEN + 18, 50, 0.82, dibujo.BLANCO, 2,
                         dibujo.FUENTE_TITULO)
            dibujo.texto(frame, "Marco del cuerpo (Pose)", _MARGEN + 120, 48, 0.56,
                         dibujo.GRIS_CLARO, 1)

            if puntos is None:
                dibujo.texto_centrado(frame, "Acerque el torso a la camara",
                                      ancho // 2, alto // 2, 1.0, dibujo.GRIS_TENUE,
                                      2, dibujo.FUENTE_TITULO)
            else:
                _dibujar_cuerpo(frame, puntos)
                marco = marco_desde_puntos(puntos)
                if marco is None:
                    dibujo.texto(frame, "Hombros poco visibles", _MARGEN + 18, 88,
                                 0.55, dibujo.AMBAR, 1)
                else:
                    dibujo.texto(frame,
                                 f"ancho de hombros: {marco.ancho:.3f}",
                                 _MARGEN + 18, 88, 0.5, dibujo.GRIS_CLARO, 1)
                    _dibujar_manos_relativas(frame, manos, marco, ancho, alto)

            cv2.imshow(ventana, frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        detector_pose.cerrar()
        detector_manos.cerrar()
        cap.release()
        cv2.destroyAllWindows()


def _dibujar_manos_relativas(frame, manos, marco, ancho, alto):
    """Por cada mano, dibuja la muneca, su posicion relativa al cuerpo y la zona."""
    if not manos:
        return
    centro_px = (int(marco.centro_x * ancho), int(marco.centro_y * alto))
    for mano in manos:
        mx, my, nx, ny = _muneca_px(mano.landmarks, ancho, alto)
        rx, ry = marco.ubicacion_relativa(nx, ny)
        zona = _zona_ilustrativa(ry)
        cv2.line(frame, centro_px, (mx, my), dibujo.VERDE, 2, cv2.LINE_AA)
        cv2.circle(frame, (mx, my), 8, dibujo.VERDE, -1, cv2.LINE_AA)
        etiqueta = f"({rx:+.2f}, {ry:+.2f})  {zona}"
        dibujo.texto(frame, etiqueta, mx + 14, my, 0.5, dibujo.BLANCO, 1)


if __name__ == "__main__":
    main()
