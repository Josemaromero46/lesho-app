"""
Marco de referencia del cuerpo (matematica pura, sin MediaPipe ni OpenCV).

Aisla el calculo del marco del cuerpo de la libreria de deteccion, igual que
`normalizacion.py` aisla el vector de las manos. Asi se puede importar y probar
sin instalar mediapipe ni opencv. El wrapper `DetectorPose` (en `pose.py`) usa
estas piezas y solo agrega la deteccion.

El marco (ver CLAUDE.md, Modelo B):
  - ORIGEN: el centro de los hombros (punto medio entre ambos).
  - ESCALA: el ancho de los hombros (distancia entre ellos).
  - La posicion de una mano (o de un punto de la cara) se expresa en "anchos de
    hombro" respecto al centro del cuerpo, invariante a la distancia a la camara
    y a donde este la persona en el cuadro.
  - ANCLAS FACIALES: nariz, ojos, boca y las dos orejas. Dan un marco de la cara
    (ojos arriba, nariz al medio, boca abajo, orejas a los lados) para distinguir
    con precision en que zona de la cara se hace una sena.

Como los landmarks de Pose y los de Hands vienen ambos normalizados al tamano de
la imagen (x, y en [0, 1]), la muneca (de Hands) y el centro de hombros (de Pose)
estan en el MISMO sistema de coordenadas y se restan directamente.
"""

import math
from dataclasses import dataclass

from .definiciones import (
    POSE_BOCA_DER,
    POSE_BOCA_IZQ,
    POSE_HOMBRO_DER,
    POSE_HOMBRO_IZQ,
    POSE_NARIZ,
    POSE_OJO_DER,
    POSE_OJO_IZQ,
    POSE_OREJA_DER,
    POSE_OREJA_IZQ,
    VISIBILIDAD_MINIMA_POSE,
)


@dataclass
class Punto3D:
    """Un landmark de Pose: coordenadas normalizadas y su visibilidad."""

    x: float
    y: float
    z: float
    visibilidad: float


@dataclass
class MarcoCuerpo:
    """Marco de referencia del cuerpo para ubicar las manos y la cara.

    Atributos
    ---------
    centro_x, centro_y : float
        Centro de los hombros, en coordenadas de imagen [0, 1]. Origen del marco.
    ancho : float
        Ancho de los hombros (distancia entre ellos), normalizado. Escala.
    nariz, ojos, boca, oreja_izq, oreja_der : (x, y) | None
        Anclas faciales en coordenadas de imagen [0, 1], o None si no se
        detectaron con confianza suficiente. `ojos` y `boca` son el punto medio
        de sus dos landmarks.
    """

    centro_x: float
    centro_y: float
    ancho: float
    nariz: "tuple[float, float] | None" = None
    ojos: "tuple[float, float] | None" = None
    boca: "tuple[float, float] | None" = None
    oreja_izq: "tuple[float, float] | None" = None
    oreja_der: "tuple[float, float] | None" = None

    def ubicacion_relativa(self, x: float, y: float) -> tuple[float, float]:
        """Convierte un punto (x, y) de la imagen a coordenadas del cuerpo.

        Devuelve la posicion en "anchos de hombro" respecto al centro de los
        hombros. En la imagen, y crece hacia abajo, asi que un punto por ENCIMA
        de los hombros (a la altura de la cara) da y negativo, y uno por debajo
        (abdomen) da y positivo.
        """
        return (x - self.centro_x) / self.ancho, (y - self.centro_y) / self.ancho

    def rel_ancla(self, ancla: "tuple[float, float] | None") -> list[float]:
        """(rx, ry) de un ancla en el marco, o [0, 0] si el ancla es None."""
        if ancla is None:
            return [0.0, 0.0]
        rx, ry = self.ubicacion_relativa(ancla[0], ancla[1])
        return [rx, ry]


def _visible(punto: "Punto3D") -> "tuple[float, float] | None":
    """(x, y) del punto si es fiable, o None si su visibilidad es baja."""
    if punto.visibilidad < VISIBILIDAD_MINIMA_POSE:
        return None
    return (punto.x, punto.y)


def _centro_visible(p1: "Punto3D", p2: "Punto3D") -> "tuple[float, float] | None":
    """Punto medio de dos landmarks si AMBOS son fiables, o None."""
    a, b = _visible(p1), _visible(p2)
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def marco_desde_puntos(puntos: "list[Punto3D]") -> "MarcoCuerpo | None":
    """Construye el MarcoCuerpo a partir de los 33 puntos de la pose.

    Devuelve None si los hombros no son fiables (poca visibilidad) o si el ancho
    de hombros es degenerado (la persona esta de perfil o muy lejos). Las anclas
    faciales que no se vean con confianza quedan en None (su ubicacion sera cero).
    """
    hombro_izq = puntos[POSE_HOMBRO_IZQ]
    hombro_der = puntos[POSE_HOMBRO_DER]

    if (hombro_izq.visibilidad < VISIBILIDAD_MINIMA_POSE
            or hombro_der.visibilidad < VISIBILIDAD_MINIMA_POSE):
        return None

    centro_x = (hombro_izq.x + hombro_der.x) / 2.0
    centro_y = (hombro_izq.y + hombro_der.y) / 2.0
    ancho = math.hypot(hombro_izq.x - hombro_der.x, hombro_izq.y - hombro_der.y)
    if ancho < 1e-4:
        return None

    return MarcoCuerpo(
        centro_x, centro_y, ancho,
        nariz=_visible(puntos[POSE_NARIZ]),
        ojos=_centro_visible(puntos[POSE_OJO_IZQ], puntos[POSE_OJO_DER]),
        boca=_centro_visible(puntos[POSE_BOCA_IZQ], puntos[POSE_BOCA_DER]),
        oreja_izq=_visible(puntos[POSE_OREJA_IZQ]),
        oreja_der=_visible(puntos[POSE_OREJA_DER]),
    )
