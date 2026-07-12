"""
Wrapper de MediaPipe Pose (Tasks API).

Este modulo es el equivalente de `comun/landmarks.py`, pero para el CUERPO en vez
de las manos. Lo usa el Modelo B (senas dinamicas) para saber DONDE, respecto al
cuerpo, se hace cada sena (lugar de articulacion), algo que la normalizacion
respecto a la muneca borra.

Solo se ocupa de la DETECCION. La matematica del marco de referencia del cuerpo
(centro y ancho de hombros, ubicacion relativa) vive en `comun/marco.py`, que es
puro y no depende de mediapipe ni opencv, para poder probarse aislado. Aqui se
re-exportan `MarcoCuerpo` y `marco_desde_puntos` por comodidad.

Optimizacion (CLAUDE.md): Pose solo corre en la ruta dinamica (entre INICIO y
FIN), no durante el deletreo del alfabeto, y con el modelo ligero. Como el torso
es estable, el marco del cuerpo se puede fijar una o pocas veces por sena.

Depende de mediapipe y opencv-python, igual que landmarks.py, asi que se importa
explicitamente con `from comun.pose import DetectorPose`.
"""

from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .definiciones import NUM_LANDMARKS_POSE
from .marco import MarcoCuerpo, Punto3D, marco_desde_puntos  # re-exportados

# Modelo de deteccion de pose (ligero), distribuido junto a este modulo.
RUTA_MODELO_POSE_POR_DEFECTO = Path(__file__).resolve().parent / "pose_landmarker.task"

# Incremento de marca de tiempo entre fotogramas (ms). El modo VIDEO exige marcas
# estrictamente crecientes; un contador propio evita colisiones.
_INCREMENTO_TIMESTAMP_MS = 33  # aproximadamente 30 fps


class DetectorPose:
    """Envuelve el PoseLandmarker de MediaPipe para extraer el cuerpo.

    Uso tipico:

        detector = DetectorPose()
        pose = detector.procesar(fotograma_bgr)   # lista de 33 Punto3D, o None
        marco = detector.marco(fotograma_bgr)      # MarcoCuerpo, o None
        detector.cerrar()
    """

    def __init__(
        self,
        confianza_deteccion: float = 0.5,
        confianza_seguimiento: float = 0.5,
        ruta_modelo=None,
    ):
        ruta = Path(ruta_modelo) if ruta_modelo else RUTA_MODELO_POSE_POR_DEFECTO
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontro el modelo de pose en {ruta}. "
                "Descargue pose_landmarker.task y coloquelo en comun/."
            )

        opciones = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ruta)),
            num_poses=1,
            min_pose_detection_confidence=confianza_deteccion,
            min_pose_presence_confidence=confianza_deteccion,
            min_tracking_confidence=confianza_seguimiento,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opciones)
        self._timestamp_ms = 0

    def procesar(self, fotograma_bgr) -> "list[Punto3D] | None":
        """Procesa un fotograma y devuelve los 33 puntos de la pose, o None.

        Devuelve None si no se detecto ninguna persona en el fotograma.
        """
        fotograma_rgb = cv2.cvtColor(fotograma_bgr, cv2.COLOR_BGR2RGB)
        imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=fotograma_rgb)

        self._timestamp_ms += _INCREMENTO_TIMESTAMP_MS
        resultado = self._landmarker.detect_for_video(imagen, self._timestamp_ms)

        if not resultado.pose_landmarks:
            return None

        puntos = [
            Punto3D(lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0))
            for lm in resultado.pose_landmarks[0]
        ]
        if len(puntos) != NUM_LANDMARKS_POSE:
            return None
        return puntos

    def marco(self, fotograma_bgr) -> "MarcoCuerpo | None":
        """Calcula el marco de referencia del cuerpo para este fotograma.

        Devuelve None si no hay pose, o si los hombros no se ven con confianza
        suficiente (el marco no seria fiable).
        """
        puntos = self.procesar(fotograma_bgr)
        if puntos is None:
            return None
        return marco_desde_puntos(puntos)

    def cerrar(self) -> None:
        """Libera el recurso de MediaPipe. Llamar al terminar de usar."""
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cerrar()
