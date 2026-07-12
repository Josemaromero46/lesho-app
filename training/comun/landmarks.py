"""
Wrapper de MediaPipe Hands (Tasks API).

Este es el unico punto del pipeline de Python que habla directamente con
MediaPipe. El resto del codigo trabaja con la clase `ManoDetectada`, que aisla
al proyecto de la API concreta de MediaPipe. Si algun dia se cambia la libreria
de deteccion, solo se reescribe este archivo.

Se usa la Tasks API de MediaPipe (`mediapipe.tasks.python.vision.HandLandmarker`),
que es la API oficial vigente. Las versiones recientes de MediaPipe (0.10.30 en
adelante) eliminaron la antigua API de Solutions (`mp.solutions.hands`). La Tasks
API requiere un archivo de modelo `hand_landmarker.task`, que se distribuye junto
a este modulo.

Depende de mediapipe y opencv-python. Por eso NO se importa desde
comun/__init__.py de forma automatica: quien necesite deteccion lo importa
explicitamente con `from comun.landmarks import DetectorLandmarks`. Asi el
modulo de normalizacion sigue siendo usable sin instalar estas librerias.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .definiciones import NUM_LANDMARKS

# Modelo de deteccion de manos, distribuido junto a este modulo.
RUTA_MODELO_POR_DEFECTO = Path(__file__).resolve().parent / "hand_landmarker.task"

# Incremento de marca de tiempo entre fotogramas (ms). El modo VIDEO de la Tasks
# API exige marcas de tiempo estrictamente crecientes; un contador propio evita
# colisiones cuando dos fotogramas llegan en el mismo milisegundo real.
_INCREMENTO_TIMESTAMP_MS = 33  # aproximadamente 30 fps


@dataclass
class ManoDetectada:
    """Una mano detectada por MediaPipe en un fotograma.

    Atributos
    ---------
    landmarks : list[tuple[float, float, float]]
        Los 21 puntos crudos (x, y, z) en el sistema de coordenadas de
        MediaPipe. x e y estan normalizados al rango [0, 1] respecto al tamano
        de la imagen; z es la profundidad relativa a la muneca. Estos valores
        son la entrada de `normalizar_respecto_muneca`.
    lateralidad : str
        "Izquierda" o "Derecha". Importa porque en LESHO la mano dominante
        define la sena. Ver la nota sobre el efecto espejo mas abajo.
    confianza : float
        Confianza de la clasificacion de lateralidad reportada por MediaPipe.
    """

    landmarks: list
    lateralidad: str
    confianza: float


class DetectorLandmarks:
    """Envuelve el HandLandmarker de MediaPipe para extraer landmarks de la mano.

    Uso tipico:

        detector = DetectorLandmarks(max_manos=2, confianza_deteccion=0.5)
        manos = detector.procesar(fotograma_bgr)
        if manos:
            vector = normalizar_respecto_muneca(manos[0].landmarks)
        detector.cerrar()

    Tambien funciona como context manager:

        with DetectorLandmarks() as detector:
            manos = detector.procesar(fotograma_bgr)
    """

    def __init__(
        self,
        max_manos: int = 2,
        confianza_deteccion: float = 0.5,
        confianza_seguimiento: float = 0.5,
        ruta_modelo=None,
    ):
        """Inicializa el detector de MediaPipe.

        Parametros
        ----------
        max_manos : int
            Numero maximo de manos a detectar. Se usa 2 para soportar senas a
            dos manos (CLAUDE.md, riesgo de MediaPipe).
        confianza_deteccion : float
            Umbral minimo para aceptar que hay una mano en el fotograma.
        confianza_seguimiento : float
            Umbral minimo para seguir una mano ya detectada entre fotogramas.
        ruta_modelo : str | Path | None
            Ruta al archivo hand_landmarker.task. Si es None, usa el que se
            distribuye junto a este modulo.
        """
        ruta = Path(ruta_modelo) if ruta_modelo else RUTA_MODELO_POR_DEFECTO
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontro el modelo de manos en {ruta}. "
                "Descargue hand_landmarker.task y coloquelo en comun/."
            )

        opciones = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ruta)),
            num_hands=max_manos,
            min_hand_detection_confidence=confianza_deteccion,
            min_hand_presence_confidence=confianza_deteccion,
            min_tracking_confidence=confianza_seguimiento,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opciones)
        self._timestamp_ms = 0

    def procesar(self, fotograma_bgr):
        """Procesa un fotograma y devuelve las manos detectadas.

        Parametros
        ----------
        fotograma_bgr : ndarray
            Fotograma en formato BGR, tal como lo entrega OpenCV (cv2).

        Retorna
        -------
        list[ManoDetectada] | None
            Lista con una entrada por mano detectada, o None si no se detecto
            ninguna mano con confianza suficiente. Devolver None significa que
            el fotograma debe descartarse (paso 2 del pipeline en CLAUDE.md).
        """
        # MediaPipe trabaja en RGB; OpenCV entrega BGR.
        fotograma_rgb = cv2.cvtColor(fotograma_bgr, cv2.COLOR_BGR2RGB)
        imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=fotograma_rgb)

        self._timestamp_ms += _INCREMENTO_TIMESTAMP_MS
        resultado = self._landmarker.detect_for_video(imagen, self._timestamp_ms)

        if not resultado.hand_landmarks:
            return None

        manos: list[ManoDetectada] = []
        for indice, lista_landmarks in enumerate(resultado.hand_landmarks):
            puntos = [(lm.x, lm.y, lm.z) for lm in lista_landmarks]
            if len(puntos) != NUM_LANDMARKS:
                continue
            lateralidad, confianza = self._leer_lateralidad(resultado, indice)
            manos.append(
                ManoDetectada(
                    landmarks=puntos,
                    lateralidad=lateralidad,
                    confianza=confianza,
                )
            )

        return manos if manos else None

    def _leer_lateralidad(self, resultado, indice: int):
        """Extrae la lateralidad (izquierda/derecha) de una mano.

        NOTA SOBRE EL EFECTO ESPEJO: la captura y la app voltean el fotograma
        horizontalmente (vista selfie) antes de procesarlo. En ese caso, la
        "Left"/"Right" que reporta MediaPipe queda invertida respecto a la mano
        real del usuario. Aqui se traduce la etiqueta cruda; la correccion del
        espejo, si hace falta, se decide al definir las senas con asesoria LESHO.
        """
        traduccion = {"Left": "Izquierda", "Right": "Derecha"}
        try:
            categoria = resultado.handedness[indice][0]
            etiqueta = traduccion.get(categoria.category_name,
                                      categoria.category_name)
            return etiqueta, float(categoria.score)
        except (AttributeError, IndexError):
            return "Desconocida", 0.0

    def cerrar(self) -> None:
        """Libera el recurso de MediaPipe. Llamar al terminar de usar."""
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cerrar()
