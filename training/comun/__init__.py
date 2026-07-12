"""
Paquete `comun`: codigo compartido entre la captura del dataset y el pipeline
de entrenamiento.

Se exportan aqui solo las piezas puras (definiciones y normalizacion), que no
dependen de librerias pesadas. El detector de MediaPipe se importa aparte con
`from comun.landmarks import DetectorLandmarks`, para que `comun` siga siendo
importable aunque MediaPipe u OpenCV no esten instalados.
"""

from .definiciones import (
    CLASES_DINAMICAS,
    CLASES_ESTATICAS,
    INDICE_MUNECA,
    LATERALIDAD_DERECHA,
    LATERALIDAD_DESCONOCIDA,
    LATERALIDAD_IZQUIERDA,
    LETRAS,
    LETRAS_CON_MOVIMIENTO,
    LETRAS_ESTATICAS,
    NUM_CLASES_A,
    NUM_CLASES_B,
    NUM_COORDENADAS,
    NUM_LANDMARKS,
    NUM_LANDMARKS_POSE,
    NUM_MANOS,
    POSE_CADERA_DER,
    POSE_CADERA_IZQ,
    POSE_HOMBRO_DER,
    POSE_HOMBRO_IZQ,
    POSE_NARIZ,
    SENAS_CONTROL,
    TAMANO_UBICACION,
    TAMANO_UBICACION_PUNTO,
    TAMANO_VECTOR,
    TAMANO_VECTOR_B,
    TAMANO_VECTOR_MANO,
    TAMANO_VENTANA_A,
    VISIBILIDAD_MINIMA_POSE,
    validar_clases_dinamicas,
    validar_clases_estaticas,
    validar_letras_con_movimiento,
)
from .marco import MarcoCuerpo, Punto3D, marco_desde_puntos
from .normalizacion import (
    componer_vector_dos_manos,
    escalar_lote,
    escalar_vector,
    normalizar_respecto_muneca,
)
from .vector_modelo_b import (
    componer_vector_modelo_b,
    separar_config_ubicacion,
)

__all__ = [
    "CLASES_DINAMICAS",
    "CLASES_ESTATICAS",
    "INDICE_MUNECA",
    "LATERALIDAD_DERECHA",
    "LATERALIDAD_DESCONOCIDA",
    "LATERALIDAD_IZQUIERDA",
    "LETRAS",
    "LETRAS_CON_MOVIMIENTO",
    "LETRAS_ESTATICAS",
    "NUM_CLASES_A",
    "NUM_CLASES_B",
    "NUM_COORDENADAS",
    "NUM_LANDMARKS",
    "NUM_LANDMARKS_POSE",
    "NUM_MANOS",
    "POSE_CADERA_DER",
    "POSE_CADERA_IZQ",
    "POSE_HOMBRO_DER",
    "POSE_HOMBRO_IZQ",
    "POSE_NARIZ",
    "SENAS_CONTROL",
    "TAMANO_UBICACION",
    "TAMANO_UBICACION_PUNTO",
    "TAMANO_VECTOR",
    "TAMANO_VECTOR_B",
    "TAMANO_VECTOR_MANO",
    "TAMANO_VENTANA_A",
    "VISIBILIDAD_MINIMA_POSE",
    "validar_clases_dinamicas",
    "validar_clases_estaticas",
    "validar_letras_con_movimiento",
    "MarcoCuerpo",
    "Punto3D",
    "marco_desde_puntos",
    "componer_vector_dos_manos",
    "escalar_vector",
    "escalar_lote",
    "normalizar_respecto_muneca",
    "componer_vector_modelo_b",
    "separar_config_ubicacion",
]
