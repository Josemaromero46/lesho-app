"""
Paquete `preprocessing`: convierte los CSV de landmarks en los datos listos para
entrenar.

Para el Modelo A, arma las ventanas temporales de forma [N, 126] a partir de las
poses estaticas y de las secuencias de letras con movimiento. Depende de pandas y
numpy, no de MediaPipe ni OpenCV.
"""

from .augmentacion import expandir_con_ruido, ruido_gaussiano, variar_velocidad
from .consolidar import cargar_estatico, cargar_movimiento, resumen_cobertura
from .ventanas import (
    construir_dataset_modelo_a,
    velocidad_media,
    ventanas_de_secuencia,
    ventanas_estaticas,
    ventanas_movimiento,
)

__all__ = [
    "cargar_estatico",
    "cargar_movimiento",
    "resumen_cobertura",
    "ventanas_de_secuencia",
    "velocidad_media",
    "ventanas_estaticas",
    "ventanas_movimiento",
    "construir_dataset_modelo_a",
    "ruido_gaussiano",
    "variar_velocidad",
    "expandir_con_ruido",
]
