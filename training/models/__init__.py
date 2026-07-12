"""
Paquete `models`: arquitectura, entrenamiento y exportacion de los modelos.

Contiene el Modelo A (alfabeto, ventana temporal). El Modelo B (senas dinamicas)
se agregara cuando esten definidas las 50 clases con asesoria LESHO.

Depende de TensorFlow, asi que solo se importa cuando se va a entrenar o exportar,
no en la captura del dataset.
"""

from .arquitectura_a import compilar_modelo_a, construir_modelo_a
from .datos import (
    cargar_dataset,
    dividir_estratificado,
    dividir_por_grupo,
    dividir_por_persona,
    pesos_de_clase,
)

__all__ = [
    "construir_modelo_a",
    "compilar_modelo_a",
    "cargar_dataset",
    "dividir_por_persona",
    "dividir_por_grupo",
    "dividir_estratificado",
    "pesos_de_clase",
]
