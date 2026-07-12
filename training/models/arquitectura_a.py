"""
Arquitectura del Modelo A (alfabeto), clasificador de ventana temporal.

Recibe una ventana de N fotogramas de landmarks (forma [N, 126]) y produce una
distribucion sobre las 33 clases del alfabeto. Reconoce tanto letras estaticas
(ventana casi quieta) como letras con movimiento (J, Ñ, Z, LL, RR).

Se usa una red convolucional 1D sobre el eje temporal en lugar de una recurrente
(GRU/LSTM) por optimizacion: en telefonos de gama baja o media, las
convoluciones son mas rapidas y paralelas, y se convierten a TFLite sin
operaciones especiales. La red es deliberadamente pequena (unos pocos miles de
parametros) para quedar muy por debajo de 2 MB y correr fluido on-device.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from comun.definiciones import NUM_CLASES_A, TAMANO_VECTOR, TAMANO_VENTANA_A


def construir_modelo_a(
    n_ventana: int = TAMANO_VENTANA_A,
    dim_vector: int = TAMANO_VECTOR,
    num_clases: int = NUM_CLASES_A,
    filtros: tuple[int, ...] = (32, 64),
    unidades_densa: int = 64,
    dropout: float = 0.3,
) -> tf.keras.Model:
    """Construye y devuelve el Modelo A sin compilar.

    filtros: numero de canales de cada bloque convolucional.
    unidades_densa: neuronas de la capa densa antes de la salida.
    dropout: regularizacion antes de la capa de clasificacion.
    """
    entrada = layers.Input(shape=(n_ventana, dim_vector), name="ventana")
    x = entrada

    # Bloques convolucionales 1D sobre el tiempo. Cada bloque reduce a la mitad
    # la longitud temporal, extrayendo patrones de movimiento cada vez mas largos.
    for canales in filtros:
        x = layers.Conv1D(canales, kernel_size=3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

    # Resumen temporal invariante a la posicion exacta del gesto en la ventana.
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(unidades_densa, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    salida = layers.Dense(num_clases, activation="softmax", name="clases")(x)

    return models.Model(entrada, salida, name="modelo_a")


def compilar_modelo_a(modelo: tf.keras.Model, tasa_aprendizaje: float = 1e-3) -> tf.keras.Model:
    """Compila el Modelo A con Adam y perdida para etiquetas enteras."""
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=tasa_aprendizaje),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return modelo
