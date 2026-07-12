"""
Arquitectura del Modelo B (senas dinamicas), clasificador de secuencias.

Recibe una secuencia de longitud fija (forma [LONGITUD, 132]) donde cada
fotograma es la configuracion de las dos manos (126) mas la ubicacion en el
cuerpo (6), y produce una distribucion sobre las 50 senas dinamicas.

Se usa un LSTM ligero (no GRU) por dos razones alineadas con el objetivo movil:
  - El LSTM captura la dependencia temporal de la sena (como evoluciona el gesto
    en el tiempo), que es justo lo que distingue una sena dinamica.
  - El LSTM de Keras se convierte a una operacion NATIVA de TFLite
    (UnidirectionalSequenceLSTM fusionado), asi que corre on-device sin
    operaciones especiales ni el delegado Flex. El GRU, en cambio, no tiene op
    fusionada en TFLite y obligaria a incluir ops de TensorFlow (mas peso y
    dependencias en el telefono).

La red es deliberadamente pequena (unas decenas de miles de parametros) para
quedar muy por debajo del limite y correr fluido en gama baja o media. Las
secuencias entran ya remuestreadas a longitud fija, asi que NO hay padding ni
capa de enmascarado (que complican la conversion a TFLite).
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from comun.definiciones import NUM_CLASES_B, TAMANO_ENTRADA_B

# Longitud fija de la secuencia de entrada (fotogramas). Coincide con
# config.LONGITUD_FIJA_SECUENCIA; se deja como default para no importar config.
LONGITUD_POR_DEFECTO = 40


def construir_modelo_b(
    longitud: int = LONGITUD_POR_DEFECTO,
    dim_vector: int = TAMANO_ENTRADA_B,
    num_clases: int = NUM_CLASES_B,
    unidades_lstm: int = 64,
    unidades_densa: int = 48,
    dropout: float = 0.3,
) -> tf.keras.Model:
    """Construye y devuelve el Modelo B sin compilar.

    unidades_lstm: tamano del estado del LSTM (capacidad temporal).
    unidades_densa: neuronas de la capa densa antes de la salida.
    dropout: regularizacion.
    """
    entrada = layers.Input(shape=(longitud, dim_vector), name="secuencia")

    # LSTM unidireccional: procesa la secuencia y entrega su estado final, un
    # resumen del gesto completo. return_sequences=False -> un vector por muestra.
    x = layers.LSTM(unidades_lstm, name="lstm")(entrada)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(unidades_densa, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    salida = layers.Dense(num_clases, activation="softmax", name="clases")(x)

    return models.Model(entrada, salida, name="modelo_b")


def compilar_modelo_b(modelo: tf.keras.Model, tasa_aprendizaje: float = 1e-3) -> tf.keras.Model:
    """Compila el Modelo B con Adam y perdida para etiquetas enteras."""
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=tasa_aprendizaje),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return modelo
