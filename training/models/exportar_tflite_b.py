"""
Punto de entrada: exportacion del Modelo B a TensorFlow Lite.

Convierte el clasificador de secuencias (LSTM) a un `.tflite` optimizado para el
telefono y genera el archivo de etiquetas en el orden de CLASES_DINAMICAS.

El LSTM de Keras se convierte a la operacion fusionada de TFLite
(UnidirectionalSequenceLSTM), asi que corre on-device con operaciones nativas.
La cuantizacion de rango dinamico (por defecto) pasa los pesos a enteros de 8
bits; la entrada y la salida siguen en float, para que la app pase la secuencia
[LONGITUD, 132] tal cual.

Uso (desde la carpeta training/):

    python models/exportar_tflite_b.py
    python models/exportar_tflite_b.py --float16
"""

import argparse
import sys
from pathlib import Path

import tensorflow as tf

# Permite ejecutar el archivo directamente: agrega training/ al path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from comun.definiciones import CLASES_DINAMICAS  # noqa: E402


def convertir(ruta_keras, precision: str = "dinamica") -> bytes:
    """Convierte el modelo Keras del Modelo B a bytes TFLite.

    precision: 'dinamica' (rango dinamico int8, por defecto), 'float16' o
    'float32' (sin cuantizar).

    El LSTM se exporta con el BATCH FIJO EN 1 (la app infiere una sena a la vez).
    Esto es necesario para que la conversion pueda fusionar el LSTM en la
    operacion nativa de TFLite (UnidirectionalSequenceLSTM); con batch dinamico,
    los tensor-list ops del LSTM no tienen forma estatica y la conversion falla o
    obliga a usar ops de TensorFlow (Flex), mas pesadas en el telefono.
    """
    modelo = tf.keras.models.load_model(ruta_keras)
    longitud = modelo.input_shape[1]
    dim = modelo.input_shape[2]

    # Se re-envuelve el modelo entrenado con un Input de BATCH FIJO EN 1. Con la
    # forma estatica, el LSTM se fusiona en la operacion nativa de TFLite y los
    # pesos quedan como constantes (from_keras_model los congela), sin variables
    # de recurso que el interprete no inicializa.
    entrada = tf.keras.Input(batch_shape=(1, longitud, dim), name="secuencia")
    modelo_fijo = tf.keras.Model(entrada, modelo(entrada))
    conversor = tf.lite.TFLiteConverter.from_keras_model(modelo_fijo)

    if precision == "dinamica":
        conversor.optimizations = [tf.lite.Optimize.DEFAULT]
    elif precision == "float16":
        conversor.optimizations = [tf.lite.Optimize.DEFAULT]
        conversor.target_spec.supported_types = [tf.float16]
    elif precision == "float32":
        pass
    else:
        raise ValueError(f"Precision desconocida: {precision!r}")

    return conversor.convert()


def escribir_etiquetas(ruta_salida) -> None:
    """Escribe las 50 clases, una por linea, en el orden de CLASES_DINAMICAS."""
    contenido = "\n".join(CLASES_DINAMICAS) + "\n"
    Path(ruta_salida).write_text(contenido, encoding="utf-8", newline="\n")


def main(precision: str) -> None:
    if not config.RUTA_MODELO_B_KERAS.exists():
        print(f"No existe el modelo Keras: {config.RUTA_MODELO_B_KERAS}")
        print("Entrene primero con: python models/entrenar_b.py")
        sys.exit(1)

    tflite = convertir(config.RUTA_MODELO_B_KERAS, precision)
    config.RUTA_MODELO_B_TFLITE.parent.mkdir(parents=True, exist_ok=True)
    config.RUTA_MODELO_B_TFLITE.write_bytes(tflite)
    escribir_etiquetas(config.RUTA_ETIQUETAS_B)

    tam_kb = len(tflite) / 1024
    print(f"Modelo TFLite ({precision}): {config.RUTA_MODELO_B_TFLITE}")
    print(f"Tamano: {tam_kb:.1f} KB")
    print(f"Etiquetas: {config.RUTA_ETIQUETAS_B}")
    if tam_kb > 2048:
        print("AVISO: el modelo supera los 2 MB esperados.")
    print("\nCopie ambos a la app: App/assets/models/ y App/assets/labels/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporta el Modelo B a TFLite.")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--float16", action="store_true", help="Cuantizacion a 16 bits.")
    grupo.add_argument("--float32", action="store_true", help="Sin cuantizar (referencia).")
    args = parser.parse_args()
    precision = "float16" if args.float16 else "float32" if args.float32 else "dinamica"
    main(precision)
