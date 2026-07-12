"""
Punto de entrada: exportacion del Modelo A a TensorFlow Lite.

Convierte el modelo Keras entrenado a un `.tflite` optimizado para el telefono y
genera el archivo de etiquetas en el orden de CLASES_ESTATICAS.

Cuantizacion (optimizacion para gama baja o media):
  - Por defecto se usa cuantizacion de rango dinamico: los pesos pasan a enteros
    de 8 bits (modelo hasta 4 veces mas pequeno y mas rapido) mientras que la
    entrada y la salida siguen siendo float. Asi la app no tiene que cuantizar
    los landmarks; el vector [N, 126] entra tal cual.
  - Con --float16 se usa cuantizacion a 16 bits (algo mas grande, maxima fidelidad).
  - Sin --optimizar se exporta en float32 (referencia, sin comprimir).

Uso (desde la carpeta training/):

    python models/exportar_tflite.py
    python models/exportar_tflite.py --float16
"""

import argparse
import sys
from pathlib import Path

import tensorflow as tf

# Permite ejecutar el archivo directamente: agrega training/ al path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from comun.definiciones import CLASES_ESTATICAS  # noqa: E402


def convertir(ruta_keras, precision: str = "dinamica") -> bytes:
    """Convierte el modelo Keras a bytes TFLite con la precision indicada.

    precision: 'dinamica' (rango dinamico int8, por defecto), 'float16' o
    'float32' (sin cuantizar).
    """
    modelo = tf.keras.models.load_model(ruta_keras)
    conversor = tf.lite.TFLiteConverter.from_keras_model(modelo)

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
    """Escribe las 33 clases, una por linea, en el orden de CLASES_ESTATICAS."""
    contenido = "\n".join(CLASES_ESTATICAS) + "\n"
    Path(ruta_salida).write_text(contenido, encoding="utf-8", newline="\n")


def main(precision: str) -> None:
    if not config.RUTA_MODELO_A_KERAS.exists():
        print(f"No existe el modelo Keras: {config.RUTA_MODELO_A_KERAS}")
        print("Entrene primero con: python models/entrenar_a.py")
        sys.exit(1)

    tflite = convertir(config.RUTA_MODELO_A_KERAS, precision)
    config.RUTA_MODELO_A_TFLITE.parent.mkdir(parents=True, exist_ok=True)
    config.RUTA_MODELO_A_TFLITE.write_bytes(tflite)
    escribir_etiquetas(config.RUTA_ETIQUETAS_A)

    tam_kb = len(tflite) / 1024
    print(f"Modelo TFLite ({precision}): {config.RUTA_MODELO_A_TFLITE}")
    print(f"Tamano: {tam_kb:.1f} KB")
    print(f"Etiquetas: {config.RUTA_ETIQUETAS_A}")
    if tam_kb > 2048:
        print("AVISO: el modelo supera los 2 MB esperados.")
    print("\nCopie ambos a la app: App/assets/models/ y App/assets/labels/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporta el Modelo A a TFLite.")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--float16", action="store_true", help="Cuantizacion a 16 bits.")
    grupo.add_argument("--float32", action="store_true", help="Sin cuantizar (referencia).")
    args = parser.parse_args()
    precision = "float16" if args.float16 else "float32" if args.float32 else "dinamica"
    main(precision)
