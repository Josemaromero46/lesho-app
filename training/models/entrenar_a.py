"""
Punto de entrada: entrenamiento del Modelo A (alfabeto).

Carga el dataset de ventanas (.npz), lo divide por persona (o por toma si hay una
sola persona), entrena la red convolucional 1D compensando el desbalance de
clases con pesos, evalua en la particion de prueba y guarda el modelo Keras.

La division por toma es la honesta con una sola persona: ninguna toma cruza de
entrenamiento a prueba, asi que la exactitud reportada no esta inflada por
ventanas casi identicas repartidas entre ambas particiones (fuga de datos).

Uso (desde la carpeta training/):

    python models/entrenar_a.py
    python models/entrenar_a.py --epocas 80 --division grupo
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# Permite ejecutar el archivo directamente: agrega training/ al path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from comun.definiciones import CLASES_ESTATICAS, NUM_CLASES_A  # noqa: E402
from models.arquitectura_a import compilar_modelo_a, construir_modelo_a  # noqa: E402
from models.datos import (  # noqa: E402
    cargar_dataset,
    dividir_por_grupo,
    dividir_por_persona,
    pesos_de_clase,
)


def _dividir(X, y, personas, grupos, metodo: str):
    """Aplica la division elegida, con respaldo por toma si hace falta.

    Con una sola persona la division por persona no es posible; se cae a la
    division por toma, que es la honesta (ninguna toma cruza de entrenamiento a
    prueba).
    """
    if metodo == "persona":
        try:
            division = dividir_por_persona(
                X, y, personas,
                config.PROPORCION_VALIDACION, config.PROPORCION_PRUEBA, config.SEMILLA,
            )
            print(f"Division por persona: {division['personas']}")
            return division
        except ValueError as error:
            print(f"Aviso: {error}\nUsando division por toma.")
    division = dividir_por_grupo(
        X, y, grupos,
        config.PROPORCION_VALIDACION, config.PROPORCION_PRUEBA, config.SEMILLA,
    )
    return division


def entrenar(
    ruta_dataset=None, metodo: str = "persona",
    epocas: int = 80, batch: int = 32, paciencia: int = 15,
):
    ruta_dataset = ruta_dataset or config.RUTA_DATASET_A
    X, y, personas, grupos = cargar_dataset(ruta_dataset)
    print(f"Dataset: X={X.shape}, {len(np.unique(y))} clases presentes de {NUM_CLASES_A}")

    division = _dividir(X, y, personas, grupos, metodo)
    X_train, y_train = division["train"]
    X_val, y_val = division["val"]
    X_test, y_test = division["test"]
    print(f"Particiones -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    pesos = pesos_de_clase(y_train, NUM_CLASES_A)

    modelo = compilar_modelo_a(construir_modelo_a())
    modelo.summary()

    parada = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=paciencia, restore_best_weights=True
    )
    historia = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epocas, batch_size=batch,
        class_weight=pesos, callbacks=[parada], verbose=2,
    )

    print("\nEvaluacion en prueba:")
    perdida, exactitud = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"  perdida: {perdida:.4f}   exactitud: {exactitud:.4f}")
    _reporte_por_clase(modelo, X_test, y_test)

    return modelo, historia


def _reporte_por_clase(modelo, X_test, y_test) -> None:
    """Imprime exactitud por clase presente en la particion de prueba."""
    if len(X_test) == 0:
        return
    pred = np.argmax(modelo.predict(X_test, verbose=0), axis=1)
    print("  exactitud por clase:")
    for indice in np.unique(y_test):
        mascara = y_test == indice
        acierto = float((pred[mascara] == indice).mean())
        print(f"    {CLASES_ESTATICAS[indice]:>7}: {acierto:.3f}  ({int(mascara.sum())} muestras)")


def main(metodo: str, epocas: int) -> None:
    modelo, _ = entrenar(metodo=metodo, epocas=epocas)
    config.RUTA_MODELO_A_KERAS.parent.mkdir(parents=True, exist_ok=True)
    modelo.save(config.RUTA_MODELO_A_KERAS)
    print(f"\nModelo guardado en: {config.RUTA_MODELO_A_KERAS}")
    print("Siguiente paso: python models/exportar_tflite.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el Modelo A del alfabeto LESHO.")
    parser.add_argument("--epocas", type=int, default=80)
    parser.add_argument(
        "--division", choices=["persona", "grupo"], default="persona",
        help="Como dividir el dataset. 'persona' requiere >=3 personas y cae a "
             "'grupo' (por toma) si hay menos.",
    )
    args = parser.parse_args()
    main(args.division, args.epocas)
