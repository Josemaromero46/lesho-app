"""
Punto de entrada: entrenamiento del Modelo B (senas dinamicas).

Carga el dataset de secuencias (.npz), lo divide por persona (o por toma si hay
pocas personas), entrena el LSTM ligero compensando el desbalance de clases con
pesos, evalua en la particion de prueba y guarda el modelo Keras.

Reutiliza las utilidades de division y pesos de `models/datos.py` (son genericas
respecto a la forma de X, asi que sirven igual para ventanas del Modelo A y
secuencias del Modelo B).

Uso (desde la carpeta training/):

    python models/entrenar_b.py
    python models/entrenar_b.py --epocas 80 --division grupo
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# Permite ejecutar el archivo directamente: agrega training/ al path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from comun.definiciones import CLASES_DINAMICAS, NUM_CLASES_B  # noqa: E402
from models.arquitectura_b import compilar_modelo_b, construir_modelo_b  # noqa: E402
from models.datos import (  # noqa: E402
    cargar_dataset,
    dividir_por_grupo,
    dividir_por_persona,
    pesos_de_clase,
)


def _dividir(X, y, personas, grupos, metodo: str):
    """Aplica la division elegida, con respaldo por toma si hace falta."""
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
    return dividir_por_grupo(
        X, y, grupos,
        config.PROPORCION_VALIDACION, config.PROPORCION_PRUEBA, config.SEMILLA,
    )


def entrenar(
    ruta_dataset=None, metodo: str = "persona",
    epocas: int = 80, batch: int = 32, paciencia: int = 15,
):
    ruta_dataset = ruta_dataset or config.RUTA_DATASET_B
    X, y, personas, grupos = cargar_dataset(ruta_dataset)
    print(f"Dataset: X={X.shape}, {len(np.unique(y))} clases presentes de {NUM_CLASES_B}")

    division = _dividir(X, y, personas, grupos, metodo)
    X_train, y_train = division["train"]
    X_val, y_val = division["val"]
    X_test, y_test = division["test"]
    print(f"Particiones -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    pesos = pesos_de_clase(y_train, NUM_CLASES_B)

    modelo = compilar_modelo_b(construir_modelo_b(longitud=config.LONGITUD_FIJA_SECUENCIA))
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
        nombre = CLASES_DINAMICAS[indice] if indice < len(CLASES_DINAMICAS) else str(indice)
        print(f"    {nombre:>12}: {acierto:.3f}  ({int(mascara.sum())} muestras)")


def main(metodo: str, epocas: int) -> None:
    if not config.RUTA_DATASET_B.exists():
        print(f"No existe el dataset del Modelo B: {config.RUTA_DATASET_B}")
        print("Construyalo primero con: python preprocessing/construir_dataset_b.py")
        sys.exit(1)

    modelo, _ = entrenar(metodo=metodo, epocas=epocas)
    config.RUTA_MODELO_B_KERAS.parent.mkdir(parents=True, exist_ok=True)
    modelo.save(config.RUTA_MODELO_B_KERAS)
    print(f"\nModelo guardado en: {config.RUTA_MODELO_B_KERAS}")
    print("Siguiente paso: python models/exportar_tflite_b.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el Modelo B de senas dinamicas.")
    parser.add_argument("--epocas", type=int, default=80)
    parser.add_argument(
        "--division", choices=["persona", "grupo"], default="persona",
        help="Como dividir el dataset. 'persona' requiere >=3 personas y cae a "
             "'grupo' (por toma) si hay menos.",
    )
    args = parser.parse_args()
    main(args.division, args.epocas)
