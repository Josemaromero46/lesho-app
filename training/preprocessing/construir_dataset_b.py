"""
Punto de entrada: construye el dataset preprocesado del Modelo B.

Lee el CSV de senas dinamicas (formato Modelo B, 132 valores por fotograma),
suaviza, escala la configuracion de las manos, remuestrea cada sena a longitud
fija, opcionalmente agrega copias con ruido, y guarda todo en un .npz listo para
entrenar.

Uso (desde la carpeta training/):

    python preprocessing/construir_dataset_b.py
    python preprocessing/construir_dataset_b.py --ruido 2

Requiere que las 50 senas esten definidas en comun/definiciones.py y que exista
el CSV capturado (config.RUTA_CSV_DINAMICO).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Permite ejecutar el archivo directamente: agrega training/ al path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from comun.definiciones import CLASES_DINAMICAS, validar_clases_dinamicas  # noqa: E402
from comun.definiciones import TAMANO_VECTOR  # noqa: E402
from preprocessing.augmentacion import (  # noqa: E402
    expandir_con_ruido,
    expandir_solo_ubicacion,
)
from preprocessing.consolidar import cargar_movimiento  # noqa: E402
from preprocessing.secuencias_b import construir_dataset_modelo_b  # noqa: E402


def construir(copias_ruido: int = 0):
    grupos = cargar_movimiento(config.RUTA_CSV_DINAMICO)

    # Se ignoran las etiquetas que no estan en las 50 clases (por ejemplo las
    # tomas de la clase NADA, descartada): quedan en el CSV pero no entrenan.
    desconocidas: dict[str, int] = {}
    filtrados = {}
    for clave, seq in grupos.items():
        if clave[0] in CLASES_DINAMICAS:
            filtrados[clave] = seq
        else:
            desconocidas[clave[0]] = desconocidas.get(clave[0], 0) + 1
    if desconocidas:
        detalle = ", ".join(f"{c} ({n} tomas)" for c, n in desconocidas.items())
        print(f"Aviso: se ignoran etiquetas fuera de las 50 clases: {detalle}")
    grupos = filtrados

    X, y, personas, grupos_id = construir_dataset_modelo_b(
        grupos,
        clases=CLASES_DINAMICAS,
        longitud=config.LONGITUD_FIJA_SECUENCIA,
        suavizar=config.SUAVIZADO_ACTIVO,
        fps=config.FPS_OBJETIVO,
        min_cutoff=config.SUAVIZADO_MIN_CUTOFF,
        beta=config.SUAVIZADO_BETA,
        d_cutoff=config.SUAVIZADO_D_CUTOFF,
    )

    # Copias "solo ubicacion" (forma borrada): fuerzan al modelo a aprender el
    # LUGAR de articulacion, no solo la forma de la mano (PAPA vs HOSPITAL).
    # Se agregan antes del ruido para que tambien reciban copias ruidosas.
    X, y, personas, grupos_id = expandir_solo_ubicacion(
        X, y, personas, grupos_id, TAMANO_VECTOR
    )

    if copias_ruido > 0:
        X, y, personas, grupos_id = expandir_con_ruido(
            X, y, personas, grupos_id, sigma=config.SIGMA_RUIDO,
            copias=copias_ruido, semilla=config.SEMILLA,
        )

    return X, y, personas, grupos_id


def _reporte(X, y, personas) -> None:
    print(f"Secuencias totales: {len(X)}")
    if len(X):
        print(f"Forma de X: {X.shape}  (dtype {X.dtype})")
        print(f"Tamano en memoria: {X.nbytes / 1e6:.1f} MB")
    print(f"Personas: {sorted(set(personas.tolist()))}")
    print("\nSecuencias por clase:")
    for i, clase in enumerate(CLASES_DINAMICAS):
        n = int((y == i).sum())
        print(f"  {clase:>12}: {n:>4}", end="")
        if (i + 1) % 3 == 0:
            print()
    print()


def main(copias_ruido: int) -> None:
    try:
        validar_clases_dinamicas()
    except ValueError as error:
        print(f"No se puede construir el dataset del Modelo B:\n  {error}")
        sys.exit(1)

    X, y, personas, grupos_id = construir(copias_ruido)
    config.RUTA_DATASET_B.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.RUTA_DATASET_B, X=X, y=y, personas=personas, grupos=grupos_id
    )
    _reporte(X, y, personas)
    print(f"Tomas distintas: {len(set(grupos_id.tolist()))}")
    print(f"\nDataset guardado en: {config.RUTA_DATASET_B}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Construye el dataset de secuencias del Modelo B."
    )
    parser.add_argument(
        "--ruido", type=int, default=0,
        help="Numero de copias con ruido gaussiano a agregar (0 = ninguna).",
    )
    args = parser.parse_args()
    main(args.ruido)
