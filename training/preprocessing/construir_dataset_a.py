"""
Punto de entrada: construye el dataset preprocesado del Modelo A.

Lee el CSV estatico y el CSV de letras con movimiento, arma las ventanas
temporales de forma [N, 126], opcionalmente agrega copias con ruido, y guarda
todo en un unico archivo .npz listo para entrenar.

Uso (desde la carpeta training/):

    python preprocessing/construir_dataset_a.py
    python preprocessing/construir_dataset_a.py --ruido 1     # 1 copia con ruido

Salida: config.RUTA_DATASET_A, con arrays X (float32), y (int16) y personas.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Permite ejecutar el archivo directamente: agrega training/ al path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from comun.definiciones import CLASES_ESTATICAS, TAMANO_VENTANA_A  # noqa: E402
from preprocessing.augmentacion import expandir_con_ruido  # noqa: E402
from preprocessing.consolidar import (  # noqa: E402
    cargar_estatico,
    cargar_movimiento,
    resumen_cobertura,
)
from preprocessing.ventanas import construir_dataset_modelo_a  # noqa: E402


def construir(copias_ruido: int = 0):
    grupos_estaticos = cargar_estatico(config.RUTA_CSV_ESTATICO)
    grupos_movimiento = cargar_movimiento(config.RUTA_CSV_LETRAS_MOVIMIENTO)

    X, y, personas, grupos = construir_dataset_modelo_a(
        grupos_estaticos,
        grupos_movimiento,
        n=TAMANO_VENTANA_A,
        paso=config.PASO_VENTANA,
        umbral_rel=config.UMBRAL_MOVIMIENTO_REL,
        umbral_abs=config.UMBRAL_MOVIMIENTO_ABS,
        por_toma_estatica=config.VENTANAS_POR_TOMA_ESTATICA,
        por_toma_movimiento=config.VENTANAS_POR_TOMA_MOVIMIENTO,
        estatico_movimiento_max=config.UMBRAL_ESTATICO_MAX,
        suavizar=config.SUAVIZADO_ACTIVO,
        fps=config.FPS_OBJETIVO,
        min_cutoff=config.SUAVIZADO_MIN_CUTOFF,
        beta=config.SUAVIZADO_BETA,
        d_cutoff=config.SUAVIZADO_D_CUTOFF,
    )

    if copias_ruido > 0:
        X, y, personas, grupos = expandir_con_ruido(
            X, y, personas, grupos, sigma=config.SIGMA_RUIDO,
            copias=copias_ruido, semilla=config.SEMILLA,
        )

    return X, y, personas, grupos, resumen_cobertura(grupos_estaticos, grupos_movimiento)


def _reporte(X, y, personas, cobertura) -> None:
    print(f"Ventanas totales: {len(X)}")
    if len(X):
        print(f"Forma de X: {X.shape}  (dtype {X.dtype})")
        print(f"Tamano en memoria: {X.nbytes / 1e6:.1f} MB")
    print(f"Personas: {sorted(set(personas.tolist()))}")
    print()
    print("Ventanas por clase:")
    conteo = {CLASES_ESTATICAS[i]: int((y == i).sum()) for i in range(len(CLASES_ESTATICAS))}
    faltantes = [c for c, n in conteo.items() if n == 0]
    for i, clase in enumerate(CLASES_ESTATICAS):
        print(f"  {clase:>7}: {conteo[clase]:>5}", end="")
        if (i + 1) % 4 == 0:
            print()
    print()
    if faltantes:
        print(f"\nClases sin ventanas todavia: {faltantes}")


def main(copias_ruido: int) -> None:
    X, y, personas, grupos, cobertura = construir(copias_ruido)
    config.RUTA_DATASET_A.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.RUTA_DATASET_A, X=X, y=y, personas=personas, grupos=grupos
    )
    _reporte(X, y, personas, cobertura)
    print(f"Tomas distintas: {len(set(grupos.tolist()))}")
    print(f"\nDataset guardado en: {config.RUTA_DATASET_A}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Construye el dataset de ventanas del Modelo A."
    )
    parser.add_argument(
        "--ruido", type=int, default=0,
        help="Numero de copias con ruido gaussiano a agregar (0 = ninguna).",
    )
    args = parser.parse_args()
    main(args.ruido)
