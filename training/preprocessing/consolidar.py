"""
Carga y agrupacion de los CSV de landmarks para el preprocesamiento.

Lee los dos CSV que alimentan al Modelo A y los agrupa en secuencias de
fotogramas listas para convertir en ventanas. Ambos usan el mismo formato de
secuencia, una fila por fotograma con un indice `frame`, y cada
(clase, persona, id_muestra) es una TOMA independiente:

  - CSV estatico (letras de pose + INICIO/FIN/REPOSO): cada toma es una formacion
    corta de la pose. Se graban muchas tomas separadas por clase para dar
    variedad (ver captura_estatica.py).
  - CSV de letras con movimiento (J, Ñ, Z, LL, RR): cada toma es una ejecucion
    del gesto.

No depende de MediaPipe ni OpenCV, solo de pandas y numpy, para poder probarse
de forma aislada con CSV sinteticos.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

_META_SECUENCIA = ["etiqueta", "persona", "id_muestra", "frame"]


def _columnas_coordenadas(df: pd.DataFrame, meta: list[str]) -> list[str]:
    """Devuelve los nombres de las columnas de coordenadas (todo lo que no es meta)."""
    return [c for c in df.columns if c not in meta]


def _cargar_secuencias(ruta, nombre) -> "OrderedDict[tuple[str, str, int], np.ndarray]":
    """Agrupa un CSV en formato secuencia en {(clase, persona, id): array (T, 126)}.

    Cada grupo es una toma, ordenada por el indice de fotograma.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro el CSV {nombre}: {ruta}")

    df = pd.read_csv(ruta)
    coords = _columnas_coordenadas(df, _META_SECUENCIA)
    grupos: "OrderedDict[tuple[str, str, int], np.ndarray]" = OrderedDict()
    for (clase, persona, idm), sub in df.groupby(
        ["etiqueta", "persona", "id_muestra"], sort=False
    ):
        sub = sub.sort_values("frame")
        grupos[(clase, persona, int(idm))] = sub[coords].to_numpy(dtype=np.float32)
    return grupos


def cargar_estatico(ruta) -> "OrderedDict[tuple[str, str, int], np.ndarray]":
    """Agrupa el CSV estatico (poses) en {(clase, persona, id): array (T, 126)}.

    Cada grupo es una TOMA corta de la pose sostenida. Se graban varias tomas
    separadas por clase para dar variedad (ver captura_estatica.py).
    """
    return _cargar_secuencias(ruta, "estatico")


def cargar_movimiento(ruta) -> "OrderedDict[tuple[str, str, int], np.ndarray]":
    """Agrupa el CSV de letras con movimiento en {(clase, persona, id): array (T, 126)}.

    Cada grupo es una secuencia (una ejecucion del gesto), ordenada por frame.
    """
    return _cargar_secuencias(ruta, "de movimiento")


def resumen_cobertura(
    grupos_estaticos: dict, grupos_movimiento: dict
) -> "OrderedDict[str, int]":
    """Cuenta cuantas TOMAS hay por clase (estaticas y de movimiento juntas).

    Sirve para revisar de un vistazo que todas las clases tengan datos antes de
    entrenar.
    """
    conteo: "OrderedDict[str, int]" = OrderedDict()
    for (clase, _persona, _idm) in grupos_estaticos:
        conteo[clase] = conteo.get(clase, 0) + 1
    for (clase, _persona, _idm) in grupos_movimiento:
        conteo[clase] = conteo.get(clase, 0) + 1
    return conteo
