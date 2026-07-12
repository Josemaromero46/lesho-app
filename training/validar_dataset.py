"""
Validador del dataset de landmarks.

Ejecutar desde la carpeta training/:

    python validar_dataset.py                     # valida el dataset estatico
    python validar_dataset.py --modo dinamico      # valida el dataset dinamico
    python validar_dataset.py --visualizar         # genera plots de esqueletos
    python validar_dataset.py --clase A            # analiza solo una clase

Genera un reporte en consola y, si se pide, imagenes PNG en training/exports/validacion/.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from comun.definiciones import (
    CLASES_ESTATICAS, CLASES_DINAMICAS,
    TAMANO_VECTOR, TAMANO_VECTOR_MANO, NUM_COORDENADAS,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

NUM_PUNTOS_MANO = TAMANO_VECTOR_MANO // NUM_COORDENADAS   # 21
TOLERANCIA_CERO = 1e-6

CONEXIONES_MANO = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (9,10),(10,11),(11,12),
    (13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

COLORES = {
    "ok":      "\033[92m✓\033[0m",
    "warn":    "\033[93m!\033[0m",
    "error":   "\033[91m✗\033[0m",
    "titulo":  "\033[1m",
    "reset":   "\033[0m",
}


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def cargar_csv(ruta: Path) -> pd.DataFrame:
    if not ruta.exists():
        print(f"{COLORES['error']} No se encontró el archivo: {ruta}")
        sys.exit(1)
    df = pd.read_csv(ruta)
    print(f"\n{COLORES['titulo']}Dataset: {ruta.name}{COLORES['reset']}")
    print(f"  Filas: {len(df):,}   Columnas: {len(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Checks individuales
# ---------------------------------------------------------------------------

def _n_meta(df: pd.DataFrame) -> int:
    """Numero de columnas de metadatos antes de las coordenadas.

    El formato de secuencia (estatico por tomas y letras con movimiento) agrega
    la columna `frame`, asi que hay 4 columnas meta; sin ella, 3.
    """
    return 4 if "frame" in df.columns else 3


def _coords(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve solo las columnas de coordenadas (todo lo que no es metadato)."""
    return df.iloc[:, _n_meta(df):]


def check_columnas(df: pd.DataFrame, modo: str) -> bool:
    esperadas = _n_meta(df) + TAMANO_VECTOR   # meta + 126 coords
    ok = len(df.columns) == esperadas
    simbolo = COLORES["ok"] if ok else COLORES["error"]
    print(f"  {simbolo} Columnas: {len(df.columns)} (esperadas {esperadas})")
    return ok


def check_nulos(df: pd.DataFrame) -> bool:
    nulos = df.isnull().sum().sum()
    ok = nulos == 0
    simbolo = COLORES["ok"] if ok else COLORES["error"]
    print(f"  {simbolo} Valores nulos: {nulos}")
    return ok


def check_infinitos(df: pd.DataFrame) -> bool:
    coords = _coords(df).astype(float)
    inf = np.isinf(coords.values).sum()
    ok = inf == 0
    simbolo = COLORES["ok"] if ok else COLORES["error"]
    print(f"  {simbolo} Valores infinitos: {inf}")
    return ok


def check_clases(df: pd.DataFrame, clases_esperadas: list) -> bool:
    clases_csv = set(df["etiqueta"].unique())
    clases_esp = set(clases_esperadas)
    extras = clases_csv - clases_esp
    faltantes = clases_esp - clases_csv
    ok = not extras and not faltantes
    simbolo = COLORES["ok"] if ok else COLORES["warn"]
    print(f"  {simbolo} Clases en CSV: {len(clases_csv)}")
    if extras:
        print(f"      Etiquetas no esperadas: {sorted(extras)}")
    if faltantes:
        print(f"      Clases sin muestras aún: {sorted(faltantes)}")
    return ok


def check_muneca_en_cero(df: pd.DataFrame) -> bool:
    """x0,y0,z0 y x21,y21,z21 deben ser exactamente 0."""
    coords = _coords(df).astype(float).values
    # Muñeca mano izquierda: índices 0,1,2
    mano_izq_ok = np.all(np.abs(coords[:, 0:3]) < TOLERANCIA_CERO, axis=1)
    # Muñeca mano derecha: índice 63,64,65 (punto 21 × 3)
    inicio_der = NUM_PUNTOS_MANO * NUM_COORDENADAS  # 63
    mano_der_ok = np.all(np.abs(coords[:, inicio_der:inicio_der+3]) < TOLERANCIA_CERO, axis=1)

    fallos_izq = int((~mano_izq_ok).sum())
    fallos_der = int((~mano_der_ok).sum())

    ok = fallos_izq == 0 and fallos_der == 0
    simbolo = COLORES["ok"] if ok else COLORES["error"]
    print(f"  {simbolo} Muñeca izquierda en cero: {fallos_izq} filas con error")
    print(f"  {simbolo} Muñeca derecha en cero:   {fallos_der} filas con error")
    return ok


def check_filas_vacias(df: pd.DataFrame) -> bool:
    """Detecta filas donde AMBAS manos son todo ceros (nada detectado)."""
    coords = _coords(df).astype(float).values
    todo_cero = np.all(np.abs(coords) < TOLERANCIA_CERO, axis=1)
    n = int(todo_cero.sum())
    ok = n == 0
    simbolo = COLORES["ok"] if ok else COLORES["error"]
    print(f"  {simbolo} Filas con ambas manos en ceros: {n}")
    if n > 0:
        indices = np.where(todo_cero)[0][:5]
        clases = df.iloc[indices]["etiqueta"].tolist()
        print(f"      Primeras afectadas (clase): {clases}")
    return ok


def check_rango(df: pd.DataFrame) -> bool:
    """Valores fuera de [-1.5, 1.5] son sospechosos para landmarks normalizados."""
    LIMITE = 1.5
    coords = _coords(df).astype(float).values
    # Ignorar columnas de muñeca (siempre cero) y de mano ausente
    fuera = np.abs(coords) > LIMITE
    n_filas = int((fuera.any(axis=1)).sum())
    max_val = float(np.abs(coords).max())
    ok = n_filas == 0
    simbolo = COLORES["ok"] if ok else COLORES["warn"]
    print(f"  {simbolo} Filas con valores > {LIMITE}: {n_filas}  (máximo absoluto: {max_val:.4f})")
    return ok


def check_una_mano_detectada(df: pd.DataFrame) -> None:
    """Informa cuántas filas tienen solo una mano vs. dos."""
    coords = _coords(df).astype(float).values
    inicio_der = NUM_PUNTOS_MANO * NUM_COORDENADAS  # 63

    izq_presente = ~np.all(np.abs(coords[:, 0:inicio_der]) < TOLERANCIA_CERO, axis=1)
    der_presente = ~np.all(np.abs(coords[:, inicio_der:]) < TOLERANCIA_CERO, axis=1)

    dos_manos = int((izq_presente & der_presente).sum())
    solo_izq  = int((izq_presente & ~der_presente).sum())
    solo_der  = int((~izq_presente & der_presente).sum())

    total = len(df)
    print(f"  {COLORES['ok']} Detección de manos:")
    print(f"      Dos manos : {dos_manos:>5}  ({100*dos_manos/total:.1f}%)")
    print(f"      Solo izq. : {solo_izq:>5}  ({100*solo_izq/total:.1f}%)")
    print(f"      Solo der. : {solo_der:>5}  ({100*solo_der/total:.1f}%)")


def resumen_muestras(df: pd.DataFrame, clases_esperadas: list,
                     muestras_esperadas: int = 40) -> None:
    """Tabla de muestras por clase y persona."""
    print(f"\n{COLORES['titulo']}Muestras por clase y persona:{COLORES['reset']}")
    personas = sorted(df["persona"].unique())
    tabla = df.groupby(["etiqueta", "persona"]).size().unstack(fill_value=0)

    # Asegurar que todas las clases esperadas aparezcan
    for c in clases_esperadas:
        if c not in tabla.index:
            tabla.loc[c] = 0
    tabla = tabla.reindex(clases_esperadas)

    # Encabezado
    cabecera = f"  {'Clase':<8}" + "".join(f"{p:>10}" for p in personas) + f"{'TOTAL':>10}"
    print(cabecera)
    print("  " + "-" * (len(cabecera) - 2))

    incompletas = []
    for clase in clases_esperadas:
        fila = tabla.loc[clase] if clase in tabla.index else pd.Series(0, index=personas)
        total = int(fila.sum())
        completa = total >= muestras_esperadas * len(personas)
        marca = "" if completa else " ←"
        valores = "".join(f"{int(fila.get(p, 0)):>10}" for p in personas)
        print(f"  {clase:<8}{valores}{total:>10}{marca}")
        if not completa:
            incompletas.append(clase)

    print(f"\n  Clases incompletas: {len(incompletas)}")
    if incompletas:
        print(f"  {incompletas}")


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def _extraer_puntos(vector: np.ndarray, mano: str) -> np.ndarray | None:
    """Extrae los 21 puntos (x,y) de la mano indicada."""
    inicio = 0 if mano == "izquierda" else NUM_PUNTOS_MANO * NUM_COORDENADAS
    segmento = vector[inicio: inicio + TAMANO_VECTOR_MANO]
    if np.all(np.abs(segmento) < TOLERANCIA_CERO):
        return None
    puntos = segmento.reshape(NUM_PUNTOS_MANO, NUM_COORDENADAS)
    return puntos[:, :2]   # solo x, y


def _dibujar_mano(ax, puntos_xy: np.ndarray, color: str, titulo: str) -> None:
    for a, b in CONEXIONES_MANO:
        ax.plot([puntos_xy[a, 0], puntos_xy[b, 0]],
                [puntos_xy[a, 1], puntos_xy[b, 1]],
                color=color, linewidth=1.2, alpha=0.7)
    ax.scatter(puntos_xy[:, 0], puntos_xy[:, 1],
               c=color, s=18, zorder=3)
    ax.scatter(puntos_xy[0, 0], puntos_xy[0, 1],
               c="red", s=40, zorder=4, label="muñeca")
    ax.set_title(titulo, fontsize=8)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")


def visualizar_clases(df: pd.DataFrame, clases: list,
                      ruta_salida: Path, clase_filtro: str | None = None) -> None:
    clases_a_mostrar = [clase_filtro] if clase_filtro else clases
    ruta_salida.mkdir(parents=True, exist_ok=True)

    for clase in clases_a_mostrar:
        sub = df[df["etiqueta"] == clase]
        if sub.empty:
            print(f"  Sin muestras para clase '{clase}', omitiendo.")
            continue

        muestra = sub.sample(min(9, len(sub)), random_state=42)
        n = len(muestra)
        cols = 3
        filas = (n + cols - 1) // cols

        fig, axes = plt.subplots(filas, cols, figsize=(9, filas * 3))
        axes = np.array(axes).flatten()

        for i, (_, row) in enumerate(muestra.iterrows()):
            vec = row.iloc[3:].astype(float).values
            ax = axes[i]
            dibujado = False
            for mano, color in [("izquierda", "#4a9eff"), ("derecha", "#ff7043")]:
                pts = _extraer_puntos(vec, mano)
                if pts is not None:
                    _dibujar_mano(ax, pts, color, f"muestra {row['id_muestra']} | {row['persona']}")
                    dibujado = True
            if not dibujado:
                ax.text(0.5, 0.5, "sin mano", ha="center", va="center",
                        transform=ax.transAxes, color="red")
                ax.axis("off")

        for j in range(n, len(axes)):
            axes[j].axis("off")

        leyenda = [
            mpatches.Patch(color="#4a9eff", label="Mano izquierda"),
            mpatches.Patch(color="#ff7043", label="Mano derecha"),
            mpatches.Patch(color="red",     label="Muñeca"),
        ]
        fig.legend(handles=leyenda, loc="lower center", ncol=3, fontsize=7)
        fig.suptitle(f"Clase: {clase}  ({n} muestras mostradas)", fontsize=11, fontweight="bold")
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        ruta_img = ruta_salida / f"clase_{clase}.png"
        plt.savefig(ruta_img, dpi=110, bbox_inches="tight")
        plt.close()
        print(f"  Guardado: {ruta_img.name}")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Valida el dataset de landmarks LESHO.")
    parser.add_argument("--modo", choices=["estatico", "dinamico"],
                        default="estatico", help="Dataset a validar.")
    parser.add_argument("--visualizar", action="store_true",
                        help="Genera imágenes PNG de los esqueletos por clase.")
    parser.add_argument("--clase", default=None,
                        help="Analiza y visualiza solo esta clase (ej. A, B, INICIO).")
    args = parser.parse_args()

    ruta = config.RUTA_CSV_ESTATICO if args.modo == "estatico" else config.RUTA_CSV_DINAMICO
    clases = CLASES_ESTATICAS if args.modo == "estatico" else CLASES_DINAMICAS

    df = cargar_csv(ruta)

    print(f"\n{COLORES['titulo']}── Checks estructurales ──{COLORES['reset']}")
    check_columnas(df, args.modo)
    check_nulos(df)
    check_infinitos(df)
    check_clases(df, clases)

    print(f"\n{COLORES['titulo']}── Checks de normalización ──{COLORES['reset']}")
    check_muneca_en_cero(df)
    check_filas_vacias(df)
    check_rango(df)
    check_una_mano_detectada(df)

    resumen_muestras(df, clases)

    if args.visualizar or args.clase:
        print(f"\n{COLORES['titulo']}── Visualización ──{COLORES['reset']}")
        ruta_salida = config.RAIZ_EXPORTS / "validacion"
        visualizar_clases(df, clases, ruta_salida, args.clase)
        print(f"\n  Imágenes guardadas en: {ruta_salida}")

    print(f"\n{COLORES['titulo']}── Validación completa ──{COLORES['reset']}\n")


if __name__ == "__main__":
    main()
