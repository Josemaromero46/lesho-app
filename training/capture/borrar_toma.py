"""
Borra una toma mal grabada del dataset dinamico (Modelo B).

Cada toma es una secuencia completa identificada por (etiqueta, persona,
id_muestra). Si te equivocaste en una toma y ya paso la ventana de la tecla R,
la toma quedo guardada; este script la quita.

Primero LISTA las tomas de una sena para que veas cual borrar; luego borra la
que indiques (por id o la ultima) y renumera las que quedan a 0..k-1, para que
al RETOMAR la captura la reanudacion asigne bien el siguiente id (sin choques).
Hace una copia de respaldo (.csv.bak) antes de escribir.

Uso (desde la carpeta training/):

    python capture/borrar_toma.py MAMA                 # lista las tomas de MAMA
    python capture/borrar_toma.py MAMA --ultima        # borra la ultima toma
    python capture/borrar_toma.py MAMA --id 5          # borra la toma con id 5
    python capture/borrar_toma.py MAMA --ultima --persona jose
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

import config  # noqa: E402


def main(clase, persona, id_toma, ultima, csv) -> None:
    ruta = Path(csv) if csv else config.RUTA_CSV_DINAMICO
    if not ruta.exists():
        print(f"No existe el CSV: {ruta}")
        sys.exit(1)

    clase = clase.strip().upper()
    df = pd.read_csv(ruta)
    sub = df[df.etiqueta == clase]
    if persona:
        sub = sub[sub.persona == persona]
    if sub.empty:
        extra = f" para la persona {persona}" if persona else ""
        print(f"No hay tomas de {clase}{extra} en {ruta.name}.")
        sys.exit(1)

    # Modo listar: sin --id ni --ultima, solo muestra las tomas.
    if id_toma is None and not ultima:
        print(f"Tomas de {clase} en {ruta.name}:")
        for (p, idm), n in sub.groupby(["persona", "id_muestra"]).size().items():
            print(f"  persona={p}   id={idm}   frames={n}")
        print("\nBorrar: agregue --ultima (la de id mas alto) o --id N.")
        return

    # Determinar la persona objetivo.
    personas = sorted(sub.persona.unique())
    if persona is None:
        if len(personas) > 1:
            print(f"Hay varias personas {personas}; indique cual con --persona.")
            sys.exit(1)
        persona = personas[0]

    ids = sorted(df[(df.etiqueta == clase) & (df.persona == persona)].id_muestra.unique())
    objetivo = ids[-1] if ultima else id_toma
    if objetivo not in ids:
        print(f"La toma id={objetivo} no existe en {clase}/{persona}. Ids: {ids}")
        sys.exit(1)

    # Respaldo antes de tocar nada.
    bak = ruta.with_suffix(".csv.bak")
    shutil.copy2(ruta, bak)

    # Borrar la toma objetivo.
    a_borrar = (df.etiqueta == clase) & (df.persona == persona) & (df.id_muestra == objetivo)
    n_frames = int(a_borrar.sum())
    df = df[~a_borrar].copy()

    # Renumerar las tomas restantes de (clase, persona) a 0..k-1.
    m = (df.etiqueta == clase) & (df.persona == persona)
    restantes = sorted(df.loc[m, "id_muestra"].unique())
    remap = {viejo: nuevo for nuevo, viejo in enumerate(restantes)}
    df.loc[m, "id_muestra"] = df.loc[m, "id_muestra"].map(remap)

    df.to_csv(ruta, index=False, encoding="utf-8")
    print(f"Borrada la toma id={objetivo} de {clase}/{persona} ({n_frames} frames).")
    print(f"Quedan {len(restantes)} tomas de {clase} (renumeradas 0..{len(restantes)-1}).")
    print(f"Respaldo del CSV anterior: {bak}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Borra una toma del dataset dinamico.")
    parser.add_argument("clase", help="Sena de la toma a borrar, por ejemplo MAMA.")
    parser.add_argument("--persona", help="Persona (si hay varias en el CSV).")
    parser.add_argument("--id", type=int, dest="id_toma", help="Id de la toma a borrar.")
    parser.add_argument("--ultima", action="store_true", help="Borra la toma de id mas alto.")
    parser.add_argument("--csv", help="Ruta del CSV (por defecto el dinamico del Modelo B).")
    args = parser.parse_args()
    main(args.clase, args.persona, args.id_toma, args.ultima, args.csv)
