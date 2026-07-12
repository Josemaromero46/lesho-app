"""
Carga y division del dataset del Modelo A para el entrenamiento.

Lee el .npz producido por `preprocessing/construir_dataset_a.py` y ofrece dos
formas de dividir en entrenamiento/validacion/prueba:

  - Por persona (recomendada): personas distintas en cada particion, para medir
    si el modelo generaliza a personas que no vio al entrenar. Requiere al menos
    3 personas.
  - Estratificada: division aleatoria manteniendo la proporcion de clases. Se usa
    como respaldo cuando hay muy pocas personas.

Tambien calcula los pesos de clase, necesarios porque las letras con movimiento
generan muchas mas ventanas que las estaticas (desbalance).
"""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight


def cargar_dataset(ruta):
    """Carga X (float32), y (int), personas y grupos (toma) desde el .npz."""
    datos = np.load(ruta, allow_pickle=True)
    X = datos["X"].astype("float32")
    y = datos["y"].astype("int32")
    personas = datos["personas"]
    # Compatibilidad con datasets viejos sin identificador de toma: se usa la
    # persona como grupo (mejor que nada, aunque no evita del todo la fuga).
    grupos = datos["grupos"] if "grupos" in datos.files else personas
    return X, y, personas, grupos


def _reparto_personas(personas_unicas, prop_val, prop_test, semilla):
    """Asigna personas completas a validacion, prueba y entrenamiento."""
    rng = np.random.default_rng(semilla)
    orden = list(rng.permutation(list(personas_unicas)))
    n = len(orden)
    n_val = max(1, round(n * prop_val))
    n_test = max(1, round(n * prop_test))
    # Garantiza al menos una persona en entrenamiento.
    while n_val + n_test >= n:
        if n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break
    val = set(orden[:n_val])
    test = set(orden[n_val:n_val + n_test])
    train = set(orden[n_val + n_test:])
    return train, val, test


def dividir_por_persona(X, y, personas, prop_val, prop_test, semilla):
    """Divide asignando personas completas a cada particion.

    Lanza ValueError si hay menos de 3 personas (usar la division estratificada).
    """
    unicas = sorted(set(personas.tolist()))
    if len(unicas) < 3:
        raise ValueError(
            f"Se necesitan al menos 3 personas para dividir por persona; "
            f"hay {len(unicas)}. Use dividir_estratificado."
        )
    train_p, val_p, test_p = _reparto_personas(unicas, prop_val, prop_test, semilla)

    def _mascara(conjunto):
        return np.array([p in conjunto for p in personas])

    return {
        "train": (X[_mascara(train_p)], y[_mascara(train_p)]),
        "val": (X[_mascara(val_p)], y[_mascara(val_p)]),
        "test": (X[_mascara(test_p)], y[_mascara(test_p)]),
        "personas": {"train": sorted(train_p), "val": sorted(val_p), "test": sorted(test_p)},
    }


def dividir_por_grupo(X, y, grupos, prop_val, prop_test, semilla):
    """Divide por TOMA: ninguna toma cruza de una particion a otra.

    Es la division honesta cuando hay una sola persona. Evita la fuga de datos de
    la division aleatoria por ventana: como las ventanas de una misma toma (y sus
    copias con ruido) son casi identicas, repartirlas entre entrenamiento y prueba
    inflaba la exactitud de forma ficticia. Aqui la prueba se mide sobre tomas que
    el modelo no vio nunca.
    """
    gss_test = GroupShuffleSplit(n_splits=1, test_size=prop_test, random_state=semilla)
    idx_tmp, idx_test = next(gss_test.split(X, y, groups=grupos))

    X_tmp, y_tmp, g_tmp = X[idx_tmp], y[idx_tmp], grupos[idx_tmp]
    prop_val_relativa = prop_val / (1.0 - prop_test)
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=prop_val_relativa, random_state=semilla
    )
    idx_train, idx_val = next(gss_val.split(X_tmp, y_tmp, groups=g_tmp))

    return {
        "train": (X_tmp[idx_train], y_tmp[idx_train]),
        "val": (X_tmp[idx_val], y_tmp[idx_val]),
        "test": (X[idx_test], y[idx_test]),
        "personas": None,
    }


def dividir_estratificado(X, y, prop_val, prop_test, semilla):
    """Division aleatoria estratificada por clase (personas mezcladas)."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=prop_test, random_state=semilla, stratify=y
    )
    prop_val_relativa = prop_val / (1.0 - prop_test)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=prop_val_relativa, random_state=semilla, stratify=y_tmp
    )
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
        "personas": None,
    }


def pesos_de_clase(y, num_clases):
    """Calcula pesos balanceados por clase para compensar el desbalance.

    Devuelve un dict {indice_clase: peso} apto para `model.fit(class_weight=...)`.
    Las clases sin muestras reciben peso 0 para no afectar el entrenamiento.
    """
    presentes = np.unique(y)
    pesos = compute_class_weight("balanced", classes=presentes, y=y)
    mapa = {int(c): float(w) for c, w in zip(presentes, pesos)}
    return {i: mapa.get(i, 0.0) for i in range(num_clases)}
