"""
Aumento de datos (data augmentation) para los landmarks.

Solo se implementan las tecnicas permitidas por CLAUDE.md:

  - Ruido gaussiano leve en las coordenadas (simula el temblor natural de la mano).
  - Variacion de velocidad por remuestreo temporal (solo para secuencias de
    letras con movimiento).

Tecnicas PROHIBIDAS y no implementadas: espejo horizontal, rotacion y escala
extrema, porque cambian el sentido anatomico de la sena en LESHO.
"""

import numpy as np


def ruido_gaussiano(X: np.ndarray, sigma: float, semilla: int | None = None) -> np.ndarray:
    """Anade ruido gaussiano a las coordenadas presentes.

    X: array de forma (..., 126) o (..., n, 126). El ruido se anade solo donde el
    valor no es exactamente cero, para no fabricar una mano donde no la hay (las
    mitades de vector de una mano ausente van en ceros) ni desplazar la muneca,
    que es el origen (0, 0, 0).
    """
    rng = np.random.default_rng(semilla)
    ruido = rng.normal(0.0, sigma, size=X.shape).astype(np.float32)
    mascara = (X != 0.0)
    return (X + ruido * mascara).astype(np.float32)


def variar_velocidad(seq: np.ndarray, factor: float) -> np.ndarray:
    """Remuestrea temporalmente una secuencia para simular otra velocidad.

    factor > 1 alarga la secuencia (gesto mas lento); factor < 1 la acorta (mas
    rapido). Interpola linealmente cada coordenada. Se usa antes de construir
    ventanas, sobre las secuencias de letras con movimiento.

    seq: array (T, 126). Devuelve (T', 126) con T' = round(T * factor).
    """
    total = len(seq)
    if total < 2:
        return seq.astype(np.float32)
    nuevo_total = max(2, int(round(total * factor)))
    destino = np.linspace(0.0, total - 1, nuevo_total)
    origen = np.arange(total)
    salida = np.empty((nuevo_total, seq.shape[1]), dtype=np.float32)
    for columna in range(seq.shape[1]):
        salida[:, columna] = np.interp(destino, origen, seq[:, columna])
    return salida


def remuestrear_a_longitud(seq: np.ndarray, n: int) -> np.ndarray:
    """Remuestrea una secuencia (T, D) a EXACTAMENTE n fotogramas.

    Interpola linealmente cada coordenada. Convierte cualquier duracion de sena a
    la longitud fija que espera el Modelo B, sin padding ni enmascarado (que
    complican la conversion a TFLite en el movil). Una sena mas larga se comprime,
    una mas corta se estira; el gesto se conserva completo.

    seq: array (T, D). Devuelve (n, D) en float32.
    """
    seq = np.asarray(seq, dtype=np.float32)
    total = len(seq)
    if total == n:
        return seq
    if total == 0:
        return np.zeros((n, seq.shape[1]), dtype=np.float32)
    if total == 1:
        return np.repeat(seq[:1], n, axis=0).astype(np.float32)
    destino = np.linspace(0.0, total - 1, n)
    origen = np.arange(total)
    salida = np.empty((n, seq.shape[1]), dtype=np.float32)
    for columna in range(seq.shape[1]):
        salida[:, columna] = np.interp(destino, origen, seq[:, columna])
    return salida


def expandir_solo_ubicacion(X, y, personas, grupos, tamano_config: int):
    """Agrega una copia de cada secuencia con la FORMA de las manos borrada.

    En las copias, los primeros `tamano_config` valores de cada fotograma (la
    configuracion de las manos) van en cero y solo queda la UBICACION en el
    cuerpo. Obliga al modelo a aprender el lugar de articulacion: sin esto,
    cuando la forma alcanza para separar las clases del entrenamiento, el modelo
    la usa como atajo e ignora el lugar, y senas de mano parecida en lugares
    distintos (PAPA en la cara, HOSPITAL en el hombro) se confunden en vivo.

    Las copias heredan el identificador de toma (no cruzan particiones).
    """
    if len(X) == 0:
        return X, y, personas, grupos
    X_solo = X.copy()
    X_solo[:, :, :tamano_config] = 0.0
    return (
        np.concatenate([X, X_solo], axis=0),
        np.concatenate([y, y], axis=0),
        np.concatenate([personas, personas], axis=0),
        np.concatenate([grupos, grupos], axis=0),
    )


def expandir_con_ruido(
    X: np.ndarray, y: np.ndarray, personas: np.ndarray, grupos: np.ndarray,
    sigma: float, copias: int = 1, semilla: int | None = None,
):
    """Duplica el dataset agregando `copias` versiones con ruido de cada ventana.

    Devuelve (X_ampliado, y_ampliado, personas_ampliado, grupos_ampliado). La
    version original se conserva sin ruido. Cada copia con ruido HEREDA el grupo
    (toma) de su ventana original, para que al dividir por toma las copias no
    crucen de entrenamiento a prueba. Util para robustecer el Modelo A ante
    pequenas variaciones sin grabar mas datos.
    """
    if copias <= 0 or len(X) == 0:
        return X, y, personas, grupos
    bloques_X = [X]
    bloques_y = [y]
    bloques_p = [personas]
    bloques_g = [grupos]
    for c in range(copias):
        semilla_c = None if semilla is None else semilla + c
        bloques_X.append(ruido_gaussiano(X, sigma, semilla_c))
        bloques_y.append(y)
        bloques_p.append(personas)
        bloques_g.append(grupos)
    return (
        np.concatenate(bloques_X, axis=0),
        np.concatenate(bloques_y, axis=0),
        np.concatenate(bloques_p, axis=0),
        np.concatenate(bloques_g, axis=0),
    )
