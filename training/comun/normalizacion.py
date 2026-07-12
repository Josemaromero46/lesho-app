"""
Normalizacion de landmarks respecto a la muneca.

Este modulo contiene el CONTRATO MAS CRITICO del proyecto. La funcion
`normalizar_respecto_muneca` produce el vector de 63 valores que alimenta a los
modelos. Este mismo algoritmo se replica en la app Flutter
(app/lib/core/normalizacion.dart). Si las dos versiones difieren aunque sea en
el orden de las coordenadas, el modelo entrenado dejara de funcionar en la app.

Cualquier cambio aqui obliga a cambiar igual la version en Dart.

El modulo es puro: no depende de MediaPipe ni de OpenCV, solo de listas de
numeros. Asi se puede importar y probar sin instalar dependencias pesadas.
"""

from .definiciones import (
    INDICE_MUNECA,
    NUM_LANDMARKS,
    TAMANO_VECTOR,
    TAMANO_VECTOR_MANO,
)

# Un landmark es una secuencia de 3 numeros: (x, y, z).
Punto = "tuple[float, float, float]"


def normalizar_respecto_muneca(landmarks_crudos) -> list[float]:
    """Normaliza los 21 landmarks de UNA mano restando la posicion de la muneca.

    Parametros
    ----------
    landmarks_crudos : secuencia de 21 puntos (x, y, z)
        Las coordenadas crudas que entrega MediaPipe Hands. Cada punto es una
        secuencia indexable de tres numeros: punto[0]=x, punto[1]=y, punto[2]=z.

    Retorna
    -------
    list[float] de longitud 63
        Vector aplanado en el orden:
        [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
        donde a cada coordenada se le resto la coordenada de la muneca
        (landmark 0). Por construccion, el landmark 0 queda en (0, 0, 0); se
        conserva en el vector para mantener los 63 valores por mano.

    Lanza
    -----
    ValueError
        Si la cantidad de landmarks recibidos no es exactamente 21.

    Detalle del contrato
    --------------------
    1. La normalizacion base es de TRASLACION: se traslada el sistema de
       coordenadas para que el origen sea la muneca. Esto hace al vector
       independiente de donde este la mano dentro del cuadro de la camara.
    2. No se aplica normalizacion por escala en esta version base. Si en el
       futuro se decide agregar escala (dividir por una distancia
       caracteristica), debe agregarse aqui y en la version Dart de forma
       identica, y volver a entrenar los modelos.
    """
    if len(landmarks_crudos) != NUM_LANDMARKS:
        raise ValueError(
            f"Se esperaban {NUM_LANDMARKS} landmarks, "
            f"se recibieron {len(landmarks_crudos)}."
        )

    muneca = landmarks_crudos[INDICE_MUNECA]
    origen_x, origen_y, origen_z = muneca[0], muneca[1], muneca[2]

    vector: list[float] = []
    for punto in landmarks_crudos:
        vector.append(float(punto[0]) - float(origen_x))
        vector.append(float(punto[1]) - float(origen_y))
        vector.append(float(punto[2]) - float(origen_z))

    # Garantia del contrato: el vector de una mano siempre tiene 63 valores.
    assert len(vector) == TAMANO_VECTOR_MANO, (
        f"El vector de una mano debe tener {TAMANO_VECTOR_MANO} valores, "
        f"tiene {len(vector)}."
    )
    return vector


def componer_vector_dos_manos(landmarks_izquierda, landmarks_derecha) -> list[float]:
    """Compone el vector de 126 valores a partir de las dos manos.

    Parametros
    ----------
    landmarks_izquierda : secuencia de 21 puntos (x, y, z) o None
        Landmarks crudos de la mano etiquetada como izquierda, o None si esa
        mano no se detecto en el fotograma.
    landmarks_derecha : secuencia de 21 puntos (x, y, z) o None
        Lo mismo para la mano derecha.

    Retorna
    -------
    list[float] de longitud 126
        Concatenacion de la mano izquierda (63 valores) y la derecha (63). Cada
        mano se normaliza respecto a SU PROPIA muneca. La mitad de una mano
        ausente se rellena con ceros.

    Detalle del contrato
    --------------------
    - Orden fijo: primero la mano izquierda, luego la derecha. Este orden debe
      ser identico en la captura y en la app.
    - Cada mano se normaliza por separado respecto a su muneca. Esta version no
      preserva la posicion relativa entre las dos manos; es una decision abierta
      a revisar si alguna sena bimanual lo requiere (ver PLAN_CODIGO.md).
    """
    izquierda = (normalizar_respecto_muneca(landmarks_izquierda)
                 if landmarks_izquierda else [0.0] * TAMANO_VECTOR_MANO)
    derecha = (normalizar_respecto_muneca(landmarks_derecha)
               if landmarks_derecha else [0.0] * TAMANO_VECTOR_MANO)

    vector = izquierda + derecha
    assert len(vector) == TAMANO_VECTOR, (
        f"El vector de dos manos debe tener {TAMANO_VECTOR} valores, "
        f"tiene {len(vector)}."
    )
    return vector


# ---------------------------------------------------------------------------
# Normalizacion por escala (invariancia a la distancia a la camara)
# ---------------------------------------------------------------------------

# Nudillos (MCP) de los cuatro dedos largos. Su distancia a la muneca es un
# tamano estable de la mano: no cambia al flexionar los dedos, pero si crece o
# se encoge segun que tan cerca o lejos este la mano de la camara.
_NUDILLOS = (5, 9, 13, 17)


def _tamano_mano(mano) -> float:
    """Tamano caracteristico de una mano ya trasladada (muneca en el origen).

    Distancia media, en el plano x-y, de la muneca a los cuatro nudillos. Se usa
    el plano x-y porque refleja el tamano aparente de la mano en la imagen, que
    es lo que cambia con la distancia a la camara.
    """
    total = 0.0
    for k in _NUDILLOS:
        x = mano[k * 3]
        y = mano[k * 3 + 1]
        total += (x * x + y * y) ** 0.5
    return total / len(_NUDILLOS)


def escalar_vector(vector) -> list[float]:
    """Normaliza la ESCALA de un vector de 126 valores (dos manos).

    Divide cada mano por su tamano caracteristico, para que el vector no dependa
    del tamano aparente de la mano, es decir, de la distancia a la camara. Cada
    mano se escala por SU propio tamano; la mitad de una mano ausente (ceros) se
    deja igual.

    Se aplica DESPUES de componer_vector_dos_manos, tanto en el preprocesamiento
    como en la inferencia (demo y app). NO se aplica en la captura: el CSV guarda
    los vectores solo trasladados, y la escala se agrega igual en entrenamiento e
    inferencia. Debe replicarse identica en la version Dart de la app.
    """
    salida: list[float] = []
    for inicio in (0, TAMANO_VECTOR_MANO):
        mano = vector[inicio:inicio + TAMANO_VECTOR_MANO]
        if all(abs(v) < 1e-9 for v in mano):
            salida.extend(mano)  # mano ausente: se queda en ceros
            continue
        escala = _tamano_mano(mano)
        if escala < 1e-6:
            salida.extend(mano)
        else:
            salida.extend(v / escala for v in mano)
    return salida


def escalar_lote(X):
    """Version vectorizada de `escalar_vector` para arrays numpy (..., 126).

    Requiere numpy. Devuelve un array de la misma forma con cada mano escalada
    por su tamano. Las manos ausentes (todo ceros) quedan igual.
    """
    import numpy as np

    X = np.asarray(X, dtype=np.float32)
    forma = X.shape
    plano = X.reshape(-1, TAMANO_VECTOR).copy()
    for inicio in (0, TAMANO_VECTOR_MANO):
        mano = plano[:, inicio:inicio + TAMANO_VECTOR_MANO]
        presente = np.abs(mano).sum(axis=1) > 1e-9
        tam = np.zeros(len(mano), dtype=np.float32)
        for k in _NUDILLOS:
            x = mano[:, k * 3]
            y = mano[:, k * 3 + 1]
            tam += np.sqrt(x * x + y * y)
        tam /= len(_NUDILLOS)
        factor = np.where(presente & (tam > 1e-6), 1.0 / np.maximum(tam, 1e-6), 1.0)
        plano[:, inicio:inicio + TAMANO_VECTOR_MANO] = mano * factor[:, None]
    return plano.reshape(forma)


# ---------------------------------------------------------------------------
# Caso de prueba de sincronizacion con la version Dart.
#
# Si se ejecuta este archivo directamente (python comun/normalizacion.py),
# imprime el resultado de un caso conocido. La version Dart debe producir
# exactamente el mismo vector para la misma entrada. Sirve para verificar a
# mano que ambas implementaciones coinciden.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Entrada artificial: 21 puntos donde el punto i es (i, i*2, i*3).
    entrada = [(i, i * 2.0, i * 3.0) for i in range(NUM_LANDMARKS)]

    una = normalizar_respecto_muneca(entrada)
    print("Una mano -> longitud:", len(una), "(esperado 63)")
    print("Landmark 0 normalizado (debe ser 0,0,0):", una[0:3])

    dos = componer_vector_dos_manos(entrada, None)
    print("Dos manos (solo izquierda) -> longitud:", len(dos), "(esperado 126)")
    print("Mitad derecha en ceros:", all(v == 0.0 for v in dos[63:]))
