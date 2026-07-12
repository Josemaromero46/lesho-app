"""
Construccion de las ventanas temporales que entran al Modelo A.

El Modelo A no clasifica un fotograma suelto, sino una ventana de N fotogramas
(TAMANO_VENTANA_A). Este modulo convierte las secuencias agrupadas (poses y
letras con movimiento) en ventanas de forma [N, 126], etiquetadas con el indice
de clase segun CLASES_ESTATICAS.

Diseno (ver PLAN_CODIGO.md y CLAUDE.md):
  - Letras estaticas + control: se graban como muchas TOMAS separadas de la pose.
    Cada toma es una formacion independiente (variedad real). Como la mano esta
    quieta dentro de una toma, sus ventanas son casi identicas, asi que se
    conservan solo unas pocas por toma; la variedad viene de tener muchas tomas.
  - Letras con movimiento (J, Ñ, Z, LL, RR): la ventana debe contener el gesto.
    Se recorren ventanas deslizantes sobre cada toma y se conservan solo las que
    tienen movimiento suficiente, con un tope por toma para no aplastar a las
    estaticas en cantidad.

Cada ventana lleva un identificador de TOMA (grupo). Todas las ventanas de una
misma toma comparten grupo, para poder dividir el dataset por toma y no dejar
ventanas casi identicas repartidas entre entrenamiento y prueba (fuga de datos
que inflaba falsamente la exactitud).

La ventana en tiempo real de la app es un buffer rodante de N fotogramas a la
misma tasa (30 fps), asi que estas ventanas de entrenamiento representan lo mismo
que la app vera en vivo.
"""

import numpy as np

from comun.definiciones import CLASES_ESTATICAS, TAMANO_VECTOR, TAMANO_VENTANA_A
from comun.normalizacion import escalar_lote
from comun.suavizado import suavizar_secuencia


def _indices_inicio_ventana(total: int, n: int, paso: int):
    """Indices de inicio de las ventanas deslizantes, o None si hay que rellenar.

    Devuelve None cuando la secuencia es mas corta que la ventana (se rellenara
    repitiendo el ultimo fotograma). Siempre incluye la ventana final (cola) para
    no perder el tramo del final.
    """
    if total < n:
        return None
    inicios = list(range(0, total - n + 1, paso))
    if inicios[-1] != total - n:
        inicios.append(total - n)
    return inicios


def _rellenar(seq: np.ndarray, n: int) -> np.ndarray:
    """Lleva una secuencia mas corta que n a longitud n repitiendo el ultimo frame."""
    faltan = n - len(seq)
    cola = np.repeat(seq[-1:], faltan, axis=0)
    return np.concatenate([seq, cola], axis=0)


def ventanas_de_secuencia(seq: np.ndarray, n: int, paso: int) -> list[np.ndarray]:
    """Parte una secuencia (T, 126) en ventanas (n, 126) deslizantes."""
    inicios = _indices_inicio_ventana(len(seq), n, paso)
    if inicios is None:
        return [_rellenar(seq, n)]
    return [seq[i : i + n] for i in inicios]


def velocidad_media(ventana: np.ndarray) -> float:
    """Movimiento medio de una ventana: desplazamiento medio entre frames.

    Se ignoran las coordenadas exactamente en cero (mano ausente o muneca en el
    origen) para no diluir la medida con mitades de vector vacias.
    """
    if len(ventana) < 2:
        return 0.0
    dif = np.abs(np.diff(ventana, axis=0))
    presentes = ventana[1:] != 0.0
    if not presentes.any():
        return 0.0
    return float(dif[presentes].mean())


def _repartidas(items: list, maximo: int) -> list:
    """Devuelve hasta `maximo` elementos repartidos uniformemente de la lista."""
    if maximo <= 0 or len(items) <= maximo:
        return list(items)
    paso = len(items) / maximo
    return [items[int(i * paso)] for i in range(maximo)]


def ventanas_estaticas(grupos_estaticos: dict, n: int, paso: int,
                       por_toma: int = 2, movimiento_max: float = 0.0):
    """Ventanas de las clases de pose fija, a partir de tomas separadas.

    Cada toma (clase, persona, id) es una formacion corta de la pose. Como la
    mano esta quieta, sus ventanas son casi identicas; se conservan solo hasta
    `por_toma` por toma. Todas las ventanas de una toma comparten identificador
    de grupo.

    Si `movimiento_max` > 0, se descartan las ventanas cuyo movimiento medio lo
    supere: son el arranque de la toma, con la mano entrando a la pose, no la pose
    quieta. Sin esto, una L entrando a cuadro se parece a una LL y contamina la
    clase estatica. Si NINGUNA ventana de la toma queda por debajo del tope, se
    descarta la toma completa: es una toma donde la mano nunca se sostuvo quieta,
    y guardar su ventana menos mala solo le ensenaria al modelo que la letra
    estatica lleva movimiento (justo lo que la confunde con la de movimiento).

    Devuelve (lista de ventanas (n,126), lista de (clase, persona, grupo)).
    """
    ventanas: list[np.ndarray] = []
    meta: list[tuple[str, str, str]] = []
    for (clase, persona, idm), seq in grupos_estaticos.items():
        grupo = f"E:{clase}:{persona}:{idm}"
        candidatas = ventanas_de_secuencia(seq, n, paso)
        if movimiento_max > 0:
            candidatas = [v for v in candidatas if velocidad_media(v) < movimiento_max]
            if not candidatas:
                continue  # toma sin pose quieta: se descarta entera
        for ventana in _repartidas(candidatas, por_toma):
            ventanas.append(ventana)
            meta.append((clase, persona, grupo))
    return ventanas, meta


def ventanas_movimiento(grupos_movimiento: dict, n: int, paso: int,
                        umbral_rel: float, umbral_abs: float = 0.0,
                        por_toma: int = 0):
    """Ventanas de las letras con movimiento, filtrando las quietas.

    De cada secuencia se conservan las ventanas cuyo movimiento medio supere DOS
    condiciones: `umbral_rel` veces la velocidad maxima de esa misma secuencia
    (adaptativo) y `umbral_abs` como piso fijo (evita que una sena lenta deje
    pasar ventanas casi quietas, que se confundirian con la letra estatica de la
    misma forma de mano). Si ninguna ventana pasa, se conserva la de mayor
    movimiento para no perder la muestra. Si `por_toma` > 0, se limita a esa
    cantidad de ventanas por toma, repartidas a lo largo de la trayectoria, para
    no aplastar en cantidad a las clases estaticas.

    Todas las ventanas de una toma comparten identificador de grupo.

    Devuelve (lista de ventanas (n,126), lista de (clase, persona, grupo)).
    """
    ventanas: list[np.ndarray] = []
    meta: list[tuple[str, str, str]] = []
    for (clase, persona, idm), seq in grupos_movimiento.items():
        grupo = f"M:{clase}:{persona}:{idm}"
        candidatas = ventanas_de_secuencia(seq, n, paso)
        velocidades = [velocidad_media(v) for v in candidatas]
        if not velocidades:
            continue
        v_max = max(velocidades)
        umbral = max(umbral_rel * v_max, umbral_abs)
        seleccion = [
            v for v, vel in zip(candidatas, velocidades) if vel >= umbral
        ]
        if not seleccion:
            seleccion = [candidatas[int(np.argmax(velocidades))]]
        for ventana in _repartidas(seleccion, por_toma):
            ventanas.append(ventana)
            meta.append((clase, persona, grupo))
    return ventanas, meta


def construir_dataset_modelo_a(
    grupos_estaticos: dict,
    grupos_movimiento: dict,
    n: int = TAMANO_VENTANA_A,
    paso: int = 5,
    umbral_rel: float = 0.4,
    umbral_abs: float = 0.0,
    por_toma_estatica: int = 2,
    por_toma_movimiento: int = 0,
    estatico_movimiento_max: float = 0.0,
    suavizar: bool = True,
    fps: float = 30.0,
    min_cutoff: float = 1.0,
    beta: float = 1.0,
    d_cutoff: float = 1.0,
):
    """Ensambla el dataset completo del Modelo A a partir de los grupos cargados.

    Devuelve:
      X        : ndarray float32 de forma (M, n, 126)
      y        : ndarray int16 de forma (M,), indice de clase en CLASES_ESTATICAS
      personas : ndarray de forma (M,), identificador de persona por ventana
      grupos   : ndarray de forma (M,), identificador de TOMA por ventana

    El indice de clase permite mapear de vuelta al nombre con
    CLASES_ESTATICAS[indice]. El identificador de toma permite dividir el dataset
    sin fuga (ventanas de una misma toma no cruzan de entrenamiento a prueba).

    Orden de las transformaciones (debe coincidir con la inferencia):
    suavizado One Euro por toma -> normalizacion por escala -> ventanas.
    """
    # Suavizado temporal: quita el temblor de los landmarks fotograma a fotograma,
    # de forma causal por toma (igual que en vivo). Se hace ANTES de la escala,
    # sobre el vector trasladado, exactamente como en la demo y la app.
    if suavizar:
        grupos_estaticos = {
            k: suavizar_secuencia(v, fps, min_cutoff, beta, d_cutoff)
            for k, v in grupos_estaticos.items()
        }
        grupos_movimiento = {
            k: suavizar_secuencia(v, fps, min_cutoff, beta, d_cutoff)
            for k, v in grupos_movimiento.items()
        }

    # Normalizacion por escala: cada fotograma se hace invariante al tamano
    # aparente de la mano (distancia a la camara) antes de armar las ventanas.
    grupos_estaticos = {k: escalar_lote(v) for k, v in grupos_estaticos.items()}
    grupos_movimiento = {k: escalar_lote(v) for k, v in grupos_movimiento.items()}

    v_est, m_est = ventanas_estaticas(
        grupos_estaticos, n, paso, por_toma_estatica, estatico_movimiento_max
    )
    v_mov, m_mov = ventanas_movimiento(
        grupos_movimiento, n, paso, umbral_rel, umbral_abs, por_toma_movimiento
    )

    ventanas = v_est + v_mov
    meta = m_est + m_mov
    if not ventanas:
        vacio = np.empty((0, n, TAMANO_VECTOR), dtype=np.float32)
        return (
            vacio,
            np.empty((0,), dtype=np.int16),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=object),
        )

    X = np.asarray(ventanas, dtype=np.float32)
    y = np.array([CLASES_ESTATICAS.index(clase) for clase, _, _ in meta], dtype=np.int16)
    personas = np.array([persona for _, persona, _ in meta], dtype=object)
    grupos = np.array([grupo for _, _, grupo in meta], dtype=object)
    return X, y, personas, grupos
