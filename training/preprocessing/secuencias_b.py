"""
Preprocesamiento de las secuencias del Modelo B.

Convierte las secuencias capturadas (de longitud variable, 132 valores por
fotograma) en tensores de longitud fija listos para entrenar el clasificador de
secuencias. El orden de las transformaciones coincide con lo que hara la app en
vivo, para que entrenamiento e inferencia vean lo mismo:

  1. Suavizado One Euro sobre los 132 valores (quita el temblor, igual que en el
     Modelo A). Se hace a la tasa original de fotogramas.
  2. Normalizacion por escala SOLO de la configuracion de las manos (los primeros
     126). La ubicacion en el cuerpo (ultimos 6) ya viene en anchos de hombro, o
     sea invariante a la distancia, asi que no se toca.
  3. Remuestreo a longitud fija (config.LONGITUD_FIJA_SECUENCIA). Toda sena queda
     con la misma cantidad de fotogramas, sin padding ni enmascarado.

No entrena ni depende de TensorFlow; produce arrays de numpy.
"""

import numpy as np

from comun.definiciones import (
    CLASES_DINAMICAS,
    GANANCIA_UBICACION,
    TAMANO_ENTRADA_B,
    TAMANO_VECTOR,
)
from comun.normalizacion import escalar_lote
from comun.suavizado import suavizar_secuencia
from preprocessing.augmentacion import remuestrear_a_longitud


# Margen (fotogramas) que se deja a cada lado del nucleo activo al recortar.
MARGEN_RECORTE = 3
# Minimo de fotogramas tras el recorte. Si el nucleo activo queda mas corto que
# esto, NO se recorta (se devuelve la secuencia entera): evita destruir senas muy
# breves o casi quietas.
MIN_TRAS_RECORTE = 12


def recortar_quietos(
    seq: np.ndarray, margen: int = MARGEN_RECORTE, min_frames: int = MIN_TRAS_RECORTE
) -> np.ndarray:
    """Recorta los tramos QUIETOS de los bordes, dejando el nucleo con movimiento.

    Muchas senas hacen su movimiento y luego la mano queda sostenida y quieta
    (HOLA: saluda y deja la mano arriba). Esa cola quieta no identifica la sena y,
    al grabarse, hacia que el modelo aprendiera la sena por cuanto se sostiene en
    vez de por su movimiento (habia que sostenerla para reconocerla). Este recorte
    deja cada sena representada por su MOVIMIENTO ACTIVO, igual en entrenamiento e
    inferencia (por eso vive aqui, en procesar_secuencia, y no solo en el demo).

    La actividad de cada fotograma combina el cambio de la FORMA de las manos
    (escalado, invariante a la distancia) y el DESPLAZAMIENTO de las manos por el
    cuerpo (para no borrar senas que viajan con la mano rigida, como IR o DAR). Se
    conserva el tramo entre la primera y la ultima actividad clara, con un margen.
    Si el nucleo resulta muy corto, se deja la secuencia intacta.

    seq: (T, TAMANO_VECTOR_B) suavizada. Devuelve (T', TAMANO_VECTOR_B) con
    T' <= T.
    """
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) < min_frames + 2 * margen:
        return seq

    conf = escalar_lote(seq[:, :TAMANO_VECTOR])
    ubic_manos = seq[:, TAMANO_VECTOR:TAMANO_VECTOR + 4]  # ubicacion de las 2 manos

    actividad = np.zeros(len(seq), dtype=np.float32)
    for i in range(1, len(seq)):
        d = np.abs(conf[i] - conf[i - 1])
        pres_c = conf[i] != 0.0
        a = float(d[pres_c].mean()) if pres_c.any() else 0.0
        du = np.abs(ubic_manos[i] - ubic_manos[i - 1])
        pres_u = ubic_manos[i] != 0.0
        b = float(du[pres_u].mean()) if pres_u.any() else 0.0
        actividad[i] = a + 2.0 * b   # el desplazamiento pesa mas (viaja la mano)

    umbral = max(0.15 * float(np.percentile(actividad, 95)), 1e-4)
    indices = np.where(actividad > umbral)[0]
    if indices.size == 0:
        return seq   # sin movimiento claro: es una sena casi quieta, se deja igual
    a0 = max(0, int(indices[0]) - margen)
    a1 = min(len(seq), int(indices[-1]) + margen + 1)
    recortada = seq[a0:a1]
    if len(recortada) < min_frames:
        return seq   # no recortar de mas
    return recortada


def _rasgos_relativos(ubic: np.ndarray) -> np.ndarray:
    """Posicion de cada mano respecto a la nariz y a la boca.

    ubic: (T, 14) el bloque de ubicacion CRUDO. Devuelve (T, 8):
    (izq-nariz, izq-boca, der-nariz, der-boca), cada uno (x, y). Si la mano o el
    ancla no estan presentes (valores en cero), el rasgo va en cero.
    """
    def presente(p):
        return (np.abs(p).sum(axis=1, keepdims=True) > 1e-9).astype(np.float32)

    # Ubicacion (18): 0-1 muñeca izq, 2-3 muñeca der, 4-5 punta indice izq,
    # 6-7 punta indice der, 8-9 nariz, 10-11 ojos, 12-13 boca, 14-15/16-17 orejas.
    # Los rasgos usan la PUNTA DEL INDICE (a donde apunta la mano).
    punta_izq, punta_der = ubic[:, 4:6], ubic[:, 6:8]
    nariz, boca = ubic[:, 8:10], ubic[:, 12:14]

    def rel(punta, ancla):
        return (punta - ancla) * (presente(punta) * presente(ancla))

    return np.concatenate(
        [rel(punta_izq, nariz), rel(punta_izq, boca),
         rel(punta_der, nariz), rel(punta_der, boca)],
        axis=1,
    ).astype(np.float32)


def procesar_secuencia(
    seq: np.ndarray, longitud: int,
    suavizar: bool = True, fps: float = 30.0,
    min_cutoff: float = 1.0, beta: float = 1.0, d_cutoff: float = 1.0,
) -> np.ndarray:
    """Aplica suavizado, escala de la configuracion y remuestreo a `longitud`.

    seq: (T, 132). Devuelve (longitud, 132).
    """
    seq = np.asarray(seq, dtype=np.float32)
    if suavizar and len(seq) > 0:
        seq = suavizar_secuencia(seq, fps, min_cutoff, beta, d_cutoff)

    # Recorte de tramos quietos: deja la sena representada por su movimiento
    # activo (no por cuanto se sostiene). Se hace ANTES de escalar y remuestrear,
    # igual en entrenamiento e inferencia.
    if len(seq) > 0:
        seq = recortar_quietos(seq)

    # Forma de las manos: solo se le aplica la escala (invariante a la distancia).
    config = escalar_lote(seq[:, :TAMANO_VECTOR])
    # Ubicacion en el cuerpo (14) mas rasgos relativos a la cara (8). Ya son
    # invariantes a la distancia; se les aplica la GANANCIA del contrato para que
    # el lugar de articulacion pese frente a la forma (senas de mano parecida en
    # lugares distintos: PAPA/HOSPITAL, PAPA/MAMA).
    ubic_cruda = seq[:, TAMANO_VECTOR:]
    relativos = _rasgos_relativos(ubic_cruda)
    ubicacion = np.concatenate([ubic_cruda, relativos], axis=1) * GANANCIA_UBICACION

    seq = np.concatenate([config, ubicacion], axis=1)
    assert seq.shape[1] == TAMANO_ENTRADA_B
    return remuestrear_a_longitud(seq, longitud)


def construir_dataset_modelo_b(
    grupos: dict,
    clases=CLASES_DINAMICAS,
    longitud: int = 40,
    suavizar: bool = True,
    fps: float = 30.0,
    min_cutoff: float = 1.0,
    beta: float = 1.0,
    d_cutoff: float = 1.0,
):
    """Ensambla el dataset del Modelo B a partir de las secuencias cargadas.

    grupos: {(clase, persona, id_muestra): array (T, 132)} (de cargar_movimiento).
    clases: lista ordenada de clases (define el indice de salida del modelo).

    Devuelve:
      X        : ndarray float32 (M, longitud, 132)
      y        : ndarray int16 (M,), indice de clase en `clases`
      personas : ndarray (M,), identificador de persona por secuencia
      grupos   : ndarray (M,), identificador de TOMA por secuencia (para dividir
                 sin fuga, igual que en el Modelo A)
    """
    X: list = []
    y: list = []
    personas: list = []
    grupos_id: list = []
    for (clase, persona, idm), seq in grupos.items():
        X.append(procesar_secuencia(
            seq, longitud, suavizar, fps, min_cutoff, beta, d_cutoff
        ))
        y.append(clases.index(clase))
        personas.append(persona)
        grupos_id.append(f"B:{clase}:{persona}:{idm}")

    if not X:
        vacio = np.empty((0, longitud, TAMANO_ENTRADA_B), dtype=np.float32)
        return (vacio, np.empty((0,), dtype=np.int16),
                np.empty((0,), dtype=object), np.empty((0,), dtype=object))

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int16),
        np.asarray(personas, dtype=object),
        np.asarray(grupos_id, dtype=object),
    )
