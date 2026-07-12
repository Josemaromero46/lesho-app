"""
Vector de entrada del Modelo B por fotograma (contrato de datos).

Es el equivalente de `normalizacion.py` pero para el Modelo B: define como se
arma el vector de 132 valores que entra al clasificador de secuencias. Igual que
aquel, es PURO (no depende de mediapipe ni opencv) para poder probarse aislado, y
cualquier cambio aqui debera replicarse en la app.

Estructura del vector (ver definiciones.py):
  [0 : 126]   CONFIGURACION de las dos manos. Exactamente el mismo vector del
              Modelo A: componer_vector_dos_manos, landmarks relativos a la
              muneca (traslacion). La escala se aplica despues, en el
              preprocesamiento, igual que en el Modelo A. Captura FORMA y
              movimiento de las manos.
  [126 : 140] UBICACION en el cuerpo, en anchos de hombro respecto al centro de
              los hombros, en este orden (7 puntos, 14 valores):
                126,127 -> mano izquierda (rx, ry)
                128,129 -> mano derecha  (rx, ry)
                130,131 -> nariz         (rx, ry)
                132,133 -> ojos (centro) (rx, ry)
                134,135 -> boca (centro) (rx, ry)
                136,137 -> oreja izquierda (rx, ry)
                138,139 -> oreja derecha   (rx, ry)
              Las manos dan DONDE se hace la sena; las cinco anclas de la cara
              forman un marco facial (ojos arriba, nariz al medio, boca abajo,
              orejas a los lados) para distinguir zonas finas de la cara. Ya es
              invariante a la distancia (dividido por el ancho de hombros), asi
              que NO se le aplica la normalizacion por escala de las manos.

Convenciones de valores ausentes (coherentes con el Modelo A):
  - Mano no detectada: sus 63 valores de configuracion Y sus 2 de ubicacion van
    en cero.
  - Sin marco del cuerpo (Pose no detecto los hombros): los 14 valores de
    ubicacion van en cero. En la practica, el flujo de captura mantiene el ultimo
    marco valido, porque el torso es estable (ver captura dinamica).
  - Ancla de la cara no fiable (por ejemplo la boca tapada): sus 2 valores van en
    cero.
"""

from .definiciones import (
    INDICE_MUNECA,
    INDICE_PUNTA_INDICE,
    TAMANO_UBICACION,
    TAMANO_VECTOR,
    TAMANO_VECTOR_B,
)
from .normalizacion import componer_vector_dos_manos


def _ubicacion_de_punto(landmarks, indice, marco) -> list[float]:
    """(rx, ry) de un landmark de la mano en el marco del cuerpo, o [0, 0]."""
    if not landmarks:
        return [0.0, 0.0]
    punto = landmarks[indice]
    rx, ry = marco.ubicacion_relativa(float(punto[0]), float(punto[1]))
    return [rx, ry]


def _ubicacion(landmarks_izquierda, landmarks_derecha, marco) -> list[float]:
    """Los 18 valores de ubicacion: muñecas + puntas del indice + anclas de la cara.

    Orden (CONTRATO): muñeca izq, muñeca der, punta indice izq, punta indice der,
    nariz, ojos, boca, oreja izq, oreja der. Si no hay marco, todo va en cero.
    Cada punto/ancla ausente aporta [0, 0].
    """
    if marco is None:
        return [0.0] * TAMANO_UBICACION

    return (
        _ubicacion_de_punto(landmarks_izquierda, INDICE_MUNECA, marco)
        + _ubicacion_de_punto(landmarks_derecha, INDICE_MUNECA, marco)
        + _ubicacion_de_punto(landmarks_izquierda, INDICE_PUNTA_INDICE, marco)
        + _ubicacion_de_punto(landmarks_derecha, INDICE_PUNTA_INDICE, marco)
        + marco.rel_ancla(marco.nariz)
        + marco.rel_ancla(marco.ojos)
        + marco.rel_ancla(marco.boca)
        + marco.rel_ancla(marco.oreja_izq)
        + marco.rel_ancla(marco.oreja_der)
    )


def componer_vector_modelo_b(landmarks_izquierda, landmarks_derecha, marco) -> list[float]:
    """Arma el vector de 132 valores de un fotograma del Modelo B.

    Parametros
    ----------
    landmarks_izquierda, landmarks_derecha : secuencia de 21 puntos (x, y, z) o None
        Landmarks crudos de cada mano (como los entrega MediaPipe Hands), o None
        si esa mano no se detecto.
    marco : MarcoCuerpo | None
        Marco de referencia del cuerpo del fotograma (de `comun.marco`), o None si
        no hay cuerpo de referencia.

    Retorna
    -------
    list[float] de longitud 132: configuracion (126) + ubicacion (6).
    """
    config = componer_vector_dos_manos(landmarks_izquierda, landmarks_derecha)  # 126
    ubic = _ubicacion(landmarks_izquierda, landmarks_derecha, marco)            # 6
    vector = config + ubic
    assert len(vector) == TAMANO_VECTOR_B, (
        f"El vector del Modelo B debe tener {TAMANO_VECTOR_B} valores, "
        f"tiene {len(vector)}."
    )
    return vector


def separar_config_ubicacion(vector_b):
    """Divide un vector del Modelo B en (configuracion 126, ubicacion 6).

    Util en el preprocesamiento: la normalizacion por escala se aplica solo a la
    configuracion de las manos; la ubicacion ya viene en anchos de hombro.
    """
    return vector_b[:TAMANO_VECTOR], vector_b[TAMANO_VECTOR:]
