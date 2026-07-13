"""
Contrato del clip de sena (Direccion 2: texto en espanol -> sena en pantalla).

Un clip es la grabacion de UNA sena como secuencia de landmarks crudos (manos +
torso superior), guardada en JSON. Es el formato que consume el muneco de
capsulas: el visor de Python (demo/visor_clips.py) y el MunecoPainter de la app
Flutter. Ver PLAN_DIRECCION2.md, seccion 4.

Decisiones del contrato (version 1):

  - Coordenadas CRUDAS normalizadas a la imagen [0, 1], la misma convencion de
    Hands y Pose (comparten espacio, por eso manos y cuerpo encajan directo).
    NO se normaliza respecto a la muneca: el objetivo es reproducir, no entrenar.
  - El campo "aspecto" (ancho/alto de la imagen de captura) es OBLIGATORIO para
    dibujar sin deformar: MediaPipe normaliza x por el ancho e y por el alto por
    separado, asi que sin el aspecto las proporciones del cuerpo salen mal
    (leccion aprendida en la app, correccion de aspecto del Modelo A).
  - 9 puntos de cuerpo bastan para torso, cuello, cabeza y brazos. Cada punto
    lleva [x, y, z, visibilidad]; la visibilidad permite descartar articulaciones
    poco fiables al limpiar o dibujar.
  - Mano ausente en un fotograma = null. La limpieza (offline o al cargar)
    interpola huecos cortos; un hueco largo amerita repetir la toma.
  - En la FASE 0 (piloto) el clip se guarda CRUDO y el visor aplica el suavizado
    One Euro al cargar. La limpieza definitiva al guardar es de la Fase 1.
  - JSON compacto (sin sangria) para no inflar los assets; sigue siendo legible
    con cualquier formateador y el visor es la herramienta real de inspeccion.

Modulo PURO (solo json y pathlib): se importa desde la captura, el visor y las
pruebas sin instalar mediapipe ni opencv.
"""

import json
from pathlib import Path

from .definiciones import (
    NUM_LANDMARKS,
    POSE_CADERA_DER,
    POSE_CADERA_IZQ,
    POSE_CODO_DER,
    POSE_CODO_IZQ,
    POSE_HOMBRO_DER,
    POSE_HOMBRO_IZQ,
    POSE_MUNECA_DER,
    POSE_MUNECA_IZQ,
    POSE_NARIZ,
)

# Version del formato. Si el contrato cambia de forma incompatible, se
# incrementa y los lectores deciden que hacer con las versiones viejas.
VERSION_CLIP = 1

# Los 9 puntos del cuerpo que guarda un clip, EN ESTE ORDEN (contrato).
PUNTOS_CUERPO_CLIP = [
    "nariz",
    "hombro_izq", "hombro_der",
    "codo_izq", "codo_der",
    "muneca_izq", "muneca_der",
    "cadera_izq", "cadera_der",
]

# Indice de cada punto del clip dentro de los 33 landmarks de MediaPipe Pose.
INDICES_POSE_CLIP = [
    POSE_NARIZ,
    POSE_HOMBRO_IZQ, POSE_HOMBRO_DER,
    POSE_CODO_IZQ, POSE_CODO_DER,
    POSE_MUNECA_IZQ, POSE_MUNECA_DER,
    POSE_CADERA_IZQ, POSE_CADERA_DER,
]

# Indices DENTRO del arreglo "cuerpo" del clip (para los renderizadores).
CUERPO_NARIZ = 0
CUERPO_HOMBRO_IZQ = 1
CUERPO_HOMBRO_DER = 2
CUERPO_CODO_IZQ = 3
CUERPO_CODO_DER = 4
CUERPO_MUNECA_IZQ = 5
CUERPO_MUNECA_DER = 6
CUERPO_CADERA_IZQ = 7
CUERPO_CADERA_DER = 8

NUM_PUNTOS_CUERPO_CLIP = len(PUNTOS_CUERPO_CLIP)  # 9

# Decimales al serializar. 4 decimales de coordenada normalizada equivalen a
# ~0.1 px en una imagen de 1280: sobra precision y el JSON queda compacto.
DECIMALES_CLIP = 4


def _redondear(valores, n_esperado, mensaje):
    """Convierte a lista de listas de float redondeados, validando el largo."""
    if len(valores) != n_esperado:
        raise ValueError(f"{mensaje}: se esperaban {n_esperado}, hay {len(valores)}.")
    return [[round(float(v), DECIMALES_CLIP) for v in punto] for punto in valores]


def fotograma_clip(cuerpo, mano_izq, mano_der) -> dict:
    """Arma el fotograma de un clip a partir de los datos crudos.

    Parametros
    ----------
    cuerpo : list[list[float]] | None
        Los 9 puntos del cuerpo como [x, y, z, visibilidad], o None si la pose
        no se detecto en este fotograma.
    mano_izq, mano_der : list[tuple] | None
        Los 21 landmarks (x, y, z) de cada mano, o None si esta ausente.
    """
    return {
        "cuerpo": (None if cuerpo is None
                   else _redondear(cuerpo, NUM_PUNTOS_CUERPO_CLIP, "puntos de cuerpo")),
        "mano_izq": (None if mano_izq is None
                     else _redondear(mano_izq, NUM_LANDMARKS, "landmarks de mano")),
        "mano_der": (None if mano_der is None
                     else _redondear(mano_der, NUM_LANDMARKS, "landmarks de mano")),
    }


def crear_clip(palabra: str, fps: float, aspecto: float, frames: list,
               persona: str = "") -> dict:
    """Arma el diccionario completo de un clip (el objeto que va al JSON).

    `frames` debe venir de `fotograma_clip`. `aspecto` es ancho/alto de la
    imagen con la que se capturo (por ejemplo 1280/720 = 1.7778). `persona` es
    metadato opcional de control de calidad; los renderizadores lo ignoran.
    """
    clip = {
        "version": VERSION_CLIP,
        "palabra": palabra,
        "fps": round(float(fps), 2),
        "aspecto": round(float(aspecto), 4),
        "num_frames": len(frames),
        "puntos_cuerpo": PUNTOS_CUERPO_CLIP,
        "frames": frames,
    }
    if persona:
        clip["persona"] = persona
    return clip


def guardar_clip(ruta, clip: dict) -> None:
    """Escribe el clip a disco como JSON compacto (UTF-8, con enie intacta)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(clip, archivo, ensure_ascii=False, separators=(",", ":"))


def cargar_clip(ruta) -> dict:
    """Lee un clip desde disco y lo valida antes de devolverlo."""
    with open(ruta, "r", encoding="utf-8") as archivo:
        clip = json.load(archivo)
    validar_clip(clip)
    return clip


def validar_clip(clip: dict) -> None:
    """Verifica la estructura minima de un clip. Lanza ValueError si no cumple."""
    for campo in ("version", "palabra", "fps", "aspecto", "num_frames", "frames"):
        if campo not in clip:
            raise ValueError(f"Al clip le falta el campo '{campo}'.")
    if clip["version"] != VERSION_CLIP:
        raise ValueError(
            f"Version de clip no soportada: {clip['version']} "
            f"(se esperaba {VERSION_CLIP})."
        )
    frames = clip["frames"]
    if clip["num_frames"] != len(frames):
        raise ValueError(
            f"num_frames ({clip['num_frames']}) no coincide con la cantidad "
            f"real de fotogramas ({len(frames)})."
        )
    if not frames:
        raise ValueError("El clip no tiene fotogramas.")
    if clip["fps"] <= 0 or clip["aspecto"] <= 0:
        raise ValueError("fps y aspecto deben ser positivos.")
    for i, frame in enumerate(frames):
        cuerpo = frame.get("cuerpo")
        if cuerpo is not None and len(cuerpo) != NUM_PUNTOS_CUERPO_CLIP:
            raise ValueError(f"Fotograma {i}: cuerpo con {len(cuerpo)} puntos.")
        for lado in ("mano_izq", "mano_der"):
            mano = frame.get(lado)
            if mano is not None and len(mano) != NUM_LANDMARKS:
                raise ValueError(f"Fotograma {i}: {lado} con {len(mano)} puntos.")
