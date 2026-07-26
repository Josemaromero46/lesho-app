"""
Escritor de clips de senas (Direccion 2: texto -> sena).

Es el equivalente de `escritor_dataset.py` pero para los clips del muneco:
en vez de anexar filas a un CSV de entrenamiento, escribe UN archivo JSON por
toma con los landmarks crudos (contrato en `comun/clips.py`). Expone la misma
interfaz que EscritorDataset (`muestras_existentes`, `escribir_muestra_dinamica`,
`cerrar`), de modo que SesionCaptura lo usa sin saber la diferencia.

Nombres de archivo: <PALABRA>_t<NN>.json (una toma por archivo). Se graban 1-3
tomas por palabra y luego se elige la mejor con el visor; por eso cada toma se
conserva por separado en vez de sobreescribirse.

Soporta reanudacion igual que el escritor del dataset: si ya existen tomas de
una palabra, la siguiente continua la numeracion.

No depende de OpenCV ni de MediaPipe: solo de `comun/clips.py` (puro).
"""

import sys
from pathlib import Path

try:
    from comun.clips import crear_clip, fotograma_clip, guardar_clip
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from comun.clips import crear_clip, fotograma_clip, guardar_clip


def fps_de_secuencia(secuencia: list, respaldo: float) -> float:
    """fps medido de una toma cruda, a partir de sus marcas de tiempo.

    MediaPipe en la computadora no siempre alcanza los fps de la camara, y el
    clip debe reproducirse a la velocidad REAL de la sena. Si las marcas no
    sirven (toma de un solo fotograma), se usa el fps de respaldo.
    """
    if len(secuencia) < 2:
        return respaldo
    duracion = secuencia[-1]["t"] - secuencia[0]["t"]
    if duracion <= 0:
        return respaldo
    return (len(secuencia) - 1) / duracion


class EscritorClips:
    """Escribe cada toma de una palabra como un clip JSON independiente.

    Uso:

        escritor = EscritorClips(carpeta, aspecto=1280 / 720)
        ya = escritor.muestras_existentes("HOLA", "autor")
        escritor.escribir_muestra_dinamica("HOLA", "autor", ya, fotogramas)
        escritor.cerrar()
    """

    def __init__(self, carpeta, aspecto: float, fps_respaldo: float = 30.0):
        self.carpeta = Path(carpeta)
        self.carpeta.mkdir(parents=True, exist_ok=True)
        # Aspecto (ancho/alto) de la imagen de captura. Va al clip porque el
        # renderizador lo necesita para no deformar el cuerpo (ver comun/clips).
        self.aspecto = float(aspecto)
        # fps a usar si la toma no trae marcas de tiempo utilizables.
        self.fps_respaldo = float(fps_respaldo)
        # Registro de lo escrito en la sesion, para el resumen de calidad.
        self._registro: list[dict] = []

    # -- Interfaz compartida con EscritorDataset ----------------------------

    def muestras_existentes(self, clase: str, persona: str) -> int:
        """Numero de tomas (archivos) ya guardadas de esa palabra.

        La persona no distingue archivos: el diccionario necesita UNA buena toma
        por palabra, la grabe quien la grabe.
        """
        return len(list(self.carpeta.glob(f"{clase}_t*.json")))

    def escribir_muestra_dinamica(self, clase: str, persona: str,
                                  id_muestra: int, secuencia: list) -> None:
        """Escribe una toma cruda como clip JSON.

        `secuencia` es la lista de fotogramas crudos que produce la sesion de
        captura: dicts con "t" (instante), "cuerpo" (9 puntos o None) y
        "mano_izq"/"mano_der" (21 landmarks o None).
        """
        fps = fps_de_secuencia(secuencia, self.fps_respaldo)
        frames = [
            fotograma_clip(f["cuerpo"], f["mano_izq"], f["mano_der"],
                           f.get("prof_izq"), f.get("prof_der"))
            for f in secuencia
        ]
        clip = crear_clip(clase, fps, self.aspecto, frames, persona=persona)
        ruta = self.carpeta / f"{clase}_t{id_muestra + 1:02d}.json"
        guardar_clip(ruta, clip)

        sin_mano = sum(
            1 for f in secuencia
            if f["mano_izq"] is None and f["mano_der"] is None
        )
        self._registro.append({
            "archivo": ruta.name,
            "frames": len(secuencia),
            "fps": fps,
            "sin_mano": sin_mano,
        })

    def cerrar(self) -> None:
        """Nada que cerrar (cada clip se escribe completo al confirmar la toma)."""

    # -- Resumen de calidad ---------------------------------------------------

    def resumen(self) -> str:
        """Tabla legible de los clips escritos en esta sesion."""
        if not self._registro:
            return "No se guardo ningun clip en esta sesion."
        lineas = ["Clips guardados en esta sesion:"]
        for r in self._registro:
            aviso = ""
            if r["sin_mano"] > r["frames"] * 0.2:
                aviso = "  <- muchos fotogramas sin mano, revisar en el visor"
            lineas.append(
                f"  {r['archivo']:<24} {r['frames']:>3} fotogramas a "
                f"{r['fps']:>5.1f} fps{aviso}"
            )
        return "\n".join(lineas)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cerrar()
