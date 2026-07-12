"""
Escritor del dataset de landmarks a CSV.

Acumula las muestras capturadas y las escribe en disco siguiendo los contratos
de formato definidos en PLAN_CODIGO.md (seccion 7). Soporta reanudacion: al
abrir, lee que muestras ya existen para no duplicarlas si una sesion se
interrumpe y se retoma.

No depende de OpenCV ni de MediaPipe, solo del modulo csv de la libreria
estandar. Por eso se puede probar de forma aislada con datos sinteticos.

Formatos (ver PLAN_CODIGO.md 7.2 y 7.3):

  Estatico  -> una fila por muestra:
    etiqueta, persona, id_muestra, x0, y0, z0, ..., x20, y20, z20

  Dinamico  -> una fila por fotograma; las filas con igual
               (etiqueta, persona, id_muestra) forman una secuencia:
    etiqueta, persona, id_muestra, frame, x0, y0, z0, ..., x20, y20, z20

  Modelo B  -> como el dinamico, pero cada fila agrega la UBICACION en el cuerpo
               (14 valores): la configuracion de las manos (126) mas la posicion
               de cada mano y de 5 anclas de la cara (nariz, ojos, boca, orejas)
               respecto al centro de los hombros:
    ..., z41, ubic_izq_x, ubic_izq_y, ..., ubic_orejader_x, ubic_orejader_y
"""

import csv
from collections import defaultdict
from pathlib import Path

import sys

# Permite ejecutar tanto como modulo del paquete como con la raiz training/ en
# el path. La normalizacion vive en comun/.
try:
    from comun.definiciones import NUM_COORDENADAS, TAMANO_VECTOR, TAMANO_VECTOR_B
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from comun.definiciones import NUM_COORDENADAS, TAMANO_VECTOR, TAMANO_VECTOR_B

MODO_ESTATICO = "estatico"
MODO_DINAMICO = "dinamico"
MODO_MODELO_B = "modelo_b"

# Modos cuyo formato es de secuencia (una fila por fotograma, con columna frame).
_MODOS_SECUENCIA = (MODO_DINAMICO, MODO_MODELO_B)
_MODOS_VALIDOS = (MODO_ESTATICO, MODO_DINAMICO, MODO_MODELO_B)

# Numero de puntos en el vector completo: 42 (21 de la mano izquierda + 21 de la
# derecha). Los puntos 0..20 son la mano izquierda; 21..41, la derecha.
_NUM_PUNTOS = TAMANO_VECTOR // NUM_COORDENADAS

# Nombres de las 18 columnas de ubicacion del Modelo B: 2 muñecas + 2 puntas del
# indice + 5 anclas de la cara, en el orden del contrato (ver comun/vector_modelo_b).
_COLUMNAS_UBICACION = [
    "ubic_munecaizq_x", "ubic_munecaizq_y",
    "ubic_munecader_x", "ubic_munecader_y",
    "ubic_puntaizq_x", "ubic_puntaizq_y",
    "ubic_puntader_x", "ubic_puntader_y",
    "ubic_nariz_x", "ubic_nariz_y",
    "ubic_ojos_x", "ubic_ojos_y",
    "ubic_boca_x", "ubic_boca_y",
    "ubic_orejaizq_x", "ubic_orejaizq_y",
    "ubic_orejader_x", "ubic_orejader_y",
]


def _columnas_coordenadas() -> list[str]:
    """Genera los nombres de las 126 columnas de coordenadas: x0,y0,z0,...,z41."""
    columnas: list[str] = []
    for i in range(_NUM_PUNTOS):
        columnas.extend([f"x{i}", f"y{i}", f"z{i}"])
    return columnas


def _columnas_de_modo(modo: str) -> list[str]:
    """Columnas de valores segun el modo: 126 coords, o 126 + 6 de ubicacion."""
    if modo == MODO_MODELO_B:
        return _columnas_coordenadas() + _COLUMNAS_UBICACION
    return _columnas_coordenadas()


class EscritorDataset:
    """Escribe muestras de landmarks a un CSV, con soporte de reanudacion.

    Uso:

        escritor = EscritorDataset(ruta_csv, MODO_ESTATICO)
        ya = escritor.muestras_existentes("A", "persona_01")
        escritor.escribir_muestra_estatica("A", "persona_01", ya, vector63)
        escritor.cerrar()
    """

    def __init__(self, ruta_csv, modo: str):
        if modo not in _MODOS_VALIDOS:
            raise ValueError(f"Modo invalido: {modo!r}")

        self.ruta = Path(ruta_csv)
        self.modo = modo
        # Tamano del vector y columnas segun el modo. El Modelo B agrega la
        # ubicacion en el cuerpo; estatico y dinamico usan 126 (sin cambios).
        self._tamano_vector = TAMANO_VECTOR_B if modo == MODO_MODELO_B else TAMANO_VECTOR
        self._columnas = _columnas_de_modo(modo)
        self._es_secuencia = modo in _MODOS_SECUENCIA

        # Indice en memoria de los id_muestra ya presentes por (clase, persona).
        self._existentes: dict[tuple[str, str], set[int]] = defaultdict(set)

        # Asegura que la carpeta del CSV exista.
        self.ruta.parent.mkdir(parents=True, exist_ok=True)

        # Carga el indice de lo ya escrito (si el archivo existe) y deja el
        # archivo listo para anexar.
        archivo_nuevo = not self.ruta.exists() or self.ruta.stat().st_size == 0
        if not archivo_nuevo:
            self._cargar_indice()

        self._archivo = open(self.ruta, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._archivo)

        if archivo_nuevo:
            self._writer.writerow(self._encabezado())
            self._archivo.flush()

    # -- API publica --------------------------------------------------------

    def muestras_existentes(self, clase: str, persona: str) -> int:
        """Numero de muestras ya guardadas para esa clase y persona.

        En estatico, una muestra es una fila. En dinamico, una muestra es una
        secuencia completa (un id_muestra). El valor devuelto sirve como punto
        de reanudacion: la siguiente muestra usa este numero como id_muestra.
        """
        return len(self._existentes[(clase, persona)])

    def escribir_muestra_estatica(
        self, clase: str, persona: str, id_muestra: int, vector: list[float]
    ) -> None:
        """Escribe una muestra estatica (un vector de 63 valores)."""
        if self.modo != MODO_ESTATICO:
            raise RuntimeError("El escritor no esta en modo estatico.")
        self._validar_vector(vector)
        fila = [clase, persona, id_muestra] + [self._fmt(v) for v in vector]
        self._writer.writerow(fila)
        self._archivo.flush()
        self._existentes[(clase, persona)].add(id_muestra)

    def escribir_muestra_dinamica(
        self, clase: str, persona: str, id_muestra: int, secuencia: list[list[float]]
    ) -> None:
        """Escribe una muestra de secuencia (dinamico o Modelo B).

        Cada vector de la secuencia se escribe como una fila con su indice de
        fotograma. Todas comparten el mismo id_muestra. El tamano del vector
        (126 dinamico, 132 Modelo B) lo fija el modo del escritor.
        """
        if self.modo not in _MODOS_SECUENCIA:
            raise RuntimeError(
                "El escritor no esta en modo de secuencia (dinamico o modelo_b)."
            )
        for indice_frame, vector in enumerate(secuencia):
            self._validar_vector(vector)
            fila = (
                [clase, persona, id_muestra, indice_frame]
                + [self._fmt(v) for v in vector]
            )
            self._writer.writerow(fila)
        self._archivo.flush()
        self._existentes[(clase, persona)].add(id_muestra)

    def cerrar(self) -> None:
        """Cierra el archivo. Llamar al terminar la sesion."""
        if self._archivo and not self._archivo.closed:
            self._archivo.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cerrar()

    # -- Internos -----------------------------------------------------------

    def _encabezado(self) -> list[str]:
        base = ["etiqueta", "persona", "id_muestra"]
        if self._es_secuencia:
            base.append("frame")
        return base + self._columnas

    def _cargar_indice(self) -> None:
        """Lee el CSV existente y reconstruye el indice de muestras presentes."""
        with open(self.ruta, "r", newline="", encoding="utf-8") as f:
            lector = csv.reader(f)
            try:
                next(lector)  # descartar encabezado
            except StopIteration:
                return
            for fila in lector:
                if len(fila) < 3:
                    continue
                clase, persona = fila[0], fila[1]
                try:
                    id_muestra = int(fila[2])
                except ValueError:
                    continue
                self._existentes[(clase, persona)].add(id_muestra)

    def _validar_vector(self, vector) -> None:
        if len(vector) != self._tamano_vector:
            raise ValueError(
                f"El vector debe tener {self._tamano_vector} valores, "
                f"tiene {len(vector)}."
            )

    @staticmethod
    def _fmt(valor: float) -> str:
        """Formatea un numero con 6 decimales para mantener el CSV compacto."""
        return f"{float(valor):.6f}"
