"""
Punto de entrada: captura del dataset estatico (Modelo A).

Graba las clases de POSE FIJA: las 25 letras estaticas del alfabeto mas INICIO,
FIN y REPOSO (28 clases en total). Las cinco letras con movimiento (J, Ñ, Z, LL,
RR) NO se graban aqui, sino con `captura_letras_movimiento.py`, porque necesitan
conservar su trayectoria.

Cada clase se graba en VARIAS TOMAS SEPARADAS (config.TOMAS_ESTATICAS_POR_CLASE).
Entre toma y toma se baja la mano y se vuelve a formar la pose, para dar variedad
real: distintos angulos, distancias y aperturas de dedos. Cada toma es una
grabacion corta (config.SEGUNDOS_GRABACION_POSE) que se guarda como una secuencia
breve de la pose sostenida.

Antes se grababa una sola pose sostenida por letra, de la que se sacaban 40
fotogramas casi identicos. Eso daba un unico ejemplo por letra y el modelo lo
memorizaba en vez de aprender la forma, confundiendo H con CH y L con LL en
cuanto la mano cambiaba de angulo. Las tomas separadas corrigen eso.

Uso (desde la carpeta training/):

    python capture/captura_estatica.py persona_01
"""

import argparse
import sys
from pathlib import Path

# Permite ejecutar el archivo directamente: agrega training/ al path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from comun.definiciones import LETRAS_ESTATICAS, SENAS_CONTROL  # noqa: E402
from comun.landmarks import DetectorLandmarks  # noqa: E402
from capture.escritor_dataset import EscritorDataset, MODO_DINAMICO  # noqa: E402
from capture.interfaz_captura import InterfazCaptura  # noqa: E402
from capture.sesion_captura import ParametrosCaptura, SesionCaptura  # noqa: E402


# Clases que se graban como pose fija: alfabeto sin las letras de movimiento,
# mas las tres senas de control. El orden no afecta al modelo (los indices de
# salida los fija CLASES_ESTATICAS en definiciones.py); aqui solo define el orden
# de grabacion.
CLASES_A_GRABAR = LETRAS_ESTATICAS + SENAS_CONTROL


def main(persona: str) -> None:
    escritor = EscritorDataset(config.RUTA_CSV_ESTATICO, MODO_DINAMICO)
    interfaz = InterfazCaptura(
        config.NOMBRE_VENTANA, config.ANCHO_CAMARA, config.ALTO_CAMARA,
        config.RUTA_REFERENCIAS,
    )
    detector = DetectorLandmarks(
        max_manos=config.MAX_MANOS,
        confianza_deteccion=config.CONFIANZA_DETECCION_MANO,
    )
    # Poses grabadas como tomas separadas y cortas: la variedad viene de repetir
    # la formacion, no de sostener la pose. Se usa el modo secuencia para que cada
    # toma quede identificada por su id_muestra (necesario para no mezclar tomas
    # al armar las ventanas ni al dividir el dataset).
    params = ParametrosCaptura(
        indice_camara=config.INDICE_CAMARA,
        ancho_camara=config.ANCHO_CAMARA,
        alto_camara=config.ALTO_CAMARA,
        segundos_cuenta=config.SEGUNDOS_CUENTA_REGRESIVA,
        segundos_grabacion=config.SEGUNDOS_GRABACION_POSE,
        segundos_entre_reps=config.SEGUNDOS_ENTRE_REPETICIONES,
        segundos_guardado=config.SEGUNDOS_MOSTRAR_GUARDADO,
        muestras_por_clase=config.TOMAS_ESTATICAS_POR_CLASE,
        min_frames=config.MIN_FRAMES_POSE,
        max_frames=config.MAX_FRAMES_SECUENCIA,
    )

    sesion = SesionCaptura(
        MODO_DINAMICO, CLASES_A_GRABAR, persona, escritor, interfaz, detector,
        params,
    )
    try:
        sesion.ejecutar()
    finally:
        detector.cerrar()
        escritor.cerrar()
        interfaz.cerrar()
    print(f"Sesion estatica finalizada. Dataset en: {config.RUTA_CSV_ESTATICO}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Captura del dataset estatico LESHO.")
    parser.add_argument(
        "persona",
        help="Identificador de la persona que graba, por ejemplo persona_01.",
    )
    args = parser.parse_args()
    main(args.persona)
