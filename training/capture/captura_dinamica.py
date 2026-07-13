"""
Punto de entrada: captura del dataset dinamico (Modelo B).

Graba las 50 senas dinamicas. Por defecto cada sena se graba 25 veces (una
secuencia de landmarks por repeticion). Entre repeticiones de la misma sena hay
una cuenta regresiva corta; al cambiar de sena, una cuenta larga. La captura
REANUDA: si una persona ya tiene tomas de una sena, completa hasta el numero
pedido (por ejemplo, sube de 20 a 25 las que ya tenian el mini).

A diferencia de la captura del alfabeto, esta agrega MediaPipe Pose: cada
fotograma guarda, ademas de la configuracion de las manos, la UBICACION de cada
mano en el cuerpo (marco de hombros). Por eso el vector es de 132 valores y el
escritor va en modo Modelo B.

Requiere que las 50 senas dinamicas esten definidas en comun/definiciones.py.
Si la lista esta vacia, el script avisa y no inicia.

Uso (desde la carpeta training/):

    python capture/captura_dinamica.py persona_01
"""

import argparse
import sys
from pathlib import Path

# Permite ejecutar el archivo directamente: agrega training/ al path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from comun.definiciones import CLASES_DINAMICAS, validar_clases_dinamicas  # noqa: E402
from comun.landmarks import DetectorLandmarks  # noqa: E402
from comun.pose import DetectorPose  # noqa: E402
from capture.escritor_dataset import EscritorDataset, MODO_DINAMICO, MODO_MODELO_B  # noqa: E402
from capture.interfaz_captura import InterfazCaptura  # noqa: E402
from capture.sesion_captura import ParametrosCaptura, SesionCaptura  # noqa: E402


def main(persona: str, clases=None, tomas=None) -> None:
    # No se puede capturar lo dinamico sin las 50 senas definidas.
    try:
        validar_clases_dinamicas()
    except ValueError as error:
        print("No se puede iniciar la captura dinamica:")
        print(f"  {error}")
        sys.exit(1)

    # `clases` permite grabar solo un subconjunto (por ejemplo, para una mini
    # prueba de 10 palabras). Deben ser un subconjunto de las 50 definidas; el
    # espacio de clases del modelo (indices) no cambia.
    clases = clases or CLASES_DINAMICAS
    desconocidas = [c for c in clases if c not in CLASES_DINAMICAS]
    if desconocidas:
        print(f"Clases no reconocidas (no estan en CLASES_DINAMICAS): {desconocidas}")
        sys.exit(1)

    # El escritor va en modo Modelo B: cada fila lleva la configuracion de las
    # manos (126) mas la ubicacion en el cuerpo (6) = 132 valores.
    escritor = EscritorDataset(config.RUTA_CSV_DINAMICO, MODO_MODELO_B)
    interfaz = InterfazCaptura(
        config.NOMBRE_VENTANA, config.ANCHO_CAMARA, config.ALTO_CAMARA,
        config.RUTA_REFERENCIAS,
    )
    detector = DetectorLandmarks(
        max_manos=config.MAX_MANOS,
        confianza_deteccion=config.CONFIANZA_DETECCION_MANO,
    )
    detector_pose = DetectorPose(
        confianza_deteccion=config.CONFIANZA_DETECCION_MANO,
    )
    params = ParametrosCaptura(
        indice_camara=config.INDICE_CAMARA,
        ancho_camara=config.ANCHO_CAMARA,
        alto_camara=config.ALTO_CAMARA,
        segundos_cuenta=config.SEGUNDOS_CUENTA_REGRESIVA,
        segundos_grabacion=config.SEGUNDOS_GRABACION,
        segundos_entre_reps=config.SEGUNDOS_ENTRE_REPETICIONES,
        segundos_guardado=config.SEGUNDOS_MOSTRAR_GUARDADO,
        muestras_por_clase=tomas or config.MUESTRAS_DINAMICAS_POR_CLASE,
        min_frames=config.MIN_FRAMES_SECUENCIA,
        max_frames=config.MAX_FRAMES_SECUENCIA,
    )

    # El modo de la sesion sigue siendo DINAMICO (formato de secuencia); la
    # presencia de detector_pose es lo que activa el vector del Modelo B.
    sesion = SesionCaptura(
        MODO_DINAMICO, clases, persona, escritor, interfaz, detector,
        params, detector_pose=detector_pose,
    )
    try:
        sesion.ejecutar()
    finally:
        detector.cerrar()
        detector_pose.cerrar()
        escritor.cerrar()
        interfaz.cerrar()
    print(f"Sesion dinamica finalizada. Dataset en: {config.RUTA_CSV_DINAMICO}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Captura del dataset dinamico LESHO.")
    parser.add_argument(
        "persona",
        help="Identificador de la persona que graba, por ejemplo persona_01.",
    )
    parser.add_argument(
        "--clases",
        help="Subconjunto de senas a grabar, separadas por coma (por ejemplo "
             "HOLA,MAMA,PAPA). Por defecto se graban las 50.",
    )
    parser.add_argument(
        "--mini", action="store_true",
        help="Graba solo las 10 senas de la mini prueba (para validar antes de las 50).",
    )
    parser.add_argument(
        "--tomas", type=int,
        help="Tomas por sena (por defecto 25). Menos tomas = mini prueba rapida.",
    )
    args = parser.parse_args()
    # Las 10 senas de la mini prueba: variadas y con distintos lugares del cuerpo.
    MINI = ["HOLA", "AGUA", "AYUDA", "MAMA", "PAPA",
            "JUGAR", "CASA", "HOSPITAL", "TRABAJO", "NOCHE"]
    if args.mini:
        subconjunto = MINI
    elif args.clases:
        subconjunto = [c.strip().upper() for c in args.clases.split(",")]
    else:
        subconjunto = None
    main(args.persona, subconjunto, args.tomas)
