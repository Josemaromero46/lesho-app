"""
Demo de captura del Modelo B (manos + marco del cuerpo).

Sirve para PROBAR el flujo de captura dinamica sin necesidad de tener definidas
las 50 senas. Usa un par de clases de prueba y escribe en un CSV aparte, dentro
de dataset/_prueba_modelo_b/, para no tocar el dataset real.

Verifica que la captura graba, ademas de las manos, la ubicacion en el cuerpo
(columnas ubic_*), que es lo que agrega el paso 2.

Uso (desde la carpeta training/):

    python demo/demo_captura_b.py prueba

Sentate de modo que se te vean los HOMBROS en camara (Pose necesita el torso).
Al terminar, revisa el CSV que te indica al final.
"""

import argparse
import sys
from pathlib import Path

# Permite ejecutar el archivo directamente: agrega training/ al path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from comun.landmarks import DetectorLandmarks  # noqa: E402
from comun.pose import DetectorPose  # noqa: E402
from capture.escritor_dataset import EscritorDataset, MODO_DINAMICO, MODO_MODELO_B  # noqa: E402
from capture.interfaz_captura import InterfazCaptura  # noqa: E402
from capture.sesion_captura import ParametrosCaptura, SesionCaptura  # noqa: E402

# Clases de PRUEBA (no son senas reales). Solo para ver que el flujo corre.
CLASES_PRUEBA = ["PRUEBA_1", "PRUEBA_2"]

# CSV de prueba, separado del dataset real. Se puede borrar sin consecuencias.
RUTA_CSV_PRUEBA = config.RAIZ_DATASET / "_prueba_modelo_b" / "landmarks_prueba_b.csv"


def main(persona: str) -> None:
    escritor = EscritorDataset(RUTA_CSV_PRUEBA, MODO_MODELO_B)
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
        muestras_por_clase=3,   # pocas tomas: es solo una prueba
        min_frames=config.MIN_FRAMES_SECUENCIA,
        max_frames=config.MAX_FRAMES_SECUENCIA,
    )

    sesion = SesionCaptura(
        MODO_DINAMICO, CLASES_PRUEBA, persona, escritor, interfaz, detector,
        params, detector_pose=detector_pose,
    )
    try:
        sesion.ejecutar()
    finally:
        detector.cerrar()
        detector_pose.cerrar()
        escritor.cerrar()
        interfaz.cerrar()
    print(f"\nPrueba finalizada. CSV en: {RUTA_CSV_PRUEBA}")
    print("Verifica que existan las columnas ubic_* y que NO sean todas cero.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo de captura del Modelo B.")
    parser.add_argument("persona", help="Identificador de prueba, por ejemplo 'prueba'.")
    args = parser.parse_args()
    main(args.persona)
