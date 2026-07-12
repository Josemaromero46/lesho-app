"""
Punto de entrada: captura de las letras del alfabeto que llevan MOVIMIENTO.

Graba las cinco letras dinamicas del alfabeto LESHO: J, Ñ, Z, LL y RR. Aunque
son clases del Modelo A (el alfabeto), se graban en modo secuencia, igual que
las senas dinamicas del Modelo B, porque su trayectoria a 30 fps es lo que las
distingue. Se guardan en un CSV aparte
(`dataset/letras_movimiento/landmarks_letras_movimiento.csv`) para que el
preprocesamiento las convierta en ventanas con movimiento, no en poses fijas.

El resto del alfabeto (las 25 letras de pose fija) mas INICIO, FIN y REPOSO se
graban con `captura_estatica.py`.

Cada letra se graba 40 veces (una secuencia por repeticion). Entre repeticiones
de la misma letra hay una cuenta regresiva corta; al cambiar de letra, una larga.

Uso (desde la carpeta training/):

    python capture/captura_letras_movimiento.py persona_01
"""

import argparse
import sys
from pathlib import Path

# Permite ejecutar el archivo directamente: agrega training/ al path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from comun.definiciones import LETRAS_CON_MOVIMIENTO  # noqa: E402
from comun.landmarks import DetectorLandmarks  # noqa: E402
from capture.escritor_dataset import EscritorDataset, MODO_DINAMICO  # noqa: E402
from capture.interfaz_captura import InterfazCaptura  # noqa: E402
from capture.sesion_captura import ParametrosCaptura, SesionCaptura  # noqa: E402


# Las cinco letras del alfabeto que se graban como secuencia de movimiento.
# La lista y su validez las garantiza definiciones.py (validar_letras_con_movimiento).
CLASES_A_GRABAR = LETRAS_CON_MOVIMIENTO


def main(persona: str) -> None:
    escritor = EscritorDataset(config.RUTA_CSV_LETRAS_MOVIMIENTO, MODO_DINAMICO)
    interfaz = InterfazCaptura(
        config.NOMBRE_VENTANA, config.ANCHO_CAMARA, config.ALTO_CAMARA,
        config.RUTA_REFERENCIAS,
    )
    detector = DetectorLandmarks(
        max_manos=config.MAX_MANOS,
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
        muestras_por_clase=config.MUESTRAS_POR_CLASE_PERSONA,
        min_frames=config.MIN_FRAMES_SECUENCIA,
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
    print(
        "Sesion de letras con movimiento finalizada. Dataset en: "
        f"{config.RUTA_CSV_LETRAS_MOVIMIENTO}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Captura de las letras con movimiento del alfabeto LESHO (J, Ñ, Z, LL, RR)."
    )
    parser.add_argument(
        "persona",
        help="Identificador de la persona que graba, por ejemplo persona_01.",
    )
    args = parser.parse_args()
    main(args.persona)
