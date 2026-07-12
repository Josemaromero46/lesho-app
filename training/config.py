"""
Configuracion central del pipeline de entrenamiento.

Reune en un solo lugar los parametros que gobiernan la captura del dataset, el
preprocesamiento y el entrenamiento. Las constantes estructurales de los
landmarks (cuantos puntos, tamano del vector, clases) viven en
`comun/definiciones.py` y se re-exportan aqui para tener una vista unica.

Los scripts del pipeline se ejecutan desde la carpeta `training/`, de modo que
`import config` y `from comun import ...` resuelven correctamente.
"""

from pathlib import Path

from comun.definiciones import (  # re-exportadas para vista unica
    NUM_CLASES_A,
    NUM_CLASES_B,
    NUM_LANDMARKS,
    TAMANO_VECTOR,
    TAMANO_VENTANA_A,
)

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------

# Carpeta training/ (donde esta este archivo).
RAIZ_TRAINING = Path(__file__).resolve().parent

# Carpeta raiz del proyecto (TESIS/).
RAIZ_PROYECTO = RAIZ_TRAINING.parent

# Carpeta del dataset versionado en Git.
RAIZ_DATASET = RAIZ_PROYECTO / "dataset"
RUTA_CSV_ESTATICO = RAIZ_DATASET / "estatico" / "landmarks_estatico.csv"
# Letras del alfabeto que se ejecutan con movimiento (J, Ñ, Z, LL, RR). Se
# graban como secuencias (formato dinamico) porque su trayectoria importa, pero
# alimentan al Modelo A, no al Modelo B.
RUTA_CSV_LETRAS_MOVIMIENTO = (
    RAIZ_DATASET / "letras_movimiento" / "landmarks_letras_movimiento.csv"
)
RUTA_CSV_DINAMICO = RAIZ_DATASET / "dinamico" / "landmarks_dinamico.csv"

# Carpeta de salida de los modelos exportados a TFLite.
RAIZ_EXPORTS = RAIZ_TRAINING / "exports"

# Dataset del Modelo A ya preprocesado (ventanas listas para entrenar), en .npz.
RUTA_DATASET_A = RAIZ_EXPORTS / "dataset_modelo_a.npz"

# Modelo A: modelo Keras entrenado, su version TFLite y sus etiquetas.
RUTA_MODELO_A_KERAS = RAIZ_EXPORTS / "modelo_a.keras"
RUTA_MODELO_A_TFLITE = RAIZ_EXPORTS / "modelo_a.tflite"
RUTA_ETIQUETAS_A = RAIZ_EXPORTS / "etiquetas_a.txt"

# Dataset del Modelo B ya preprocesado (secuencias de longitud fija), en .npz.
RUTA_DATASET_B = RAIZ_EXPORTS / "dataset_modelo_b.npz"

# Modelo B: modelo Keras entrenado, su version TFLite y sus etiquetas.
RUTA_MODELO_B_KERAS = RAIZ_EXPORTS / "modelo_b.keras"
RUTA_MODELO_B_TFLITE = RAIZ_EXPORTS / "modelo_b.tflite"
RUTA_ETIQUETAS_B = RAIZ_EXPORTS / "etiquetas_b.txt"


# ---------------------------------------------------------------------------
# Parametros de captura del dataset
# ---------------------------------------------------------------------------

# Repeticiones a grabar por clase y por persona.
MUESTRAS_POR_CLASE_PERSONA = 40

# Numero de TOMAS SEPARADAS por clase estatica. Cada toma es una formacion
# independiente de la pose: se baja la mano y se vuelve a formar antes de grabar.
# Antes cada letra se grababa como UNA sola pose sostenida (un solo ejemplo), y
# el modelo memorizaba esa pose en vez de aprender la forma, confundiendo H con
# CH y L con LL en cuanto la mano cambiaba un poco de angulo. Con muchas tomas
# separadas el modelo ve la variedad real de cada pose.
TOMAS_ESTATICAS_POR_CLASE = 20

# Segundos de grabacion de cada toma de pose. Corto a proposito: la pose solo se
# sostiene un instante, la variedad viene de repetir la formacion, no de aguantar
# la mano quieta mucho tiempo.
SEGUNDOS_GRABACION_POSE = 1.2

# Minimo de fotogramas para aceptar una toma de pose valida.
MIN_FRAMES_POSE = 12

# Numero de personas que graban (autor + 3 colaboradores).
NUM_PERSONAS = 4

# Segundos de cuenta regresiva antes de cada toma (para que la persona se prepare).
SEGUNDOS_CUENTA_REGRESIVA = 3

# Segundos de grabacion de cada toma.
SEGUNDOS_GRABACION = 3

# Tasa de fotogramas objetivo de la captura.
FPS_OBJETIVO = 30

# Indice de la camara a usar en OpenCV (0 = camara por defecto).
INDICE_CAMARA = 0

# Resolucion solicitada a la camara. 1280x720 da buen encuadre para dos manos
# a 60-100 cm de distancia.
ANCHO_CAMARA = 1280
ALTO_CAMARA = 720

# Segundos que se muestra el aviso de "guardado" tras cada toma. Durante esta
# ventana la persona puede repetir la toma (tecla R) antes de confirmarla.
SEGUNDOS_MOSTRAR_GUARDADO = 1.2

# Cuenta regresiva corta entre repeticiones de la MISMA sena dinamica. La
# cuenta larga (SEGUNDOS_CUENTA_REGRESIVA) se usa solo al cambiar de sena.
SEGUNDOS_ENTRE_REPETICIONES = 1.0

# Nombre de la ventana de OpenCV.
NOMBRE_VENTANA = "Captura LESHO"

# Carpeta opcional de imagenes de referencia de las senas. Si existe un archivo
# <CLASE>.png o <CLASE>.jpg, se muestra como guia de como se hace la sena.
RUTA_REFERENCIAS = RAIZ_TRAINING / "capture" / "referencias"


# ---------------------------------------------------------------------------
# Parametros de las secuencias dinamicas (Modelo B)
# ---------------------------------------------------------------------------

# Longitud minima y maxima de una secuencia valida (en fotogramas).
MIN_FRAMES_SECUENCIA = 30
MAX_FRAMES_SECUENCIA = 60

# Longitud fija a la que se llevan las secuencias al entrenar (padding/truncado).
# Provisional: confirmar tras ver la distribucion real de longitudes del dataset.
LONGITUD_FIJA_SECUENCIA = 40


# ---------------------------------------------------------------------------
# Parametros de preprocesamiento (ventanas del Modelo A)
# ---------------------------------------------------------------------------

# Paso (stride) entre ventanas deslizantes consecutivas, en fotogramas. Un paso
# menor genera mas ventanas (mas datos) con mas solape, a costa de mas memoria.
# Se usa 2 porque cada toma de pose deja pocos fotogramas; un paso mayor dejaria
# muy pocas ventanas por toma.
PASO_VENTANA = 2

# Cuantas ventanas conservar por cada toma de pose estatica. Dentro de una misma
# toma la mano esta quieta, asi que sus ventanas son casi identicas y basta con
# unas pocas; la variedad real viene de tener MUCHAS tomas distintas, no muchas
# ventanas de la misma toma.
VENTANAS_POR_TOMA_ESTATICA = 2

# Tope de ventanas por toma de letra con movimiento (J, Ñ, Z, LL, RR). Sin tope,
# cada seña genera cientos de ventanas y aplasta a las estaticas: la LL le ganaba
# a la L por pura cantidad. Se toman repartidas a lo largo de la trayectoria para
# no perder el gesto.
VENTANAS_POR_TOMA_MOVIMIENTO = 6

# Umbral relativo de movimiento para conservar una ventana de una letra dinamica.
# Se conservan las ventanas cuya velocidad media supere esta fraccion de la
# velocidad maxima de su secuencia; asi se descartan las ventanas "quietas" del
# inicio o el final de la grabacion, que no contienen el gesto. Se afina con
# datos reales.
UMBRAL_MOVIMIENTO_REL = 0.4

# Desviacion estandar del ruido gaussiano para augmentation, en unidades de
# landmark normalizado. Pequeno: simula el temblor natural de la mano.
SIGMA_RUIDO = 0.01

# --- Suavizado temporal de landmarks (filtro One Euro) ---
# La Tasks API de MediaPipe no filtra los landmarks, asi que tiemblan fotograma a
# fotograma. El filtro One Euro los suaviza: fuerte cuando la mano esta quieta
# (letras estaticas firmes) y suave cuando se mueve rapido (J/Z/LL sin retraso).
# Se aplica sobre el vector trasladado, IGUAL en el preprocesamiento y en la
# inferencia (demo y app), para que entrenamiento e inferencia coincidan.
SUAVIZADO_ACTIVO = True
# Frecuencia de corte minima (Hz). Gobierna el suavizado con la mano QUIETA. Mas
# baja = mas firme cuando no te movés, pero mas retraso. Calibrado: 0.5 quita ~88%
# del temblor con la mano quieta. Subir si una pose estatica se siente perezosa.
SUAVIZADO_MIN_CUTOFF = 0.5
# Ganancia por velocidad. Gobierna el suavizado con la mano en MOVIMIENTO. Mas
# alta = reacciona mas rapido (menos retraso en J/Z/LL/Ñ), a costa de dejar pasar
# algo mas de temblor. Calibrado: 10 conserva ~100% del movimiento real. Bajar si
# el esqueleto tiembla al moverse; subir si las letras de movimiento van con lag.
SUAVIZADO_BETA = 10.0
# Corte del suavizado de la derivada (Hz). Se deja en 1.0 (valor tipico).
SUAVIZADO_D_CUTOFF = 1.0

# Piso ABSOLUTO de movimiento para aceptar una ventana de una letra dinamica.
# Una ventana de J/Ñ/Z/LL/RR solo cuenta si su movimiento medio supera este
# valor. Evita que las ventanas quietas del inicio o el final de una sena lenta
# se etiqueten como movimiento y se confundan con la letra estatica de la misma
# forma de mano.
#
# Calibrado sobre los datos REALES (estaticas de tomas separadas + movimiento),
# ya suavizados con One Euro y escalados. Una mano quieta da: mediana 0.0007,
# p95 0.0022. Las letras de movimiento: LL 0.0031 (la mas lenta), RR 0.0074,
# Ñ 0.0116, Z 0.0168, J 0.0435. Se pone el umbral en el p95 de las estaticas: una
# ventana solo cuenta como movimiento si se mueve mas que el 95% de las poses
# quietas. Con esto la LL activa (0.0031) supera el umbral y una L quieta (0.0007)
# no. Antes del suavizado este valor era 0.0090 (las magnitudes bajan ~3x al
# suavizar).
UMBRAL_MOVIMIENTO_ABS = 0.0022

# Tope de movimiento para aceptar una ventana ESTATICA. Una toma de pose fija se
# graba re-formando la mano cada vez, asi que el ARRANQUE de la toma captura la
# mano entrando a la pose (en movimiento). Esas ventanas tienen movimiento de
# rango dinamico (~0.011) y, al etiquetarse como la letra estatica, el modelo las
# confunde con la letra de movimiento de la misma forma (una L entrando se ve
# como LL). Solo se conservan las ventanas por debajo de este tope: la pose
# quieta de verdad. Es el espejo de UMBRAL_MOVIMIENTO_ABS. Medido: las estaticas
# quietas quedan bajo ~0.0022 (p95); se deja un margen hasta 0.0045.
UMBRAL_ESTATICO_MAX = 0.0045

# --- Compuerta simetrica de la inferencia en vivo (demo/app) ---
# La compuerta de UMBRAL_MOVIMIENTO_ABS bloquea las letras de movimiento cuando la
# mano esta quieta (una N quieta no es Ñ). Pero le faltaba el sentido inverso: con
# la mano en pleno movimiento, una letra ESTATICA que comparte forma de mano con
# una de movimiento (N con Ñ, L con LL, R con RR) no deberia ganar. En medio del
# vaiven de una Ñ hay instantes que se ven como una N quieta y la N se colaba. Con
# movimiento por encima de este umbral se bloquean esas estaticas gemelas para
# dejar pasar la letra de movimiento. Medido en vivo (suavizado): N/L/R sostenidas
# nunca superan 0.0013, asi que este umbral (0.006) NO las afecta al sostenerlas;
# solo actua cuando de verdad hay movimiento. Recupera la Ñ (97%->98%) sin tocar
# la N. Es un ajuste SOLO de inferencia: no afecta al modelo ni al entrenamiento.
UMBRAL_MOVIMIENTO_MOVIENDO = 0.006

# Letras estaticas que comparten forma de mano con una letra de movimiento y por
# eso se confunden con ella durante el gesto: N (con Ñ), L (con LL), R (con RR).
# J y Z no tienen gemela estatica.
ESTATICAS_GEMELAS_DE_MOVIMIENTO = ["N", "L", "R"]


# ---------------------------------------------------------------------------
# Parametros de MediaPipe
# ---------------------------------------------------------------------------

# Maximo de manos a detectar (2 para soportar senas a dos manos).
MAX_MANOS = 2

# Umbral de confianza para aceptar una mano detectada. Se subio de 0.5 a 0.6
# para reducir falsos positivos (por ejemplo, la cara detectada como mano). Si
# en condiciones de poca luz se pierden manos legitimas, bajar un poco; si
# aparecen falsos positivos, subir hacia 0.7.
CONFIANZA_DETECCION_MANO = 0.6


# ---------------------------------------------------------------------------
# Parametros de entrenamiento (valores iniciales, se afinan al entrenar)
# ---------------------------------------------------------------------------

# Proporciones de division del dataset.
PROPORCION_ENTRENAMIENTO = 0.70
PROPORCION_VALIDACION = 0.15
PROPORCION_PRUEBA = 0.15

# Semilla para reproducibilidad.
SEMILLA = 42
