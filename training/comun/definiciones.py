"""
Definiciones del dominio del problema LESHO.

Este modulo es la fuente unica de verdad para:
  - las constantes estructurales de los landmarks (cuantos puntos, cuantas
    coordenadas, tamano del vector de entrada de los modelos),
  - las listas ordenadas de clases de los Modelos A y B.

No tiene dependencias externas. Cualquier otro modulo del pipeline (captura,
normalizacion, entrenamiento) importa desde aqui para no duplicar estos valores.

IMPORTANTE: el ORDEN de las listas de clases es un contrato. El indice de cada
clase en la lista es el indice de la neurona de salida del modelo y la linea
correspondiente en el archivo de etiquetas (.txt). Una vez entrenado un modelo,
este orden no se reordena nunca.
"""

# ---------------------------------------------------------------------------
# Constantes estructurales de los landmarks (fijas, las define MediaPipe Hands)
# ---------------------------------------------------------------------------

# MediaPipe Hands siempre devuelve 21 puntos por mano.
NUM_LANDMARKS = 21

# Cada landmark tiene tres coordenadas: x, y, z.
NUM_COORDENADAS = 3

# El sistema soporta dos manos (muchas senas del LESHO son bimanuales).
NUM_MANOS = 2

# Vector de UNA mano: 21 landmarks * 3 coordenadas = 63 valores.
TAMANO_VECTOR_MANO = NUM_LANDMARKS * NUM_COORDENADAS  # 63

# Vector de entrada de los modelos: dos manos = 126 valores. La primera mitad
# (63) corresponde a la mano izquierda y la segunda a la derecha. Si en un
# fotograma solo hay una mano, la mitad de la mano ausente va en ceros.
TAMANO_VECTOR = TAMANO_VECTOR_MANO * NUM_MANOS  # 126

# Indice del landmark de la muneca dentro de la lista de MediaPipe.
# Es el punto 0 y se usa como origen para normalizar los demas.
INDICE_MUNECA = 0

# Etiquetas de lateralidad. Determinan en que mitad del vector va cada mano.
# NOTA: por el efecto espejo (vista selfie), la etiqueta que reporta MediaPipe
# no coincide con la mano fisica real, pero es consistente entre la captura y la
# app, que es lo que importa para el modelo.
LATERALIDAD_IZQUIERDA = "Izquierda"
LATERALIDAD_DERECHA = "Derecha"
LATERALIDAD_DESCONOCIDA = "Desconocida"


# ---------------------------------------------------------------------------
# Constantes de MediaPipe Pose (Modelo B: ubicacion de la mano en el cuerpo)
# ---------------------------------------------------------------------------

# Muchas senas dinamicas se distinguen por DONDE se hacen en el cuerpo (lugar de
# articulacion), no solo por la forma o el movimiento de la mano. La
# normalizacion respecto a la muneca captura la forma pero borra la ubicacion,
# asi que el Modelo B agrega un marco de referencia del cuerpo con MediaPipe
# Pose. Pose devuelve 33 puntos; el marco usa solo unos pocos. Indices fijados
# por MediaPipe (no cambiarlos):
NUM_LANDMARKS_POSE = 33
POSE_NARIZ = 0          # centro de la cara
POSE_OJO_IZQ = 2        # ojo izquierdo (nivel de los ojos = zona alta de la cara)
POSE_OJO_DER = 5        # ojo derecho
POSE_OREJA_IZQ = 7      # oreja izquierda (extremo lateral / ancho de la cabeza)
POSE_OREJA_DER = 8      # oreja derecha
POSE_BOCA_IZQ = 9       # comisura izquierda (nivel de la boca = zona baja de la cara)
POSE_BOCA_DER = 10      # comisura derecha
POSE_HOMBRO_IZQ = 11    # el centro de los hombros es el origen del marco
POSE_HOMBRO_DER = 12    # la distancia entre hombros es la escala del marco
POSE_CADERA_IZQ = 23    # define el tronco (pecho, abdomen, cintura)
POSE_CADERA_DER = 24

# Visibilidad minima (0..1) para confiar en un punto de Pose. Si los hombros no
# se ven con esta confianza, el marco del cuerpo de ese fotograma no es fiable.
VISIBILIDAD_MINIMA_POSE = 0.5


# ---------------------------------------------------------------------------
# Vector de entrada del Modelo B (por fotograma)
# ---------------------------------------------------------------------------

# Cada fotograma que entra al Modelo B tiene dos partes:
#   1. CONFIGURACION de las dos manos: los mismos 126 valores del Modelo A
#      (landmarks relativos a la muñeca). Captura la FORMA y el movimiento.
#   2. UBICACION en el cuerpo: (rx, ry) en anchos de hombro respecto al centro de
#      los hombros, para 9 puntos, en este orden fijo (CONTRATO):
#        muñeca izquierda, muñeca derecha,
#        punta del indice izquierda, punta del indice derecha,
#        nariz, ojos (centro), boca (centro), oreja izquierda, oreja derecha.
#      Se agrega la PUNTA DEL INDICE ademas de la muñeca porque muchas senas se
#      hacen APUNTANDO con los dedos a una zona (HOLA a la frente, AGUA a la boca):
#      la muñeca puede quedar al costado mientras los dedos apuntan a otra altura,
#      asi que la punta del indice captura DONDE se hace la sena mejor que la
#      muñeca. Los cinco puntos de la cara son el marco facial (ojos arriba, nariz
#      al medio, boca abajo, orejas a los lados) para ubicar con precision en la
#      cara. Todo esto lo borra la normalizacion respecto a la muñeca; por eso se
#      agrega aparte. El orden (config primero, ubicacion despues) es un CONTRATO.
INDICE_PUNTA_INDICE = 8                          # landmark de la punta del indice
TAMANO_UBICACION_PUNTO = 2                       # (rx, ry) de un punto
# 9 puntos: 2 muñecas + 2 puntas del indice + 5 anclas de la cara.
TAMANO_UBICACION = TAMANO_UBICACION_PUNTO * 9    # 18
TAMANO_VECTOR_B = TAMANO_VECTOR + TAMANO_UBICACION  # 126 + 18 = 144

# Rasgos RELATIVOS de la PUNTA DEL INDICE a las anclas de la cara: por cada mano,
# su posicion respecto a la nariz y a la boca (punta - nariz, punta - boca). Hacen
# EXPLICITO "a que zona de la cara APUNTA la mano" (frente = arriba de la nariz,
# boca, menton = debajo de la boca). Es la senal mas fuerte para separar senas
# faciales (HOLA en la frente vs AGUA en la boca, PAPA sobre el labio vs MAMA en el
# menton). Se DERIVAN en el preprocesamiento; el CSV se graba con 144.
TAMANO_RELATIVO = 8   # (izq-nariz, izq-boca, der-nariz, der-boca) x (x, y)

# Dimension del vector que ENTRA al Modelo B (lo que produce procesar_secuencia):
# forma (126) + ubicacion (18) + rasgos relativos (8) = 152.
TAMANO_ENTRADA_B = TAMANO_VECTOR_B + TAMANO_RELATIVO  # 152

# Ganancia del bloque de UBICACION+RELATIVOS en el preprocesamiento del Modelo B
# (se aplica en procesar_secuencia, igual en entrenamiento e inferencia).
# CONTRATO: si la app implementa su preprocesamiento en Dart, debe aplicar la
# misma ganancia. Por que: la forma ocupa 126 dimensiones y la ubicacion pocas,
# asi que sin ganancia dos senas de mano casi igual en lugar distinto (PAPA en la
# cara, HOSPITAL en el hombro) se confundian. La ganancia balancea la varianza
# del bloque de lugar (22 dims) con la de la forma (22 x 2.5^2 ~= 138 ~ 126).
GANANCIA_UBICACION = 2.5


# ---------------------------------------------------------------------------
# Ventana temporal del Modelo A
# ---------------------------------------------------------------------------

# El Modelo A no clasifica un fotograma suelto, sino una ventana de los ultimos
# N fotogramas. Asi reconoce tanto letras estaticas (ventana casi quieta) como
# letras con movimiento (J, Ñ, Z, LL, RR) con una sola arquitectura. La entrada
# del modelo es un tensor de forma [TAMANO_VENTANA_A, TAMANO_VECTOR].
#
# Valor inicial 20 (~0.7 s a 30 fps). Es un compromiso de optimizacion: mas
# fotogramas capturan mejor el movimiento pero encarecen cada inferencia en el
# telefono, porque el modelo corre sobre la ventana en cada fotograma. En gama
# baja o media conviene el menor N que aun reconozca bien las letras con
# movimiento. Se afina al entrenar.
TAMANO_VENTANA_A = 20


# ---------------------------------------------------------------------------
# Clases del Modelo A (alfabeto): 30 letras + INICIO + FIN + REPOSO = 33
# ---------------------------------------------------------------------------

# Las 30 letras del alfabeto dactilologico del LESHO, en orden alfabetico
# espanol tradicional, con los digrafos CH (tras C), LL (tras L) y RR (tras R),
# y la enie tras la N. El orden es un CONTRATO: define los indices de salida del
# Modelo A y debe coincidir exactamente con app/assets/labels/etiquetas_a.txt.
LETRAS = [
    "A", "B", "C", "CH", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "LL", "M", "N", "Ñ", "O", "P", "Q",
    "R", "RR", "S", "T", "U", "V", "W", "X", "Y", "Z",
]

# Subconjunto del alfabeto que se ejecuta CON MOVIMIENTO. Siguen siendo clases
# del Modelo A (ocupan las mismas salidas y el mismo orden), pero se graban con
# la captura dinamica para conservar su trayectoria a 30 fps, y en el
# preprocesamiento se convierten en ventanas con movimiento, no en ventanas
# quietas. CH y R son estaticas; solo estas cinco letras llevan movimiento.
LETRAS_CON_MOVIMIENTO = ["J", "Ñ", "Z", "LL", "RR"]

# El resto del alfabeto se graba como pose fija.
LETRAS_ESTATICAS = [letra for letra in LETRAS if letra not in LETRAS_CON_MOVIMIENTO]

# Senas de control que NO son letras:
#   INICIO -> activa el modo de grabacion dinamica (Modelo B)
#   FIN    -> cierra la grabacion dinamica y dispara la inferencia del Modelo B
#   REPOSO -> mano visible pero relajada, sin ejecutar sena (estado neutro)
# INICIO y FIN son dos senas DISTINTAS entre si (decision cerrada en CLAUDE.md).
SENAS_CONTROL = ["INICIO", "FIN", "REPOSO"]

# Lista completa y ordenada de las 33 clases del Modelo A.
CLASES_ESTATICAS = LETRAS + SENAS_CONTROL

# Numero esperado de clases del Modelo A.
NUM_CLASES_A = 33


# ---------------------------------------------------------------------------
# Clases del Modelo B (dinamico): 50 senas dinamicas
# ---------------------------------------------------------------------------

# Las 50 senas dinamicas del Modelo B, definidas con el autor (2026-07-06) sobre
# la lista de vocabulario cotidiano. El ORDEN es un CONTRATO: fija el indice de
# salida del modelo y la linea en etiquetas_b.txt; no se reordena una vez
# entrenado. Identificadores en MAYUSCULA, sin acentos en las vocales, con enie,
# y con guion bajo en lugar de espacio (POR_FAVOR). La articulacion EXACTA de
# cada sena en LESHO la ejecuta la persona que graba (asesoria LESHO).
#
# Notas de la lista: el unico cambio respecto a la lista original es que TRABAJAR
# (verbo) se reemplaza por TRABAJO (lugar), porque suelen ser la misma sena y
# TRABAJO es mas util como lugar. MANANA puede significar "manana" (dia siguiente)
# o "en la manana"; que la persona que graba use la sena del sentido pretendido
# (dia siguiente).
CLASES_DINAMICAS: list[str] = [
    # Cortesia (5)
    "HOLA", "ADIOS", "GRACIAS", "POR_FAVOR", "PERDON",
    # Respuestas (3)
    "SI", "NO", "BIEN",
    # Necesidades (8)
    "AGUA", "COMER", "BAÑO", "DORMIR", "AYUDA", "DOLOR", "HAMBRE", "ENFERMO",
    # Personas / familia (7)
    "MAMA", "PAPA", "FAMILIA", "AMIGO", "NIÑO", "MAESTRO", "DOCTOR",
    # Verbos de uso diario (11)
    "QUERER", "NECESITAR", "TENER", "DAR", "IR", "VENIR", "JUGAR", "ESTUDIAR",
    "APRENDER", "COMPRAR", "ESPERAR",
    # Lugares (6)
    "CASA", "ESCUELA", "HOSPITAL", "TIENDA", "CALLE", "TRABAJO",
    # Emociones / estados (5)
    "FELIZ", "TRISTE", "ENOJADO", "CANSADO", "MIEDO",
    # Tiempo (5)
    "HOY", "MAÑANA", "AYER", "DIA", "NOCHE",
]

# NOTA (2026-07-06): se probo una clase de rechazo "NADA" (no-senas) y se
# DESCARTO: con pocas tomas, una clase tan variada se superpone con las senas
# reales y degrada el reconocimiento. La app usa segmentacion explicita (INICIO,
# boton en pantalla o ESPACIO abre la ventana de palabra), asi que no hace falta
# rechazo aprendido. Las tomas de NADA que existan en el CSV se ignoran al
# construir el dataset.

# Numero esperado de clases dinamicas (objetivo final).
NUM_CLASES_B = 50


# ---------------------------------------------------------------------------
# Validaciones de integridad
# ---------------------------------------------------------------------------

def validar_clases_estaticas() -> None:
    """Verifica que las clases estaticas cumplan el contrato.

    Lanza ValueError si hay un numero incorrecto de clases o duplicados.
    """
    if len(CLASES_ESTATICAS) != NUM_CLASES_A:
        raise ValueError(
            f"Se esperaban {NUM_CLASES_A} clases estaticas, "
            f"hay {len(CLASES_ESTATICAS)}."
        )
    if len(set(CLASES_ESTATICAS)) != len(CLASES_ESTATICAS):
        raise ValueError("Hay clases estaticas duplicadas.")


def validar_letras_con_movimiento() -> None:
    """Verifica que toda letra con movimiento pertenezca al alfabeto.

    Lanza ValueError si `LETRAS_CON_MOVIMIENTO` incluye algo que no esta en
    `LETRAS`. Evita que un error de escritura deje una letra sin capturar.
    """
    fuera = [letra for letra in LETRAS_CON_MOVIMIENTO if letra not in LETRAS]
    if fuera:
        raise ValueError(
            f"LETRAS_CON_MOVIMIENTO tiene letras que no estan en LETRAS: {fuera}."
        )
    if len(set(LETRAS_CON_MOVIMIENTO)) != len(LETRAS_CON_MOVIMIENTO):
        raise ValueError("Hay letras con movimiento duplicadas.")


def validar_clases_dinamicas() -> None:
    """Verifica que las clases dinamicas esten definidas y sean coherentes.

    Lanza ValueError si la lista esta vacia, tiene un numero distinto al
    esperado, o tiene duplicados. Se llama antes de capturar o entrenar el
    Modelo B, no antes del Modelo A.
    """
    if len(CLASES_DINAMICAS) == 0:
        raise ValueError(
            "CLASES_DINAMICAS esta vacia. Defina las 50 senas dinamicas con "
            "asesoria LESHO antes de capturar o entrenar el Modelo B."
        )
    if len(CLASES_DINAMICAS) != NUM_CLASES_B:
        raise ValueError(
            f"Se esperaban {NUM_CLASES_B} clases dinamicas, "
            f"hay {len(CLASES_DINAMICAS)}."
        )
    if len(set(CLASES_DINAMICAS)) != len(CLASES_DINAMICAS):
        raise ValueError("Hay clases dinamicas duplicadas.")


# Se valida lo estatico al importar, porque es fijo y deberia estar siempre bien.
validar_clases_estaticas()
validar_letras_con_movimiento()
