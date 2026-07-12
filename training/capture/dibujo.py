"""
Primitivas de dibujo para la interfaz de captura.

Reune las funciones de bajo nivel que dibujan sobre el fotograma de OpenCV:
paneles translucidos, texto, barras de progreso, chips de estado y el esqueleto
de la mano. Mantiene una paleta de colores unica y sobria, en linea con la
estetica de los diagramas de la tesis.

El texto se renderiza con Pillow y una fuente TrueType real (Segoe UI en
Windows, con respaldo a DejaVu Sans). Esto es necesario porque las fuentes
internas de OpenCV no soportan caracteres no ASCII, y el proyecto usa la enie y
las tildes (por ejemplo la clase "Ñ" o la palabra "Señas"). Las formas
(paneles, lineas, barras) se siguen dibujando con OpenCV.

Todos los colores estan en formato BGR, porque es el que usa OpenCV.
"""

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paleta (BGR)
# ---------------------------------------------------------------------------

FONDO_PANEL = (28, 26, 24)      # casi negro calido, para paneles translucidos
BLANCO = (245, 245, 245)
GRIS_CLARO = (185, 185, 185)
GRIS_TENUE = (130, 130, 130)

VERDE = (96, 196, 70)           # estado correcto / mano detectada
ROJO = (66, 66, 230)            # grabando
AMBAR = (40, 175, 240)          # preparacion / atencion
TEAL = (200, 165, 80)           # acento neutro

MANO_LINEA = (210, 200, 165)
MANO_PUNTO = (96, 196, 70)

# Identificadores de estilo de fuente (regular o titulo). Se conservan estos
# nombres por compatibilidad con la interfaz; el titulo se dibuja en negrita.
FUENTE_TITULO = "titulo"
FUENTE_TEXTO = "texto"

# ---------------------------------------------------------------------------
# Conexiones del esqueleto de la mano (21 landmarks de MediaPipe).
# ---------------------------------------------------------------------------

CONEXIONES_MANO = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),          # indice
    (9, 10), (10, 11), (11, 12),             # medio
    (13, 14), (14, 15), (15, 16),            # anular
    (0, 17), (17, 18), (18, 19), (19, 20),   # menique
    (5, 9), (9, 13), (13, 17),               # palma
]


# ---------------------------------------------------------------------------
# Gestion de fuentes TrueType (Pillow)
# ---------------------------------------------------------------------------

_RUTAS_FUENTE = {
    False: [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"],
    True: [r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf"],
}
_cache_fuentes: dict = {}
_DRAW_DUMMY = ImageDraw.Draw(Image.new("RGB", (4, 4)))


def _ruta_fuente(negrita: bool):
    for ruta in _RUTAS_FUENTE[negrita]:
        if os.path.exists(ruta):
            return ruta
    # Respaldo multiplataforma: DejaVu Sans, que viene con matplotlib.
    try:
        import matplotlib
        sub = "DejaVuSans-Bold.ttf" if negrita else "DejaVuSans.ttf"
        ruta = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", sub)
        if os.path.exists(ruta):
            return ruta
    except Exception:
        pass
    return None


def _fuente(px: int, negrita: bool):
    clave = (px, negrita)
    fuente = _cache_fuentes.get(clave)
    if fuente is None:
        ruta = _ruta_fuente(negrita)
        fuente = ImageFont.truetype(ruta, px) if ruta else ImageFont.load_default()
        _cache_fuentes[clave] = fuente
    return fuente


def _px(escala: float) -> int:
    """Convierte la escala (estilo OpenCV) a tamano de fuente en pixeles."""
    return max(11, int(round(escala * 32)))


def _es_negrita(grosor: int, fuente) -> bool:
    return grosor >= 2 or fuente == FUENTE_TITULO


# ---------------------------------------------------------------------------
# Texto (Pillow sobre una region recortada del frame, por eficiencia)
# ---------------------------------------------------------------------------

def medir(texto, escala, grosor=1, fuente=None):
    """Devuelve (ancho, alto) en pixeles del texto."""
    f = _fuente(_px(escala), _es_negrita(grosor, fuente))
    izq, arriba, der, abajo = _DRAW_DUMMY.textbbox((0, 0), texto, font=f)
    return der - izq, abajo - arriba


def texto(frame, cadena, x, y, escala=0.7, color=BLANCO, grosor=1, fuente=None):
    """Dibuja texto Unicode. (x, y) es la esquina inferior izquierda (baseline)."""
    if not cadena:
        return
    x, y = int(x), int(y)
    f = _fuente(_px(escala), _es_negrita(grosor, fuente))
    ascenso, descenso = f.getmetrics()
    izq, arriba, der, abajo = _DRAW_DUMMY.textbbox((0, 0), cadena, font=f)
    ancho = der - izq

    # OpenCV ancla el texto por el baseline (y); Pillow por la cima de la linea.
    cima_x, cima_y = x, y - ascenso
    alto, anchura = frame.shape[:2]
    pad = 3
    x0 = max(0, cima_x - pad)
    y0 = max(0, cima_y - pad)
    x1 = min(anchura, cima_x + ancho + pad)
    y1 = min(alto, cima_y + ascenso + descenso + pad)
    if x1 <= x0 or y1 <= y0:
        return

    roi = frame[y0:y1, x0:x1]
    img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    dibujante = ImageDraw.Draw(img)
    color_rgb = (color[2], color[1], color[0])
    dibujante.text((cima_x - x0, cima_y - y0), cadena, font=f, fill=color_rgb)
    frame[y0:y1, x0:x1] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def texto_centrado(frame, cadena, centro_x, y, escala=0.7, color=BLANCO,
                   grosor=1, fuente=None):
    """Dibuja texto centrado horizontalmente en centro_x."""
    ancho, _ = medir(cadena, escala, grosor, fuente)
    texto(frame, cadena, int(centro_x - ancho / 2), y, escala, color, grosor,
          fuente)


# ---------------------------------------------------------------------------
# Formas (OpenCV)
# ---------------------------------------------------------------------------

def _rect_redondeado(img, x, y, w, h, radio, color) -> None:
    """Dibuja un rectangulo relleno con esquinas redondeadas."""
    radio = max(0, min(radio, w // 2, h // 2))
    x2, y2 = x + w, y + h
    cv2.rectangle(img, (x + radio, y), (x2 - radio, y2), color, -1)
    cv2.rectangle(img, (x, y + radio), (x2, y2 - radio), color, -1)
    for cx, cy in ((x + radio, y + radio), (x2 - radio, y + radio),
                   (x + radio, y2 - radio), (x2 - radio, y2 - radio)):
        cv2.circle(img, (cx, cy), radio, color, -1)


def panel(frame, x, y, w, h, color=FONDO_PANEL, alpha=0.66, radio=14) -> None:
    """Dibuja un panel translucido con esquinas redondeadas sobre el frame."""
    overlay = frame.copy()
    _rect_redondeado(overlay, int(x), int(y), int(w), int(h), radio, color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def barra_progreso(frame, x, y, w, h, fraccion, color=VERDE,
                   color_fondo=(60, 58, 55)) -> None:
    """Dibuja una barra de progreso horizontal. fraccion en [0, 1]."""
    fraccion = max(0.0, min(1.0, fraccion))
    radio = h // 2
    _rect_redondeado(frame, int(x), int(y), int(w), int(h), radio, color_fondo)
    ancho_relleno = int(w * fraccion)
    if ancho_relleno > 2 * radio:
        _rect_redondeado(frame, int(x), int(y), ancho_relleno, int(h), radio,
                         color)


def chip_estado(frame, x_derecha, y, etiqueta, color_acento) -> None:
    """Dibuja un chip con un punto de color y una etiqueta (indicador de estado).

    Se ancla por su borde derecho en x_derecha, de modo que crece hacia la
    izquierda y nunca se sale del margen sin importar el largo del texto.
    """
    escala, grosor = 0.6, 1
    ancho_txt, alto_txt = medir(etiqueta, escala, grosor)
    pad = 12
    radio_punto = 6
    alto = alto_txt + 2 * pad
    ancho = ancho_txt + 4 * pad + 2 * radio_punto
    x = int(x_derecha - ancho)
    panel(frame, x, y, ancho, alto, alpha=0.72, radio=alto // 2)
    cy = y + alto // 2
    cv2.circle(frame, (int(x + pad + radio_punto), cy), radio_punto,
               color_acento, -1, cv2.LINE_AA)
    texto(frame, etiqueta, x + 2 * pad + 2 * radio_punto, cy + alto_txt // 2,
          escala, BLANCO, grosor)


def dibujar_mano(frame, landmarks, color_linea=MANO_LINEA,
                 color_punto=MANO_PUNTO) -> None:
    """Dibuja el esqueleto de la mano sobre el frame.

    landmarks: lista de 21 puntos (x, y, z) con x, y normalizados en [0, 1].
    """
    alto, ancho = frame.shape[:2]
    puntos_px = [(int(p[0] * ancho), int(p[1] * alto)) for p in landmarks]

    for a, b in CONEXIONES_MANO:
        if a < len(puntos_px) and b < len(puntos_px):
            cv2.line(frame, puntos_px[a], puntos_px[b], color_linea, 2,
                     cv2.LINE_AA)
    for px in puntos_px:
        cv2.circle(frame, px, 4, color_punto, -1, cv2.LINE_AA)


# Indices de MediaPipe Pose para dibujar el cuerpo (torso, brazos y cara).
_POSE_CONEXIONES = [
    (11, 12),                        # hombros
    (11, 13), (13, 15),              # brazo izquierdo: hombro-codo-muneca
    (12, 14), (14, 16),              # brazo derecho
    (11, 23), (12, 24), (23, 24),    # tronco
]
_POSE_ARTICULACIONES = [11, 12, 13, 14, 15, 16, 23, 24]
_POSE_CARA = [0, 2, 5, 7, 8, 9, 10]  # nariz, ojos, orejas y comisuras de la boca


def dibujar_cuerpo(frame, puntos, vis_min: float = 0.5) -> None:
    """Dibuja el esqueleto del cuerpo (MediaPipe Pose) sobre el frame.

    puntos: lista de 33 landmarks con `.x`, `.y` normalizados en [0, 1] y
    `.visibilidad`. Solo se dibuja lo visible con confianza suficiente. Los puntos
    de la cara (nariz, ojos, orejas, boca) se resaltan en ambar porque son las
    anclas que usa el marco del cuerpo del Modelo B.
    """
    alto, ancho = frame.shape[:2]

    def _px(i):
        return int(puntos[i].x * ancho), int(puntos[i].y * alto)

    def _vis(i):
        return i < len(puntos) and puntos[i].visibilidad >= vis_min

    for a, b in _POSE_CONEXIONES:
        if _vis(a) and _vis(b):
            cv2.line(frame, _px(a), _px(b), TEAL, 2, cv2.LINE_AA)
    for i in _POSE_ARTICULACIONES:
        if _vis(i):
            cv2.circle(frame, _px(i), 4, VERDE, -1, cv2.LINE_AA)
    for i in _POSE_CARA:
        if _vis(i):
            cv2.circle(frame, _px(i), 3, AMBAR, -1, cv2.LINE_AA)

    # Centro de los hombros: es el ORIGEN del marco del cuerpo (respecto a el se
    # mide la ubicacion de las manos y la cara). Se resalta con un anillo ambar.
    if _vis(11) and _vis(12):
        (x1, y1), (x2, y2) = _px(11), _px(12)
        centro = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.circle(frame, centro, 6, AMBAR, -1, cv2.LINE_AA)
        cv2.circle(frame, centro, 11, AMBAR, 1, cv2.LINE_AA)
