"""
Visor de clips de senas: reproduce un clip como muneco de capsulas (Fase 0).

Herramienta de control de calidad de la Direccion 2 (texto -> sena). Carga los
clips JSON grabados con capture/captura_diccionario.py y los reproduce sobre el
muneco volumetrico de capsulas descrito en PLAN_DIRECCION2.md (seccion 6):
capsulas conicas con luz fija, articulaciones soldadas, cabeza esferica sin
rostro, orden de dibujado por profundidad. Sirve para validar la LEGIBILIDAD de
cada sena (sobre todo la orientacion de la palma) con una persona que conozca
LESHO, antes de grabar el diccionario completo.

Limpieza al cargar (el clip se guarda CRUDO en la Fase 0):
  - interpolacion de huecos cortos de manos (mano perdida por unos fotogramas),
  - cuerpo continuo (interpolacion total + bordes extendidos),
  - suavizado One Euro por tramo, con los parametros del proyecto (config).

Uso (desde la carpeta training/):

    python demo/visor_clips.py                      # todos los clips del piloto
    python demo/visor_clips.py clips/piloto/HOLA_t01.json
    python demo/visor_clips.py --exportar salida --cada 5   # PNGs, sin ventana

Teclas: ESPACIO pausa | A/D fotograma anterior/siguiente | V velocidad |
        M espejo | N/P clip siguiente/anterior | G guardar PNG | Q salir
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Permite ejecutar el archivo directamente: agrega training/ al path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from comun.clips import (  # noqa: E402
    CUERPO_CADERA_DER,
    CUERPO_CADERA_IZQ,
    CUERPO_CODO_DER,
    CUERPO_CODO_IZQ,
    CUERPO_HOMBRO_DER,
    CUERPO_HOMBRO_IZQ,
    CUERPO_MUNECA_DER,
    CUERPO_MUNECA_IZQ,
    CUERPO_NARIZ,
    NUM_PUNTOS_CUERPO_CLIP,
    cargar_clip,
)
from comun.suavizado import suavizar_secuencia  # noqa: E402
from capture import dibujo  # noqa: E402

# ---------------------------------------------------------------------------
# Parametros del muneco (proporciones en fracciones del ANCHO DE HOMBROS)
# ---------------------------------------------------------------------------

# Grosores y radios (PLAN_DIRECCION2 seccion 6.2; se calibran aqui, en el
# piloto, y el MunecoPainter de Flutter debe copiar los valores finales).
PROP_BRAZO = 0.34          # grosor del brazo en el hombro (hombro a codo)
PROP_ANTEBRAZO = 0.25      # grosor del antebrazo (codo a muneca)
PROP_CUELLO = 0.28         # grosor del cuello
PROP_DEDO = 0.10           # grosor base de un dedo
MARGEN_PALMA = 1.15        # engorde del blob de la palma, en radios de dedo
BOLA_NUDILLO = 0.54        # radio de la bola de cada articulacion del dedo,
                           # en radios de dedo (maniqui: 21 landmarks visibles)
PROP_RADIO_CABEZA = 0.44   # radio de la cabeza (horizontal)
OVALO_CABEZA = 1.07        # la cabeza es levemente ovalada (mas alta que ancha)
PROP_RADIO_TORSO = 0.15    # redondeo de las esquinas del torso
ANGOSTE_CADERAS = 0.86     # las caderas se dibujan mas angostas que lo medido
PROP_BOLA_CODO = 0.15      # radio de la bola de articulacion del codo
PROP_BOLA_MUNECA = 0.10    # radio de la bola de articulacion de la muneca

# Torso de maniqui en TRES piezas (como la figura de referencia): pecho ancho,
# cintura angosta y bloque de cadera. Cada tupla es (t_inicio, t_fin,
# ancho_inicio, ancho_fin): t recorre de la linea de hombros (0) a la de
# caderas (1) y el ancho es relativo al ancho local interpolado.
PIEZAS_TORSO = [
    (0.00, 0.62, 1.00, 0.82),   # pecho
    (0.50, 0.85, 0.62, 0.54),   # cintura
    (0.74, 1.00, 1.00, 1.08),   # cadera
]

# Cuanto del ancho del lienzo ocupa el ancho de hombros del muneco.
FRACCION_HOMBROS_CANVAS = 0.335

# Posicion vertical del centro de hombros en el lienzo (fraccion de la altura).
ALTURA_HOMBROS_CANVAS = 0.46

# Direccion de la luz (fija, arriba a la izquierda), normalizada.
_LUZ = np.array([-0.45, -0.89])

# Huecos de mano de hasta este tiempo se interpolan; mas largos, la mano no se
# dibuja en ese tramo (y la toma probablemente amerite repetirse).
MAX_HUECO_MANO_S = 0.35

# Modulacion de grosor por profundidad (pseudo-3D): un dedo mas cerca de la
# camara se dibuja mas grueso. factor = 1 - z * GANANCIA_Z, acotado.
GANANCIA_Z_DEDOS = 5.0
FACTOR_Z_MIN, FACTOR_Z_MAX = 0.72, 1.30

# Cadenas de falanges de cada dedo: (landmark_inicio, landmark_fin) encadenados.
_DEDOS = [
    [1, 2, 3, 4],        # pulgar
    [5, 6, 7, 8],        # indice
    [9, 10, 11, 12],     # medio
    [13, 14, 15, 16],    # anular
    [17, 18, 19, 20],    # menique
]
# Puntos que definen el blob de la palma: muneca, base del pulgar y nudillos.
_PALMA = [0, 1, 2, 5, 9, 13, 17]

# Paletas del muneco (BGR). El color del cuerpo y el de las manos difieren un
# paso de tono para que una mano frente al pecho se lea sin esfuerzo.
PALETAS = {
    # Azul de juguete (la referencia visual aprobada en el plan).
    "azul": {"cuerpo": (196, 124, 56), "mano": (222, 168, 108)},
    # Terracota, por si se prefiere alinear con la paleta de la app.
    "terracota": {"cuerpo": (43, 96, 186), "mano": (96, 150, 226)},
}
COLOR_FONDO = (238, 248, 255)  # crema, el fondo de la app


def _tonos(base_bgr):
    """Deriva (oscuro, base, claro) de un color base, para el sombreado."""
    base = np.array(base_bgr, dtype=np.float64)
    oscuro = base * 0.60
    claro = base + (255.0 - base) * 0.38
    return (
        tuple(int(v) for v in oscuro),
        tuple(int(v) for v in base),
        tuple(int(v) for v in claro),
    )


# ---------------------------------------------------------------------------
# Preparacion del clip: arrays, interpolacion de huecos y suavizado
# ---------------------------------------------------------------------------

def _interpolar_huecos(arr, presentes, max_hueco, extender_bordes):
    """Rellena huecos de una secuencia (T, D) por interpolacion lineal.

    `presentes` marca los fotogramas con dato real. Un hueco interior de hasta
    `max_hueco` fotogramas (None = sin limite) se interpola entre sus extremos.
    Con `extender_bordes`, los tramos vacios del inicio y el final se rellenan
    con el primer/ultimo valor real (solo tiene sentido para el cuerpo).
    Devuelve (arr_relleno, presentes_actualizado) sin tocar los originales.
    """
    arr = arr.copy()
    presentes = presentes.copy()
    indices = np.where(presentes)[0]
    if len(indices) == 0:
        return arr, presentes

    for a, b in zip(indices[:-1], indices[1:]):
        largo = b - a - 1
        if largo == 0:
            continue
        if max_hueco is not None and largo > max_hueco:
            continue
        for k in range(1, largo + 1):
            alfa = k / (largo + 1)
            arr[a + k] = (1 - alfa) * arr[a] + alfa * arr[b]
            presentes[a + k] = True

    if extender_bordes:
        arr[: indices[0]] = arr[indices[0]]
        arr[indices[-1] + 1:] = arr[indices[-1]]
        presentes[:] = True

    return arr, presentes


def _suavizar_tramos(arr, presentes, fps):
    """Aplica One Euro a cada tramo contiguo presente de la secuencia (T, D)."""
    arr = arr.copy()
    t = 0
    T = len(arr)
    while t < T:
        if not presentes[t]:
            t += 1
            continue
        fin = t
        while fin < T and presentes[fin]:
            fin += 1
        if fin - t >= 2:
            arr[t:fin] = suavizar_secuencia(
                arr[t:fin], fps,
                config.SUAVIZADO_MIN_CUTOFF, config.SUAVIZADO_BETA,
                config.SUAVIZADO_D_CUTOFF,
            )
        t = fin
    return arr


class ClipPreparado:
    """Un clip listo para renderizar: arrays limpios en espacio fisico.

    El "espacio fisico" corrige el aspecto de la imagen de captura: MediaPipe
    normaliza x por el ancho e y por el alto por separado, asi que x (y z, que
    esta en la escala de x) se multiplican por el aspecto para que las
    distancias sean reales y el muneco no salga deformado.
    """

    def __init__(self, clip):
        self.palabra = clip["palabra"]
        self.fps = float(clip["fps"])
        self.aspecto = float(clip["aspecto"])
        frames = clip["frames"]
        T = len(frames)

        cuerpo = np.full((T, NUM_PUNTOS_CUERPO_CLIP, 3), np.nan)
        cuerpo_ok = np.zeros(T, dtype=bool)
        manos = {
            "izq": np.full((T, 21, 3), np.nan),
            "der": np.full((T, 21, 3), np.nan),
        }
        mano_ok = {"izq": np.zeros(T, dtype=bool), "der": np.zeros(T, dtype=bool)}

        for t, frame in enumerate(frames):
            if frame["cuerpo"] is not None:
                cuerpo[t] = [p[:3] for p in frame["cuerpo"]]
                cuerpo_ok[t] = True
            for lado, clave in (("izq", "mano_izq"), ("der", "mano_der")):
                if frame[clave] is not None:
                    manos[lado][t] = frame[clave]
                    mano_ok[lado][t] = True

        # A espacio fisico: x y z en la escala del alto de la imagen.
        for arr in (cuerpo, manos["izq"], manos["der"]):
            arr[..., 0] *= self.aspecto
            arr[..., 2] *= self.aspecto

        # Cuerpo: continuo de punta a punta (el torso no puede parpadear).
        plano = cuerpo.reshape(T, -1)
        plano, cuerpo_ok = _interpolar_huecos(plano, cuerpo_ok, None, True)
        plano = _suavizar_tramos(plano, cuerpo_ok, self.fps)
        self.cuerpo = plano.reshape(T, NUM_PUNTOS_CUERPO_CLIP, 3)

        # Manos: interpolar solo huecos cortos; los largos quedan sin mano.
        max_hueco = max(1, int(round(MAX_HUECO_MANO_S * self.fps)))
        self.manos = {}
        self.mano_ok = {}
        for lado in ("izq", "der"):
            plano = manos[lado].reshape(T, -1)
            plano, ok = _interpolar_huecos(plano, mano_ok[lado], max_hueco, False)
            plano = _suavizar_tramos(plano, ok, self.fps)
            self.manos[lado] = plano.reshape(T, 21, 3)
            self.mano_ok[lado] = ok

        self.num_frames = T

        # Marco del clip: mediana del centro y del ancho de hombros. La mediana
        # da una escala y un encuadre estables (no "respiran" con el ruido).
        hi = self.cuerpo[:, CUERPO_HOMBRO_IZQ, :2]
        hd = self.cuerpo[:, CUERPO_HOMBRO_DER, :2]
        self.centro = np.median((hi + hd) / 2.0, axis=0)
        self.ancho_hombros = float(np.median(np.linalg.norm(hi - hd, axis=1)))

        # Caja que abarca TODO el clip (cuerpo, manos y cabeza estimada), para
        # que el encuadre garantice que ninguna sena se corte en el lienzo.
        xs = [self.cuerpo[..., 0].ravel()]
        ys = [self.cuerpo[..., 1].ravel()]
        for lado in ("izq", "der"):
            if self.mano_ok[lado].any():
                xs.append(self.manos[lado][self.mano_ok[lado], :, 0].ravel())
                ys.append(self.manos[lado][self.mano_ok[lado], :, 1].ravel())
        xs, ys = np.concatenate(xs), np.concatenate(ys)
        # La cabeza sobresale de la nariz hacia arriba y a los lados.
        alcance_cabeza = (0.10 + PROP_RADIO_CABEZA + 0.06) * self.ancho_hombros
        nariz_x = self.cuerpo[:, CUERPO_NARIZ, 0]
        nariz_y = self.cuerpo[:, CUERPO_NARIZ, 1]
        self.bbox = (
            float(min(np.nanmin(xs), np.nanmin(nariz_x) - alcance_cabeza)),
            float(min(np.nanmin(ys), np.nanmin(nariz_y) - alcance_cabeza)),
            float(max(np.nanmax(xs), np.nanmax(nariz_x) + alcance_cabeza)),
            float(np.nanmax(ys)),
        )

        # Estadistica simple para el HUD de control de calidad.
        self.frames_sin_mano = {
            lado: int((~self.mano_ok[lado]).sum()) for lado in ("izq", "der")
        }

    def fotograma(self, indice_flotante):
        """Devuelve (cuerpo, mano_izq, mano_der) interpolados entre fotogramas.

        La interpolacion entre fotogramas vecinos hace fluida la reproduccion a
        velocidades bajas (0.5x). Una mano solo se interpola si esta presente en
        ambos fotogramas; si no, se usa la del fotograma mas cercano disponible.
        """
        i0 = int(np.floor(indice_flotante))
        i0 = max(0, min(i0, self.num_frames - 1))
        i1 = min(i0 + 1, self.num_frames - 1)
        alfa = float(indice_flotante - i0)

        cuerpo = (1 - alfa) * self.cuerpo[i0] + alfa * self.cuerpo[i1]
        salida = [cuerpo]
        for lado in ("izq", "der"):
            ok0, ok1 = self.mano_ok[lado][i0], self.mano_ok[lado][i1]
            if ok0 and ok1:
                mano = (1 - alfa) * self.manos[lado][i0] + alfa * self.manos[lado][i1]
            elif ok0:
                mano = self.manos[lado][i0]
            elif ok1:
                mano = self.manos[lado][i1]
            else:
                mano = None
            salida.append(mano)
        return salida


# ---------------------------------------------------------------------------
# Renderizador: muneco de capsulas
# ---------------------------------------------------------------------------

class MunecoCapsulas:
    """Dibuja un fotograma del clip como muneco volumetrico de capsulas."""

    def __init__(self, ancho=720, alto=900, paleta="azul"):
        self.w = ancho
        self.h = alto
        colores = PALETAS[paleta]
        self.tonos_cuerpo = _tonos(colores["cuerpo"])
        self.tonos_mano = _tonos(colores["mano"])

    # -- Mapeo del espacio fisico al lienzo ---------------------------------

    def preparar_marco(self, clip: ClipPreparado, vista_espejo: bool):
        """Fija escala y traslacion para encuadrar el muneco en el lienzo.

        La escala base hace que el ancho de hombros ocupe una fraccion fija del
        lienzo; si con eso alguna parte del clip (una mano en alto, por ejemplo)
        se saliera, la escala se reduce y el muneco se desplaza para que TODA la
        sena quepa con margen.

        VISTA (lateralidad): la captura es en espejo (selfie), asi que el clip
        guarda la imagen especular del firmante. La vista POR DEFECTO ("de
        frente") voltea horizontalmente al dibujar, para que el muneco sea una
        persona de frente que firma con su mano derecha real (la sena de mano
        derecha se ve a la IZQUIERDA del espectador, como con un interprete).
        Con `vista_espejo` se dibuja tal cual se grabo (el muneco como reflejo,
        util para que un nino IMITE la sena); se valida con asesoria LESHO.
        """
        self._centro = clip.centro
        self._vista_espejo = vista_espejo

        margen = 30.0
        piso = self.h - 84.0  # margen inferior + panel del HUD
        x0, y0, x1, y1 = clip.bbox
        escala = FRACCION_HOMBROS_CANVAS * self.w / clip.ancho_hombros
        if x1 - x0 > 1e-6:
            escala = min(escala, (self.w - 2 * margen) / (x1 - x0))
        if y1 - y0 > 1e-6:
            escala = min(escala, (piso - margen) / (y1 - y0))
        self._escala = escala

        # Altura de los hombros: la nominal, corregida para que el clip quepa.
        y_hombros = self.h * ALTURA_HOMBROS_CANVAS
        arriba = y_hombros + (y0 - clip.centro[1]) * escala
        abajo = y_hombros + (y1 - clip.centro[1]) * escala
        if arriba < margen:
            y_hombros += margen - arriba
        elif abajo > piso:
            y_hombros -= abajo - piso
        self._y_hombros = y_hombros

        # Grosores en pixeles, derivados del ancho de hombros EN EL LIENZO.
        self.S = clip.ancho_hombros * self._escala

    def _px(self, p):
        """Punto fisico (x, y[, z]) -> pixel (x, y) del lienzo.

        Sin `vista_espejo` se voltea la x: el clip viene en espejo (selfie) y
        el volteo lo convierte en una persona vista de frente.
        """
        dx = (p[0] - self._centro[0]) * self._escala
        if not self._vista_espejo:
            dx = -dx
        x = self.w / 2.0 + dx
        y = self._y_hombros + (p[1] - self._centro[1]) * self._escala
        return np.array([x, y])

    # -- Primitivas con sombreado --------------------------------------------

    # Capas del sombreado: (indice de tono, factor de radio, corrimiento a la luz).
    _CAPAS = ((0, 1.00, 0.00), (1, 0.80, 0.16), (2, 0.46, 0.34))

    def _cadena(self, img, segmentos, tonos):
        """Cadena de capsulas conicas sombreadas, dibujada POR CAPA.

        `segmentos` es una lista de (a, b, ra, rb) en pixeles. Dibujar toda la
        cadena capa por capa (primero toda la sombra, luego toda la base, luego
        todo el brillo) funde las articulaciones sin costuras: un dedo o un
        brazo se ve como una sola pieza continua, no como capsulas apiladas.
        """
        for indice, factor, corr in self._CAPAS:
            color = tonos[indice]
            for a, b, ra, rb in segmentos:
                da = _LUZ * ra * corr
                db = _LUZ * rb * corr
                self._capsula_solida(img, a + da, b + db,
                                     ra * factor, rb * factor, color)

    def _capsula(self, img, a, b, ra, rb, tonos):
        """Una capsula conica sombreada suelta (caso particular de _cadena)."""
        self._cadena(img, [(a, b, ra, rb)], tonos)

    @staticmethod
    def _capsula_solida(img, a, b, ra, rb, color):
        """Capsula conica plana: cuadrilatero + circulos en los extremos."""
        ra_i, rb_i = max(1, int(round(ra))), max(1, int(round(rb)))
        d = b - a
        largo = np.hypot(*d)
        if largo > 1e-3:
            n = np.array([-d[1], d[0]]) / largo
            poligono = np.array([
                a + n * ra, b + n * rb, b - n * rb, a - n * ra,
            ], dtype=np.int32)
            cv2.fillConvexPoly(img, poligono, color, lineType=cv2.LINE_AA)
        cv2.circle(img, tuple(np.int32(a)), ra_i, color, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(np.int32(b)), rb_i, color, -1, cv2.LINE_AA)

    def _esfera(self, img, centro, radio, tonos, brillo=True, ovalo=1.0):
        """Esfera sombreada: elipses concentricas corridas hacia la luz.

        Con `ovalo` > 1 la esfera se estira en vertical (cabeza levemente
        ovalada, silueta mas humana que una bola perfecta).
        """
        oscuro, base, claro = tonos
        capas = [(oscuro, 1.00, 0.00), (base, 0.86, 0.12), (claro, 0.58, 0.26)]
        for color, factor, corr in capas:
            c = centro + _LUZ * radio * corr
            ejes = (max(1, int(radio * factor)),
                    max(1, int(radio * factor * ovalo)))
            cv2.ellipse(img, tuple(np.int32(c)), ejes, 0, 0, 360,
                        color, -1, cv2.LINE_AA)
        if brillo:
            c = centro + _LUZ * radio * 0.42
            tenue = tuple(int(v + (255 - v) * 0.55) for v in claro)
            cv2.circle(img, tuple(np.int32(c)), max(1, int(radio * 0.20)),
                       tenue, -1, cv2.LINE_AA)

    # -- Partes del muneco ----------------------------------------------------

    def _torso(self, img, cuerpo):
        """Torso con degradado vertical y esquinas redondeadas, via mascara."""
        hi = self._px(cuerpo[CUERPO_HOMBRO_IZQ])
        hd = self._px(cuerpo[CUERPO_HOMBRO_DER])
        ci = self._px(cuerpo[CUERPO_CADERA_IZQ])
        cd = self._px(cuerpo[CUERPO_CADERA_DER])

        # Silueta de maniqui: las caderas se angostan respecto a lo medido,
        # para que el pecho domine y el torso no sea un bloque parejo.
        medio_caderas = (ci + cd) / 2.0
        ci = medio_caderas + (ci - medio_caderas) * ANGOSTE_CADERAS
        cd = medio_caderas + (cd - medio_caderas) * ANGOSTE_CADERAS

        # Torso de maniqui en TRES piezas (pecho ancho, cintura angosta, bloque
        # de cadera), unidas en una sola mascara que se dilata con un kernel
        # circular: los empalmes quedan con filetes redondeados y la silueta se
        # lee como figura articulada de juguete, no como una losa.
        radio = max(2, int(PROP_RADIO_TORSO * self.S))

        def par(t, ancho):
            """(izq, der) a la altura t (0 = hombros, 1 = caderas)."""
            a = hi + (ci - hi) * t
            b = hd + (cd - hd) * t
            medio = (a + b) / 2.0
            return medio + (a - medio) * ancho, medio + (b - medio) * ancho

        mascara = np.zeros((self.h, self.w), dtype=np.uint8)
        for t0, t1, w0, w1 in PIEZAS_TORSO:
            a0, b0 = par(t0, w0)
            a1, b1 = par(t1, w1)
            pieza = np.array([a0, b0, b1, a1], dtype=np.float64)
            centroide = pieza.mean(axis=0)
            hacia = centroide - pieza
            normas = np.maximum(np.linalg.norm(hacia, axis=1, keepdims=True),
                                1e-6)
            # El encogimiento se acota para que una pieza angosta no colapse.
            paso = np.minimum(radio, normas * 0.6)
            interior = pieza + hacia / normas * paso
            cv2.fillPoly(mascara, [np.int32(interior)], 255,
                         lineType=cv2.LINE_AA)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radio + 1, 2 * radio + 1))
        mascara = cv2.dilate(mascara, kernel)

        # Degradado vertical claro (hombros) -> oscuro (caderas).
        ys, xs = np.where(mascara > 0)
        if len(ys) == 0:
            return
        y0, y1 = ys.min(), ys.max()
        oscuro, base, claro = self.tonos_cuerpo
        arriba = np.array(claro, dtype=np.float64) * 0.35 + np.array(base) * 0.65
        abajo = np.array(oscuro, dtype=np.float64) * 0.75 + np.array(base) * 0.25
        alfa = (ys - y0) / max(1, y1 - y0)
        colores = (1 - alfa[:, None]) * arriba + alfa[:, None] * abajo
        # Borde lateral oscuro sutil (lado contrario a la luz) para dar volumen.
        centro_x = (hi[0] + hd[0]) / 2.0
        semiancho = max(1.0, abs(hd[0] - hi[0]) / 2.0 + radio)
        lateral = np.clip((xs - centro_x) / semiancho, -1, 1)
        sombra = np.clip(lateral * -np.sign(_LUZ[0]), 0, 1) ** 2 * 0.18
        colores *= (1 - sombra[:, None])
        img[ys, xs] = colores.astype(np.uint8)

    def _centro_cabeza(self, cuerpo):
        """Centro de la cabeza: un poco por encima de la nariz."""
        nariz = self._px(cuerpo[CUERPO_NARIZ])
        return nariz + np.array([0.0, -0.10 * self.S])

    def _cuello(self, img, cuerpo):
        """El cuello se dibuja ANTES del torso, para que este tape su base y no
        aparezca un 'medallon' claro sobre el pecho."""
        hi = self._px(cuerpo[CUERPO_HOMBRO_IZQ])
        hd = self._px(cuerpo[CUERPO_HOMBRO_DER])
        centro_hombros = (hi + hd) / 2.0
        self._capsula(img, centro_hombros, self._centro_cabeza(cuerpo),
                      PROP_CUELLO * self.S / 2, PROP_CUELLO * self.S / 2,
                      self.tonos_cuerpo)

    def _cabeza(self, img, cuerpo):
        self._esfera(img, self._centro_cabeza(cuerpo),
                     PROP_RADIO_CABEZA * self.S, self.tonos_cuerpo,
                     ovalo=OVALO_CABEZA)

    def _brazo(self, img, cuerpo, mano, lado):
        """Brazo completo (hombro-codo-muneca) soldado, y su mano si existe."""
        if lado == "izq":
            hombro = self._px(cuerpo[CUERPO_HOMBRO_IZQ])
            codo = self._px(cuerpo[CUERPO_CODO_IZQ])
            muneca_pose = self._px(cuerpo[CUERPO_MUNECA_IZQ])
        else:
            hombro = self._px(cuerpo[CUERPO_HOMBRO_DER])
            codo = self._px(cuerpo[CUERPO_CODO_DER])
            muneca_pose = self._px(cuerpo[CUERPO_MUNECA_DER])

        # Si hay mano, el antebrazo termina en la muneca de Hands (mas precisa):
        # asi el brazo y la mano quedan soldados sin hueco.
        muneca = self._px(mano[0]) if mano is not None else muneca_pose

        # Brazo con taper (grueso en el hombro, fino en la muneca) y un bulto
        # de deltoide en el arranque, para una silueta mas humana.
        r_brazo = PROP_BRAZO * self.S / 2
        r_ante = PROP_ANTEBRAZO * self.S / 2
        self._cadena(img, [
            (hombro, hombro, r_brazo * 1.12, r_brazo * 1.12),  # deltoide
            (hombro, codo, r_brazo, r_ante * 1.02),
            (codo, muneca, r_ante, r_ante * 0.85),
        ], self.tonos_cuerpo)
        # Bolas de articulacion en codo y muneca (estilo maniqui, como las
        # manos en tono claro): hacen legibles los quiebres del brazo.
        self._esfera(img, codo, PROP_BOLA_CODO * self.S, self.tonos_mano,
                     brillo=False)
        self._esfera(img, muneca, PROP_BOLA_MUNECA * self.S, self.tonos_mano,
                     brillo=False)

        if mano is not None:
            self._mano(img, mano)

    def _mano(self, img, mano):
        """Mano estilo maniqui: palma llena + dedos INDIVIDUALES articulados.

        Cada dedo se dibuja completo por su cuenta, con contorno oscuro propio
        y bolitas en sus articulaciones, para que un dedo DOBLADO sobre la
        palma siga leyendose como dedo (con la mano unificada en un solo tono,
        los dedos recogidos desaparecian en la masa). El orden por profundidad
        se conserva: dedos tras la palma primero, luego la palma con sus
        nudillos, luego los dedos delanteros, cada grupo del mas lejano al mas
        cercano.
        """
        puntos = np.array([self._px(p) for p in mano])
        z = mano[:, 2]
        r_dedo = PROP_DEDO * self.S / 2

        z_palma = float(np.mean(z[_PALMA]))
        detras, delante = [], []
        for dedo in _DEDOS:
            z_dedo = float(np.mean(z[dedo]))
            (detras if z_dedo > z_palma + 0.004 else delante).append((z_dedo, dedo))
        detras.sort(key=lambda par: -par[0])
        delante.sort(key=lambda par: -par[0])

        for _, dedo in detras:
            self._dedo(img, puntos, z, dedo, r_dedo)
        self._palma(img, puntos, r_dedo)
        for _, dedo in delante:
            self._dedo(img, puntos, z, dedo, r_dedo)

    def _palma(self, img, puntos, r_dedo):
        """Blob de la palma (casco convexo dilatado) + bolitas de nudillos.

        El casco de muneca y nudillos es flaco cuando la mano esta de canto;
        engordarlo con un margen (relleno + bordes gruesos + circulos en los
        vertices, equivalente a dilatar) le da cuerpo de palma real. Las
        bolitas donde nacen los dedos hacen visibles esas articulaciones.
        """
        indices = np.array(_PALMA)
        for indice, factor, corr in self._CAPAS:
            color = self.tonos_mano[indice]
            margen = r_dedo * MARGEN_PALMA * factor
            casco = cv2.convexHull(np.float32(puntos[indices])).reshape(-1, 2)
            centroide = casco.mean(axis=0)
            pts = (casco - centroide) * factor + centroide \
                + _LUZ * r_dedo * 2 * corr
            cv2.fillConvexPoly(img, np.int32(pts), color, lineType=cv2.LINE_AA)
            n = len(pts)
            for i in range(n):
                a, b = pts[i], pts[(i + 1) % n]
                cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), color,
                         max(1, int(2 * margen)), cv2.LINE_AA)
                cv2.circle(img, tuple(np.int32(a)), max(1, int(margen)),
                           color, -1, cv2.LINE_AA)
        # Nudillos: bolita donde nace cada dedo (base del pulgar incluida).
        for i in (2, 5, 9, 13, 17):
            self._bolita(img, puntos[i], r_dedo * BOLA_NUDILLO * 1.05)

    # Capas de un dedo: contorno oscuro MAS grueso que el de otras piezas,
    # para que un dedo doblado sobre la palma no se funda con ella.
    _CAPAS_DEDO = ((0, 1.14, 0.00), (1, 0.88, 0.14), (2, 0.50, 0.30))

    def _dedo(self, img, puntos, z, cadena, r_base):
        """Un dedo articulado: capsulas conicas + bolitas en las falanges.

        Las bolitas en las articulaciones intermedias y en la punta (los 21
        landmarks de MediaPipe hechos visibles) hacen legible el dedo en
        cualquier postura, doblado o extendido, al estilo maniqui.
        """
        radios = [1.00, 0.92, 0.85, 0.80]  # de la base a la punta

        def rz(k):
            factor = float(np.clip(1.0 - z[cadena[k]] * GANANCIA_Z_DEDOS,
                                   FACTOR_Z_MIN, FACTOR_Z_MAX))
            return r_base * radios[k] * factor

        for indice, factor, corr in self._CAPAS_DEDO:
            color = self.tonos_mano[indice]
            for k in range(len(cadena) - 1):
                ra, rb = rz(k), rz(k + 1)
                self._capsula_solida(
                    img,
                    puntos[cadena[k]] + _LUZ * ra * corr,
                    puntos[cadena[k + 1]] + _LUZ * rb * corr,
                    ra * factor, rb * factor, color,
                )
        # Articulaciones intermedias y punta del dedo.
        self._bolita(img, puntos[cadena[1]], r_base * BOLA_NUDILLO)
        self._bolita(img, puntos[cadena[2]], r_base * BOLA_NUDILLO * 0.9)
        self._bolita(img, puntos[cadena[3]], r_base * BOLA_NUDILLO * 0.75)

    def _bolita(self, img, centro, radio):
        """Bolita de articulacion (mini esfera en el tono de la mano)."""
        oscuro, base, claro = self.tonos_mano
        for color, factor, corr in ((oscuro, 1.00, 0.00),
                                    (base, 0.78, 0.18),
                                    (claro, 0.42, 0.36)):
            c = centro + _LUZ * radio * corr
            cv2.circle(img, tuple(np.int32(c)), max(1, int(radio * factor)),
                       color, -1, cv2.LINE_AA)

    # -- Fotograma completo ---------------------------------------------------

    def dibujar(self, cuerpo, mano_izq, mano_der):
        """Renderiza un fotograma completo y devuelve la imagen BGR."""
        img = np.full((self.h, self.w, 3), COLOR_FONDO, dtype=np.uint8)

        # Orden por profundidad: cuello, torso y cabeza primero; en las senas
        # las manos van casi siempre DELANTE del cuerpo, asi que los brazos se
        # dibujan despues, y entre ellos, primero el mas lejano (z de la muneca
        # de Pose). Nota: esto refina el orden de PLAN_DIRECCION2 seccion 6,
        # que ponia el brazo lejano DETRAS del torso y ocultaba una mano al
        # firmar frente al pecho.
        self._cuello(img, cuerpo)
        self._torso(img, cuerpo)
        self._cabeza(img, cuerpo)

        z_izq = cuerpo[CUERPO_MUNECA_IZQ][2]
        z_der = cuerpo[CUERPO_MUNECA_DER][2]
        brazos = [("izq", mano_izq, z_izq), ("der", mano_der, z_der)]
        brazos.sort(key=lambda b: -b[2])  # mayor z (mas lejos) primero
        for lado, mano, _ in brazos:
            self._brazo(img, cuerpo, mano, lado)

        return img


# ---------------------------------------------------------------------------
# Reproduccion y exportacion
# ---------------------------------------------------------------------------

VELOCIDADES = [1.0, 0.75, 0.5, 0.25]


def _hud(img, clip: ClipPreparado, indice, velocidad, vista_espejo, pausado):
    """Panel inferior con la palabra y el estado de la reproduccion."""
    alto_panel = 74
    panel = img[img.shape[0] - alto_panel:, :]
    panel[:] = (panel * 0.25 + np.array(dibujo.FONDO_PANEL) * 0.75).astype(np.uint8)
    y0 = img.shape[0] - alto_panel
    dibujo.texto(img, clip.palabra, 18, y0 + 34, 0.95, dibujo.BLANCO, 2,
                 dibujo.FUENTE_TITULO)
    estado = "PAUSA" if pausado else f"{velocidad:g}x"
    vista_txt = "vista: espejo" if vista_espejo else "vista: de frente"
    info = (f"fotograma {int(indice) + 1}/{clip.num_frames}   {estado}   "
            f"{vista_txt}   fps clip: {clip.fps:g}")
    dibujo.texto(img, info, 18, y0 + 62, 0.52, dibujo.GRIS_CLARO)
    perdidos = [f"{lado}: {n}" for lado, n in clip.frames_sin_mano.items() if n]
    if perdidos:
        dibujo.texto(img, "sin mano " + ", ".join(perdidos),
                     img.shape[1] - 250, y0 + 34, 0.5, dibujo.AMBAR)


def _listar_clips(rutas):
    """Expande archivos y carpetas a la lista ordenada de clips JSON."""
    encontrados = []
    for ruta in rutas:
        ruta = Path(ruta)
        if ruta.is_dir():
            encontrados.extend(sorted(ruta.glob("*.json")))
        elif ruta.exists():
            encontrados.append(ruta)
        else:
            print(f"Aviso: no existe {ruta}, se omite.")
    return encontrados


def exportar(clips, carpeta, cada, paleta, vista_espejo, ancho, alto):
    """Renderiza fotogramas a PNG sin abrir ventana (control de calidad)."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    for ruta in clips:
        clip = ClipPreparado(cargar_clip(ruta))
        muneco = MunecoCapsulas(ancho, alto, paleta)
        muneco.preparar_marco(clip, vista_espejo)
        for i in range(0, clip.num_frames, cada):
            cuerpo, mi, md = clip.fotograma(float(i))
            img = muneco.dibujar(cuerpo, mi, md)
            nombre = f"{ruta.stem}_f{i:03d}.png"
            cv2.imwrite(str(carpeta / nombre), img)
        print(f"{ruta.stem}: {len(range(0, clip.num_frames, cada))} PNG en {carpeta}")


def reproducir(rutas_clips, paleta, ancho, alto, vista_espejo=False):
    """Bucle interactivo del visor."""
    indice_clip = 0
    vel_idx = 0
    pausado = False

    clip = ClipPreparado(cargar_clip(rutas_clips[indice_clip]))
    muneco = MunecoCapsulas(ancho, alto, paleta)
    muneco.preparar_marco(clip, vista_espejo)
    indice = 0.0

    nombre_ventana = "Visor de clips LESHO"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_AUTOSIZE)

    while True:
        cuerpo, mi, md = clip.fotograma(indice)
        img = muneco.dibujar(cuerpo, mi, md)
        _hud(img, clip, indice, VELOCIDADES[vel_idx], vista_espejo, pausado)
        cv2.imshow(nombre_ventana, img)

        espera_ms = max(1, int(1000.0 / clip.fps))
        tecla = cv2.waitKey(espera_ms) & 0xFF

        if not pausado:
            indice += VELOCIDADES[vel_idx]
            if indice >= clip.num_frames:
                indice = 0.0  # bucle

        cambiar = 0
        if tecla in (ord("q"), 27):
            break
        elif tecla == ord(" "):
            pausado = not pausado
        elif tecla == ord("m"):
            vista_espejo = not vista_espejo
            muneco.preparar_marco(clip, vista_espejo)
        elif tecla == ord("v"):
            vel_idx = (vel_idx + 1) % len(VELOCIDADES)
        elif tecla == ord("d"):
            indice = min(clip.num_frames - 1, np.floor(indice) + 1)
            pausado = True
        elif tecla == ord("a"):
            indice = max(0.0, np.floor(indice) - 1)
            pausado = True
        elif tecla == ord("g"):
            salida = Path(f"{clip.palabra}_f{int(indice):03d}.png")
            cv2.imwrite(str(salida), img)
            print(f"Guardado {salida}")
        elif tecla == ord("n"):
            cambiar = 1
        elif tecla == ord("p"):
            cambiar = -1

        if cambiar:
            indice_clip = (indice_clip + cambiar) % len(rutas_clips)
            clip = ClipPreparado(cargar_clip(rutas_clips[indice_clip]))
            muneco.preparar_marco(clip, vista_espejo)
            indice = 0.0
            pausado = False

        try:
            if cv2.getWindowProperty(nombre_ventana, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Visor de clips de senas (muneco de capsulas)."
    )
    parser.add_argument(
        "rutas", nargs="*",
        help="Clips JSON o carpetas con clips. Por defecto training/clips/piloto.",
    )
    parser.add_argument("--paleta", choices=sorted(PALETAS), default="azul")
    parser.add_argument("--espejo", action="store_true",
                        help="Arrancar en vista espejo (por defecto: de frente).")
    parser.add_argument("--ancho", type=int, default=720)
    parser.add_argument("--alto", type=int, default=840)
    parser.add_argument("--exportar", metavar="CARPETA",
                        help="Exportar fotogramas a PNG en vez de abrir ventana.")
    parser.add_argument("--cada", type=int, default=1,
                        help="Al exportar, guardar 1 de cada N fotogramas.")
    args = parser.parse_args()

    rutas = args.rutas or [config.RAIZ_TRAINING / "clips" / "piloto"]
    clips = _listar_clips(rutas)
    if not clips:
        print("No se encontro ningun clip. Grabe primero con "
              "capture/captura_diccionario.py")
        sys.exit(1)

    if args.exportar:
        exportar(clips, args.exportar, max(1, args.cada), args.paleta,
                 args.espejo, args.ancho, args.alto)
    else:
        reproducir(clips, args.paleta, args.ancho, args.alto,
                   vista_espejo=args.espejo)


if __name__ == "__main__":
    main()
