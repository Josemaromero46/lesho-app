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
        E entra/sale del MODO EDICION

Modo edicion (para arreglar temblor o dedos perdidos SIN colocarlos a mano):
  A/D moverse frame a frame, I marca el inicio del tramo malo, O marca el fin,
  F lo arregla (regenera ese tramo desde los frames buenos de al lado, para las
  dos manos), Z deshace, S guarda el clip (respaldo del original en .bak),
  C limpia la marca, E vuelve a reproducir.
"""

import argparse
import copy
import shutil
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
    guardar_clip,
)
from comun.suavizado import suavizar_secuencia  # noqa: E402
from capture import dibujo  # noqa: E402

# ---------------------------------------------------------------------------
# Parametros del muneco (proporciones en fracciones del ANCHO DE HOMBROS)
# ---------------------------------------------------------------------------

# Estilo PERSONAJE ILUSTRADO (no maniqui monocromo): piel calida + camiseta de
# color + contorno oscuro limpio, para que se lea a simple vista, pensado para
# ninos. El MunecoPainter de Flutter copia estos valores.

# Grosores y radios (fracciones del ANCHO DE HOMBROS).
PROP_BRAZO = 0.32          # grosor del brazo en el hombro (manga)
PROP_ANTEBRAZO = 0.24      # grosor del antebrazo (piel)
PROP_CUELLO = 0.32         # grosor del cuello
# Grosor del dedo COMO FRACCION DEL ANCHO DE NUDILLOS (distancia 5..17 de la
# mano), no del ancho de hombros: asi los dedos se proporcionan al tamano real
# de la mano en pantalla y NO se amontonan cuando la mano se ve chica. El
# radio del dedo es la mitad de este valor por el ancho de nudillos. Se bajo de
# 0.30 a 0.24 para que los dedos JUNTOS (letra B) queden con un hilo de
# separacion y no se pisen los contornos (que hacia ver un dedo "hundido").
PROP_DEDO_MANO = 0.24
PROP_RADIO_CABEZA = 0.46   # radio de la cabeza (horizontal)
OVALO_CABEZA = 1.14        # la cabeza es ovalada (mas alta que ancha)
ANGOSTE_CADERAS = 0.80     # las caderas se dibujan mas angostas que lo medido

# Grosor del CONTORNO oscuro (linea que rodea cada pieza), en anchos de hombro.
# Es la clave del estilo: separa dedos, brazos y torso de un vistazo.
GROSOR_CONTORNO = 0.028

# Perfil de la silueta del torso: (t, ancho relativo al medio-hombro). t va de
# la linea de hombros (0) a la de caderas (1). Cintura marcada = mas humano.
PERFIL_TORSO = [
    (0.00, 1.00), (0.30, 0.95), (0.58, 0.76), (0.82, 0.84), (1.00, 0.90),
]

# Cuanto del ancho del lienzo ocupa el ancho de hombros del muneco.
FRACCION_HOMBROS_CANVAS = 0.325

# Posicion vertical del centro de hombros en el lienzo (fraccion de la altura).
ALTURA_HOMBROS_CANVAS = 0.46

# Direccion de la luz (fija, arriba a la izquierda), normalizada.
_LUZ = np.array([-0.45, -0.89])

# Huecos de mano de hasta este tiempo se rellenan solos (interpolando entre el
# ultimo frame con mano y el primero que reaparece); mas largos, la mano no se
# dibuja en ese tramo (o se arregla a mano con el editor). Subido de 0.35 a 0.60
# para cubrir mejor perdidas cortas (por ejemplo al pasar la mano por la cara).
MAX_HUECO_MANO_S = 0.60

# Modulacion de grosor por profundidad (pseudo-3D): un dedo mas cerca de la
# camara se dibuja mas grueso. factor = 1 - z * GANANCIA_Z, acotado.
GANANCIA_Z_DEDOS = 5.0
FACTOR_Z_MIN, FACTOR_Z_MAX = 0.72, 1.30

# Margen de profundidad para dibujar un TRAMO de dedo DETRAS de la palma. Por
# defecto los tramos van ENCIMA de la palma (visibles); un tramo se pone detras
# solo si esta CLARAMENTE mas lejos que la palma por mas de este margen (por
# ejemplo la punta enrollada de un dedo, o el pulgar en vista de dorso). El
# margen evita el parpadeo de los tramos que quedan al ras de la palma.
MARGEN_DORSO_DEDO = 0.012

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

# Paletas del muneco (BGR). Piel calida + camiseta de color con buen contraste,
# para que el brazo se distinga del torso y la mano de la piel del brazo.
PALETAS = {
    "humano": {
        "piel": (128, 176, 226),     # tan calido
        "camisa": (110, 150, 86),    # verde bosque (paleta de la app)
        "pelo": (46, 56, 78),        # castano oscuro
    },
    "humano_terracota": {
        "piel": (128, 176, 226),
        "camisa": (52, 96, 205),     # terracota (paleta de la app)
        "pelo": (46, 56, 78),
    },
}
# Contorno (casi negro calido, mas suave que el negro puro para ninos) y uña.
COL_CONTORNO = (40, 44, 58)
COL_UNA = (206, 228, 248)
COL_OJO = (52, 46, 42)          # casi negro calido, para los ojos
COL_BRILLO = (245, 245, 245)    # brillo del ojo
COLOR_FONDO = (238, 248, 255)   # crema, el fondo de la app


def _tonos(base_bgr):
    """Deriva (oscuro, base, claro) de un color base, para el sombreado."""
    base = np.array(base_bgr, dtype=np.float64)
    oscuro = base * 0.68
    claro = base + (255.0 - base) * 0.40
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
        # Profundidad de oclusion por mano (z del MUNDO si el clip la trae, que es
        # fiable; si no, cae a la z de imagen, como antes). 21 valores por mano.
        prof = {"izq": np.full((T, 21), np.nan), "der": np.full((T, 21), np.nan)}

        for t, frame in enumerate(frames):
            if frame["cuerpo"] is not None:
                cuerpo[t] = [p[:3] for p in frame["cuerpo"]]
                cuerpo_ok[t] = True
            for lado, clave, pclave in (("izq", "mano_izq", "prof_izq"),
                                        ("der", "mano_der", "prof_der")):
                if frame[clave] is not None:
                    manos[lado][t] = frame[clave]
                    mano_ok[lado][t] = True
                    if frame.get(pclave) is not None:
                        prof[lado][t] = frame[pclave]
                    else:
                        prof[lado][t] = [p[2] for p in frame[clave]]

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
        self.prof = {}
        for lado in ("izq", "der"):
            plano = manos[lado].reshape(T, -1)
            plano, ok = _interpolar_huecos(plano, mano_ok[lado], max_hueco, False)
            plano = _suavizar_tramos(plano, ok, self.fps)
            self.manos[lado] = plano.reshape(T, 21, 3)
            self.mano_ok[lado] = ok
            # Profundidad: mismos huecos y suavizado que la mano.
            pplano, pok = _interpolar_huecos(
                prof[lado], mano_ok[lado], max_hueco, False)
            self.prof[lado] = _suavizar_tramos(pplano, pok, self.fps)

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
                prof = (1 - alfa) * self.prof[lado][i0] + alfa * self.prof[lado][i1]
            elif ok0:
                mano, prof = self.manos[lado][i0], self.prof[lado][i0]
            elif ok1:
                mano, prof = self.manos[lado][i1], self.prof[lado][i1]
            else:
                mano = None
            # Se adjunta la profundidad de oclusion como 4ta columna (21, 4).
            salida.append(None if mano is None
                          else np.concatenate([mano, prof[:, None]], axis=1))
        return salida


# ---------------------------------------------------------------------------
# Renderizador: muneco de capsulas
# ---------------------------------------------------------------------------

class MunecoCapsulas:
    """Dibuja un fotograma del clip como personaje ilustrado.

    Estilo (pedido del usuario, orientado a ninos): piel calida, camiseta de
    color y contorno oscuro limpio, con manos muy definidas (dedos contorneados
    y uñas). El nombre se conserva por compatibilidad con quien lo importa.
    """

    def __init__(self, ancho=720, alto=900, paleta="humano"):
        self.w = ancho
        self.h = alto
        colores = PALETAS[paleta]
        self.tonos_piel = _tonos(colores["piel"])
        self.tonos_camisa = _tonos(colores["camisa"])
        self.col_pelo = colores["pelo"]
        self._g = 3.0  # grosor de contorno en px (se fija en preparar_marco)

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
        self._g = max(2.0, GROSOR_CONTORNO * self.S)  # grosor del contorno

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

    # -- Primitivas con contorno + sombreado ---------------------------------

    # Capas del sombreado sobre el relleno: (indice de tono, factor, corr luz).
    # La definicion la da el CONTORNO; el sombreado es suave (estilo ilustrado).
    _CAPAS = ((1, 1.00, 0.00), (2, 0.66, 0.24))  # base, luego brillo

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

    def _cadena(self, img, segmentos, tonos, contorno=True, grosor=None):
        """Cadena de capsulas con CONTORNO oscuro y sombreado suave.

        `segmentos` es una lista de (a, b, ra, rb) en pixeles. Primero se dibuja
        todo el contorno (capsulas engordadas en color oscuro), luego el relleno
        capa por capa. Como cada pieza (dedo, brazo) lleva su propio contorno y
        se dibuja en orden de profundidad, las piezas quedan separadas por una
        linea oscura: un dedo doblado no se funde con el de al lado.

        `grosor` permite un contorno mas fino que el del cuerpo (self._g): para
        los DEDOS, que son piezas chicas y cercanas, un contorno grueso funde
        los de dedos vecinos y parece que las bases convergen.
        """
        if contorno:
            g = self._g if grosor is None else grosor
            for a, b, ra, rb in segmentos:
                self._capsula_solida(img, a, b, ra + g, rb + g, COL_CONTORNO)
        for indice, factor, corr in self._CAPAS:
            color = tonos[indice]
            for a, b, ra, rb in segmentos:
                self._capsula_solida(img, a + _LUZ * ra * corr,
                                     b + _LUZ * rb * corr,
                                     ra * factor, rb * factor, color)

    def _elipse_contorneada(self, img, centro, rx, ry, tonos, ang=0.0):
        """Elipse con contorno oscuro y sombreado suave (cabeza, articulacion)."""
        c = tuple(np.int32(centro))
        g = self._g
        cv2.ellipse(img, c, (int(rx + g), int(ry + g)), ang, 0, 360,
                    COL_CONTORNO, -1, cv2.LINE_AA)
        for indice, factor, corr in self._CAPAS:
            cc = centro + _LUZ * ry * corr
            cv2.ellipse(img, tuple(np.int32(cc)),
                        (max(1, int(rx * factor)), max(1, int(ry * factor))),
                        ang, 0, 360, tonos[indice], -1, cv2.LINE_AA)

    # -- Partes del muneco ----------------------------------------------------

    def _perfil_torso(self, cuerpo):
        """Devuelve (linea de puntos del contorno del torso, hombros, caderas)."""
        hi = self._px(cuerpo[CUERPO_HOMBRO_IZQ])
        hd = self._px(cuerpo[CUERPO_HOMBRO_DER])
        ci = self._px(cuerpo[CUERPO_CADERA_IZQ])
        cd = self._px(cuerpo[CUERPO_CADERA_DER])
        medio_cad = (ci + cd) / 2.0
        ci = medio_cad + (ci - medio_cad) * ANGOSTE_CADERAS
        cd = medio_cad + (cd - medio_cad) * ANGOSTE_CADERAS

        medio_sup = (hi + hd) / 2.0
        eje = medio_cad - medio_sup
        semi_hombro = np.linalg.norm(hd - hi) / 2.0
        u = np.array([eje[1], -eje[0]])
        nu = np.linalg.norm(u)
        u = u / nu if nu > 1e-6 else np.array([1.0, 0.0])

        izq, der = [], []
        for t, ancho in PERFIL_TORSO:
            centro = medio_sup + eje * t
            medio = semi_hombro * ancho
            izq.append(centro + u * medio)
            der.append(centro - u * medio)
        contorno = np.array(izq + der[::-1], dtype=np.float64)
        return contorno, medio_sup, medio_cad

    def _torso(self, img, cuerpo):
        """Torso (camiseta) como silueta suave con cintura, contorno y degradado."""
        contorno, medio_sup, medio_cad = self._perfil_torso(cuerpo)
        centroide = contorno.mean(axis=0)

        # Contorno oscuro: la silueta empujada hacia afuera.
        hacia = contorno - centroide
        normas = np.maximum(np.linalg.norm(hacia, axis=1, keepdims=True), 1e-6)
        externo = contorno + hacia / normas * self._g
        cv2.fillPoly(img, [np.int32(externo)], COL_CONTORNO, lineType=cv2.LINE_AA)

        # Relleno con degradado vertical (camiseta): claro arriba, oscuro abajo.
        mascara = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.fillPoly(mascara, [np.int32(contorno)], 255, lineType=cv2.LINE_AA)
        ys, xs = np.where(mascara > 0)
        if len(ys) == 0:
            return
        oscuro, base, claro = self.tonos_camisa
        y0, y1 = ys.min(), ys.max()
        arriba = np.array(base, np.float64) * 0.55 + np.array(claro) * 0.45
        abajo = np.array(base, np.float64) * 0.7 + np.array(oscuro) * 0.3
        alfa = (ys - y0) / max(1, y1 - y0)
        colores = (1 - alfa[:, None]) * arriba + alfa[:, None] * abajo
        # Sombra lateral suave del lado contrario a la luz.
        semiancho = max(1.0, (xs.max() - xs.min()) / 2.0)
        lateral = np.clip((xs - centroide[0]) / semiancho, -1, 1)
        sombra = np.clip(lateral * -np.sign(_LUZ[0]), 0, 1) ** 2 * 0.16
        colores *= (1 - sombra[:, None])
        img[ys, xs] = colores.astype(np.uint8)

    def _centro_cabeza(self, cuerpo):
        """Centro de la cabeza: un poco por encima de la nariz."""
        nariz = self._px(cuerpo[CUERPO_NARIZ])
        return nariz + np.array([0.0, -0.10 * self.S])

    def _cuello(self, img, cuerpo):
        """Cuello (piel). Se dibuja ANTES del torso, que le tapa la base."""
        hi = self._px(cuerpo[CUERPO_HOMBRO_IZQ])
        hd = self._px(cuerpo[CUERPO_HOMBRO_DER])
        centro_hombros = (hi + hd) / 2.0
        r = PROP_CUELLO * self.S / 2
        self._cadena(img, [(centro_hombros, self._centro_cabeza(cuerpo), r, r)],
                     self.tonos_piel)

    def _cabeza(self, img, cuerpo):
        """Cabeza ovalada (piel) con casquete de pelo y rostro amigable.

        El rostro es estatico y frontal (el muneco siempre se ve de frente).
        Sirve para que las senas que se hacen EN LA CARA (cerca de los ojos, la
        boca, la frente) se lean respecto a esos rasgos, y para que la figura
        sea calida para ninos. La mano, que se dibuja despues, tapa el rostro
        cuando la sena pasa por delante de la cara.
        """
        c = self._centro_cabeza(cuerpo)
        rx = PROP_RADIO_CABEZA * self.S
        ry = rx * OVALO_CABEZA
        self._elipse_contorneada(img, c, rx, ry, self.tonos_piel)
        self._pelo(img, c, rx, ry)
        self._rostro(img, c, rx, ry)

    def _rostro(self, img, c, rx, ry):
        """Ojos, cejas y sonrisa simples sobre la mitad baja de la cara."""
        ojo_dx = 0.36 * rx
        ojo_y = c[1] + 0.06 * ry
        r_ojo = 0.135 * rx
        for signo in (-1, 1):
            oc = np.array([c[0] + signo * ojo_dx, ojo_y])
            # Ojo: ovalo oscuro con brillo; ceja: arco corto por encima.
            cv2.ellipse(img, tuple(np.int32(oc)),
                        (max(1, int(r_ojo * 0.82)), max(1, int(r_ojo))),
                        0, 0, 360, COL_OJO, -1, cv2.LINE_AA)
            brillo = oc + np.array([-r_ojo * 0.28, -r_ojo * 0.40])
            cv2.circle(img, tuple(np.int32(brillo)),
                       max(1, int(r_ojo * 0.30)), COL_BRILLO, -1, cv2.LINE_AA)
            ceja = oc + np.array([0.0, -r_ojo * 1.7])
            cv2.ellipse(img, tuple(np.int32(ceja)),
                        (int(r_ojo * 1.05), int(r_ojo * 0.7)),
                        0, 200, 340, COL_CONTORNO,
                        max(2, int(self._g * 0.8)), cv2.LINE_AA)
        # Sonrisa: arco abierto hacia arriba en la mitad baja de la cara.
        boca = np.array([c[0], c[1] + 0.40 * ry])
        cv2.ellipse(img, tuple(np.int32(boca)),
                    (int(0.26 * rx), int(0.20 * ry)), 0, 22, 158,
                    COL_CONTORNO, max(2, int(self._g * 0.9)), cv2.LINE_AA)

    def _pelo(self, img, c, rx, ry):
        """Casquete de pelo sobre la parte de arriba de la cabeza.

        Se dibuja como un recorte de la elipse de la cabeza por encima de una
        linea de cabello (curva suave). Da lectura de 'persona' sin necesidad
        de rostro (que esta fuera del alcance).
        """
        # Linea de cabello: arco que baja a los lados (mas pelo en las sienes).
        pasos = 40
        arriba = []
        for i in range(pasos + 1):
            ang = np.pi + np.pi * i / pasos     # de 180 a 360 grados (borde sup)
            arriba.append((c[0] + rx * np.cos(ang), c[1] + ry * np.sin(ang)))
        linea = []
        for i in range(pasos + 1):
            x = c[0] - rx + 2 * rx * i / pasos
            # La linea del flequillo: mas abajo a los lados, mas arriba al centro.
            frac = (x - c[0]) / rx
            y = c[1] - ry * 0.22 - ry * 0.34 * (1 - frac * frac)
            linea.append((x, y))
        poly = np.array(arriba + linea[::-1], dtype=np.float64)
        # Recorta a la elipse de la cabeza para que el pelo no sobresalga.
        mascara = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.ellipse(mascara, tuple(np.int32(c)), (int(rx), int(ry)), 0, 0, 360,
                    255, -1, cv2.LINE_AA)
        pelo = np.zeros_like(mascara)
        cv2.fillPoly(pelo, [np.int32(poly)], 255, lineType=cv2.LINE_AA)
        pelo = cv2.bitwise_and(pelo, mascara)
        oscuro = tuple(int(v * 0.72) for v in self.col_pelo)
        img[pelo > 0] = self.col_pelo
        # Sombreado suave del pelo (mitad inferior del casquete mas oscura).
        ys, xs = np.where(pelo > 0)
        if len(ys):
            alfa = np.clip((ys - ys.min()) / max(1, np.ptp(ys)), 0, 1)
            base = np.array(self.col_pelo, np.float64)
            osc = np.array(oscuro, np.float64)
            img[ys, xs] = ((1 - alfa[:, None]) * base
                           + alfa[:, None] * osc).astype(np.uint8)

    def _brazo(self, img, cuerpo, mano, lado):
        """Brazo: manga (camiseta) del hombro al codo, antebrazo de piel, mano."""
        if lado == "izq":
            hombro = self._px(cuerpo[CUERPO_HOMBRO_IZQ])
            codo = self._px(cuerpo[CUERPO_CODO_IZQ])
            muneca_pose = self._px(cuerpo[CUERPO_MUNECA_IZQ])
        else:
            hombro = self._px(cuerpo[CUERPO_HOMBRO_DER])
            codo = self._px(cuerpo[CUERPO_CODO_DER])
            muneca_pose = self._px(cuerpo[CUERPO_MUNECA_DER])
        muneca = self._px(mano[0]) if mano is not None else muneca_pose

        r_brazo = PROP_BRAZO * self.S / 2
        r_ante = PROP_ANTEBRAZO * self.S / 2

        # Manga: hombro (con deltoide) a codo. Antebrazo: codo a muneca (piel).
        # Se dibujan por separado; el contorno del antebrazo sobre la manga en
        # el codo hace de dobladillo de la manga corta.
        self._cadena(img, [
            (hombro, hombro, r_brazo * 1.10, r_brazo * 1.10),  # deltoide
            (hombro, codo, r_brazo, r_ante * 1.06),
        ], self.tonos_camisa)
        self._cadena(img, [(codo, muneca, r_ante * 1.02, r_ante * 0.9)],
                     self.tonos_piel)

        # `invertir` (lado "der") corrige la lateralidad de la uña. Como la mano
        # ya se asocio al brazo por cercania, el lado del brazo corresponde a la
        # mano fisica correcta, asi que este signo es estable y confiable.
        if mano is not None:
            self._mano(img, mano, invertir=(lado == "der"))

    # -- Mano ilustrada: dedos contorneados con uñas --------------------------

    def _facing_una(self, mano, cadena, invertir):
        """Cuanto se ve la UÑA del dedo (en [-1, 1]): >0 uña, <0 yema.

        Combina dos señales ESTABLES (sin productos cruz degenerados, que eran
        la causa del temblor y del parpadeo cuando la mano se ve de canto):

        1. PALMA: la normal del plano de la palma (muneca, base del indice, del
           menique) dice si se ve el dorso o la palma. Es la señal confiable
           para los dedos ESTIRADOS. `invertir` viene del brazo (ya asociado por
           cercania), asi que la lateralidad es correcta sin recalcularla.
        2. ENROLLADO: aunque la palma mire a la camara, si el dedo se enrolla y
           su punta va HACIA la camara, se ve la uña. Se detecta con el angulo
           de doblez del dedo y con la punta acercandose (mas cerca que el
           nudillo). No usa la normal de la palma, asi que sirve justo cuando
           esa normal no cambia (palma quieta, dedos enrollando).

        Se toma el maximo de las dos: la uña aparece si el dorso mira a la
        camara O si el dedo enrollado apunta a la camara.
        """
        w = mano[0]
        n = np.cross(mano[5] - w, mano[17] - w)
        norma = np.linalg.norm(n)
        nz = (n[2] / norma) if norma > 1e-9 else 0.0
        if invertir:
            nz = -nz
        cara_palma = nz if self._vista_espejo else -nz

        mcp, pip, dip, tip = (mano[i] for i in cadena)
        v_prox = pip - mcp
        v_dist = tip - dip
        n1 = np.linalg.norm(v_prox)
        n2 = np.linalg.norm(v_dist)
        cara_enrollado = 0.0
        if n1 > 1e-9 and n2 > 1e-9:
            coseno = float(np.clip(np.dot(v_prox, v_dist) / (n1 * n2), -1, 1))
            doblez = np.arccos(coseno)                  # 0 recto .. pi enrollado
            enrollado = float(np.clip((doblez - 1.05) / 0.9, 0.0, 1.0))  # ~60-110
            escala = float(np.linalg.norm(mano[9] - w)) + 1e-9
            hacia = float(np.clip((mcp[2] - tip[2]) / escala * 2.0, 0.0, 1.0))
            cara_enrollado = enrollado * hacia
        return max(cara_palma, cara_enrollado)

    def _mano(self, img, mano, invertir):
        """Mano definida: palma con contorno + dedos contorneados con uñas.

        Cada dedo lleva su propio contorno oscuro y una uña clara en la punta
        que aparece segun se vea su dorso (palma girada o dedo enrollado hacia la
        camara). El orden por profundidad se conserva (dedos detras de la palma,
        palma, dedos delante), del mas lejano al mas cercano.
        """
        # La mano llega como (21, 4): las 3 primeras columnas son la posicion en
        # espacio fisico y la 4ta es la PROFUNDIDAD de oclusion (z del mundo si el
        # clip la trae, fiable; si no, la z de imagen). Se separan: la posicion y
        # la z de imagen (para el grosor por perspectiva) usan las 3 primeras; la
        # oclusion usa la 4ta.
        prof = mano[:, 3] if mano.shape[1] > 3 else mano[:, 2]
        mano = mano[:, :3]
        puntos = np.array([self._px(p) for p in mano])
        z = mano[:, 2]
        # Grosor del dedo proporcional al TAMANO REAL de la mano en pantalla
        # (ancho de nudillos 5..17, estable con los dedos abiertos o juntos), no
        # al ancho de hombros: asi los dedos no se amontonan cuando la mano se ve
        # chica. Piso relativo a S para que nunca desaparezcan.
        ancho_nudillos = float(np.linalg.norm(puntos[5] - puntos[17]))
        r_dedo = max(PROP_DEDO_MANO * ancho_nudillos / 2, 0.035 * self.S)
        # Contorno de los dedos MAS FINO que el del cuerpo: proporcional al
        # grosor del dedo, para que dos dedos vecinos no fundan sus contornos y
        # parezca que las bases convergen. Nunca mas grueso que el del cuerpo.
        g_dedo = min(self._g, 0.42 * r_dedo)

        # Profundidad POR TRAMO del dedo (no por dedo entero): un dedo enrollado
        # tiene el nudillo por delante (se ve) y la punta doblada hacia atras
        # (se oculta tras la mano). Cada tramo se manda ATRAS solo si esta
        # claramente mas lejos que la palma (mas alla de MARGEN_DORSO_DEDO); por
        # defecto va ADELANTE (visible). Los tramos contiguos de la misma clase
        # se unen en una cadena para que el dedo siga siendo liso; la costura
        # entre el tramo de adelante y el de atras cae en el borde de la palma,
        # que la tapa.
        # Profundidad de la palma y margen ADAPTATIVO al rango de profundidad de
        # la mano (sirve igual con z del mundo en metros que con z de imagen): un
        # tramo va detras solo si esta claramente mas lejos que la palma.
        prof_palma = float(np.mean(prof[_PALMA]))
        # Margen chico: una falange de un dedo DOBLADO se oculta apenas queda
        # detras de la palma. No afecta a los dedos ESTIRADOS (la B), que van
        # siempre adelante por la compuerta de "doblado".
        margen_prof = max(1e-6, 0.10 * float(np.max(prof) - np.min(prof)))
        radios = [1.00, 0.94, 0.88, 0.82]

        # Cuanto se ve el DORSO de la mano (normal de la palma; dato ESTABLE, a
        # diferencia de la z de las puntas). Sirve para ocultar geometricamente
        # las falanges enrolladas, sin depender de la z ruidosa de la punta.
        w = mano[0]
        n = np.cross(mano[5] - w, mano[17] - w)
        nn = np.linalg.norm(n)
        nz = (n[2] / nn) if nn > 1e-9 else 0.0
        if invertir:
            nz = -nz
        dorso_cam = nz if self._vista_espejo else -nz

        def cadenas_dedo(dedo):
            # El pulgar es mas grueso que los demas dedos.
            r_base = r_dedo * (1.35 if dedo is _DEDOS[0] else 1.0)

            def rz(k):
                factor = float(np.clip(1.0 - z[dedo[k]] * GANANCIA_Z_DEDOS,
                                       FACTOR_Z_MIN, FACTOR_Z_MAX))
                return r_base * radios[k] * factor

            # SOLO un dedo DOBLADO puede ocultar tramos tras la mano. Un dedo
            # ESTIRADO (como los de la letra B) va SIEMPRE adelante, aunque su
            # punta quede un pelo mas lejos: asi no se "hunde" en la palma. Se
            # mide el doblez con el angulo entre la falange proximal y la distal.
            v_prox = mano[dedo[1]] - mano[dedo[0]]
            v_dist = mano[dedo[3]] - mano[dedo[2]]
            np1, np2 = np.linalg.norm(v_prox), np.linalg.norm(v_dist)
            doblez = 0.0
            if np1 > 1e-9 and np2 > 1e-9:
                doblez = np.arccos(
                    np.clip(np.dot(v_prox, v_dist) / (np1 * np2), -1.0, 1.0))
            # ~86 grados: un dedo ESTIRADO (aunque venga algo curvado, como el
            # del medio en la B) no llega; solo un dedo ENROLLADO de verdad (Y,
            # puno) lo supera. Umbral alto a proposito para no "doblar" de mas.
            doblado = doblez > 1.5
            dorso = dorso_cam > 0.05

            clases = []
            for k in range(3):
                if k == 0 or not doblado:
                    # El nudillo, y todo dedo estirado, van adelante (visibles).
                    clases.append(False)
                    continue
                # Falange (de PIP en adelante) de un dedo DOBLADO: se oculta si la
                # mano muestra el dorso, o si esa falange esta claramente mas
                # lejos que la palma (profundidad del mundo, fiable).
                prof_seg = (prof[dedo[k]] + prof[dedo[k + 1]]) / 2
                clases.append(dorso or prof_seg > prof_palma + margen_prof)
            j = 0
            while j < 3:
                clase = clases[j]
                ini = j
                while j < 3 and clases[j] == clase:
                    j += 1
                segs = [(puntos[dedo[k]], puntos[dedo[k + 1]], rz(k), rz(k + 1))
                        for k in range(ini, j)]
                prof_media = float(np.mean(
                    [(prof[dedo[k]] + prof[dedo[k + 1]]) / 2 for k in range(ini, j)]))
                tiene_punta = (j == 3)  # el ultimo tramo llega a la punta
                yield clase, prof_media, segs, tiene_punta, rz(3)

        atras, adelante = [], []
        for dedo in _DEDOS:
            for clase, z_media, segs, tiene_punta, r_punta in cadenas_dedo(dedo):
                item = (z_media, segs, dedo, tiene_punta, r_punta)
                (atras if clase else adelante).append(item)
        atras.sort(key=lambda t: -t[0])
        adelante.sort(key=lambda t: -t[0])

        for _, segs, _dedo, _tp, _rp in atras:
            self._cadena(img, segs, self.tonos_piel, grosor=g_dedo)
        self._palma(img, puntos, r_dedo)
        for _, segs, dedo, tiene_punta, r_punta in adelante:
            self._cadena(img, segs, self.tonos_piel, grosor=g_dedo)
            if tiene_punta:
                una_vis = float(np.clip(
                    self._facing_una(mano, dedo, invertir) * 1.7 + 0.15, 0, 1))
                self._una(img, puntos[dedo[3]], puntos[dedo[2]], r_punta,
                          una_vis)

    def _palma(self, img, puntos, r_dedo):
        """Palma ANCHA con contorno y sombreado.

        No se usa el casco simple de muneca + nudillos, porque la muneca es UN
        solo punto y da una palma TRIANGULAR que se angosta hacia abajo: los
        dedos parecian nacer de un vertice comun. En una mano real la palma es
        casi tan ancha abajo (el talon) como arriba (los nudillos). Por eso la
        base se ensancha: se llevan dos esquinas al nivel de la muneca con el
        ancho de los nudillos, formando una palma rectangular.
        """
        mcp_i, mcp_p, muneca = puntos[5], puntos[17], puntos[0]
        centro_mcp = (mcp_i + mcp_p) / 2.0
        base_i = muneca + 0.90 * (mcp_i - centro_mcp)
        base_p = muneca + 0.90 * (mcp_p - centro_mcp)
        pts = np.array(
            [puntos[5], puntos[9], puntos[13], puntos[17], puntos[1],
             base_p, base_i],
            dtype=np.float32,
        )
        casco = cv2.convexHull(pts).reshape(-1, 2)
        centroide = casco.mean(axis=0)
        margen = r_dedo * 1.05

        def dibujar_casco(escala, extra, color):
            pts = (casco - centroide) * escala + centroide
            radio = margen * escala + extra
            cv2.fillConvexPoly(img, np.int32(pts), color, lineType=cv2.LINE_AA)
            n = len(pts)
            for i in range(n):
                a, b = pts[i], pts[(i + 1) % n]
                cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), color,
                         max(1, int(2 * radio)), cv2.LINE_AA)
                cv2.circle(img, tuple(np.int32(a)), max(1, int(radio)), color,
                           -1, cv2.LINE_AA)

        dibujar_casco(1.0, self._g, COL_CONTORNO)   # contorno
        for indice, factor, corr in self._CAPAS:
            dibujar_casco(factor, 0.0, self.tonos_piel[indice])

    def _una(self, img, punta, previa, r, vis):
        """Uña en la punta, orientada a lo largo del dedo.

        Se muestra solo cuando se ve el dorso (`vis` alto). Se dibuja con tamano
        fijo (no se encoge), cerca de la punta, en el lado del dorso.
        """
        if vis <= 0.35:
            return
        d = punta - previa
        largo = np.hypot(*d)
        direccion = d / largo if largo > 1e-3 else np.array([0.0, -1.0])
        # Cerca de la punta (offset chico = mas arriba en el dedo).
        centro = punta - direccion * r * 0.22
        ang = np.degrees(np.arctan2(direccion[1], direccion[0]))
        ejes = (max(1, int(r * 0.80)), max(1, int(r * 0.60)))
        cv2.ellipse(img, tuple(np.int32(centro)), (ejes[0] + 1, ejes[1] + 1),
                    ang, 0, 360, COL_CONTORNO, -1, cv2.LINE_AA)
        cv2.ellipse(img, tuple(np.int32(centro)), ejes, ang, 0, 360,
                    COL_UNA, -1, cv2.LINE_AA)

    # -- Fotograma completo ---------------------------------------------------

    @staticmethod
    def _asociar_manos(cuerpo, mano_izq, mano_der):
        """Asocia cada mano al brazo cuya MUÑECA DE POSE tiene mas cerca.

        No se usa la etiqueta izquierda/derecha con que se grabaron las manos,
        porque MediaPipe la INTERCAMBIA cuando las dos manos se juntan o apuntan
        a la camara, y entonces el brazo agarra la mano equivocada y se cruza.
        En cambio, cada mano se pega al brazo cuya muneca (de la Pose) esta mas
        cerca de la muneca de la mano; con dos manos se elige la asignacion de
        menor distancia total. Devuelve {"izq": mano|None, "der": mano|None}.
        """
        objetivo = {"izq": np.asarray(cuerpo[CUERPO_MUNECA_IZQ][:2]),
                    "der": np.asarray(cuerpo[CUERPO_MUNECA_DER][:2])}
        presentes = [m for m in (mano_izq, mano_der) if m is not None]
        res = {"izq": None, "der": None}
        if not presentes:
            return res

        def dist(mano, lado):
            return float(np.linalg.norm(np.asarray(mano[0][:2]) - objetivo[lado]))

        if len(presentes) == 1:
            m = presentes[0]
            lado = "izq" if dist(m, "izq") <= dist(m, "der") else "der"
            res[lado] = m
            return res
        a, b = presentes
        recto = dist(a, "izq") + dist(b, "der")
        cruzado = dist(b, "izq") + dist(a, "der")
        if recto <= cruzado:
            res["izq"], res["der"] = a, b
        else:
            res["izq"], res["der"] = b, a
        return res

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

        manos = self._asociar_manos(cuerpo, mano_izq, mano_der)
        z_izq = cuerpo[CUERPO_MUNECA_IZQ][2]
        z_der = cuerpo[CUERPO_MUNECA_DER][2]
        brazos = [("izq", manos["izq"], z_izq), ("der", manos["der"], z_der)]
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


def _rellenar_rango_crudo(frames, clave, i0, i1) -> bool:
    """Regenera un tramo [i0, i1] de una mano desde los frames buenos de al lado.

    Reemplaza los fotogramas marcados (temblorosos o sin mano) por la
    interpolacion lineal entre el ultimo frame BUENO antes de i0 y el primero
    BUENO despues de i1. Si solo hay dato de un lado, lo copia. Trabaja sobre el
    clip CRUDO (coordenadas [0,1]); luego se re-suaviza al reconstruir. Devuelve
    True si pudo rellenar.
    """
    n = len(frames)
    i0, i1 = max(0, i0), min(n - 1, i1)
    a = i0 - 1
    while a >= 0 and frames[a].get(clave) is None:
        a -= 1
    b = i1 + 1
    while b < n and frames[b].get(clave) is None:
        b += 1
    tiene_a, tiene_b = a >= 0, b < n
    if not tiene_a and not tiene_b:
        return False
    for i in range(i0, i1 + 1):
        if tiene_a and tiene_b:
            t = (i - a) / (b - a)
            va = np.asarray(frames[a][clave], dtype=float)
            vb = np.asarray(frames[b][clave], dtype=float)
            frames[i][clave] = ((1 - t) * va + t * vb).tolist()
        else:
            fuente = frames[a if tiene_a else b][clave]
            frames[i][clave] = [list(p) for p in fuente]
    return True


def _guardar_editado(ruta, raw) -> None:
    """Guarda el clip editado, respaldando el original en .bak la primera vez."""
    ruta = Path(ruta)
    bak = ruta.with_suffix(ruta.suffix + ".bak")
    if not bak.exists():
        shutil.copy(ruta, bak)
    guardar_clip(ruta, raw)


def _hud_edicion(img, rango, hay_marca, editado):
    """Panel superior con el estado del modo edicion y los controles."""
    ancho = img.shape[1]
    dibujo.panel(img, 12, 10, ancho - 24, 74, alpha=0.72)
    dibujo.texto(img, "MODO EDICION", 26, 40, 0.8, dibujo.AMBAR, 2,
                 dibujo.FUENTE_TITULO)
    if hay_marca:
        estado = f"tramo malo: {rango[0]} a {rango[1]}"
        color = dibujo.VERDE
    else:
        estado = "marca I inicio, O fin del tramo malo"
        color = dibujo.GRIS_CLARO
    dibujo.texto(img, estado, 230, 40, 0.6, color, 1)
    ctrl = ("A/D frame   I inicio   O fin   F arreglar   Z deshacer   "
            "S guardar   C limpiar   E salir")
    dibujo.texto(img, ctrl, 26, 70, 0.5, dibujo.GRIS_CLARO, 1)
    if editado:
        dibujo.texto(img, "sin guardar", ancho - 150, 40, 0.55, dibujo.ROJO, 1)


def reproducir(rutas_clips, paleta, ancho, alto, vista_espejo=False):
    """Bucle interactivo del visor, con modo edicion de clips."""
    indice_clip = 0
    vel_idx = 0
    pausado = False

    def cargar(i):
        raw = cargar_clip(rutas_clips[i])
        return raw, ClipPreparado(raw)

    raw, clip = cargar(indice_clip)
    muneco = MunecoCapsulas(ancho, alto, paleta)
    muneco.preparar_marco(clip, vista_espejo)
    indice = 0.0

    # Estado del editor.
    modo_edicion = False
    rango = [0, 0]
    hay_marca = False
    editado = False
    historial = []  # copias de raw["frames"] para deshacer

    def reconstruir():
        nonlocal clip
        clip = ClipPreparado(raw)
        muneco.preparar_marco(clip, vista_espejo)

    nombre_ventana = "Visor de clips LESHO"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_AUTOSIZE)

    while True:
        cuerpo, mi, md = clip.fotograma(indice)
        img = muneco.dibujar(cuerpo, mi, md)
        _hud(img, clip, indice, VELOCIDADES[vel_idx], vista_espejo, pausado)
        if modo_edicion:
            _hud_edicion(img, rango, hay_marca, editado)
        cv2.imshow(nombre_ventana, img)

        espera_ms = max(1, int(1000.0 / clip.fps))
        tecla = cv2.waitKey(espera_ms) & 0xFF

        # En modo edicion la reproduccion queda en pausa (se trabaja frame a frame).
        if not pausado and not modo_edicion:
            indice += VELOCIDADES[vel_idx]
            if indice >= clip.num_frames:
                indice = 0.0  # bucle

        cambiar = 0
        cuadro = int(np.floor(indice))
        if tecla in (ord("q"), 27):
            break
        elif tecla == ord("e"):
            modo_edicion = not modo_edicion
            pausado = True
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
        elif not modo_edicion and tecla == ord(" "):
            pausado = not pausado
        elif not modo_edicion and tecla == ord("n"):
            cambiar = 1
        elif not modo_edicion and tecla == ord("p"):
            cambiar = -1
        # -- Teclas del modo edicion --
        elif modo_edicion and tecla == ord("i"):
            rango[0] = cuadro
            rango[1] = max(rango[1], cuadro)
            hay_marca = True
        elif modo_edicion and tecla == ord("o"):
            rango[1] = cuadro
            rango[0] = min(rango[0], cuadro)
            hay_marca = True
        elif modo_edicion and tecla == ord("c"):
            hay_marca = False
        elif modo_edicion and tecla == ord("f") and hay_marca:
            historial.append(copy.deepcopy(raw["frames"]))
            ok = False
            for clave in ("mano_izq", "mano_der"):
                ok = _rellenar_rango_crudo(
                    raw["frames"], clave, rango[0], rango[1]) or ok
            if ok:
                editado = True
                reconstruir()
                print(f"Arreglado el tramo {rango[0]}-{rango[1]}")
            else:
                historial.pop()
                print("No hay frames buenos de referencia para rellenar.")
        elif modo_edicion and tecla == ord("z") and historial:
            raw["frames"] = historial.pop()
            editado = len(historial) > 0
            reconstruir()
            print("Deshecho el ultimo arreglo")
        elif modo_edicion and tecla == ord("s"):
            _guardar_editado(rutas_clips[indice_clip], raw)
            editado = False
            print(f"Guardado {rutas_clips[indice_clip]} (respaldo en .bak)")

        if cambiar:
            indice_clip = (indice_clip + cambiar) % len(rutas_clips)
            raw, clip = cargar(indice_clip)
            muneco.preparar_marco(clip, vista_espejo)
            indice = 0.0
            pausado = False
            hay_marca = False
            editado = False
            historial = []

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
    parser.add_argument("--paleta", choices=sorted(PALETAS), default="humano")
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
