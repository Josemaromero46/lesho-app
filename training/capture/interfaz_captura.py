"""
Interfaz visual de la captura.

Compone, fotograma a fotograma, la capa de informacion que se dibuja sobre el
video: barra de progreso global, chip de estado, indicador de deteccion de
mano, cuenta regresiva, panel con la sena actual y los controles, y el aviso de
guardado. Toda la presentacion vive aqui; la logica de la sesion (cuando grabar,
que guardar) vive en sesion_captura.py.

El metodo central es `componer(frame, ctx)`, que recibe el fotograma y un objeto
`ContextoUI` con el estado a mostrar, y dibuja encima.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from . import dibujo


# Estados de la interfaz (coinciden con los de la maquina de la sesion).
INTRO = "intro"
PREPARACION = "preparacion"
GRABANDO = "grabando"
GUARDADO = "guardado"
PAUSA = "pausa"
FIN = "fin"


@dataclass
class ContextoUI:
    """Datos que la interfaz necesita para dibujar un fotograma."""

    estado: str = INTRO
    modo: str = "estatico"
    persona: str = ""

    clase_actual: str = ""
    indice_clase: int = 0          # 1-based, para mostrar "3 de 30"
    total_clases: int = 0

    rep_actual: int = 0            # 1-based
    total_reps: int = 1

    cuenta_restante: float = 0.0   # segundos que faltan en la cuenta regresiva
    progreso_grabacion: float = 0.0  # 0..1 dentro de la ventana de grabacion
    mano_detectada: bool = False
    num_manos: int = 0             # manos detectadas en el fotograma actual
    num_acumulado: int = 0         # frames o muestras acumuladas hasta ahora

    mensaje: str = ""              # mensaje libre (por ejemplo en GUARDADO)
    manos: list = field(default_factory=list)  # landmarks de las manos a dibujar

    # Modelo B: cuerpo (Pose). Vacio y False en la captura del alfabeto.
    usa_pose: bool = False         # True si la sesion usa Pose (captura dinamica)
    cuerpo: list = field(default_factory=list)  # 33 puntos de Pose a dibujar
    cuerpo_detectado: bool = False  # cuerpo (hombros) visible en este fotograma


class InterfazCaptura:
    """Dibuja y gestiona la ventana de OpenCV de la captura."""

    def __init__(self, nombre_ventana: str, ancho: int, alto: int,
                 ruta_referencias=None):
        self.nombre = nombre_ventana
        self.ancho = ancho
        self.alto = alto
        self._referencias = self._cargar_referencias(ruta_referencias)
        cv2.namedWindow(self.nombre, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.nombre, ancho, alto)

    # -- Ciclo de ventana ---------------------------------------------------

    def mostrar(self, frame) -> None:
        cv2.imshow(self.nombre, frame)

    def leer_tecla(self, espera_ms: int = 1) -> str:
        """Devuelve la tecla presionada en minuscula, o cadena vacia."""
        codigo = cv2.waitKey(espera_ms) & 0xFF
        if codigo == 255:
            return ""
        if codigo == 27:       # ESC
            return "esc"
        if codigo == 32:       # barra espaciadora
            return "espacio"
        return chr(codigo).lower()

    def ventana_cerrada(self) -> bool:
        """True si el usuario cerro la ventana con la X."""
        try:
            return cv2.getWindowProperty(self.nombre, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def cerrar(self) -> None:
        cv2.destroyAllWindows()

    # -- Composicion del overlay -------------------------------------------

    def componer(self, frame, ctx: ContextoUI):
        """Dibuja toda la capa de informacion sobre el frame (in place)."""
        # 1. Esqueletos al fondo del overlay: primero el cuerpo (Pose), luego las
        #    manos encima. El cuerpo solo existe en la captura dinamica (Modelo B).
        if ctx.cuerpo:
            dibujo.dibujar_cuerpo(frame, ctx.cuerpo)
        for mano in ctx.manos:
            dibujo.dibujar_mano(frame, mano)

        # 2. Pantallas que cubren todo el frame.
        if ctx.estado == INTRO:
            self._overlay_intro(frame, ctx)
            return frame
        if ctx.estado == FIN:
            self._overlay_fin(frame, ctx)
            return frame

        # 3. Elementos comunes a los estados de captura.
        self._barra_global(frame, ctx)
        self._panel_inferior(frame, ctx)
        self._referencia_lateral(frame, ctx)

        # 4. Elementos especificos de cada estado.
        if ctx.estado == PREPARACION:
            self._overlay_preparacion(frame, ctx)
        elif ctx.estado == GRABANDO:
            self._overlay_grabando(frame, ctx)
        elif ctx.estado == GUARDADO:
            self._overlay_guardado(frame, ctx)
        elif ctx.estado == PAUSA:
            self._overlay_pausa(frame, ctx)

        return frame

    # -- Bloques reutilizables ---------------------------------------------

    def _barra_global(self, frame, ctx: ContextoUI):
        """Barra fina de progreso global en el borde superior."""
        margen = 24
        ancho_barra = self.ancho - 2 * margen
        fraccion = (ctx.indice_clase / ctx.total_clases) if ctx.total_clases else 0
        dibujo.barra_progreso(frame, margen, 18, ancho_barra, 8, fraccion,
                              color=dibujo.TEAL)
        etiqueta = f"Seña {ctx.indice_clase} de {ctx.total_clases}"
        dibujo.texto(frame, etiqueta, margen, 52, 0.6, dibujo.GRIS_CLARO, 1)

        # Chips de deteccion (solo cuando importa), anclados al margen derecho.
        if ctx.estado in (PREPARACION, GRABANDO):
            x_der = self.ancho - 24
            if ctx.num_manos >= 2:
                dibujo.chip_estado(frame, x_der, 36, "Dos manos detectadas",
                                   dibujo.VERDE)
            elif ctx.num_manos == 1:
                dibujo.chip_estado(frame, x_der, 36, "Una mano detectada",
                                   dibujo.VERDE)
            else:
                dibujo.chip_estado(frame, x_der, 36, "Acerque la mano",
                                   dibujo.AMBAR)

            # Chip del cuerpo, solo en la captura dinamica (Modelo B). Avisa si el
            # torso/cara se salio del cuadro, porque el marco del cuerpo depende
            # de que se vean los hombros y la cara.
            if ctx.usa_pose:
                if ctx.cuerpo_detectado:
                    dibujo.chip_estado(frame, x_der, 78, "Cuerpo detectado",
                                       dibujo.VERDE)
                else:
                    dibujo.chip_estado(frame, x_der, 78, "Muestre el torso y la cara",
                                       dibujo.AMBAR)

    def _panel_inferior(self, frame, ctx: ContextoUI):
        """Panel con la sena actual, la repeticion y los controles."""
        h = 116
        y = self.alto - h - 20
        margen = 24
        w = self.ancho - 2 * margen
        dibujo.panel(frame, margen, y, w, h, alpha=0.68)

        # Nombre de la sena, grande.
        dibujo.texto(frame, ctx.clase_actual, margen + 26, y + 56, 1.5,
                     dibujo.BLANCO, 2, dibujo.FUENTE_TITULO)

        # Repeticion (solo tiene sentido si hay mas de una).
        if ctx.total_reps > 1:
            rep = f"Repeticion {ctx.rep_actual} de {ctx.total_reps}"
            dibujo.texto(frame, rep, margen + 26, y + 92, 0.7,
                         dibujo.GRIS_CLARO, 1)

        # Controles, alineados a la derecha del panel.
        controles = "ESPACIO pausa    R repetir    Q salir"
        ancho_txt, _ = dibujo.medir(controles, 0.6, 1)
        dibujo.texto(frame, controles, margen + w - ancho_txt - 26, y + 92,
                     0.6, dibujo.GRIS_TENUE, 1)

    def _referencia_lateral(self, frame, ctx: ContextoUI):
        """Muestra la imagen de referencia de la sena, si existe."""
        imagen = self._referencias.get(ctx.clase_actual)
        if imagen is None:
            return
        lado = 180
        x = self.ancho - lado - 24
        y = 70
        recorte = cv2.resize(imagen, (lado, lado))
        dibujo.panel(frame, x - 8, y - 8, lado + 16, lado + 36, alpha=0.6)
        frame[y:y + lado, x:x + lado] = recorte
        dibujo.texto_centrado(frame, "referencia", x + lado // 2, y + lado + 20,
                              0.55, dibujo.GRIS_CLARO, 1)

    # -- Overlays por estado ------------------------------------------------

    def _overlay_intro(self, frame, ctx: ContextoUI):
        self._velo(frame, 0.55)
        cx = self.ancho // 2
        dibujo.texto_centrado(frame, "Captura de señas LESHO", cx, 220, 1.6,
                              dibujo.BLANCO, 2, dibujo.FUENTE_TITULO)
        modo_legible = "estático" if ctx.modo == "estatico" else "dinámico"
        sub = f"Modo {modo_legible}   ·   {ctx.persona}"
        dibujo.texto_centrado(frame, sub, cx, 270, 0.8, dibujo.TEAL, 1)

        lineas = [
            "Siéntese a 60-100 cm de la cámara, con el dispositivo apoyado.",
            "Imite la seña que aparezca en pantalla cuando inicie la cuenta.",
            "Mantenga la mano dentro del cuadro y bien iluminada.",
        ]
        y = 340
        for linea in lineas:
            dibujo.texto_centrado(frame, linea, cx, y, 0.68, dibujo.GRIS_CLARO, 1)
            y += 40

        dibujo.texto_centrado(frame, "Presione ESPACIO para comenzar", cx,
                              y + 30, 0.85, dibujo.VERDE, 2)

    def _overlay_fin(self, frame, ctx: ContextoUI):
        self._velo(frame, 0.6)
        cx = self.ancho // 2
        dibujo.texto_centrado(frame, "Captura completada", cx, 300, 1.5,
                              dibujo.VERDE, 2, dibujo.FUENTE_TITULO)
        if ctx.mensaje:
            dibujo.texto_centrado(frame, ctx.mensaje, cx, 360, 0.8,
                                  dibujo.GRIS_CLARO, 1)
        dibujo.texto_centrado(frame, "Gracias. Presione Q para salir.", cx, 420,
                              0.8, dibujo.BLANCO, 1)

    def _overlay_preparacion(self, frame, ctx: ContextoUI):
        cx = self.ancho // 2
        dibujo.texto_centrado(frame, "Prepárese", cx, 150, 1.0, dibujo.AMBAR, 2,
                              dibujo.FUENTE_TITULO)
        numero = str(int(ctx.cuenta_restante) + 1)
        dibujo.texto_centrado(frame, numero, cx, 320, 4.5, dibujo.AMBAR, 6,
                              dibujo.FUENTE_TITULO)

    def _overlay_grabando(self, frame, ctx: ContextoUI):
        # Punto REC parpadeante en la esquina superior izquierda del area util.
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (self.ancho // 2 - 90, 150), 12, dibujo.ROJO, -1,
                       cv2.LINE_AA)
        dibujo.texto(frame, "GRABANDO", self.ancho // 2 - 60, 158, 1.0,
                     dibujo.ROJO, 2, dibujo.FUENTE_TITULO)

        # Barra de tiempo de la toma, sobre el panel inferior.
        margen = 24
        w = self.ancho - 2 * margen
        y = self.alto - 116 - 20 - 16
        dibujo.barra_progreso(frame, margen, y, w, 8, ctx.progreso_grabacion,
                              color=dibujo.ROJO)

    def _overlay_guardado(self, frame, ctx: ContextoUI):
        cx = self.ancho // 2
        # Marca de verificacion simple dibujada con dos lineas.
        cv2.line(frame, (cx - 26, 240), (cx - 8, 262), dibujo.VERDE, 4,
                 cv2.LINE_AA)
        cv2.line(frame, (cx - 8, 262), (cx + 30, 214), dibujo.VERDE, 4,
                 cv2.LINE_AA)
        mensaje = ctx.mensaje or "Guardado"
        dibujo.texto_centrado(frame, mensaje, cx, 320, 1.0, dibujo.VERDE, 2,
                              dibujo.FUENTE_TITULO)
        dibujo.texto_centrado(frame, "R para repetir esta toma", cx, 360, 0.65,
                              dibujo.GRIS_CLARO, 1)

    def _overlay_pausa(self, frame, ctx: ContextoUI):
        self._velo(frame, 0.5)
        cx = self.ancho // 2
        dibujo.texto_centrado(frame, "PAUSA", cx, 320, 1.8, dibujo.BLANCO, 2,
                              dibujo.FUENTE_TITULO)
        dibujo.texto_centrado(frame, "Presione ESPACIO para continuar", cx, 380,
                              0.8, dibujo.GRIS_CLARO, 1)

    # -- Auxiliares ---------------------------------------------------------

    def _velo(self, frame, alpha: float):
        """Oscurece todo el frame para resaltar el texto encima."""
        overlay = frame.copy()
        overlay[:] = (18, 17, 16)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    @staticmethod
    def _cargar_referencias(ruta_referencias):
        """Carga las imagenes de referencia disponibles en un diccionario."""
        referencias: dict = {}
        if not ruta_referencias:
            return referencias
        carpeta = Path(ruta_referencias)
        if not carpeta.is_dir():
            return referencias
        for archivo in carpeta.iterdir():
            if archivo.suffix.lower() in (".png", ".jpg", ".jpeg"):
                imagen = cv2.imread(str(archivo))
                if imagen is not None:
                    referencias[archivo.stem] = imagen
        return referencias
