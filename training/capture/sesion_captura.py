"""
Sesion de captura: bucle de camara y maquina de estados.

Contiene la logica comun a la captura estatica y dinamica. Lee la camara,
extrae landmarks con MediaPipe, conduce al usuario por las fases (preparacion,
grabacion, guardado) y entrega las muestras al escritor del dataset. La
presentacion la resuelve `InterfazCaptura`; aqui solo vive el "cuando" y el
"que se guarda".

Maquina de estados de la captura:

    INTRO ──ESPACIO──► PREPARACION ──cuenta a 0──► GRABANDO ──tiempo──► GUARDADO
                            ▲                                              │
                            └──────────── repetir / siguiente toma ────────┘
                                              │ (clase completa)
                                              ▼
                                             FIN

Convencion de orientacion (CONTRATO con la app): cada fotograma se voltea
horizontalmente (vista espejo, modo selfie) y MediaPipe procesa el fotograma ya
volteado. La app debe hacer lo mismo, o los landmarks no coincidiran con el
dataset.
"""

import time
from dataclasses import dataclass

import cv2

from comun.clips import (
    CUERPO_CADERA_DER,
    CUERPO_CADERA_IZQ,
    CUERPO_CODO_DER,
    CUERPO_CODO_IZQ,
    INDICES_POSE_CLIP,
)
from comun.definiciones import VISIBILIDAD_MINIMA_POSE
from comun.definiciones import LATERALIDAD_DERECHA, LATERALIDAD_IZQUIERDA
from comun.marco import marco_desde_puntos
from comun.normalizacion import componer_vector_dos_manos
from comun.vector_modelo_b import componer_vector_modelo_b
from .escritor_dataset import MODO_DINAMICO, MODO_ESTATICO
from .interfaz_captura import (
    ContextoUI,
    FIN,
    GRABANDO,
    GUARDADO,
    INTRO,
    PAUSA,
    PREPARACION,
)


@dataclass
class ParametrosCaptura:
    """Parametros temporales y de conteo de una sesion de captura."""

    indice_camara: int = 0
    ancho_camara: int = 1280
    alto_camara: int = 720
    segundos_cuenta: float = 3.0
    segundos_grabacion: float = 3.0
    segundos_entre_reps: float = 1.0
    segundos_guardado: float = 1.2
    muestras_por_clase: int = 40
    min_frames: int = 30
    max_frames: int = 60


class SesionCaptura:
    """Conduce una sesion completa de captura de una persona."""

    def __init__(self, modo, clases, persona, escritor, interfaz, detector,
                 params: ParametrosCaptura, detector_pose=None,
                 captura_cruda=False):
        self.modo = modo
        self.clases = clases
        self.persona = persona
        self.escritor = escritor
        self.interfaz = interfaz
        self.detector = detector
        self.params = params
        # Detector de Pose OPCIONAL. Si se pasa (captura del Modelo B), cada
        # fotograma guarda ademas la ubicacion de las manos en el cuerpo, y el
        # vector pasa de 126 a 132 valores. Si es None, la captura es identica a
        # antes (alfabeto: solo manos).
        self.detector_pose = detector_pose
        # Captura CRUDA (clips del diccionario, Direccion 2): en vez de vectores
        # normalizados, el buffer acumula fotogramas crudos (landmarks [0,1] de
        # manos y cuerpo, tal como salen de MediaPipe) para reproducirlos en el
        # muneco. Apagada por defecto: el dataset (alfabeto y Modelo B) no cambia.
        self.captura_cruda = captura_cruda
        # Ultimo marco del cuerpo valido. Se mantiene (carry-forward) porque el
        # torso es estable: si un fotograma no detecta bien la pose, se reusa el
        # ultimo marco fiable en vez de perder la ubicacion.
        self._marco_actual = None
        self._cuerpo_actual: list = []   # puntos de Pose del fotograma, a dibujar
        self._cuerpo_visible = False     # cuerpo (hombros) visible este fotograma

        self.estado = INTRO
        self.clase_idx = 0
        self.t_estado = time.time()

        self._buffer: list = []          # vectores acumulados en GRABANDO
        self._pendiente = None           # toma a confirmar (o None si invalida)
        self._tomas_en_clase = 0         # cuantas tomas se hicieron de la clase
        self._total_guardado = 0         # muestras escritas en esta sesion
        self._manos_actuales: list = []  # landmarks a dibujar este fotograma
        self._num_manos = 0              # manos detectadas en el fotograma actual

        # Para reanudar la fase exacta tras una pausa.
        self._pausa_estado = None
        self._pausa_transcurrido = 0.0

    # -- Bucle principal ----------------------------------------------------

    def ejecutar(self) -> None:
        cap = self._abrir_camara()
        # Si todas las clases ya estan completas, ir directo al cierre.
        if self._siguiente_clase_pendiente(0) is None:
            self.estado = FIN
        try:
            fallos_lectura = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    fallos_lectura += 1
                    if fallos_lectura > 30:
                        raise RuntimeError("La camara dejo de entregar imagen.")
                    continue
                fallos_lectura = 0

                frame = cv2.flip(frame, 1)  # vista espejo (selfie)
                ahora = time.time()

                ctx = self._procesar_estado(frame, ahora)
                self.interfaz.componer(frame, ctx)
                self.interfaz.mostrar(frame)

                if not self._manejar_teclas():
                    break
                if self.interfaz.ventana_cerrada():
                    break
        finally:
            self._confirmar_pendiente()  # no perder la ultima toma valida
            cap.release()

    # -- Despacho por estado ------------------------------------------------

    def _procesar_estado(self, frame, ahora) -> ContextoUI:
        self._manos_actuales = []
        self._num_manos = 0
        self._cuerpo_actual = []
        self._cuerpo_visible = False

        # Solo se corre MediaPipe cuando aporta (preparacion y grabacion).
        if self.estado in (PREPARACION, GRABANDO):
            manos = self.detector.procesar(frame)
            # Modelo B: Pose para el marco del cuerpo y para dibujarlo.
            if self.detector_pose is not None:
                puntos = self.detector_pose.procesar(frame)
                if puntos is not None:
                    self._cuerpo_actual = puntos
                    marco = marco_desde_puntos(puntos)
                    self._cuerpo_visible = marco is not None
                    # Carry-forward: solo se actualiza si el marco es fiable.
                    if marco is not None:
                        self._marco_actual = marco
            if manos:
                self._num_manos = len(manos)
                self._manos_actuales = [m.landmarks for m in manos]
            if self.estado == GRABANDO:
                if self.captura_cruda:
                    # Clip del diccionario: se guarda TODO fotograma, haya manos
                    # o no (una mano ausente queda como null y se limpia offline).
                    self._buffer.append(self._fotograma_crudo(manos))
                elif manos:
                    self._buffer.append(self._componer_vector(manos))

        ctx = self._contexto_base()

        if self.estado == PREPARACION:
            self._paso_preparacion(ahora, ctx)
        elif self.estado == GRABANDO:
            self._paso_grabando(ahora, ctx)
        elif self.estado == GUARDADO:
            self._paso_guardado(ahora, ctx)

        return ctx

    def _paso_preparacion(self, ahora, ctx: ContextoUI):
        duracion = (self.params.segundos_cuenta if self._tomas_en_clase == 0
                    else self.params.segundos_entre_reps)
        restante = duracion - (ahora - self.t_estado)
        ctx.cuenta_restante = max(0.0, restante)
        if restante <= 0:
            self._buffer = []
            self._cambiar_estado(GRABANDO, ahora)

    def _paso_grabando(self, ahora, ctx: ContextoUI):
        transcurrido = ahora - self.t_estado
        ctx.progreso_grabacion = min(1.0, transcurrido / self.params.segundos_grabacion)
        if transcurrido >= self.params.segundos_grabacion:
            self._procesar_toma()
            self._cambiar_estado(GUARDADO, ahora)

    def _paso_guardado(self, ahora, ctx: ContextoUI):
        if self._pendiente is not None:
            ctx.mensaje = self._pendiente["mensaje"]
        else:
            ctx.mensaje = self._mensaje_invalido
        if ahora - self.t_estado >= self.params.segundos_guardado:
            self._cerrar_toma()

    # -- Procesamiento de la toma -------------------------------------------

    def _procesar_toma(self) -> None:
        """Convierte el buffer en una muestra pendiente (o la marca invalida)."""
        existentes = self.escritor.muestras_existentes(
            self._clase_actual, self.persona
        )
        if self.modo == MODO_ESTATICO:
            faltan = self.params.muestras_por_clase - existentes
            muestras = self._submuestrear(self._buffer, faltan)
            if not muestras:
                self._marcar_invalido("No se detectó la mano, repita")
                return
            self._pendiente = {
                "tipo": MODO_ESTATICO,
                "datos": muestras,
                "base": existentes,
                "mensaje": f"Guardado: {len(muestras)} muestras",
            }
        else:  # dinamico
            if len(self._buffer) < self.params.min_frames:
                self._marcar_invalido("Toma muy corta, repita")
                return
            if self.captura_cruda and not self._toma_cruda_valida():
                return
            secuencia = self._buffer[: self.params.max_frames]
            self._pendiente = {
                "tipo": MODO_DINAMICO,
                "datos": secuencia,
                "base": existentes,
                "mensaje": f"Secuencia de {len(secuencia)} fotogramas",
            }

    def _cerrar_toma(self) -> None:
        """Confirma la toma pendiente y decide la siguiente fase."""
        hubo_muestra = self._pendiente is not None
        self._confirmar_pendiente()
        self._tomas_en_clase += 1

        if hubo_muestra and self._esta_completa(self._clase_actual):
            siguiente = self._siguiente_clase_pendiente(self.clase_idx + 1)
            if siguiente is None:
                self._cambiar_estado(FIN, time.time())
                return
            self.clase_idx = siguiente
            self._tomas_en_clase = 0
        # En cualquier otro caso se hace otra toma de la misma clase.
        self._cambiar_estado(PREPARACION, time.time())

    def _confirmar_pendiente(self) -> None:
        """Escribe la toma pendiente en el dataset, si la hay."""
        if not self._pendiente:
            return
        clase = self._clase_actual
        base = self._pendiente["base"]
        if self._pendiente["tipo"] == MODO_ESTATICO:
            for i, vector in enumerate(self._pendiente["datos"]):
                self.escritor.escribir_muestra_estatica(
                    clase, self.persona, base + i, vector
                )
                self._total_guardado += 1
        else:
            self.escritor.escribir_muestra_dinamica(
                clase, self.persona, base, self._pendiente["datos"]
            )
            self._total_guardado += 1
        self._pendiente = None

    def _marcar_invalido(self, mensaje: str) -> None:
        self._pendiente = None
        self._mensaje_invalido = mensaje

    # -- Teclado ------------------------------------------------------------

    def _manejar_teclas(self) -> bool:
        """Procesa la tecla del fotograma. Devuelve False para terminar."""
        tecla = self.interfaz.leer_tecla(1)
        if not tecla:
            return True

        if tecla in ("q", "esc"):
            return False

        if tecla == "espacio":
            if self.estado == INTRO:
                primera = self._siguiente_clase_pendiente(0)
                if primera is None:
                    self._cambiar_estado(FIN, time.time())
                else:
                    self.clase_idx = primera
                    self._tomas_en_clase = 0
                    self._cambiar_estado(PREPARACION, time.time())
            elif self.estado == PAUSA:
                self._reanudar()
            elif self.estado not in (FIN,):
                self._pausar()

        elif tecla == "r":
            # Repetir: descartar la toma en curso y volver a preparacion.
            if self.estado in (GRABANDO, GUARDADO):
                self._pendiente = None
                self._buffer = []
                self._cambiar_estado(PREPARACION, time.time())

        return True

    def _pausar(self) -> None:
        self._pausa_estado = self.estado
        self._pausa_transcurrido = time.time() - self.t_estado
        self.estado = PAUSA

    def _reanudar(self) -> None:
        self.estado = self._pausa_estado or PREPARACION
        self.t_estado = time.time() - self._pausa_transcurrido

    # -- Auxiliares ---------------------------------------------------------

    @property
    def _clase_actual(self) -> str:
        return self.clases[self.clase_idx]

    def _contexto_base(self) -> ContextoUI:
        existentes = self.escritor.muestras_existentes(
            self._clase_actual, self.persona
        ) if self.estado != INTRO else 0
        total_reps = self.params.muestras_por_clase if self.modo == MODO_DINAMICO else 1
        return ContextoUI(
            estado=self.estado,
            modo=self.modo,
            persona=self.persona,
            clase_actual=self._clase_actual,
            indice_clase=self.clase_idx + 1,
            total_clases=len(self.clases),
            rep_actual=min(existentes + 1, total_reps),
            total_reps=total_reps,
            mano_detectada=self._num_manos > 0,
            num_manos=self._num_manos,
            num_acumulado=len(self._buffer),
            manos=self._manos_actuales,
            mensaje=self._mensaje_fin if self.estado == FIN else "",
            usa_pose=self.detector_pose is not None,
            cuerpo=self._cuerpo_actual,
            cuerpo_detectado=self._cuerpo_visible,
        )

    def _asignar_lateralidad(self, manos):
        """Separa las manos detectadas en (izquierda, derecha) por lateralidad.

        Las manos con lateralidad desconocida o repetida se colocan en el primer
        slot libre, de modo que una sena a una mano siempre produce datos.
        """
        izquierda = derecha = None
        sin_asignar = []
        for mano in manos:
            if mano.lateralidad == LATERALIDAD_IZQUIERDA and izquierda is None:
                izquierda = mano.landmarks
            elif mano.lateralidad == LATERALIDAD_DERECHA and derecha is None:
                derecha = mano.landmarks
            else:
                sin_asignar.append(mano.landmarks)
        for landmarks in sin_asignar:
            if izquierda is None:
                izquierda = landmarks
            elif derecha is None:
                derecha = landmarks
        return izquierda, derecha

    def _fotograma_crudo(self, manos) -> dict:
        """Fotograma crudo para un clip del diccionario (Direccion 2).

        Guarda las coordenadas tal como salen de MediaPipe ([0, 1] de la
        imagen), sin normalizar respecto a la muneca, porque el objetivo es
        REPRODUCIR el movimiento sobre el muneco, no entrenar. Incluye el
        instante de captura para que el escritor calcule el fps real de la toma.
        """
        izquierda = derecha = None
        if manos:
            izquierda, derecha = self._asignar_lateralidad(manos)
        cuerpo = None
        if self._cuerpo_actual:
            cuerpo = [
                [p.x, p.y, p.z, p.visibilidad]
                for p in (self._cuerpo_actual[i] for i in INDICES_POSE_CLIP)
            ]
        return {
            "t": time.time(),
            "cuerpo": cuerpo,
            "mano_izq": izquierda,
            "mano_der": derecha,
        }

    def _toma_cruda_valida(self) -> bool:
        """Valida una toma cruda: manos y CUERPO COMPLETO en suficientes frames.

        En la captura cruda el buffer crece con TODOS los fotogramas, asi que el
        minimo de frames no garantiza que la sena se haya visto. Se exige que al
        menos la mitad de los fotogramas tengan alguna mano y que al menos la
        mitad tengan cuerpo (sin cuerpo no hay marco para centrar el muneco).

        Ademas, el muneco dibuja BRAZOS y TORSO COMPLETO, asi que codos y
        caderas deben verse (el dataset del Modelo B no los necesitaba, por eso
        este encuadre es mas exigente: la persona debe alejarse un poco mas de
        la camara que en la captura del dataset).
        """
        total = len(self._buffer)
        con_mano = sum(
            1 for f in self._buffer
            if f["mano_izq"] is not None or f["mano_der"] is not None
        )
        cuerpos = [f["cuerpo"] for f in self._buffer if f["cuerpo"] is not None]
        if con_mano < total * 0.5:
            self._marcar_invalido("Se perdieron las manos, repita")
            return False
        if len(cuerpos) < total * 0.5:
            self._marcar_invalido("No se detecto el cuerpo, repita")
            return False

        def visibles(indice_a, indice_b) -> int:
            return sum(
                1 for c in cuerpos
                if c[indice_a][3] >= VISIBILIDAD_MINIMA_POSE
                and c[indice_b][3] >= VISIBILIDAD_MINIMA_POSE
            )

        if visibles(CUERPO_CODO_IZQ, CUERPO_CODO_DER) < len(cuerpos) * 0.5:
            self._marcar_invalido("Codos fuera de cuadro, alejese un poco")
            return False
        if visibles(CUERPO_CADERA_IZQ, CUERPO_CADERA_DER) < len(cuerpos) * 0.5:
            self._marcar_invalido("Caderas fuera de cuadro, alejese un poco")
            return False
        return True

    def _componer_vector(self, manos) -> list:
        """Compone el vector del fotograma a partir de las manos detectadas.

        Sin Pose: 126 valores (solo manos, como el alfabeto). Con Pose (Modelo B):
        132 valores, agregando la ubicacion de las manos en el marco del cuerpo.
        """
        izquierda, derecha = self._asignar_lateralidad(manos)
        if self.detector_pose is not None:
            return componer_vector_modelo_b(izquierda, derecha, self._marco_actual)
        return componer_vector_dos_manos(izquierda, derecha)

    @property
    def _mensaje_fin(self) -> str:
        return f"Se guardaron {self._total_guardado} muestras de {self.persona}."

    def _cambiar_estado(self, estado, ahora) -> None:
        self.estado = estado
        self.t_estado = ahora

    def _esta_completa(self, clase: str) -> bool:
        existentes = self.escritor.muestras_existentes(clase, self.persona)
        return existentes >= self.params.muestras_por_clase

    def _siguiente_clase_pendiente(self, desde: int):
        """Devuelve el indice de la primera clase no completa, o None."""
        for idx in range(desde, len(self.clases)):
            if not self._esta_completa(self.clases[idx]):
                return idx
        return None

    @staticmethod
    def _submuestrear(vectores: list, n: int) -> list:
        """Toma hasta n elementos espaciados uniformemente de la lista."""
        if n <= 0 or not vectores:
            return []
        if len(vectores) <= n:
            return list(vectores)
        paso = len(vectores) / n
        return [vectores[int(i * paso)] for i in range(n)]

    # Mensaje de la ultima toma invalida (se inicializa por si se consulta antes).
    _mensaje_invalido = "Repita la toma"

    def _abrir_camara(self):
        cap = cv2.VideoCapture(self.params.indice_camara)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.params.ancho_camara)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.params.alto_camara)
        if not cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la camara (indice {self.params.indice_camara}). "
                "Verifique que este conectada y no la use otra aplicacion."
            )
        return cap
