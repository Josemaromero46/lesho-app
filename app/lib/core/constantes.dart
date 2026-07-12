/// Constantes globales del sistema LESHO.
///
/// Único lugar donde se definen los parámetros que controlan el
/// comportamiento del pipeline de reconocimiento y los modelos.
/// Cualquier otro archivo importa desde aquí; no se escriben estos
/// números a mano en otros lugares.
abstract final class Constantes {
  // -------------------------------------------------------------------------
  // Estructura de landmarks (fijada por MediaPipe Hands)
  // -------------------------------------------------------------------------
  static const int numLandmarks = 21;
  static const int numCoordenadas = 3;
  static const int numManos = 2;
  static const int tamanoVectorMano = 63; // 21 × 3
  static const int tamanoVector = 126; // 2 × 63

  // -------------------------------------------------------------------------
  // Clases de los modelos
  // -------------------------------------------------------------------------
  static const int numClasesA = 33; // 30 letras (con CH, LL, RR) + INICIO + FIN + REPOSO
  static const int numClasesB = 50; // señas dinámicas

  // -------------------------------------------------------------------------
  // Modelo B (señas dinámicas): vector de entrada por fotograma
  // -------------------------------------------------------------------------
  // Config de las 2 manos (126) + ubicación en el cuerpo (18) = 144. Réplica del
  // contrato de definiciones.py / vector_modelo_b.py.
  static const int indicePuntaIndice = 8; // landmark de la punta del índice
  // 9 puntos x (rx, ry): 2 muñecas + 2 puntas del índice + 5 anclas de la cara.
  static const int tamanoUbicacion = 18;
  static const int tamanoVectorB = 126 + tamanoUbicacion; // 144
  static const int tamanoRelativo = 8; // (puntaIzq-nariz, puntaIzq-boca, puntaDer-...)
  static const int tamanoEntradaB = tamanoVectorB + tamanoRelativo; // 152 (lo que ve el modelo)
  static const double gananciaUbicacion = 2.5; // pesa el lugar frente a la forma

  // Índices de MediaPipe Pose (33 puntos) usados por el marco del cuerpo.
  static const int poseNariz = 0;
  static const int poseOjoIzq = 2;
  static const int poseOjoDer = 5;
  static const int poseOrejaIzq = 7;
  static const int poseOrejaDer = 8;
  static const int poseBocaIzq = 9;
  static const int poseBocaDer = 10;
  static const int poseHombroIzq = 11;
  static const int poseHombroDer = 12;
  static const int numLandmarksPose = 33;
  static const double visibilidadMinimaPose = 0.5;

  /// Fotogramas de la ventana que entra al Modelo A.
  ///
  /// El Modelo A no clasifica un fotograma suelto sino una ventana de los
  /// últimos [tamanoVentanaA] fotogramas, para reconocer tanto las letras
  /// estáticas como las de movimiento (J, Ñ, Z, LL, RR). Entrada del modelo:
  /// [tamanoVentanaA] × [tamanoVector]. Debe coincidir con TAMANO_VENTANA_A del
  /// pipeline de entrenamiento. En cada fotograma el modelo corre sobre toda la
  /// ventana, así que este valor es la palanca principal de costo en el teléfono.
  static const int tamanoVentanaA = 20;

  // Nombres de las clases de control del Modelo A.
  static const String claseInicio = 'INICIO';
  static const String claseFin = 'FIN';
  static const String claseReposo = 'REPOSO';

  /// Letras del alfabeto que se ejecutan CON movimiento (J, Ñ, Z, LL, RR).
  /// Se usan para la compuerta de movimiento (ver [umbralMovimientoAbs]).
  static const List<String> letrasConMovimiento = ['J', 'Ñ', 'Z', 'LL', 'RR'];

  /// Letras estáticas GEMELAS de una de movimiento (N de Ñ, L de LL, R de RR).
  /// Se bloquean cuando la mano se mueve claro (ver [umbralMovimientoMoviendo]).
  static const List<String> estaticasGemelas = ['N', 'L', 'R'];

  // -------------------------------------------------------------------------
  // Filtros de reconocimiento (deben coincidir con demo_deletreo.py)
  // -------------------------------------------------------------------------

  /// Ventanas consecutivas con la misma clase para confirmar una detección.
  static const int fotogramasPersistencia = 5;

  /// Confianza mínima del Modelo A por ventana para contar hacia la persistencia.
  static const double umbralConfianza = 0.60;

  /// Milisegundos bloqueados tras una detección confirmada. Da tiempo a ver la
  /// letra escrita y formar la siguiente; permite repetir la misma letra.
  static const int cooldownMs = 1200;

  /// Confianza mínima de MediaPipe para aceptar una mano detectada.
  static const double confianzaDeteccionMano = 0.60;

  // -------------------------------------------------------------------------
  // Corrección de relación de aspecto (CRÍTICO para que el modelo reconozca)
  // -------------------------------------------------------------------------

  /// Relación de aspecto (ancho/alto) con la que se entrenó el modelo: cámara
  /// 1280x720. MediaPipe normaliza X por el ancho e Y por el alto por separado, así
  /// que la forma de la mano depende del aspecto de la imagen.
  static const double aspectoEntrenamiento = 1280.0 / 720.0;

  /// Corrección de aspecto POR DEFECTO (teléfono típico 3:4). CRÍTICA para que el
  /// modelo reconozca: en el teléfono la imagen llega vertical 3:4, distinto al
  /// 16:9 del entrenamiento, y sin corregir la mano le llega estirada y confunde
  /// letras (verificado: 30/30 a 16:9 caen a 15/30 a 3:4, vuelven a 30/30 al
  /// corregir). Se multiplican X y Z por (aspecto_cámara / aspecto_entrenamiento).
  /// En la app se calcula EN VIVO con las dimensiones reales del fotograma (ver
  /// DetectorManos.factorAspecto); este valor es solo el respaldo típico.
  static const double correccionAspectoX = (3.0 / 4.0) / aspectoEntrenamiento;

  // -------------------------------------------------------------------------
  // Suavizado temporal (filtro One Euro) — igual que en el entrenamiento
  // -------------------------------------------------------------------------

  /// Frecuencia de corte mínima (Hz). Gobierna el suavizado con la mano QUIETA.
  static const double suavizadoMinCutoff = 0.5;

  /// Ganancia por velocidad. Gobierna el suavizado con la mano en MOVIMIENTO.
  static const double suavizadoBeta = 10.0;

  /// Corte del suavizado de la derivada (Hz). Valor típico.
  static const double suavizadoDCutoff = 1.0;

  // -------------------------------------------------------------------------
  // Compuerta de movimiento (evita N->Ñ, L->LL, R->RR)
  // -------------------------------------------------------------------------

  /// Movimiento por debajo del cual la ventana se considera QUIETA: se bloquean
  /// las letras de movimiento (una N quieta no puede leerse como Ñ).
  static const double umbralMovimientoAbs = 0.0022;

  /// Movimiento por encima del cual la mano se mueve CLARO: se bloquean las
  /// estáticas gemelas (en pleno vaivén de una Ñ no puede ganar la N).
  static const double umbralMovimientoMoviendo = 0.006;

  // -------------------------------------------------------------------------
  // Cámara y secuencias
  // -------------------------------------------------------------------------
  static const int fpsObjetivo = 30;
  static const int minFramesSecuencia = 30;
  static const int maxFramesSecuencia = 60;

  // Longitud fija de secuencia para el Modelo B (padding/truncado).
  static const int longitudFijaSecuencia = 40;

  // -------------------------------------------------------------------------
  // Rutas de assets
  // -------------------------------------------------------------------------
  static const String rutaModeloA = 'assets/models/modelo_a.tflite';
  static const String rutaModeloB = 'assets/models/modelo_b.tflite';
  static const String rutaEtiquetasA = 'assets/labels/etiquetas_a.txt';
  static const String rutaEtiquetasB = 'assets/labels/etiquetas_b.txt';
  static const String directorioVideos = 'assets/videos/';
}
