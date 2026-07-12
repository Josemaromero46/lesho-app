import 'package:camera/camera.dart';

/// Gestiona la cámara frontal y entrega el flujo de fotogramas.
///
/// La app abre siempre la cámara frontal en modo selfie. El flip horizontal
/// se aplica ANTES de pasar el fotograma a MediaPipe (contrato compartido
/// con el pipeline de entrenamiento).
class ControladorCamara {
  CameraController? _controlador;
  bool _inicializado = false;
  bool _fluyendo = false;
  int _orientacionSensor = 270;

  bool get estaInicializado => _inicializado;
  bool get estaFluyendo => _fluyendo;
  CameraController? get controlador => _controlador;

  /// Orientación del sensor de la cámara frontal (grados). El nativo la usa para
  /// rotar el fotograma a vertical antes de pasarlo a MediaPipe.
  int get orientacionSensor => _orientacionSensor;

  /// Abre la cámara frontal.
  ///
  /// Usa resolución BAJA a propósito: MediaPipe extrae los landmarks igual de bien
  /// con poca resolución (internamente reescala la imagen), y con menos datos por
  /// fotograma la detección corre mucho más rápido en gama baja. Se pide NV21, que
  /// llega en un solo plano y es lo que el puente nativo convierte para MediaPipe.
  Future<void> inicializar() async {
    final camaras = await availableCameras();
    if (camaras.isEmpty) throw StateError('No se encontraron cámaras.');

    final camaraFrontal = camaras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => camaras.first,
    );
    _orientacionSensor = camaraFrontal.sensorOrientation;

    _controlador = CameraController(
      camaraFrontal,
      ResolutionPreset.low,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.nv21,
    );

    await _controlador!.initialize();
    _inicializado = true;
  }

  /// Inicia la entrega de fotogramas al [procesador].
  ///
  /// El [procesador] se llama en cada frame. Debe ser rápido o usar un
  /// flag para descartar frames mientras procesa el anterior.
  Future<void> iniciarFlujo(
    Future<void> Function(CameraImage imagen) procesador,
  ) async {
    if (!_inicializado) throw StateError('Cámara no inicializada.');
    if (_fluyendo) return;
    await _controlador!.startImageStream(procesador);
    _fluyendo = true;
  }

  /// Detiene la entrega de fotogramas.
  Future<void> detenerFlujo() async {
    if (!_fluyendo) return;
    await _controlador!.stopImageStream();
    _fluyendo = false;
  }

  /// Libera todos los recursos.
  Future<void> liberar() async {
    if (_fluyendo) await detenerFlujo();
    await _controlador?.dispose();
    _controlador = null;
    _inicializado = false;
  }
}
