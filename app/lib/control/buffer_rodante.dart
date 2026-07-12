import 'package:lesho_app/core/constantes.dart';

/// Buffer rodante de los últimos N fotogramas de landmarks.
///
/// Mantiene una ventana deslizante de tamaño [Constantes.tamanoVentanaA] que
/// alimenta al Modelo A. En cada fotograma se agrega el vector nuevo y, si se
/// supera la capacidad, se descarta el más viejo. El Modelo A solo clasifica
/// cuando la ventana está llena (ver [listo]).
///
/// Con esta ventana, el mismo modelo reconoce tanto las letras estáticas
/// (ventana casi quieta) como las de movimiento (J, Ñ, Z, LL, RR), sin usar
/// las señas de INICIO y FIN para el deletreo.
class BufferRodante {
  final int _capacidad;
  final List<List<double>> _frames = [];

  /// [capacidad] es el número de fotogramas de la ventana. Por defecto usa
  /// [Constantes.tamanoVentanaA], que debe coincidir con TAMANO_VENTANA_A del
  /// pipeline de entrenamiento.
  BufferRodante([int? capacidad])
      : _capacidad = capacidad ?? Constantes.tamanoVentanaA;

  /// Agrega el vector de un fotograma (126 valores), descartando el más viejo
  /// si la ventana ya estaba llena.
  void agregar(List<double> vector) {
    _frames.add(List<double>.from(vector));
    if (_frames.length > _capacidad) {
      _frames.removeAt(0);
    }
  }

  /// True cuando la ventana ya tiene los N fotogramas que espera el Modelo A.
  bool get listo => _frames.length >= _capacidad;

  /// Número de fotogramas acumulados hasta ahora.
  int get longitud => _frames.length;

  /// Copia de la ventana actual como matriz N × 126. Solo tiene sentido usarla
  /// cuando [listo] es true.
  List<List<double>> get ventana =>
      _frames.map((f) => List<double>.from(f)).toList();

  /// Vacía el buffer. Se usa al cambiar de estado para que la ventana no
  /// mezcle fotogramas de antes y después de la transición.
  void limpiar() => _frames.clear();
}
