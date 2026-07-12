import 'package:flutter/services.dart';
import 'package:lesho_app/core/constantes.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// Error lanzado cuando un modelo no está disponible en los assets.
class ModeloNoDisponibleError extends Error {
  final String mensaje;
  ModeloNoDisponibleError(this.mensaje);

  @override
  String toString() => 'ModeloNoDisponibleError: $mensaje';
}

/// Carga los modelos TFLite y sus etiquetas desde los assets de la app.
class CargadorModelos {
  Interpreter? _interpreteA;
  Interpreter? _interpreteB;
  List<String>? _etiquetasA;
  List<String>? _etiquetasB;

  bool _cargado = false;

  bool get estaCargado => _cargado;

  Interpreter get interpreteA {
    if (_interpreteA == null) throw StateError('Modelo A no cargado.');
    return _interpreteA!;
  }

  Interpreter get interpreteB {
    if (_interpreteB == null) throw StateError('Modelo B no cargado.');
    return _interpreteB!;
  }

  List<String> get etiquetasA {
    if (_etiquetasA == null) throw StateError('Etiquetas A no cargadas.');
    return _etiquetasA!;
  }

  List<String> get etiquetasB {
    if (_etiquetasB == null) throw StateError('Etiquetas B no cargadas.');
    return _etiquetasB!;
  }

  /// Indica si el Modelo B (señas dinámicas) está disponible.
  bool get tieneModeloB => _interpreteB != null;

  /// Carga el Modelo A (requerido) y, si está, el Modelo B (opcional).
  ///
  /// En esta etapa se prioriza el deletreo (Modelo A). El Modelo B se integra
  /// después (necesita MediaPipe Pose), así que si su .tflite todavía no está en
  /// los assets simplemente se ignora, sin romper la app.
  ///
  /// Lanza [ModeloNoDisponibleError] si falta el Modelo A.
  Future<void> cargar() async {
    try {
      _interpreteA = await Interpreter.fromAsset(Constantes.rutaModeloA);
    } on Exception catch (e) {
      throw ModeloNoDisponibleError(
        'No se pudo cargar el Modelo A. Debe entrenarse y copiarse a '
        'assets/models/ primero. Detalle: $e',
      );
    }
    _etiquetasA = await _cargarEtiquetas(Constantes.rutaEtiquetasA);

    // Modelo B: opcional por ahora. Si su .tflite todavía no está en los assets,
    // se ignora. Nota: un asset ausente se lanza como FlutterError (que es un
    // Error, NO un Exception), así que hay que capturar con un catch amplio; un
    // `on Exception` lo dejaría pasar y tumbaría todo el arranque.
    try {
      _interpreteB = await Interpreter.fromAsset(Constantes.rutaModeloB);
      _etiquetasB = await _cargarEtiquetas(Constantes.rutaEtiquetasB);
    } catch (_) {
      _interpreteB = null;
      _etiquetasB = null;
    }

    _cargado = true;
  }

  Future<List<String>> _cargarEtiquetas(String ruta) async {
    final texto = await rootBundle.loadString(ruta);
    return texto
        .split('\n')
        .map((l) => l.trim())
        .where((l) => l.isNotEmpty)
        .toList();
  }

  void liberar() {
    _interpreteA?.close();
    _interpreteB?.close();
    _interpreteA = null;
    _interpreteB = null;
    _cargado = false;
  }
}
