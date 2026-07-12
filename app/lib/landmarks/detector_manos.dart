import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:lesho_app/core/constantes.dart';
import 'package:lesho_app/core/marco.dart';
import 'package:lesho_app/core/normalizacion.dart';

/// Resultado de la detección en un fotograma: manos y, si se pidió, la pose.
class ResultadoDeteccion {
  /// Landmarks de la mano etiquetada como izquierda por MediaPipe, o null.
  ///
  /// NOTA: por el efecto espejo (vista selfie), la etiqueta de MediaPipe no
  /// coincide con la mano física real, pero es consistente entre la captura del
  /// dataset y la app, que es lo que importa para el modelo.
  final List<Punto>? manoIzquierda;

  /// Landmarks de la mano etiquetada como derecha por MediaPipe, o null.
  final List<Punto>? manoDerecha;

  /// Los 33 puntos de la pose (Modelo B), o null si no se pidió o no se detectó.
  final List<Punto3D>? pose;

  const ResultadoDeteccion({
    this.manoIzquierda,
    this.manoDerecha,
    this.pose,
  });

  bool get hayMano => manoIzquierda != null || manoDerecha != null;
}

/// Detecta manos con MediaPipe HandLandmarker y extrae los 21 landmarks por mano.
///
/// La detección real corre en el lado nativo (Android/Kotlin, MediaPipe Tasks) a
/// través del canal de métodos "lesho/manos". Esta clase arma el mensaje, invoca
/// al nativo y traduce el resultado a [ResultadoDeteccion]. Se usa el mismo
/// modelo (hand_landmarker.task) y la misma configuración que el pipeline de
/// entrenamiento, para no romper la paridad con el modelo.
class DetectorManos {
  static const MethodChannel _canal = MethodChannel('lesho/manos');
  bool _inicializado = false;

  /// Rotación (grados) que el nativo aplica al fotograma para ponerlo vertical.
  /// Se fija con la orientación del sensor de la cámara frontal.
  int rotacion = 270;

  /// Factor de corrección de aspecto calculado en vivo con las dimensiones reales
  /// del fotograma. Sirve para cualquier teléfono, sin importar la proporción de
  /// su cámara. Se actualiza en cada [procesar]. Ver [Constantes.correccionAspectoX].
  double factorAspecto = Constantes.correccionAspectoX;

  bool get estaInicializado => _inicializado;

  /// Inicializa el detector nativo. [numManos] es 1 para el deletreo (alfabeto de
  /// una mano) y 2 para las señas dinámicas (bimanuales). [conPose] activa
  /// MediaPipe Pose para la ubicación en el cuerpo (Modelo B).
  Future<void> inicializar({int numManos = 1, bool conPose = false}) async {
    await _canal.invokeMethod('inicializar', {
      'numManos': numManos,
      'conPose': conPose,
    });
    _inicializado = true;
  }

  /// Procesa un fotograma NV21 y devuelve los landmarks detectados.
  ///
  /// Retorna [ResultadoDeteccion] vacío si no se detectó ninguna mano.
  Future<ResultadoDeteccion> procesar(CameraImage imagen,
      {bool conPose = true}) {
    // Con ImageFormatGroup.nv21 el fotograma llega en un solo plano.
    return procesarBytes(
        imagen.planes.first.bytes, imagen.width, imagen.height,
        conPose: conPose);
  }

  /// Procesa un fotograma NV21 dado en bytes (para procesar fotogramas guardados
  /// fuera de línea: se capturan a la velocidad de la cámara y se corren después).
  Future<ResultadoDeteccion> procesarBytes(
      Uint8List bytes, int width, int height,
      {bool conPose = true}) async {
    if (!_inicializado) throw StateError('DetectorManos no inicializado.');

    // Aspecto REAL de la imagen tal como la ve MediaPipe (ya rotada). Con rotación
    // de 90 o 270 grados el ancho y el alto se intercambian. Así el factor sirve
    // para cualquier teléfono, sin depender de una proporción fija.
    final rotado = rotacion % 180 == 90;
    final aspectoImagen = rotado ? height / width : width / height;
    factorAspecto = aspectoImagen / Constantes.aspectoEntrenamiento;

    final salida = await _canal.invokeMethod<Map<dynamic, dynamic>>('detectar', {
      'bytes': bytes,
      'width': width,
      'height': height,
      'rotation': rotacion,
      'conPose': conPose,
    });

    if (salida == null) return const ResultadoDeteccion();
    final manos = (salida['manos'] as List<dynamic>?) ?? const [];
    final pose = _aPuntosPose(salida['pose']);
    return _asignarManos(manos, pose);
  }

  /// Convierte los 132 valores planos (x,y,z,vis x 33) en 33 [Punto3D], o null.
  List<Punto3D>? _aPuntosPose(dynamic crudos) {
    if (crudos is! List) return null;
    final v = crudos.cast<double>();
    if (v.length != Constantes.numLandmarksPose * 4) return null;
    return [
      for (var i = 0; i < Constantes.numLandmarksPose; i++)
        Punto3D(v[i * 4], v[i * 4 + 1], v[i * 4 + 2], v[i * 4 + 3])
    ];
  }

  /// Separa las manos en (izquierda, derecha) por lateralidad. Réplica de
  /// `_asignar_manos` de la demo: "Left" va a la izquierda, "Right" a la derecha,
  /// y una mano de lateralidad desconocida o repetida cae en el primer slot libre
  /// (así una seña a una mano siempre produce datos).
  ResultadoDeteccion _asignarManos(List<dynamic> manos, List<Punto3D>? pose) {
    List<Punto>? izquierda;
    List<Punto>? derecha;
    final sinAsignar = <List<Punto>>[];

    for (final cruda in manos) {
      final mapa = (cruda as Map).cast<String, dynamic>();
      final puntos = _aPuntos(mapa['landmarks']);
      if (puntos == null) continue;
      final lateralidad = mapa['lateralidad'] as String? ?? 'Desconocida';

      if (lateralidad == 'Left' && izquierda == null) {
        izquierda = puntos;
      } else if (lateralidad == 'Right' && derecha == null) {
        derecha = puntos;
      } else {
        sinAsignar.add(puntos);
      }
    }
    for (final puntos in sinAsignar) {
      if (izquierda == null) {
        izquierda = puntos;
      } else if (derecha == null) {
        derecha = puntos;
      }
    }
    return ResultadoDeteccion(
      manoIzquierda: izquierda,
      manoDerecha: derecha,
      pose: pose,
    );
  }

  /// Convierte los 63 valores planos (x,y,z x 21) en una lista de 21 [Punto].
  List<Punto>? _aPuntos(dynamic crudos) {
    if (crudos is! List) return null;
    final valores = crudos.cast<double>();
    if (valores.length != Constantes.tamanoVectorMano) return null;
    final puntos = <Punto>[];
    for (var i = 0; i < Constantes.numLandmarks; i++) {
      puntos.add(Punto(valores[i * 3], valores[i * 3 + 1], valores[i * 3 + 2]));
    }
    return puntos;
  }

  /// Libera el recurso nativo.
  Future<void> liberar() async {
    if (!_inicializado) return;
    await _canal.invokeMethod('liberar');
    _inicializado = false;
  }
}
