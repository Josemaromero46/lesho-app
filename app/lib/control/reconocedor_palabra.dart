import 'dart:typed_data';

import 'package:flutter/foundation.dart';

import 'package:lesho_app/control/secuencias_b.dart';
import 'package:lesho_app/core/constantes.dart';
import 'package:lesho_app/core/marco.dart';
import 'package:lesho_app/core/normalizacion.dart';
import 'package:lesho_app/core/vector_modelo_b.dart';
import 'package:lesho_app/inferencia/modelo_b.dart';
import 'package:lesho_app/landmarks/detector_manos.dart';

class ResultadoPalabra {
  final String sena;
  final double confianza;
  const ResultadoPalabra(this.sena, this.confianza);
}

/// Un fotograma crudo guardado durante la grabación (NV21 + dimensiones + tiempo).
class _FrameCrudo {
  final Uint8List bytes;
  final int width;
  final int height;
  final double tMs;
  const _FrameCrudo(this.bytes, this.width, this.height, this.tMs);
}

/// Graba una seña dinámica y la clasifica con el Modelo B.
///
/// CLAVE DE RENDIMIENTO: mientras graba, SOLO guarda los fotogramas crudos de la
/// cámara (rápido, a la velocidad de la cámara sin perder movimiento). MediaPipe
/// (manos + pose) es lento en gama baja, así que correrlo en vivo perdería
/// fotogramas y se comería el movimiento. Al soltar, procesa TODOS los fotogramas
/// guardados fuera de línea (en orden, con la Pose cada 3 fotogramas): arma el
/// vector por fotograma (corrección de aspecto + último marco del cuerpo),
/// preprocesa (suavizado, recorte de quietos, escala, ubicación+relativos,
/// remuestreo a 40) y corre el Modelo B. Así el movimiento se captura a fondo
/// aunque el teléfono sea lento.
class ReconocedorPalabra {
  final ModeloB _modeloB;

  // Mínimo de fotogramas para intentar clasificar (a bajo fps una seña son pocos).
  static const int _minFrames = 8;
  // Tope de fotogramas guardados (memoria): ~30 fps x 7 s.
  static const int _maxCrudos = 210;

  // Solo estas 10 señas están entrenadas en el mini-modelo; la salida del modelo
  // (50 clases) se enmascara a estas para no elegir una clase sin datos.
  static const List<String> clasesEntrenadas = [
    'HOLA', 'AGUA', 'AYUDA', 'MAMA', 'PAPA',
    'JUGAR', 'CASA', 'HOSPITAL', 'TRABAJO', 'NOCHE',
  ];

  final List<_FrameCrudo> _crudos = [];
  final List<List<double>> _secuencia = [];
  MarcoCuerpo? _ultimoMarco;
  bool _grabando = false;

  /// Diagnóstico: cuántos fotogramas de la última grabación tuvieron cuerpo.
  int framesConMarco = 0;

  /// Factor de corrección de aspecto (se fija desde el detector en vivo).
  double correccionAspecto = Constantes.correccionAspectoX;

  List<int>? _idxPresentes;

  ReconocedorPalabra(this._modeloB);

  bool get grabando => _grabando;
  int get frames => _secuencia.length;
  int get framesCrudos => _crudos.length;

  void iniciar() {
    _crudos.clear();
    _secuencia.clear();
    _ultimoMarco = null;
    framesConMarco = 0;
    _grabando = true;
  }

  /// Guarda un fotograma CRUDO (rápido). Los bytes deben ser una copia propia (la
  /// cámara reutiliza su buffer). No corre MediaPipe: eso se hace al detener.
  void agregarFrameCrudo(Uint8List bytes, int width, int height) {
    if (!_grabando || _crudos.length >= _maxCrudos) return;
    _crudos.add(_FrameCrudo(bytes, width, height, _ahora()));
  }

  /// Detiene la grabación, procesa TODOS los fotogramas guardados con MediaPipe y
  /// clasifica. Es async porque corre el detector sobre cada fotograma.
  Future<ResultadoPalabra?> detener(DetectorManos detector) async {
    _grabando = false;
    final crono = Stopwatch()..start();
    _secuencia.clear();
    _ultimoMarco = null;
    framesConMarco = 0;

    if (_crudos.length < _minFrames) {
      _crudos.clear();
      return null;
    }

    final tIni = _crudos.first.tMs;
    final tFin = _crudos.last.tMs;
    for (var i = 0; i < _crudos.length; i++) {
      final c = _crudos[i];
      // Pose solo cada 3 fotogramas (el cuerpo es estable, se reusa el marco).
      final d = await detector.procesarBytes(c.bytes, c.width, c.height,
          conPose: i % 3 == 0);
      correccionAspecto = detector.factorAspecto;
      _agregarDeteccion(d);
    }
    _crudos.clear();

    final n = _secuencia.length;
    final dt = n > 1
        ? ((tFin - tIni) / (n - 1) / 1000.0).clamp(1.0 / 60.0, 0.5)
        : 1.0 / 30.0;

    final procesada = procesarSecuenciaB(_secuencia, dt: dt);
    final probs = _modeloB.predecir(procesada);

    _idxPresentes ??= _calcularPresentes();
    var mejor = -1;
    var mejorP = -1.0;
    var suma = 0.0;
    for (final i in _idxPresentes!) {
      suma += probs[i];
      if (probs[i] > mejorP) {
        mejorP = probs[i];
        mejor = i;
      }
    }
    // Diagnóstico liviano: tiempo desde soltar hasta el resultado y ritmo de fps.
    debugPrint('LESHO_T detener ${crono.elapsedMilliseconds}ms '
        'n=$n dt=${dt.toStringAsFixed(4)}');
    if (mejor < 0 || suma <= 0) return null;
    return ResultadoPalabra(_modeloB.etiquetas[mejor], mejorP / suma);
  }

  // Arma el vector de 140 de un fotograma a partir de su detección (manos + pose),
  // corrige el aspecto y mantiene el último marco válido del cuerpo.
  void _agregarDeteccion(ResultadoDeteccion d) {
    final izq = _corregirMano(d.manoIzquierda);
    final der = _corregirMano(d.manoDerecha);
    if (d.pose != null) {
      final marco = marcoDesdePuntos(_corregirPose(d.pose!));
      if (marco != null) _ultimoMarco = marco;
    }
    if (_ultimoMarco != null) framesConMarco++;
    _secuencia.add(componerVectorModeloB(izq, der, _ultimoMarco));
  }

  List<int> _calcularPresentes() {
    final etiquetas = _modeloB.etiquetas;
    final indices = <int>[];
    for (final c in clasesEntrenadas) {
      final i = etiquetas.indexOf(c);
      if (i >= 0) indices.add(i);
    }
    return indices;
  }

  List<Punto>? _corregirMano(List<Punto>? mano) {
    if (mano == null) return null;
    final f = correccionAspecto;
    return [for (final p in mano) Punto(p.x * f, p.y, p.z * f)];
  }

  List<Punto3D> _corregirPose(List<Punto3D> pose) {
    final f = correccionAspecto;
    return [for (final p in pose) Punto3D(p.x * f, p.y, p.z, p.visibilidad)];
  }

  double _ahora() => DateTime.now().millisecondsSinceEpoch.toDouble();
}
