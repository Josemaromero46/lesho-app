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
/// CLAVE DE RENDIMIENTO: el procesamiento corre EN PARALELO con la grabación.
/// Mientras se graba, se muestrea un fotograma cada ~[_intervaloMuestreoMs] ms
/// (el modelo remuestrea a 40 de todas formas, capturar a 30 fps es de sobra) y
/// un consumidor lo va procesando con MediaPipe de inmediato, en orden. El Modelo
/// A no corre durante la grabación, así que el detector está libre. Al soltar el
/// botón solo queda drenar la cola final (los últimos fotogramas), preprocesar
/// (suavizado, recorte de quietos, escala, ubicación+relativos, remuestreo a 40)
/// y correr el Modelo B: la espera baja de decenas de segundos a unos pocos.
class ReconocedorPalabra {
  final ModeloB _modeloB;

  // Mínimo de fotogramas para intentar clasificar (a bajo fps una seña son pocos).
  static const int _minFrames = 8;
  // Muestreo temporal durante la grabación: un fotograma cada ~150 ms (~6.7 por
  // segundo). Es la perilla que intercambia densidad de la trayectoria contra
  // tiempo de espera al soltar; con el remuestreo a 40 del preprocesamiento, esta
  // densidad conserva la seña completa.
  static const int _intervaloMuestreoMs = 150;
  // Tope de fotogramas muestreados (memoria y tiempo; ~12 s de seña).
  static const int _maxMuestreados = 80;
  // La Pose corre cada este número de fotogramas procesados: el torso es estable
  // y el marco del cuerpo se reusa entre corridas (carry-forward).
  static const int _poseCada = 4;

  final List<_FrameCrudo> _cola = [];
  final List<List<double>> _secuencia = [];
  MarcoCuerpo? _ultimoMarco;
  bool _grabando = false;
  DetectorManos? _detector;
  Future<void>? _consumidor;
  double _ultimoMuestreoMs = 0;
  int _muestreados = 0;
  int _procesados = 0;
  double _tPrimero = 0;
  double _tUltimo = 0;

  /// Diagnóstico: cuántos fotogramas de la última grabación tuvieron cuerpo.
  int framesConMarco = 0;

  /// Aviso por cada detección procesada durante la grabación (para que la
  /// pantalla dibuje el esqueleto sin correr detecciones aparte).
  void Function(ResultadoDeteccion d)? onDeteccionVivo;

  /// Factor de corrección de aspecto (se fija desde el detector en vivo).
  double correccionAspecto = Constantes.correccionAspectoX;

  ReconocedorPalabra(this._modeloB);

  bool get grabando => _grabando;
  int get frames => _secuencia.length;
  int get framesCrudos => _muestreados;

  /// Abre la grabación. El [detector] queda fijo para el consumidor en paralelo.
  void iniciar(DetectorManos detector) {
    _detector = detector;
    _cola.clear();
    _secuencia.clear();
    _ultimoMarco = null;
    framesConMarco = 0;
    _muestreados = 0;
    _procesados = 0;
    _tPrimero = 0;
    _tUltimo = 0;
    _ultimoMuestreoMs = 0;
    _grabando = true;
  }

  /// Verdadero si este fotograma cae en el muestreo temporal. La pantalla lo
  /// consulta ANTES de copiar los bytes, para no copiar los fotogramas que no se
  /// van a usar (la cámara reutiliza su buffer, copiar cuesta).
  bool necesitaFrame() {
    if (!_grabando || _muestreados >= _maxMuestreados) return false;
    return _muestreados == 0 ||
        _ahora() - _ultimoMuestreoMs >= _intervaloMuestreoMs;
  }

  /// Encola un fotograma muestreado (los bytes deben ser copia propia) y arranca
  /// el consumidor si no está corriendo.
  void agregarFrameCrudo(Uint8List bytes, int width, int height) {
    if (!_grabando || _muestreados >= _maxMuestreados) return;
    _ultimoMuestreoMs = _ahora();
    _muestreados++;
    _cola.add(_FrameCrudo(bytes, width, height, _ultimoMuestreoMs));
    _consumidor ??= _consumir();
  }

  /// Consumidor: procesa la cola con MediaPipe, un fotograma a la vez y en orden,
  /// mientras la grabación sigue. Termina cuando la cola queda vacía (y se
  /// relanza si llega otro fotograma).
  Future<void> _consumir() async {
    final detector = _detector;
    if (detector == null) {
      _consumidor = null;
      return;
    }
    while (_cola.isNotEmpty) {
      final c = _cola.removeAt(0);
      try {
        final d = await detector.procesarBytes(c.bytes, c.width, c.height,
            conPose: _procesados % _poseCada == 0);
        _procesados++;
        if (_tPrimero == 0) _tPrimero = c.tMs;
        _tUltimo = c.tMs;
        correccionAspecto = detector.factorAspecto;
        _agregarDeteccion(d);
        onDeteccionVivo?.call(d);
      } catch (_) {
        // Un fotograma fallido no debe tumbar la palabra completa.
      }
    }
    _consumidor = null;
  }

  /// Cierra la grabación, espera a que el consumidor drene la cola restante y
  /// clasifica la secuencia con el Modelo B.
  Future<ResultadoPalabra?> detener() async {
    _grabando = false;
    final crono = Stopwatch()..start();
    final colaFinal = _cola.length;

    // Drena lo que falte (con la grabación cerrada ya no entran fotogramas).
    while (_consumidor != null) {
      await _consumidor;
    }

    // Si tras descartar los fotogramas sin manos quedan muy pocos, la seña no se
    // capturó bien (no se hizo, o se perdió la mano casi todo el tiempo). Es mejor
    // no responder que devolver una clase basura (típicamente AGUA).
    if (_secuencia.length < _minFrames) return null;

    final n = _secuencia.length;
    final dt = n > 1
        ? ((_tUltimo - _tPrimero) / (n - 1) / 1000.0).clamp(1.0 / 60.0, 0.5)
        : 1.0 / 30.0;

    final procesada = procesarSecuenciaB(_secuencia, dt: dt);
    final probs = _modeloB.predecir(procesada);

    // Argmax sobre las 50 clases. Ya no hay filtro: el modelo tiene datos de todas.
    var mejor = -1;
    var mejorP = -1.0;
    for (var i = 0; i < probs.length; i++) {
      if (probs[i] > mejorP) {
        mejorP = probs[i];
        mejor = i;
      }
    }
    // Diagnóstico: espera real al soltar, tamaño de la secuencia y cuántos
    // fotogramas seguían en cola al soltar (mide qué tanto se adelantó el
    // consumidor durante la grabación).
    debugPrint('LESHO_T detener ${crono.elapsedMilliseconds}ms '
        'n=$n cola_final=$colaFinal dt=${dt.toStringAsFixed(4)}');
    if (mejor < 0) return null;
    return ResultadoPalabra(_modeloB.etiquetas[mejor], mejorP);
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
    // Solo se agregan fotogramas con AL MENOS UNA MANO, igual que el capturador
    // del dataset (que solo guardaba fotogramas con manos). Un fotograma sin manos
    // entra como ceros: el modelo lo interpreta como AGUA/PAPA y, cuando son una
    // fracción grande (teléfono lento, o detección perdida durante el movimiento),
    // arrastran el resultado a la clase equivocada. Validado en Python contra el
    // modelo: filtrarlos recupera la respuesta correcta (HOLA que caía en AGUA
    // vuelve a HOLA; PAPA que caía en AGUA vuelve a PAPA).
    if (izq == null && der == null) return;
    if (_ultimoMarco != null) framesConMarco++;
    _secuencia.add(componerVectorModeloB(izq, der, _ultimoMarco));
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
