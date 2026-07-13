import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/services.dart' show rootBundle;
import 'package:lesho_app/core/suavizado.dart';

/// Modelo y carga de un CLIP de seña (Dirección 2: texto -> seña).
///
/// Un clip es una seña grabada como secuencia de landmarks crudos (manos +
/// torso superior) en JSON, con el contrato de `training/comun/clips.py`
/// (versión 1). Esta clase es la réplica en Dart de `ClipPreparado` del visor
/// de Python (`training/demo/visor_clips.py`): carga el JSON, limpia la
/// secuencia (interpola huecos cortos de manos, cuerpo continuo, suavizado One
/// Euro) y la deja en "espacio físico" lista para que `MunecoPainter` la dibuje.
///
/// Espacio físico: MediaPipe normaliza x por el ancho de la imagen e y por el
/// alto por separado, así que x y z se multiplican por el aspecto del clip para
/// que las distancias sean reales y el muñeco no salga deformado (la misma
/// corrección de aspecto que usa el reconocimiento).

/// Índices de los 9 puntos de cuerpo del clip (contrato con clips.py).
class PuntosCuerpo {
  static const nariz = 0;
  static const hombroIzq = 1;
  static const hombroDer = 2;
  static const codoIzq = 3;
  static const codoDer = 4;
  static const munecaIzq = 5;
  static const munecaDer = 6;
  static const caderaIzq = 7;
  static const caderaDer = 8;
  static const total = 9;
}

/// Un punto 3D del clip, ya en espacio físico.
class P3 {
  final double x;
  final double y;
  final double z;
  const P3(this.x, this.y, this.z);
}

/// Un fotograma limpio del clip: cuerpo siempre presente, manos opcionales.
class FotogramaSena {
  final List<P3> cuerpo; // 9 puntos
  final List<P3>? manoIzq; // 21 landmarks o null
  final List<P3>? manoDer;
  const FotogramaSena(this.cuerpo, this.manoIzq, this.manoDer);
}

/// Parámetros del suavizado One Euro para reproducir el clip. Son los mismos
/// del pipeline (training/config.py); aquí solo alisan el dibujo, no alimentan
/// a ningún modelo.
const _suavizadoMinCutoff = 0.5;
const _suavizadoBeta = 10.0;
const _suavizadoDCutoff = 1.0;

/// Huecos de mano de hasta este tiempo se interpolan; más largos, la mano no
/// se dibuja en ese tramo.
const _maxHuecoManoSegundos = 0.35;

class ClipSena {
  final String palabra;
  final double fps;
  final double aspecto;
  final int numFrames;

  /// Secuencias limpias en espacio físico.
  final List<List<double>> _cuerpo; // T x 27 (9 puntos x 3)
  final List<List<double>?> _manoIzq; // T x (21 x 3) o null
  final List<List<double>?> _manoDer;

  /// Marco del clip: mediana del centro y el ancho de hombros (estables).
  final double centroX;
  final double centroY;
  final double anchoHombros;

  /// Caja que abarca todo el clip (cuerpo, manos y cabeza), para encuadrar.
  final double bboxMinX, bboxMinY, bboxMaxX, bboxMaxY;

  ClipSena._({
    required this.palabra,
    required this.fps,
    required this.aspecto,
    required this.numFrames,
    required List<List<double>> cuerpo,
    required List<List<double>?> manoIzq,
    required List<List<double>?> manoDer,
    required this.centroX,
    required this.centroY,
    required this.anchoHombros,
    required this.bboxMinX,
    required this.bboxMinY,
    required this.bboxMaxX,
    required this.bboxMaxY,
  })  : _cuerpo = cuerpo,
        _manoIzq = manoIzq,
        _manoDer = manoDer;

  /// Carga un clip desde los assets de la app.
  static Future<ClipSena> desdeAsset(String ruta) async {
    final texto = await rootBundle.loadString(ruta);
    return ClipSena.desdeJson(jsonDecode(texto) as Map<String, dynamic>);
  }

  /// Construye el clip desde el JSON ya decodificado, aplicando la limpieza.
  factory ClipSena.desdeJson(Map<String, dynamic> json) {
    if (json['version'] != 1) {
      throw FormatException('Version de clip no soportada: ${json['version']}');
    }
    final palabra = json['palabra'] as String;
    final fps = (json['fps'] as num).toDouble();
    final aspecto = (json['aspecto'] as num).toDouble();
    final framesJson = json['frames'] as List<dynamic>;
    final t = framesJson.length;

    // Extrae las secuencias crudas a listas planas, en espacio físico
    // (x y z multiplicados por el aspecto).
    final cuerpo = List<List<double>?>.filled(t, null);
    final manoIzq = List<List<double>?>.filled(t, null);
    final manoDer = List<List<double>?>.filled(t, null);

    // Cada punto trae [x, y, z, ...] (el cuerpo agrega visibilidad, que aqui
    // no se usa); se toman las tres primeras coordenadas.
    List<double> aplanar(List<dynamic> puntos) {
      final salida = <double>[];
      for (final punto in puntos) {
        final p = punto as List<dynamic>;
        salida.add((p[0] as num).toDouble() * aspecto);
        salida.add((p[1] as num).toDouble());
        salida.add((p[2] as num).toDouble() * aspecto);
      }
      return salida;
    }

    for (var i = 0; i < t; i++) {
      final frame = framesJson[i] as Map<String, dynamic>;
      if (frame['cuerpo'] != null) {
        cuerpo[i] = aplanar(frame['cuerpo'] as List<dynamic>);
      }
      if (frame['mano_izq'] != null) {
        manoIzq[i] = aplanar(frame['mano_izq'] as List<dynamic>);
      }
      if (frame['mano_der'] != null) {
        manoDer[i] = aplanar(frame['mano_der'] as List<dynamic>);
      }
    }

    // Limpieza: cuerpo continuo de punta a punta; manos con huecos cortos
    // interpolados; suavizado One Euro por tramo.
    final cuerpoLimpio = _limpiar(
      cuerpo, fps,
      maxHueco: null, extenderBordes: true,
      tamano: PuntosCuerpo.total * 3,
    );
    final maxHueco = math.max(1, (_maxHuecoManoSegundos * fps).round());
    final izqLimpia = _limpiar(manoIzq, fps,
        maxHueco: maxHueco, extenderBordes: false, tamano: 63);
    final derLimpia = _limpiar(manoDer, fps,
        maxHueco: maxHueco, extenderBordes: false, tamano: 63);

    // El cuerpo debe existir en todos los fotogramas tras la limpieza.
    final cuerpoFinal = <List<double>>[];
    for (final fila in cuerpoLimpio) {
      cuerpoFinal.add(fila ?? List<double>.filled(PuntosCuerpo.total * 3, 0));
    }

    // Marco: mediana del centro y el ancho de hombros.
    final centrosX = <double>[], centrosY = <double>[], anchos = <double>[];
    for (final fila in cuerpoFinal) {
      final hiX = fila[PuntosCuerpo.hombroIzq * 3];
      final hiY = fila[PuntosCuerpo.hombroIzq * 3 + 1];
      final hdX = fila[PuntosCuerpo.hombroDer * 3];
      final hdY = fila[PuntosCuerpo.hombroDer * 3 + 1];
      centrosX.add((hiX + hdX) / 2);
      centrosY.add((hiY + hdY) / 2);
      anchos.add(math.sqrt(math.pow(hiX - hdX, 2) + math.pow(hiY - hdY, 2)));
    }
    final centroX = _mediana(centrosX);
    final centroY = _mediana(centrosY);
    final anchoHombros = math.max(1e-6, _mediana(anchos));

    // Caja del clip (incluye el alcance de la cabeza sobre la nariz).
    var minX = double.infinity, minY = double.infinity;
    var maxX = -double.infinity, maxY = -double.infinity;
    void incluir(double x, double y) {
      minX = math.min(minX, x);
      minY = math.min(minY, y);
      maxX = math.max(maxX, x);
      maxY = math.max(maxY, y);
    }

    // 0.10 (centro de cabeza sobre la nariz) + 0.44 (radio) + margen.
    final alcanceCabeza = 0.60 * anchoHombros;
    for (var i = 0; i < t; i++) {
      final fila = cuerpoFinal[i];
      for (var p = 0; p < PuntosCuerpo.total; p++) {
        incluir(fila[p * 3], fila[p * 3 + 1]);
      }
      final narizX = fila[PuntosCuerpo.nariz * 3];
      final narizY = fila[PuntosCuerpo.nariz * 3 + 1];
      incluir(narizX - alcanceCabeza, narizY - alcanceCabeza);
      incluir(narizX + alcanceCabeza, narizY);
      for (final mano in [izqLimpia[i], derLimpia[i]]) {
        if (mano == null) continue;
        for (var p = 0; p < 21; p++) {
          incluir(mano[p * 3], mano[p * 3 + 1]);
        }
      }
    }

    return ClipSena._(
      palabra: palabra,
      fps: fps,
      aspecto: aspecto,
      numFrames: t,
      cuerpo: cuerpoFinal,
      manoIzq: izqLimpia,
      manoDer: derLimpia,
      centroX: centroX,
      centroY: centroY,
      anchoHombros: anchoHombros,
      bboxMinX: minX,
      bboxMinY: minY,
      bboxMaxX: maxX,
      bboxMaxY: maxY,
    );
  }

  /// Duración del clip en segundos.
  double get duracionSegundos => numFrames / fps;

  /// Fotograma interpolado en un índice flotante (reproducción fluida).
  ///
  /// Una mano solo se interpola si está presente en ambos fotogramas vecinos;
  /// si no, se usa la del fotograma disponible.
  FotogramaSena fotograma(double indice) {
    final i0 = indice.floor().clamp(0, numFrames - 1);
    final i1 = math.min(i0 + 1, numFrames - 1);
    final alfa = (indice - i0).clamp(0.0, 1.0);

    List<double> lerp(List<double> a, List<double> b) {
      final salida = List<double>.filled(a.length, 0);
      for (var i = 0; i < a.length; i++) {
        salida[i] = a[i] + (b[i] - a[i]) * alfa;
      }
      return salida;
    }

    List<P3> aPuntos(List<double> plano) {
      final puntos = <P3>[];
      for (var i = 0; i < plano.length; i += 3) {
        puntos.add(P3(plano[i], plano[i + 1], plano[i + 2]));
      }
      return puntos;
    }

    List<P3>? mano(List<List<double>?> serie) {
      final a = serie[i0], b = serie[i1];
      if (a != null && b != null) return aPuntos(lerp(a, b));
      if (a != null) return aPuntos(a);
      if (b != null) return aPuntos(b);
      return null;
    }

    return FotogramaSena(
      aPuntos(lerp(_cuerpo[i0], _cuerpo[i1])),
      mano(_manoIzq),
      mano(_manoDer),
    );
  }

  // -- Limpieza (réplica de _interpolar_huecos + _suavizar_tramos) ----------

  static List<List<double>?> _limpiar(
    List<List<double>?> serie,
    double fps, {
    required int? maxHueco,
    required bool extenderBordes,
    required int tamano,
  }) {
    final t = serie.length;
    final salida = List<List<double>?>.generate(
        t, (i) => serie[i] == null ? null : List<double>.from(serie[i]!));

    final presentes = <int>[];
    for (var i = 0; i < t; i++) {
      if (salida[i] != null) presentes.add(i);
    }
    if (presentes.isEmpty) return salida;

    // Interpola huecos interiores de hasta maxHueco fotogramas.
    for (var k = 0; k + 1 < presentes.length; k++) {
      final a = presentes[k], b = presentes[k + 1];
      final largo = b - a - 1;
      if (largo == 0) continue;
      if (maxHueco != null && largo > maxHueco) continue;
      for (var j = 1; j <= largo; j++) {
        final alfa = j / (largo + 1);
        final fila = List<double>.filled(tamano, 0);
        for (var c = 0; c < tamano; c++) {
          fila[c] = salida[a]![c] * (1 - alfa) + salida[b]![c] * alfa;
        }
        salida[a + j] = fila;
      }
    }

    if (extenderBordes) {
      final primero = presentes.first, ultimo = presentes.last;
      for (var i = 0; i < primero; i++) {
        salida[i] = List<double>.from(salida[primero]!);
      }
      for (var i = ultimo + 1; i < t; i++) {
        salida[i] = List<double>.from(salida[ultimo]!);
      }
    }

    // Suavizado One Euro por tramo contiguo presente (causal, como el visor).
    var inicio = 0;
    while (inicio < t) {
      if (salida[inicio] == null) {
        inicio++;
        continue;
      }
      var fin = inicio;
      while (fin < t && salida[fin] != null) {
        fin++;
      }
      if (fin - inicio >= 2) {
        final filtro = FiltroUnEuro(
          fps: fps,
          minCutoff: _suavizadoMinCutoff,
          beta: _suavizadoBeta,
          dCutoff: _suavizadoDCutoff,
        );
        for (var i = inicio; i < fin; i++) {
          salida[i] = filtro.filtrar(salida[i]!, preservarCeros: false);
        }
      }
      inicio = fin;
    }
    return salida;
  }

  static double _mediana(List<double> valores) {
    final orden = List<double>.from(valores)..sort();
    final n = orden.length;
    if (n == 0) return 0;
    return n.isOdd ? orden[n ~/ 2] : (orden[n ~/ 2 - 1] + orden[n ~/ 2]) / 2;
  }
}
