import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:lesho_app/texto_a_sena/clip_sena.dart';

/// Muñeco volumétrico de cápsulas (PLAN_DIRECCION2, sección 6).
///
/// Dibuja un fotograma de un [ClipSena] como figura de juguete: cápsulas
/// cónicas con luz fija, articulaciones soldadas, cabeza esférica sin rostro y
/// orden de dibujado por profundidad. Es la réplica en CustomPainter del
/// renderizador calibrado en `training/demo/visor_clips.py`; las proporciones
/// y el sombreado deben mantenerse iguales en ambos.
///
/// Sin plugins, sin motor 3D: solo Canvas de Flutter.

// Proporciones en fracciones del ANCHO DE HOMBROS (calibradas en el visor).
const _propBrazo = 0.34;
const _propAntebrazo = 0.25;
const _propCuello = 0.28;
const _propDedo = 0.10;
const _margenPalma = 1.15; // engorde del blob de la palma, en radios de dedo
const _bolaNudillo = 0.54; // radio de la bola de cada articulacion del dedo
const _propRadioCabeza = 0.44;
const _ovaloCabeza = 1.07; // cabeza levemente ovalada (mas alta que ancha)
const _propRadioTorso = 0.15;
const _angosteCaderas = 0.86; // caderas mas angostas que lo medido (maniqui)
const _propBolaCodo = 0.15; // radio de la bola de articulacion del codo
const _propBolaMuneca = 0.10; // radio de la bola de la muneca

// Torso de maniqui en TRES piezas (pecho ancho, cintura angosta, bloque de
// cadera). Cada tupla es (t inicio, t fin, ancho inicio, ancho fin): t recorre
// de la linea de hombros (0) a la de caderas (1) y el ancho es relativo al
// ancho local interpolado. Calibrado en el visor de Python (mantener paridad).
const _piezasTorso = [
  (0.00, 0.62, 1.00, 0.82), // pecho
  (0.50, 0.85, 0.62, 0.54), // cintura
  (0.74, 1.00, 1.00, 1.08), // cadera
];

/// Cuánto del ancho del lienzo ocupa el ancho de hombros del muñeco.
const _fraccionHombros = 0.335;

/// Posición vertical nominal del centro de hombros (fracción de la altura).
const _alturaHombros = 0.46;

/// Dirección de la luz (fija, arriba a la izquierda), normalizada.
const _luzX = -0.45, _luzY = -0.89;

/// Pseudo-profundidad de los dedos: más cerca de la cámara = más grueso.
const _gananciaZDedos = 5.0;
const _factorZMin = 0.72, _factorZMax = 1.30;

/// Cadenas de falanges de cada dedo y puntos del blob de la palma.
const _dedos = [
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16],
  [17, 18, 19, 20],
];
const _palma = [0, 1, 2, 5, 9, 13, 17];

/// Colores del muñeco. Azul de juguete por defecto (referencia del plan).
class ColoresMuneco {
  final Color cuerpo;
  final Color mano;
  const ColoresMuneco({required this.cuerpo, required this.mano});

  static const azul = ColoresMuneco(
    cuerpo: Color(0xFF387CC4),
    mano: Color(0xFF6CA8DE),
  );
}

/// Tres tonos derivados de un color base, para el sombreado por capas.
class _Tonos {
  final Color oscuro;
  final Color base;
  final Color claro;
  _Tonos(Color c)
      : oscuro = _mezclar(c, Colors.black, 0.40),
        base = c,
        claro = _mezclar(c, Colors.white, 0.38);

  static Color _mezclar(Color a, Color b, double t) => Color.lerp(a, b, t)!;
}

// Capas del sombreado: (tono, factor de radio, corrimiento hacia la luz).
const _capas = [(0, 1.00, 0.00), (1, 0.80, 0.16), (2, 0.46, 0.34)];

class MunecoPainter extends CustomPainter {
  final ClipSena clip;
  final FotogramaSena fotograma;

  /// Vista (lateralidad): la captura es en espejo (selfie), asi que el clip
  /// guarda la imagen especular del firmante. Con [vistaEspejo] en false (el
  /// valor por defecto) se voltea horizontalmente al dibujar: el muneco es una
  /// persona DE FRENTE que firma con su mano derecha real, como un interprete.
  /// Con true se dibuja tal cual se grabo (como un reflejo, util para que un
  /// nino imite la sena). Se valida con asesoria LESHO.
  final bool vistaEspejo;
  final ColoresMuneco colores;

  late final _Tonos _tonosCuerpo = _Tonos(colores.cuerpo);
  late final _Tonos _tonosMano = _Tonos(colores.mano);

  // Marco del fotograma actual (se calcula en paint segun el tamano real).
  double _escala = 1;
  double _yHombros = 0;
  double _s = 1; // ancho de hombros en pixeles
  Size _tamano = Size.zero;

  MunecoPainter({
    required this.clip,
    required this.fotograma,
    this.vistaEspejo = false,
    this.colores = ColoresMuneco.azul,
  });

  @override
  void paint(Canvas canvas, Size size) {
    _tamano = size;
    _prepararMarco(size);

    final cuerpo = fotograma.cuerpo;
    _cuello(canvas, cuerpo);
    _torso(canvas, cuerpo);
    _cabeza(canvas, cuerpo);

    // Brazos: primero el mas lejano (z de la muneca de Pose); las manos van
    // casi siempre delante del cuerpo, por eso se dibujan tras torso y cabeza.
    final zIzq = cuerpo[PuntosCuerpo.munecaIzq].z;
    final zDer = cuerpo[PuntosCuerpo.munecaDer].z;
    if (zIzq >= zDer) {
      _brazo(canvas, cuerpo, fotograma.manoIzq, izquierdo: true);
      _brazo(canvas, cuerpo, fotograma.manoDer, izquierdo: false);
    } else {
      _brazo(canvas, cuerpo, fotograma.manoDer, izquierdo: false);
      _brazo(canvas, cuerpo, fotograma.manoIzq, izquierdo: true);
    }
  }

  // -- Marco y mapeo ---------------------------------------------------------

  void _prepararMarco(Size size) {
    const margen = 16.0;
    final piso = size.height - margen;
    var escala = _fraccionHombros * size.width / clip.anchoHombros;
    final anchoClip = clip.bboxMaxX - clip.bboxMinX;
    final altoClip = clip.bboxMaxY - clip.bboxMinY;
    if (anchoClip > 1e-6) {
      escala = math.min(escala, (size.width - 2 * margen) / anchoClip);
    }
    if (altoClip > 1e-6) {
      escala = math.min(escala, (piso - margen) / altoClip);
    }
    _escala = escala;

    var yHombros = size.height * _alturaHombros;
    final arriba = yHombros + (clip.bboxMinY - clip.centroY) * escala;
    final abajo = yHombros + (clip.bboxMaxY - clip.centroY) * escala;
    if (arriba < margen) {
      yHombros += margen - arriba;
    } else if (abajo > piso) {
      yHombros -= abajo - piso;
    }
    _yHombros = yHombros;
    _s = clip.anchoHombros * escala;
  }

  Offset _px(P3 p) {
    var dx = (p.x - clip.centroX) * _escala;
    // Sin vista espejo se voltea la x: el clip viene en espejo (selfie) y el
    // volteo lo convierte en una persona vista de frente.
    if (!vistaEspejo) dx = -dx;
    return Offset(_tamano.width / 2 + dx,
        _yHombros + (p.y - clip.centroY) * _escala);
  }

  // -- Primitivas con sombreado ----------------------------------------------

  /// Cadena de cápsulas cónicas sombreadas, dibujada POR CAPA para que las
  /// articulaciones queden fundidas sin costuras (un brazo o un dedo es una
  /// sola pieza continua).
  void _cadena(Canvas canvas, List<(Offset, Offset, double, double)> segmentos,
      _Tonos tonos) {
    for (final (indice, factor, corr) in _capas) {
      final color = switch (indice) {
        0 => tonos.oscuro,
        1 => tonos.base,
        _ => tonos.claro,
      };
      final pintura = Paint()
        ..color = color
        ..isAntiAlias = true;
      for (final (a, b, ra, rb) in segmentos) {
        final da = Offset(_luzX * ra * corr, _luzY * ra * corr);
        final db = Offset(_luzX * rb * corr, _luzY * rb * corr);
        _capsulaSolida(
            canvas, a + da, b + db, ra * factor, rb * factor, pintura);
      }
    }
  }

  /// Cápsula cónica plana: cuadrilátero + círculos en los extremos.
  void _capsulaSolida(
      Canvas canvas, Offset a, Offset b, double ra, double rb, Paint pintura) {
    final d = b - a;
    final largo = d.distance;
    if (largo > 1e-3) {
      final n = Offset(-d.dy / largo, d.dx / largo);
      final camino = Path()
        ..moveTo(a.dx + n.dx * ra, a.dy + n.dy * ra)
        ..lineTo(b.dx + n.dx * rb, b.dy + n.dy * rb)
        ..lineTo(b.dx - n.dx * rb, b.dy - n.dy * rb)
        ..lineTo(a.dx - n.dx * ra, a.dy - n.dy * ra)
        ..close();
      canvas.drawPath(camino, pintura);
    }
    canvas.drawCircle(a, math.max(1, ra), pintura);
    canvas.drawCircle(b, math.max(1, rb), pintura);
  }

  // -- Partes del muñeco -------------------------------------------------------

  Offset _centroCabeza(List<P3> cuerpo) =>
      _px(cuerpo[PuntosCuerpo.nariz]) + Offset(0, -0.10 * _s);

  void _cuello(Canvas canvas, List<P3> cuerpo) {
    // Antes del torso, para que este tape su base (sin "medallon" en el pecho).
    final hi = _px(cuerpo[PuntosCuerpo.hombroIzq]);
    final hd = _px(cuerpo[PuntosCuerpo.hombroDer]);
    final centro = Offset((hi.dx + hd.dx) / 2, (hi.dy + hd.dy) / 2);
    final r = _propCuello * _s / 2;
    _cadena(canvas, [(centro, _centroCabeza(cuerpo), r, r)], _tonosCuerpo);
  }

  void _cabeza(Canvas canvas, List<P3> cuerpo) {
    final centro = _centroCabeza(cuerpo);
    final radio = _propRadioCabeza * _s;
    // Esfera con degradado radial corrido hacia la luz, mas un brillo suave.
    // Levemente ovalada en vertical (silueta mas humana que una bola): se
    // dibuja el circulo con el canvas escalado en y alrededor del centro.
    final foco = centro + Offset(_luzX * radio * 0.35, _luzY * radio * 0.35);
    final pintura = Paint()
      ..isAntiAlias = true
      ..shader = ui.Gradient.radial(
        foco,
        radio * 1.55,
        [_tonosCuerpo.claro, _tonosCuerpo.base, _tonosCuerpo.oscuro],
        [0.0, 0.55, 1.0],
      );
    canvas.save();
    canvas.translate(centro.dx, centro.dy);
    canvas.scale(1.0, _ovaloCabeza);
    canvas.translate(-centro.dx, -centro.dy);
    canvas.drawCircle(centro, radio, pintura);
    canvas.restore();
    final brillo = Paint()
      ..isAntiAlias = true
      ..color = Color.lerp(_tonosCuerpo.claro, Colors.white, 0.55)!;
    canvas.drawCircle(
        centro + Offset(_luzX * radio * 0.42, _luzY * radio * 0.42),
        radio * 0.20,
        brillo);
  }

  void _torso(Canvas canvas, List<P3> cuerpo) {
    final hi = _px(cuerpo[PuntosCuerpo.hombroIzq]);
    final hd = _px(cuerpo[PuntosCuerpo.hombroDer]);
    var ci = _px(cuerpo[PuntosCuerpo.caderaIzq]);
    var cd = _px(cuerpo[PuntosCuerpo.caderaDer]);

    // Silueta de maniqui: las caderas se angostan respecto a lo medido, para
    // que el pecho domine y el torso no sea un bloque parejo.
    final medioCaderas = Offset((ci.dx + cd.dx) / 2, (ci.dy + cd.dy) / 2);
    ci = medioCaderas + (ci - medioCaderas) * _angosteCaderas;
    cd = medioCaderas + (cd - medioCaderas) * _angosteCaderas;

    // Torso de maniqui en TRES piezas (pecho, cintura, cadera). Cada pieza se
    // encoge hacia su centro y se dibuja rellena MAS trazada con esquinas
    // redondas (equivale a la dilatacion de la mascara en el visor de Python);
    // como las tres comparten el mismo degradado vertical, se funden sin
    // costuras y con filetes redondeados en los empalmes.
    final radio = math.max(2.0, _propRadioTorso * _s);
    final centroX = (hi.dx + hd.dx) / 2;

    final arriba = Color.lerp(_tonosCuerpo.base, _tonosCuerpo.claro, 0.35)!;
    final abajo = Color.lerp(_tonosCuerpo.base, _tonosCuerpo.oscuro, 0.75)!;
    final yArriba = math.min(hi.dy, hd.dy) - radio;
    final yAbajo = math.max(ci.dy, cd.dy) + radio;
    final degradado = ui.Gradient.linear(
      Offset(centroX, yArriba),
      Offset(centroX, yAbajo),
      [arriba, abajo],
    );
    final relleno = Paint()
      ..isAntiAlias = true
      ..shader = degradado;
    final borde = Paint()
      ..isAntiAlias = true
      ..shader = degradado
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2 * radio
      ..strokeJoin = StrokeJoin.round
      ..strokeCap = StrokeCap.round;

    (Offset, Offset) par(double t, double ancho) {
      final a = hi + (ci - hi) * t;
      final b = hd + (cd - hd) * t;
      final medio = Offset((a.dx + b.dx) / 2, (a.dy + b.dy) / 2);
      return (medio + (a - medio) * ancho, medio + (b - medio) * ancho);
    }

    for (final (t0, t1, w0, w1) in _piezasTorso) {
      final (a0, b0) = par(t0, w0);
      final (a1, b1) = par(t1, w1);
      final pieza = [a0, b0, b1, a1];
      final centroide = Offset(
        pieza.map((p) => p.dx).reduce((a, b) => a + b) / 4,
        pieza.map((p) => p.dy).reduce((a, b) => a + b) / 4,
      );
      final interior = pieza.map((p) {
        final hacia = centroide - p;
        final norma = math.max(1e-6, hacia.distance);
        // El encogimiento se acota para que una pieza angosta no colapse.
        final paso = math.min(radio, norma * 0.6);
        return p + hacia * (paso / norma);
      }).toList();
      final camino = Path()
        ..moveTo(interior[0].dx, interior[0].dy)
        ..lineTo(interior[1].dx, interior[1].dy)
        ..lineTo(interior[2].dx, interior[2].dy)
        ..lineTo(interior[3].dx, interior[3].dy)
        ..close();
      canvas.drawPath(camino, borde);
      canvas.drawPath(camino, relleno);
    }
  }

  void _brazo(Canvas canvas, List<P3> cuerpo, List<P3>? mano,
      {required bool izquierdo}) {
    final hombro = _px(
        cuerpo[izquierdo ? PuntosCuerpo.hombroIzq : PuntosCuerpo.hombroDer]);
    final codo =
        _px(cuerpo[izquierdo ? PuntosCuerpo.codoIzq : PuntosCuerpo.codoDer]);
    final munecaPose = _px(
        cuerpo[izquierdo ? PuntosCuerpo.munecaIzq : PuntosCuerpo.munecaDer]);

    // Si hay mano, el antebrazo termina en la muneca de Hands (mas precisa):
    // brazo y mano quedan soldados sin hueco.
    final muneca = mano != null ? _px(mano[0]) : munecaPose;

    // Brazo con taper (grueso en el hombro, fino en la muneca) y un bulto de
    // deltoide en el arranque, para una silueta mas humana.
    final rBrazo = _propBrazo * _s / 2;
    final rAnte = _propAntebrazo * _s / 2;
    _cadena(canvas, [
      (hombro, hombro, rBrazo * 1.12, rBrazo * 1.12), // deltoide
      (hombro, codo, rBrazo, rAnte * 1.02),
      (codo, muneca, rAnte, rAnte * 0.85),
    ], _tonosCuerpo);
    // Bolas de articulacion en codo y muneca (estilo maniqui, como las manos
    // en tono claro): hacen legibles los quiebres del brazo.
    _bola(canvas, codo, _propBolaCodo * _s, _tonosMano);
    _bola(canvas, muneca, _propBolaMuneca * _s, _tonosMano);

    if (mano != null) _mano(canvas, mano);
  }

  /// Bola de articulacion (esfera chica con degradado radial hacia la luz).
  void _bola(Canvas canvas, Offset centro, double radio, _Tonos tonos) {
    final foco = centro + Offset(_luzX * radio * 0.35, _luzY * radio * 0.35);
    final pintura = Paint()
      ..isAntiAlias = true
      ..shader = ui.Gradient.radial(
        foco,
        radio * 1.55,
        [tonos.claro, tonos.base, tonos.oscuro],
        [0.0, 0.55, 1.0],
      );
    canvas.drawCircle(centro, radio, pintura);
  }

  /// Mano como UNA pieza solida: palma llena + dedos, capa por capa.
  ///
  /// En vez de dibujar la palma y cada dedo como piezas sombreadas
  /// independientes (se ven tubos sueltos), toda la mano se dibuja capa por
  /// capa de tono: donde palma y dedos se solapan, el mismo tono se funde sin
  /// costura y la mano se lee como un solo volumen, tipo manopla de juguete.
  /// El orden por profundidad se conserva: los dedos DETRAS de la palma se
  /// dibujan como pieza previa, y la palma con los dedos delanteros encima.
  void _mano(Canvas canvas, List<P3> mano) {
    final puntos = mano.map(_px).toList();
    final rDedo = _propDedo * _s / 2;

    final zPalma =
        _palma.map((i) => mano[i].z).reduce((a, b) => a + b) / _palma.length;
    final detras = <(double, List<int>)>[];
    final delante = <(double, List<int>)>[];
    for (final dedo in _dedos) {
      final zDedo =
          dedo.map((i) => mano[i].z).reduce((a, b) => a + b) / dedo.length;
      (zDedo > zPalma + 0.004 ? detras : delante).add((zDedo, dedo));
    }
    detras.sort((a, b) => b.$1.compareTo(a.$1));
    delante.sort((a, b) => b.$1.compareTo(a.$1));

    for (final (_, dedo) in detras) {
      _dedo(canvas, puntos, mano, dedo, rDedo);
    }
    _palmaBlob(canvas, puntos, rDedo);
    for (final (_, dedo) in delante) {
      _dedo(canvas, puntos, mano, dedo, rDedo);
    }
  }

  /// Blob de la palma (casco convexo dilatado) + bolitas de nudillos.
  ///
  /// El casco de muneca y nudillos es flaco cuando la mano esta de canto; el
  /// margen (relleno + trazo grueso redondo, equivale a dilatar) le da cuerpo
  /// de palma real. Las bolitas donde nacen los dedos hacen visibles esas
  /// articulaciones.
  void _palmaBlob(Canvas canvas, List<Offset> puntos, double rDedo) {
    final casco = _cascoConvexo(_palma.map((i) => puntos[i]).toList());
    final centroide = Offset(
      casco.map((p) => p.dx).reduce((a, b) => a + b) / casco.length,
      casco.map((p) => p.dy).reduce((a, b) => a + b) / casco.length,
    );
    for (final (indice, factor, corr) in _capas) {
      final pintura = Paint()
        ..color = _tonoMano(indice)
        ..isAntiAlias = true;
      final margen = rDedo * _margenPalma * factor;
      final desplazamiento =
          Offset(_luzX * rDedo * 2 * corr, _luzY * rDedo * 2 * corr);
      final camino = Path();
      for (var i = 0; i < casco.length; i++) {
        final p =
            centroide + (casco[i] - centroide) * factor + desplazamiento;
        if (i == 0) {
          camino.moveTo(p.dx, p.dy);
        } else {
          camino.lineTo(p.dx, p.dy);
        }
      }
      camino.close();
      canvas.drawPath(camino, pintura);
      final borde = Paint()
        ..color = pintura.color
        ..isAntiAlias = true
        ..style = PaintingStyle.stroke
        ..strokeWidth = math.max(1, 2 * margen)
        ..strokeJoin = StrokeJoin.round
        ..strokeCap = StrokeCap.round;
      canvas.drawPath(camino, borde);
    }
    // Nudillos: bolita donde nace cada dedo (base del pulgar incluida).
    for (final i in const [2, 5, 9, 13, 17]) {
      _bolita(canvas, puntos[i], rDedo * _bolaNudillo * 1.05);
    }
  }

  // Capas de un dedo: contorno oscuro MAS grueso que el de otras piezas, para
  // que un dedo doblado sobre la palma no se funda con ella.
  static const _capasDedo = [(0, 1.14, 0.00), (1, 0.88, 0.14), (2, 0.50, 0.30)];

  /// Un dedo articulado: capsulas conicas + bolitas en las falanges (los 21
  /// landmarks de MediaPipe hechos visibles, estilo maniqui).
  void _dedo(Canvas canvas, List<Offset> puntos, List<P3> mano,
      List<int> cadena, double rBase) {
    const radios = [1.00, 0.92, 0.85, 0.80];
    double rz(int k) {
      final factor = (1.0 - mano[cadena[k]].z * _gananciaZDedos)
          .clamp(_factorZMin, _factorZMax);
      return rBase * radios[k] * factor;
    }

    for (final (indice, factor, corr) in _capasDedo) {
      final pintura = Paint()
        ..color = _tonoMano(indice)
        ..isAntiAlias = true;
      for (var k = 0; k + 1 < cadena.length; k++) {
        final ra = rz(k), rb = rz(k + 1);
        _capsulaSolida(
          canvas,
          puntos[cadena[k]] + Offset(_luzX * ra * corr, _luzY * ra * corr),
          puntos[cadena[k + 1]] + Offset(_luzX * rb * corr, _luzY * rb * corr),
          ra * factor,
          rb * factor,
          pintura,
        );
      }
    }
    // Articulaciones intermedias y punta del dedo.
    _bolita(canvas, puntos[cadena[1]], rBase * _bolaNudillo);
    _bolita(canvas, puntos[cadena[2]], rBase * _bolaNudillo * 0.9);
    _bolita(canvas, puntos[cadena[3]], rBase * _bolaNudillo * 0.75);
  }

  Color _tonoMano(int indice) => switch (indice) {
        0 => _tonosMano.oscuro,
        1 => _tonosMano.base,
        _ => _tonosMano.claro,
      };

  /// Bolita de articulacion (mini esfera en el tono de la mano).
  void _bolita(Canvas canvas, Offset centro, double radio) {
    for (final (indice, factor, corr) in const [
      (0, 1.00, 0.00),
      (1, 0.78, 0.18),
      (2, 0.42, 0.36),
    ]) {
      final pintura = Paint()
        ..color = _tonoMano(indice)
        ..isAntiAlias = true;
      final c = centro + Offset(_luzX * radio * corr, _luzY * radio * corr);
      canvas.drawCircle(c, math.max(1, radio * factor), pintura);
    }
  }

  /// Casco convexo (cadena monotona de Andrew) para el blob de la palma.
  static List<Offset> _cascoConvexo(List<Offset> puntos) {
    if (puntos.length < 3) return puntos;
    final orden = List<Offset>.from(puntos)
      ..sort((a, b) =>
          a.dx != b.dx ? a.dx.compareTo(b.dx) : a.dy.compareTo(b.dy));
    double cruz(Offset o, Offset a, Offset b) =>
        (a.dx - o.dx) * (b.dy - o.dy) - (a.dy - o.dy) * (b.dx - o.dx);

    final inferior = <Offset>[];
    for (final p in orden) {
      while (inferior.length >= 2 &&
          cruz(inferior[inferior.length - 2], inferior.last, p) <= 0) {
        inferior.removeLast();
      }
      inferior.add(p);
    }
    final superior = <Offset>[];
    for (final p in orden.reversed) {
      while (superior.length >= 2 &&
          cruz(superior[superior.length - 2], superior.last, p) <= 0) {
        superior.removeLast();
      }
      superior.add(p);
    }
    inferior.removeLast();
    superior.removeLast();
    return [...inferior, ...superior];
  }

  @override
  bool shouldRepaint(MunecoPainter anterior) =>
      anterior.fotograma != fotograma ||
      anterior.vistaEspejo != vistaEspejo ||
      anterior.clip != clip;
}
