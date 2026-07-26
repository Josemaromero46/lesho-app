import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:lesho_app/texto_a_sena/clip_sena.dart';

/// Muñeco de señas dibujado como PERSONAJE ILUSTRADO (PLAN_DIRECCION2, sección 6).
///
/// Estilo pensado para niños y para que se entienda a simple vista: piel cálida,
/// camiseta de color (para que el brazo se distinga del torso), contorno oscuro
/// limpio, y manos muy definidas con dedos contorneados y uñas en las puntas.
/// Es la réplica en CustomPainter del renderizador calibrado en
/// `training/demo/visor_clips.py`; las proporciones, colores y capas deben
/// mantenerse iguales en ambos. Sin plugins ni motor 3D: solo Canvas de Flutter.

// Proporciones en fracciones del ANCHO DE HOMBROS (calibradas en el visor).
const _propBrazo = 0.32; // grosor de la manga (hombro a codo)
const _propAntebrazo = 0.24; // grosor del antebrazo (piel)
const _propCuello = 0.32;
const _propDedo = 0.118;
const _propRadioCabeza = 0.46;
const _ovaloCabeza = 1.14; // cabeza ovalada (mas alta que ancha)
const _angosteCaderas = 0.80; // caderas mas angostas que lo medido
const _grosorContorno = 0.028; // ancho del contorno oscuro

// Perfil de la silueta del torso: (t, ancho relativo al medio-hombro). t va de
// la linea de hombros (0) a la de caderas (1). Cintura marcada = mas humano.
const _perfilTorso = [
  (0.00, 1.00),
  (0.30, 0.95),
  (0.58, 0.76),
  (0.82, 0.84),
  (1.00, 0.90),
];

const _fraccionHombros = 0.325;
const _alturaHombros = 0.46;

const _luzX = -0.45, _luzY = -0.89;
const _ganZDedos = 5.0;
const _factorZMin = 0.72, _factorZMax = 1.30;

const _dedos = [
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16],
  [17, 18, 19, 20],
];
const _palmaIdx = [0, 1, 2, 5, 9, 13, 17];

// Capas del sombreado sobre el relleno: (indice de tono, factor, corr luz). La
// definicion la da el contorno; el sombreado es suave (base y un brillo).
const _capas = [(1, 1.00, 0.00), (2, 0.66, 0.24)];

/// Colores del personaje. Piel cálida + camiseta de color + pelo castaño.
class ColoresMuneco {
  final Color piel;
  final Color camisa;
  final Color pelo;
  const ColoresMuneco(
      {required this.piel, required this.camisa, required this.pelo});

  static const humano = ColoresMuneco(
    piel: Color(0xFFE2B080), // tan calido
    camisa: Color(0xFF56966E), // verde bosque
    pelo: Color(0xFF4E382E), // castano oscuro
  );
}

/// Contorno oscuro (casi negro cálido) y color de las uñas.
const _contorno = Color(0xFF3A2C28);
const _colUna = Color(0xFFF8E4CE);

/// Tres tonos derivados de un color base, para el sombreado.
class _Tonos {
  final Color oscuro;
  final Color base;
  final Color claro;
  _Tonos(Color c)
      : oscuro = Color.lerp(c, Colors.black, 0.32)!,
        base = c,
        claro = Color.lerp(c, Colors.white, 0.40)!;

  Color tono(int i) => switch (i) {
        0 => oscuro,
        1 => base,
        _ => claro,
      };
}

class MunecoPainter extends CustomPainter {
  final ClipSena clip;
  final FotogramaSena fotograma;

  /// Vista (lateralidad): false (por defecto) dibuja una persona DE FRENTE que
  /// firma con su mano derecha real; true la dibuja como reflejo (para imitar).
  final bool vistaEspejo;
  final ColoresMuneco colores;

  late final _Tonos _piel = _Tonos(colores.piel);
  late final _Tonos _camisa = _Tonos(colores.camisa);

  double _escala = 1;
  double _yHombros = 0;
  double _s = 1; // ancho de hombros en pixeles
  double _g = 3; // grosor del contorno en pixeles
  Size _tamano = Size.zero;

  MunecoPainter({
    required this.clip,
    required this.fotograma,
    this.vistaEspejo = false,
    this.colores = ColoresMuneco.humano,
  });

  @override
  void paint(Canvas canvas, Size size) {
    _tamano = size;
    _prepararMarco(size);

    final cuerpo = fotograma.cuerpo;
    _cuello(canvas, cuerpo);
    _torso(canvas, cuerpo);
    _cabeza(canvas, cuerpo);

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
    _g = math.max(2.0, _grosorContorno * _s);
  }

  Offset _px(P3 p) {
    var dx = (p.x - clip.centroX) * _escala;
    if (!vistaEspejo) dx = -dx;
    return Offset(_tamano.width / 2 + dx,
        _yHombros + (p.y - clip.centroY) * _escala);
  }

  // -- Primitivas con contorno + sombreado -----------------------------------

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

  /// Cadena de cápsulas con contorno oscuro y sombreado suave. Cada pieza lleva
  /// su propio contorno, así un dedo o un brazo se separa del de al lado por una
  /// línea oscura, sin fundirse.
  void _cadena(
      Canvas canvas, List<(Offset, Offset, double, double)> segmentos, _Tonos t,
      {bool contorno = true}) {
    if (contorno) {
      final p = Paint()
        ..color = _contorno
        ..isAntiAlias = true;
      for (final (a, b, ra, rb) in segmentos) {
        _capsulaSolida(canvas, a, b, ra + _g, rb + _g, p);
      }
    }
    for (final (indice, factor, corr) in _capas) {
      final p = Paint()
        ..color = t.tono(indice)
        ..isAntiAlias = true;
      for (final (a, b, ra, rb) in segmentos) {
        _capsulaSolida(
          canvas,
          a + Offset(_luzX * ra * corr, _luzY * ra * corr),
          b + Offset(_luzX * rb * corr, _luzY * rb * corr),
          ra * factor,
          rb * factor,
          p,
        );
      }
    }
  }

  void _elipseContorneada(
      Canvas canvas, Offset centro, double rx, double ry, _Tonos t) {
    canvas.drawOval(
      Rect.fromCenter(
          center: centro, width: 2 * (rx + _g), height: 2 * (ry + _g)),
      Paint()
        ..color = _contorno
        ..isAntiAlias = true,
    );
    for (final (indice, factor, corr) in _capas) {
      final cc = centro + Offset(_luzX * ry * corr, _luzY * ry * corr);
      canvas.drawOval(
        Rect.fromCenter(
            center: cc, width: 2 * rx * factor, height: 2 * ry * factor),
        Paint()
          ..color = t.tono(indice)
          ..isAntiAlias = true,
      );
    }
  }

  // -- Partes del muñeco -----------------------------------------------------

  Offset _centroCabeza(List<P3> cuerpo) =>
      _px(cuerpo[PuntosCuerpo.nariz]) + Offset(0, -0.10 * _s);

  void _cuello(Canvas canvas, List<P3> cuerpo) {
    final hi = _px(cuerpo[PuntosCuerpo.hombroIzq]);
    final hd = _px(cuerpo[PuntosCuerpo.hombroDer]);
    final centro = Offset((hi.dx + hd.dx) / 2, (hi.dy + hd.dy) / 2);
    final r = _propCuello * _s / 2;
    _cadena(canvas, [(centro, _centroCabeza(cuerpo), r, r)], _piel);
  }

  void _torso(Canvas canvas, List<P3> cuerpo) {
    final hi = _px(cuerpo[PuntosCuerpo.hombroIzq]);
    final hd = _px(cuerpo[PuntosCuerpo.hombroDer]);
    var ci = _px(cuerpo[PuntosCuerpo.caderaIzq]);
    var cd = _px(cuerpo[PuntosCuerpo.caderaDer]);
    final medioCad = Offset((ci.dx + cd.dx) / 2, (ci.dy + cd.dy) / 2);
    ci = medioCad + (ci - medioCad) * _angosteCaderas;
    cd = medioCad + (cd - medioCad) * _angosteCaderas;

    final medioSup = Offset((hi.dx + hd.dx) / 2, (hi.dy + hd.dy) / 2);
    final eje = medioCad - medioSup;
    final semiHombro = (hd - hi).distance / 2;
    var u = Offset(eje.dy, -eje.dx);
    final nu = u.distance;
    u = nu > 1e-6 ? u / nu : const Offset(1, 0);

    final izq = <Offset>[];
    final der = <Offset>[];
    for (final (t, ancho) in _perfilTorso) {
      final centro = medioSup + eje * t;
      final medio = semiHombro * ancho;
      izq.add(centro + u * medio);
      der.add(centro - u * medio);
    }
    final path = Path()..addPolygon([...izq, ...der.reversed], true);

    // Contorno oscuro (trazo grueso) y relleno con degradado vertical.
    canvas.drawPath(
      path,
      Paint()
        ..color = _contorno
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2 * _g
        ..strokeJoin = StrokeJoin.round
        ..isAntiAlias = true,
    );
    final arriba = Color.lerp(_camisa.base, _camisa.claro, 0.45)!;
    final abajo = Color.lerp(_camisa.base, _camisa.oscuro, 0.30)!;
    final shader = ui.Gradient.linear(
      Offset(medioSup.dx, medioSup.dy),
      Offset(medioCad.dx, medioCad.dy),
      [arriba, abajo],
    );
    canvas.drawPath(
      path,
      Paint()
        ..shader = shader
        ..isAntiAlias = true,
    );
  }

  void _cabeza(Canvas canvas, List<P3> cuerpo) {
    final c = _centroCabeza(cuerpo);
    final rx = _propRadioCabeza * _s;
    final ry = rx * _ovaloCabeza;
    _elipseContorneada(canvas, c, rx, ry, _piel);
    _pelo(canvas, c, rx, ry);
  }

  /// Casquete de pelo sobre la parte de arriba de la cabeza (sin rostro).
  void _pelo(Canvas canvas, Offset c, double rx, double ry) {
    const pasos = 40;
    final pts = <Offset>[];
    for (var i = 0; i <= pasos; i++) {
      final ang = math.pi + math.pi * i / pasos;
      pts.add(Offset(c.dx + rx * math.cos(ang), c.dy + ry * math.sin(ang)));
    }
    for (var i = pasos; i >= 0; i--) {
      final x = c.dx - rx + 2 * rx * i / pasos;
      final frac = (x - c.dx) / rx;
      final y = c.dy - ry * 0.22 - ry * 0.34 * (1 - frac * frac);
      pts.add(Offset(x, y));
    }
    final hairPath = Path()..addPolygon(pts, true);
    final ovalPath = Path()
      ..addOval(Rect.fromCenter(center: c, width: 2 * rx, height: 2 * ry));

    canvas.save();
    canvas.clipPath(ovalPath);
    final baseHair = colores.pelo;
    final oscHair = Color.lerp(colores.pelo, Colors.black, 0.28)!;
    canvas.drawPath(
      hairPath,
      Paint()
        ..shader = ui.Gradient.linear(
          Offset(c.dx, c.dy - ry),
          Offset(c.dx, c.dy + ry),
          [baseHair, oscHair],
        )
        ..isAntiAlias = true,
    );
    canvas.restore();
  }

  void _brazo(Canvas canvas, List<P3> cuerpo, List<P3>? mano,
      {required bool izquierdo}) {
    final hombro = _px(
        cuerpo[izquierdo ? PuntosCuerpo.hombroIzq : PuntosCuerpo.hombroDer]);
    final codo =
        _px(cuerpo[izquierdo ? PuntosCuerpo.codoIzq : PuntosCuerpo.codoDer]);
    final munecaPose = _px(
        cuerpo[izquierdo ? PuntosCuerpo.munecaIzq : PuntosCuerpo.munecaDer]);
    final muneca = mano != null ? _px(mano[0]) : munecaPose;

    final rBrazo = _propBrazo * _s / 2;
    final rAnte = _propAntebrazo * _s / 2;

    // Manga (camiseta): hombro con deltoide hasta el codo. Antebrazo (piel):
    // codo a muneca. El contorno del antebrazo sobre la manga hace de dobladillo.
    _cadena(canvas, [
      (hombro, hombro, rBrazo * 1.10, rBrazo * 1.10), // deltoide
      (hombro, codo, rBrazo, rAnte * 1.06),
    ], _camisa);
    _cadena(canvas, [(codo, muneca, rAnte * 1.02, rAnte * 0.9)], _piel);

    if (mano != null) _mano(canvas, mano);
  }

  // -- Mano ilustrada: dedos contorneados con uñas ---------------------------

  void _mano(Canvas canvas, List<P3> mano) {
    final puntos = mano.map(_px).toList();
    final rDedo = _propDedo * _s / 2;

    final zPalma =
        _palmaIdx.map((i) => mano[i].z).reduce((a, b) => a + b) /
            _palmaIdx.length;
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
    _palma(canvas, puntos, rDedo);
    for (final (_, dedo) in delante) {
      _dedo(canvas, puntos, mano, dedo, rDedo);
    }
  }

  void _palma(Canvas canvas, List<Offset> puntos, double rDedo) {
    final hull = _cascoConvexo(_palmaIdx.map((i) => puntos[i]).toList());
    final centroide = Offset(
      hull.map((p) => p.dx).reduce((a, b) => a + b) / hull.length,
      hull.map((p) => p.dy).reduce((a, b) => a + b) / hull.length,
    );
    final margen = rDedo * 1.05;

    void casco(double escala, double extra, Color color) {
      final pts =
          hull.map((q) => centroide + (q - centroide) * escala).toList();
      final path = Path()..addPolygon(pts, true);
      final radio = margen * escala + extra;
      canvas.drawPath(
        path,
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = math.max(1, 2 * radio)
          ..strokeJoin = StrokeJoin.round
          ..strokeCap = StrokeCap.round
          ..isAntiAlias = true,
      );
      canvas.drawPath(
        path,
        Paint()
          ..color = color
          ..isAntiAlias = true,
      );
    }

    casco(1.0, _g, _contorno); // contorno
    for (final (indice, factor, _) in _capas) {
      casco(factor, 0.0, _piel.tono(indice));
    }
  }

  void _dedo(Canvas canvas, List<Offset> puntos, List<P3> mano,
      List<int> cadena, double rBase) {
    const radios = [1.00, 0.94, 0.88, 0.82];
    double rz(int k) {
      final factor = (1.0 - mano[cadena[k]].z * _ganZDedos)
          .clamp(_factorZMin, _factorZMax);
      return rBase * radios[k] * factor;
    }

    final segmentos = <(Offset, Offset, double, double)>[
      for (var k = 0; k + 1 < cadena.length; k++)
        (puntos[cadena[k]], puntos[cadena[k + 1]], rz(k), rz(k + 1)),
    ];
    _cadena(canvas, segmentos, _piel);
    _una(canvas, puntos[cadena[3]], puntos[cadena[2]], rz(3));
  }

  /// Uña: óvalo claro en la punta, orientado a lo largo del dedo.
  void _una(Canvas canvas, Offset punta, Offset previa, double r) {
    final d = punta - previa;
    final largo = d.distance;
    final dir = largo > 1e-3 ? d / largo : const Offset(0, -1);
    final centro = punta - dir * (r * 0.30);
    final ang = math.atan2(dir.dy, dir.dx);
    final ejeX = r * 0.78, ejeY = r * 0.58;

    canvas.save();
    canvas.translate(centro.dx, centro.dy);
    canvas.rotate(ang);
    canvas.drawOval(
      Rect.fromCenter(
          center: Offset.zero, width: 2 * (ejeX + 1), height: 2 * (ejeY + 1)),
      Paint()
        ..color = _contorno
        ..isAntiAlias = true,
    );
    canvas.drawOval(
      Rect.fromCenter(center: Offset.zero, width: 2 * ejeX, height: 2 * ejeY),
      Paint()
        ..color = _colUna
        ..isAntiAlias = true,
    );
    canvas.restore();
  }

  /// Casco convexo (cadena monótona de Andrew) para el blob de la palma.
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
