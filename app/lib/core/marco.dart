import 'dart:math' as math;

import 'package:lesho_app/core/constantes.dart';

/// CONTRATO: replica `comun/marco.py`. Marco de referencia del cuerpo para el
/// Modelo B: ubica las manos y la cara respecto al centro de los hombros, escalado
/// por el ancho de los hombros (invariante a la distancia a la cámara).

/// Un landmark de Pose: coordenadas normalizadas y su visibilidad.
class Punto3D {
  final double x;
  final double y;
  final double z;
  final double visibilidad;

  const Punto3D(this.x, this.y, this.z, this.visibilidad);
}

/// Un ancla facial (x, y) en coordenadas de imagen, o null si no es fiable.
typedef Ancla = ({double x, double y})?;

/// Marco de referencia del cuerpo para ubicar las manos y la cara.
class MarcoCuerpo {
  final double centroX;
  final double centroY;
  final double ancho;
  final Ancla nariz;
  final Ancla ojos;
  final Ancla boca;
  final Ancla orejaIzq;
  final Ancla orejaDer;

  const MarcoCuerpo(
    this.centroX,
    this.centroY,
    this.ancho, {
    this.nariz,
    this.ojos,
    this.boca,
    this.orejaIzq,
    this.orejaDer,
  });

  /// Convierte un punto (x, y) de la imagen a coordenadas del cuerpo, en "anchos
  /// de hombro" respecto al centro de los hombros.
  List<double> ubicacionRelativa(double x, double y) =>
      [(x - centroX) / ancho, (y - centroY) / ancho];

  /// (rx, ry) de un ancla en el marco, o [0, 0] si el ancla es null.
  List<double> relAncla(Ancla ancla) =>
      ancla == null ? const [0.0, 0.0] : ubicacionRelativa(ancla.x, ancla.y);
}

({double x, double y})? _visible(Punto3D p) =>
    p.visibilidad < Constantes.visibilidadMinimaPose ? null : (x: p.x, y: p.y);

({double x, double y})? _centroVisible(Punto3D p1, Punto3D p2) {
  final a = _visible(p1);
  final b = _visible(p2);
  if (a == null || b == null) return null;
  return (x: (a.x + b.x) / 2.0, y: (a.y + b.y) / 2.0);
}

/// Construye el MarcoCuerpo a partir de los 33 puntos de la pose. Devuelve null si
/// los hombros no son fiables o el ancho es degenerado (persona de perfil o lejos).
MarcoCuerpo? marcoDesdePuntos(List<Punto3D> puntos) {
  if (puntos.length != Constantes.numLandmarksPose) return null;
  final hombroIzq = puntos[Constantes.poseHombroIzq];
  final hombroDer = puntos[Constantes.poseHombroDer];

  if (hombroIzq.visibilidad < Constantes.visibilidadMinimaPose ||
      hombroDer.visibilidad < Constantes.visibilidadMinimaPose) {
    return null;
  }

  final centroX = (hombroIzq.x + hombroDer.x) / 2.0;
  final centroY = (hombroIzq.y + hombroDer.y) / 2.0;
  final ancho = math.sqrt(
    math.pow(hombroIzq.x - hombroDer.x, 2) +
        math.pow(hombroIzq.y - hombroDer.y, 2),
  );
  if (ancho < 1e-4) return null;

  return MarcoCuerpo(
    centroX,
    centroY,
    ancho,
    nariz: _visible(puntos[Constantes.poseNariz]),
    ojos: _centroVisible(
        puntos[Constantes.poseOjoIzq], puntos[Constantes.poseOjoDer]),
    boca: _centroVisible(
        puntos[Constantes.poseBocaIzq], puntos[Constantes.poseBocaDer]),
    orejaIzq: _visible(puntos[Constantes.poseOrejaIzq]),
    orejaDer: _visible(puntos[Constantes.poseOrejaDer]),
  );
}
