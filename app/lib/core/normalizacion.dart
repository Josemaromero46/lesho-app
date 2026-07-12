import 'dart:math';

import 'package:lesho_app/core/constantes.dart';

/// CONTRATO CRITICO: este archivo replica exactamente
/// [training/comun/normalizacion.py]. Si las dos versiones difieren,
/// el modelo entrenado no funcionará en la app.
///
/// Cualquier cambio aquí obliga a cambiar igual la versión en Python
/// y a reentrenar los modelos.

/// Punto tridimensional con coordenadas x, y, z.
class Punto {
  final double x;
  final double y;
  final double z;

  const Punto(this.x, this.y, this.z);
}

/// Normaliza los 21 landmarks de UNA mano restando la posición de la muñeca.
///
/// Parámetros:
///   [landmarksCrudos] - Lista de 21 puntos (x, y, z) de MediaPipe.
///
/// Retorna [List<double>] de 63 valores en el orden:
///   [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]
/// donde a cada coordenada se le restó la de la muñeca (punto 0).
/// El punto 0 queda en (0,0,0) pero se conserva en el vector.
List<double> normalizarRespectMuneca(List<Punto> landmarksCrudos) {
  if (landmarksCrudos.length != Constantes.numLandmarks) {
    throw ArgumentError(
      'Se esperaban ${Constantes.numLandmarks} landmarks, '
      'se recibieron ${landmarksCrudos.length}.',
    );
  }

  final muneca = landmarksCrudos[0]; // landmark 0 = muñeca
  final vector = <double>[];

  for (final punto in landmarksCrudos) {
    vector.add(punto.x - muneca.x);
    vector.add(punto.y - muneca.y);
    vector.add(punto.z - muneca.z);
  }

  assert(vector.length == Constantes.tamanoVectorMano);
  return vector;
}

/// Compone el vector de 126 valores a partir de las dos manos.
///
/// Orden fijo: primero la mano izquierda (63 valores), luego la derecha (63).
/// Si una mano no se detectó, su mitad va en ceros.
/// Cada mano se normaliza respecto a SU PROPIA muñeca.
List<double> componerVectorDosManos(
  List<Punto>? landmarksIzquierda,
  List<Punto>? landmarksDerecha,
) {
  final izquierda = landmarksIzquierda != null
      ? normalizarRespectMuneca(landmarksIzquierda)
      : List.filled(Constantes.tamanoVectorMano, 0.0);

  final derecha = landmarksDerecha != null
      ? normalizarRespectMuneca(landmarksDerecha)
      : List.filled(Constantes.tamanoVectorMano, 0.0);

  final vector = [...izquierda, ...derecha];
  assert(vector.length == Constantes.tamanoVector);
  return vector;
}

/// Nudillos (MCP) de los cuatro dedos largos. Su distancia a la muñeca es un
/// tamaño estable de la mano, que crece o se encoge según la distancia a la
/// cámara.
const List<int> _nudillos = [5, 9, 13, 17];

/// Tamaño característico de una mano ya trasladada (muñeca en el origen):
/// distancia media, en el plano x-y, de la muñeca a los cuatro nudillos.
double _tamanoMano(List<double> mano) {
  var total = 0.0;
  for (final k in _nudillos) {
    final x = mano[k * 3];
    final y = mano[k * 3 + 1];
    total += sqrt(x * x + y * y);
  }
  return total / _nudillos.length;
}

/// Normaliza la ESCALA del vector de 126 valores, dividiendo cada mano por su
/// tamaño. Hace el vector invariante al tamaño aparente de la mano, es decir, a
/// la distancia a la cámara. La mitad de una mano ausente (ceros) se deja igual.
///
/// CONTRATO: replica exactamente `escalar_vector` de
/// [training/comun/normalizacion.py]. Se aplica DESPUÉS de
/// [componerVectorDosManos], antes de pasar el vector al modelo.
List<double> escalarVector(List<double> vector) {
  final salida = <double>[];
  for (final inicio in [0, Constantes.tamanoVectorMano]) {
    final mano =
        vector.sublist(inicio, inicio + Constantes.tamanoVectorMano);
    final ausente = mano.every((v) => v.abs() < 1e-9);
    if (ausente) {
      salida.addAll(mano);
      continue;
    }
    final escala = _tamanoMano(mano);
    if (escala < 1e-6) {
      salida.addAll(mano);
    } else {
      salida.addAll(mano.map((v) => v / escala));
    }
  }
  return salida;
}
