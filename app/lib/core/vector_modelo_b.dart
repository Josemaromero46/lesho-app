import 'package:lesho_app/core/constantes.dart';
import 'package:lesho_app/core/marco.dart';
import 'package:lesho_app/core/normalizacion.dart';

/// CONTRATO: replica `comun/vector_modelo_b.py`. Arma el vector de 140 valores de
/// un fotograma del Modelo B: configuración de las 2 manos (126) + ubicación en el
/// cuerpo (14). La escala se aplica después, en el preprocesamiento.
///
/// Los landmarks de mano y de pose que se pasan aquí deben venir ya con la
/// corrección de aspecto aplicada (igual que en el Modelo A), para que coincidan
/// con el entrenamiento.

/// (rx, ry) de un landmark de la mano en el marco del cuerpo, o [0, 0].
List<double> _ubicacionDePunto(List<Punto>? mano, int indice, MarcoCuerpo marco) {
  if (mano == null || mano.length <= indice) return const [0.0, 0.0];
  final p = mano[indice];
  return marco.ubicacionRelativa(p.x, p.y);
}

/// Los 18 valores de ubicación (orden CONTRATO): muñeca izq, muñeca der, punta
/// índice izq, punta índice der, nariz, ojos, boca, oreja izq, oreja der. La punta
/// del índice captura a DÓNDE apunta la mano (frente vs boca). Sin marco, todo va
/// en cero.
List<double> _ubicacion(
    List<Punto>? izquierda, List<Punto>? derecha, MarcoCuerpo? marco) {
  if (marco == null) return List<double>.filled(Constantes.tamanoUbicacion, 0.0);
  const puntaIndice = Constantes.indicePuntaIndice;
  return [
    ..._ubicacionDePunto(izquierda, 0, marco), // muñeca izq
    ..._ubicacionDePunto(derecha, 0, marco), // muñeca der
    ..._ubicacionDePunto(izquierda, puntaIndice, marco), // punta índice izq
    ..._ubicacionDePunto(derecha, puntaIndice, marco), // punta índice der
    ...marco.relAncla(marco.nariz),
    ...marco.relAncla(marco.ojos),
    ...marco.relAncla(marco.boca),
    ...marco.relAncla(marco.orejaIzq),
    ...marco.relAncla(marco.orejaDer),
  ];
}

/// Arma el vector de 140 valores de un fotograma del Modelo B.
List<double> componerVectorModeloB(
  List<Punto>? landmarksIzquierda,
  List<Punto>? landmarksDerecha,
  MarcoCuerpo? marco,
) {
  final config = componerVectorDosManos(landmarksIzquierda, landmarksDerecha); // 126
  final ubic = _ubicacion(landmarksIzquierda, landmarksDerecha, marco); // 14
  final vector = [...config, ...ubic];
  assert(vector.length == Constantes.tamanoVectorB);
  return vector;
}
