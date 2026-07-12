import 'dart:math' as math;

import 'package:lesho_app/core/constantes.dart';
import 'package:lesho_app/core/normalizacion.dart';
import 'package:lesho_app/core/suavizado.dart';

/// CONTRATO: replica `preprocessing/secuencias_b.py` (procesar_secuencia). Toma
/// una secuencia de fotogramas de 140 valores (config 126 + ubicación 14) y la
/// convierte en el tensor de entrada del Modelo B: [longitud, 148].
///
/// Orden (igual que en el entrenamiento):
///   1. Suavizado One Euro sobre los 140 valores.
///   2. Recorte de tramos quietos (deja la seña por su movimiento activo).
///   3. Escala SOLO de la configuración de las manos (126).
///   4. Ubicación (14) + rasgos relativos a la cara (8), por la ganancia (2.5).
///   5. Remuestreo a longitud fija (40).

const int _tv = 126; // TAMANO_VECTOR (config de las manos)

// Punta del índice de cada mano relativa a nariz y boca: 8 valores. Hace explícito
// a qué zona de la cara APUNTA la mano (frente vs boca). Cero si falta la mano o el
// ancla. Ubicación (18): 0-1 muñeca izq, 2-3 muñeca der, 4-5 punta índice izq,
// 6-7 punta índice der, 8-9 nariz, 10-11 ojos, 12-13 boca, 14-17 orejas.
List<double> _rasgosRelativos(List<double> ubic) {
  double sx(int i) => ubic[i].abs();
  bool pres(int a, int b) => (sx(a) + sx(b)) > 1e-9;
  final puntaIzqPres = pres(4, 5);
  final puntaDerPres = pres(6, 7);
  final narizPres = pres(8, 9);
  final bocaPres = pres(12, 13);

  List<double> rel(int p0, bool pPres, int a0, bool aPres) {
    if (!(pPres && aPres)) return const [0.0, 0.0];
    return [ubic[p0] - ubic[a0], ubic[p0 + 1] - ubic[a0 + 1]];
  }

  return [
    ...rel(4, puntaIzqPres, 8, narizPres), // punta izq - nariz
    ...rel(4, puntaIzqPres, 12, bocaPres), // punta izq - boca
    ...rel(6, puntaDerPres, 8, narizPres), // punta der - nariz
    ...rel(6, puntaDerPres, 12, bocaPres), // punta der - boca
  ];
}

double _percentil(List<double> valores, double p) {
  if (valores.isEmpty) return 0.0;
  final orden = [...valores]..sort();
  final idx = ((p / 100.0) * (orden.length - 1)).round().clamp(0, orden.length - 1);
  return orden[idx];
}

// Recorta los tramos QUIETOS de los bordes, dejando el núcleo con movimiento.
// Réplica de recortar_quietos: actividad = cambio de forma (escalada) + 2x
// desplazamiento de las manos por el cuerpo.
List<List<double>> _recortarQuietos(List<List<double>> seq,
    {int margen = 3, int minFrames = 12}) {
  if (seq.length < minFrames + 2 * margen) return seq;

  final conf = [for (final f in seq) escalarVector(f.sublist(0, _tv))];
  final actividad = List<double>.filled(seq.length, 0.0);
  for (var i = 1; i < seq.length; i++) {
    var sumaC = 0.0, cuentaC = 0;
    for (var k = 0; k < _tv; k++) {
      if (conf[i][k] != 0.0) {
        sumaC += (conf[i][k] - conf[i - 1][k]).abs();
        cuentaC++;
      }
    }
    final a = cuentaC == 0 ? 0.0 : sumaC / cuentaC;
    var sumaU = 0.0, cuentaU = 0;
    for (var k = _tv; k < _tv + 4; k++) {
      if (seq[i][k] != 0.0) {
        sumaU += (seq[i][k] - seq[i - 1][k]).abs();
        cuentaU++;
      }
    }
    final b = cuentaU == 0 ? 0.0 : sumaU / cuentaU;
    actividad[i] = a + 2.0 * b;
  }

  final umbral = math.max(0.15 * _percentil(actividad, 95), 1e-4);
  var primero = -1, ultimo = -1;
  for (var i = 0; i < actividad.length; i++) {
    if (actividad[i] > umbral) {
      if (primero < 0) primero = i;
      ultimo = i;
    }
  }
  if (primero < 0) return seq;
  final a0 = math.max(0, primero - margen);
  final a1 = math.min(seq.length, ultimo + margen + 1);
  final recortada = seq.sublist(a0, a1);
  return recortada.length < minFrames ? seq : recortada;
}

// Remuestrea una secuencia (T, D) a n fotogramas por interpolación lineal.
List<List<double>> _remuestrear(List<List<double>> seq, int n) {
  if (seq.isEmpty) return List.generate(n, (_) => List<double>.filled(0, 0.0));
  if (seq.length == 1) return List.generate(n, (_) => List<double>.from(seq[0]));
  final dim = seq[0].length;
  final salida = <List<double>>[];
  for (var i = 0; i < n; i++) {
    final pos = (seq.length - 1) * (i / (n - 1));
    final i0 = pos.floor();
    final i1 = math.min(i0 + 1, seq.length - 1);
    final a = pos - i0;
    final fila = List<double>.filled(dim, 0.0);
    for (var k = 0; k < dim; k++) {
      fila[k] = seq[i0][k] * (1 - a) + seq[i1][k] * a;
    }
    salida.add(fila);
  }
  return salida;
}

/// Procesa la secuencia de 140 valores y devuelve [longitud, 148] lista para el
/// Modelo B. [dt] es el paso de tiempo (s) para el suavizado One Euro.
List<List<double>> procesarSecuenciaB(
  List<List<double>> secuencia, {
  int longitud = 40,
  double dt = 1.0 / 30.0,
}) {
  if (secuencia.isEmpty) {
    return List.generate(
        longitud, (_) => List<double>.filled(Constantes.tamanoEntradaB, 0.0));
  }

  // 1. Suavizado One Euro sobre los 140 valores.
  final filtro = FiltroUnEuro(
    fps: Constantes.fpsObjetivo.toDouble(),
    minCutoff: Constantes.suavizadoMinCutoff,
    beta: Constantes.suavizadoBeta,
    dCutoff: Constantes.suavizadoDCutoff,
  );
  var seq = [for (final f in secuencia) filtro.filtrar(f, dt: dt)];

  // 2. Recorte de tramos quietos.
  seq = _recortarQuietos(seq);

  // 3-4. Config escalada (126) + [ubicación(14) + relativos(8)] x ganancia.
  final procesada = <List<double>>[];
  for (final f in seq) {
    final config = escalarVector(f.sublist(0, _tv)); // 126
    final ubicCruda = f.sublist(_tv, Constantes.tamanoVectorB); // 14
    final relativos = _rasgosRelativos(ubicCruda); // 8
    final ubicacion = [
      for (final v in [...ubicCruda, ...relativos]) v * Constantes.gananciaUbicacion
    ]; // 22
    procesada.add([...config, ...ubicacion]); // 148
  }

  // 5. Remuestreo a longitud fija.
  return _remuestrear(procesada, longitud);
}
