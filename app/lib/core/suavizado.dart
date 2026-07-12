import 'dart:math' as math;

/// CONTRATO CRITICO: replica exactamente `FiltroUnEuro` de
/// [training/comun/suavizado.py]. Suaviza los landmarks temblorosos que entrega
/// MediaPipe (la Tasks API no filtra). Se aplica sobre el vector YA TRASLADADO
/// (muneca en el origen), igual que en el preprocesamiento y en la demo, antes de
/// la escala. Sin este filtro la app veria landmarks con temblor que el modelo
/// nunca vio al entrenar.
///
/// Suaviza fuerte cuando la mano se mueve lento (letra estatica firme) y poco
/// cuando se mueve rapido (la J o la LL no se retrasan): un promedio movil no
/// sirve porque agregaria retraso a las letras con movimiento.
double _alpha(double cutoff, double dt) {
  final tau = 1.0 / (2.0 * math.pi * cutoff);
  return 1.0 / (1.0 + tau / dt);
}

class FiltroUnEuro {
  final double dt;
  final double minCutoff;
  final double beta;
  final double dCutoff;

  List<double>? _xPrev;
  List<double>? _dxPrev;

  FiltroUnEuro({
    double fps = 30.0,
    this.minCutoff = 1.0,
    this.beta = 1.0,
    this.dCutoff = 1.0,
  }) : dt = 1.0 / fps;

  /// Olvida el estado. Llamar cuando la mano desaparece, para que al volver el
  /// filtro arranque limpio y no interpole un salto.
  void reiniciar() {
    _xPrev = null;
    _dxPrev = null;
  }

  /// Filtra un vector de coordenadas y devuelve la version suavizada.
  ///
  /// [dt] es el tiempo real (segundos) desde el fotograma anterior. Es IMPORTANTE
  /// pasarlo cuando la tasa de fotogramas no es la nominal (en el telefono corre
  /// mas lento que 30 fps): con el dt correcto el suavizado se comporta igual que
  /// a 30 fps para un mismo movimiento real. Si no se pasa, usa el dt nominal.
  ///
  /// Si [preservarCeros], las coordenadas que entran en exactamente cero salen en
  /// cero (muneca en el origen y mitad de una mano ausente).
  List<double> filtrar(List<double> x, {double? dt, bool preservarCeros = true}) {
    final paso = dt ?? this.dt;
    final n = x.length;
    if (_xPrev == null) {
      _xPrev = List<double>.from(x);
      _dxPrev = List<double>.filled(n, 0.0);
      return List<double>.from(x); // primera llamada: pasa igual (los ceros ya son cero)
    }

    final xp = _xPrev!;
    final dxp = _dxPrev!;
    final aD = _alpha(dCutoff, paso);
    final salida = List<double>.filled(n, 0.0);
    for (var i = 0; i < n; i++) {
      final dx = (x[i] - xp[i]) / paso;
      final dxHat = aD * dx + (1.0 - aD) * dxp[i];
      final cutoff = minCutoff + beta * dxHat.abs();
      final a = _alpha(cutoff, paso); // un alpha por coordenada
      final suave = a * x[i] + (1.0 - a) * xp[i];
      // El estado guarda el valor suave crudo; la salida preserva los ceros.
      xp[i] = suave;
      dxp[i] = dxHat;
      salida[i] = (preservarCeros && x[i] == 0.0) ? 0.0 : suave;
    }
    return salida;
  }
}
