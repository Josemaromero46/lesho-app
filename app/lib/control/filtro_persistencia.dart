import 'package:lesho_app/core/constantes.dart';

/// Clase detectada y confirmada tras superar el filtro de persistencia.
class ResultadoConfirmado {
  final String clase;
  final double confianza;

  const ResultadoConfirmado({required this.clase, required this.confianza});
}

/// Filtro temporal: exige que la misma clase aparezca en frames consecutivos
/// antes de confirmar una detección.
///
/// Reglas:
///   - Se confirma una clase cuando aparece en [Constantes.fotogramasPersistencia]
///     frames consecutivos con confianza >= [Constantes.umbralConfianza].
///   - Tras confirmar, bloquea nuevas detecciones durante [Constantes.cooldownMs] ms.
///   - Si la clase cambia antes de llegar al umbral, el contador se reinicia.
///   - Si la confianza cae bajo el umbral, el contador se reinicia.
class FiltroPersistencia {
  String? _claseActual;
  int _contadorFrames = 0;
  DateTime? _finCooldown;

  /// Registra el resultado de un frame.
  ///
  /// Retorna [ResultadoConfirmado] cuando se cumple la persistencia, o null.
  ResultadoConfirmado? registrar(String clase, double confianza) {
    if (_enCooldown()) return null;

    if (confianza < Constantes.umbralConfianza) {
      _reiniciarContador();
      return null;
    }

    if (clase == _claseActual) {
      _contadorFrames++;
    } else {
      _claseActual = clase;
      _contadorFrames = 1;
    }

    if (_contadorFrames >= Constantes.fotogramasPersistencia) {
      final confirmada = ResultadoConfirmado(
        clase: clase,
        confianza: confianza,
      );
      _reiniciarContador();
      _activarCooldown();
      return confirmada;
    }

    return null;
  }

  /// Reinicia el filtro completamente (contador y cooldown).
  ///
  /// Se llama al cambiar de estado en la máquina de estados para que el
  /// primer frame del nuevo estado no herede el contexto del anterior.
  void reiniciar() {
    _reiniciarContador();
    _finCooldown = null;
  }

  bool _enCooldown() {
    if (_finCooldown == null) return false;
    if (DateTime.now().isBefore(_finCooldown!)) return true;
    _finCooldown = null;
    return false;
  }

  void _reiniciarContador() {
    _claseActual = null;
    _contadorFrames = 0;
  }

  void _activarCooldown() {
    _finCooldown = DateTime.now().add(
      const Duration(milliseconds: Constantes.cooldownMs),
    );
  }
}
