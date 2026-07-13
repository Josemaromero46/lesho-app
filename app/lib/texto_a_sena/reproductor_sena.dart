import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:lesho_app/texto_a_sena/clip_sena.dart';
import 'package:lesho_app/texto_a_sena/muneco_painter.dart';

/// Reproductor de un [ClipSena] sobre el muñeco de cápsulas.
///
/// Avanza el clip con un ticker a los fps del propio clip, con interpolación
/// entre fotogramas (fluido incluso a 0.5x) y control de velocidad, una
/// ventaja directa sobre reproducir video. Al terminar puede repetir en bucle
/// (pantalla de prueba) o avisar con [alTerminar] (cola de clips de una frase).
class ReproductorSena extends StatefulWidget {
  final ClipSena clip;

  /// Velocidad de reproducción (1.0 = la velocidad real de la seña).
  final double velocidad;

  /// Vista espejo: false (por defecto) muestra una persona DE FRENTE que
  /// firma con su mano derecha real; true muestra el reflejo tal cual se
  /// grabó (para imitar). Decisión a validar con asesoría LESHO.
  final bool vistaEspejo;

  /// Si es true, el clip se repite en bucle.
  final bool repetir;

  /// Se llama al terminar el clip (solo si [repetir] es false).
  final VoidCallback? alTerminar;

  final ColoresMuneco colores;

  const ReproductorSena({
    super.key,
    required this.clip,
    this.velocidad = 1.0,
    this.vistaEspejo = false,
    this.repetir = true,
    this.alTerminar,
    this.colores = ColoresMuneco.azul,
  });

  @override
  State<ReproductorSena> createState() => _ReproductorSenaState();
}

class _ReproductorSenaState extends State<ReproductorSena>
    with SingleTickerProviderStateMixin {
  late final Ticker _ticker;
  Duration _anterior = Duration.zero;
  double _indice = 0;
  bool _termino = false;

  @override
  void initState() {
    super.initState();
    _ticker = createTicker(_alTick)..start();
  }

  @override
  void didUpdateWidget(ReproductorSena anterior) {
    super.didUpdateWidget(anterior);
    if (anterior.clip != widget.clip) {
      _indice = 0;
      _termino = false;
    }
  }

  void _alTick(Duration transcurrido) {
    final dt = (transcurrido - _anterior).inMicroseconds / 1e6;
    _anterior = transcurrido;
    if (_termino) return;

    var indice = _indice + dt * widget.clip.fps * widget.velocidad;
    final tope = (widget.clip.numFrames - 1).toDouble();
    if (indice >= tope) {
      if (widget.repetir) {
        indice = 0;
      } else {
        indice = tope;
        _termino = true;
        widget.alTerminar?.call();
      }
    }
    setState(() => _indice = indice);
  }

  @override
  void dispose() {
    _ticker.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: MunecoPainter(
        clip: widget.clip,
        fotograma: widget.clip.fotograma(_indice),
        vistaEspejo: widget.vistaEspejo,
        colores: widget.colores,
      ),
      child: const SizedBox.expand(),
    );
  }
}
