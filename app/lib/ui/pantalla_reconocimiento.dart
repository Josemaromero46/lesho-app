import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:lesho_app/core/constantes.dart';
import 'package:lesho_app/core/normalizacion.dart';
import 'package:lesho_app/captura/controlador_camara.dart';
import 'package:lesho_app/control/maquina_estados.dart';
import 'package:lesho_app/inferencia/cargador_modelos.dart';
import 'package:lesho_app/inferencia/modelo_a.dart';
import 'package:lesho_app/landmarks/detector_manos.dart';

/// Pantalla de reconocimiento del alfabeto (Dirección 1: niño deletrea -> texto).
///
/// Muestra la cámara frontal en modo selfie y el texto deletreado. Reproduce la
/// lógica de la demo validada: ventana temporal, suavizado, compuerta de
/// movimiento, persistencia y cooldown (todo en [MaquinaEstados]).
class PantallaReconocimiento extends StatefulWidget {
  const PantallaReconocimiento({super.key});

  @override
  State<PantallaReconocimiento> createState() => _EstadoPantallaReconocimiento();
}

class _EstadoPantallaReconocimiento extends State<PantallaReconocimiento> {
  final _camara = ControladorCamara();
  final _detector = DetectorManos();
  final _cargador = CargadorModelos();
  MaquinaEstados? _maquina;

  bool _cargando = true;
  String? _errorCarga;
  bool _procesandoFrame = false;
  Timer? _timerHud;

  // Últimas manos detectadas, para dibujar el esqueleto sobre el preview.
  List<Punto>? _manoIzq;
  List<Punto>? _manoDer;

  @override
  void initState() {
    super.initState();
    // La rotación del fotograma asume orientación vertical.
    SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
    _inicializar();
  }

  Future<void> _inicializar() async {
    try {
      final permiso = await Permission.camera.request();
      if (!permiso.isGranted) {
        throw StateError('Permiso de cámara denegado.');
      }

      await _cargador.cargar();
      _maquina = MaquinaEstados(modeloA: ModeloA(_cargador));
      _maquina!.onTextoCambiado = () {
        if (mounted) setState(() {});
      };

      await _detector.inicializar();
      await _camara.inicializar();
      _detector.rotacion = _camara.orientacionSensor;
      await _camara.iniciarFlujo(_procesarFrame);

      // Refresca el HUD de diagnóstico a ~7 Hz, sin atarlo al ritmo de frames.
      _timerHud = Timer.periodic(const Duration(milliseconds: 150), (_) {
        if (mounted) setState(() {});
      });

      if (mounted) setState(() => _cargando = false);
    } catch (e, stack) {
      debugPrint('LESHO error de arranque: $e\n$stack');
      if (mounted) {
        setState(() {
          _cargando = false;
          _errorCarga = e.toString();
        });
      }
    }
  }

  Future<void> _procesarFrame(CameraImage imagen) async {
    if (_procesandoFrame || _maquina == null) return;
    _procesandoFrame = true;
    try {
      final deteccion = await _detector.procesar(imagen);
      _manoIzq = deteccion.manoIzquierda;
      _manoDer = deteccion.manoDerecha;
      if (deteccion.hayMano) {
        _maquina!.procesarManos(
          deteccion.manoIzquierda,
          deteccion.manoDerecha,
          correccionAspecto: _detector.factorAspecto,
        );
      } else {
        _maquina!.sinManos();
      }
    } finally {
      _procesandoFrame = false;
    }
  }

  @override
  void dispose() {
    _timerHud?.cancel();
    _camara.liberar();
    _detector.liberar();
    _cargador.liberar();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final colores = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        foregroundColor: colores.onSurface,
        elevation: 0,
        title: const Text('Deletrear'),
        actions: [
          if (_maquina != null)
            IconButton(
              icon: const Icon(Icons.backspace_rounded),
              tooltip: 'Borrar última letra',
              onPressed: () => _maquina!.borrarUltima(),
            ),
          if (_maquina != null)
            IconButton(
              icon: const Icon(Icons.clear_all_rounded),
              tooltip: 'Limpiar todo',
              onPressed: () => _maquina!.limpiar(),
            ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: Stack(
              children: [
                Positioned.fill(
                  child: _AreaCamara(
                    camara: _camara,
                    cargando: _cargando,
                    error: _errorCarga,
                  ),
                ),
                if (_errorCarga == null && !_cargando)
                  Positioned.fill(
                    child: CustomPaint(
                      painter: _PintorManos(_manoIzq, _manoDer),
                    ),
                  ),
                if (_maquina != null && _errorCarga == null && !_cargando)
                  Positioned(
                    top: 8,
                    left: 8,
                    child: _HudDiagnostico(maquina: _maquina!),
                  ),
              ],
            ),
          ),
          _AreaResultados(texto: _maquina?.texto ?? ''),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Widgets internos
// ---------------------------------------------------------------------------

class _AreaCamara extends StatelessWidget {
  final ControladorCamara camara;
  final bool cargando;
  final String? error;

  const _AreaCamara({
    required this.camara,
    required this.cargando,
    required this.error,
  });

  @override
  Widget build(BuildContext context) {
    final colores = Theme.of(context).colorScheme;

    if (error != null) {
      return _PanelEstado(
        icono: Icons.error_outline_rounded,
        color: colores.error,
        titulo: 'No se pudo iniciar',
        detalle: error!,
      );
    }

    if (cargando || !camara.estaInicializado) {
      return const Center(child: CircularProgressIndicator());
    }

    return ClipRRect(child: CameraPreview(camara.controlador!));
  }
}

class _PanelEstado extends StatelessWidget {
  final IconData icono;
  final Color color;
  final String titulo;
  final String detalle;

  const _PanelEstado({
    required this.icono,
    required this.color,
    required this.titulo,
    required this.detalle,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icono, color: color, size: 56),
            const SizedBox(height: 16),
            Text(
              titulo,
              textAlign: TextAlign.center,
              style: Theme.of(context)
                  .textTheme
                  .titleLarge
                  ?.copyWith(color: color),
            ),
            const SizedBox(height: 10),
            Text(
              detalle,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyMedium,
            ),
          ],
        ),
      ),
    );
  }
}

/// Dibuja el esqueleto de la(s) mano(s) sobre el preview, con los landmarks
/// normalizados [0,1] que devuelve MediaPipe. Sirve para VER si la orientación es
/// correcta: si el esqueleto cae sobre la mano real, el fotograma que recibe el
/// modelo está bien orientado; si aparece girado o espejado, hay que corregir la
/// rotación en el puente nativo.
class _PintorManos extends CustomPainter {
  final List<Punto>? izquierda;
  final List<Punto>? derecha;

  _PintorManos(this.izquierda, this.derecha);

  // Conexiones de los 21 landmarks de la mano (topología de MediaPipe Hands).
  static const List<List<int>> _huesos = [
    [0, 1], [1, 2], [2, 3], [3, 4], // pulgar
    [0, 5], [5, 6], [6, 7], [7, 8], // índice
    [5, 9], [9, 10], [10, 11], [11, 12], // medio
    [9, 13], [13, 14], [14, 15], [15, 16], // anular
    [13, 17], [17, 18], [18, 19], [19, 20], // meñique
    [0, 17], // base de la palma
  ];

  @override
  void paint(Canvas canvas, Size size) {
    for (final mano in [izquierda, derecha]) {
      if (mano == null || mano.length != Constantes.numLandmarks) continue;
      _dibujarMano(canvas, size, mano);
    }
  }

  void _dibujarMano(Canvas canvas, Size size, List<Punto> mano) {
    final lineas = Paint()
      ..color = const Color(0xCC00E5A0)
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;
    final puntos = Paint()..color = const Color(0xFFFFFFFF);

    Offset aPantalla(Punto p) => Offset(p.x * size.width, p.y * size.height);

    for (final hueso in _huesos) {
      canvas.drawLine(aPantalla(mano[hueso[0]]), aPantalla(mano[hueso[1]]), lineas);
    }
    for (final p in mano) {
      canvas.drawCircle(aPantalla(p), 4, puntos);
    }
  }

  @override
  bool shouldRepaint(_PintorManos old) =>
      old.izquierda != izquierda || old.derecha != derecha;
}

/// HUD de diagnóstico para la prueba en el teléfono: cuántas manos ve MediaPipe,
/// la letra candidata, el movimiento y la confianza. Sirve para saber si el
/// problema es la cámara/detección o el reconocimiento. Se quita en la versión
/// final para niños.
class _HudDiagnostico extends StatelessWidget {
  final MaquinaEstados maquina;

  const _HudDiagnostico({required this.maquina});

  @override
  Widget build(BuildContext context) {
    final hayManos = maquina.manosVisibles > 0;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.55),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                hayManos ? Icons.front_hand_rounded : Icons.do_not_touch_rounded,
                size: 16,
                color: hayManos ? Colors.greenAccent : Colors.white38,
              ),
              const SizedBox(width: 6),
              Text(
                'Manos: ${maquina.manosVisibles}',
                style: const TextStyle(color: Colors.white, fontSize: 13),
              ),
            ],
          ),
          const SizedBox(height: 2),
          Text(
            maquina.candidato.isEmpty
                ? 'Candidata: -'
                : 'Candidata: ${maquina.candidato} (${(maquina.confianza * 100).toStringAsFixed(0)}%)',
            style: const TextStyle(color: Colors.white, fontSize: 13),
          ),
          Text(
            'Mov: ${maquina.movimiento.toStringAsFixed(4)}',
            style: const TextStyle(color: Colors.white70, fontSize: 12),
          ),
          Text(
            'Ventana: ${maquina.nivelVentana}/${Constantes.tamanoVentanaA}',
            style: TextStyle(
              color: maquina.nivelVentana >= Constantes.tamanoVentanaA
                  ? Colors.greenAccent
                  : Colors.white70,
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
          Text(
            'Det/s: ${maquina.fps.toStringAsFixed(1)}',
            style: TextStyle(
              color: maquina.fps >= 15
                  ? Colors.greenAccent
                  : (maquina.fps >= 8 ? Colors.amberAccent : Colors.redAccent),
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }
}

class _AreaResultados extends StatelessWidget {
  final String texto;

  const _AreaResultados({required this.texto});

  @override
  Widget build(BuildContext context) {
    final colores = Theme.of(context).colorScheme;

    return Container(
      color: Colors.white,
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
      child: Row(
        children: [
          Text(
            'Texto:',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: colores.onSurface.withValues(alpha: 0.5),
                ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              texto.isEmpty ? '---' : texto,
              style: Theme.of(context).textTheme.displayLarge?.copyWith(
                    fontSize: 32,
                    color: texto.isEmpty
                        ? colores.onSurface.withValues(alpha: 0.2)
                        : colores.onSurface,
                    letterSpacing: 4,
                  ),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }
}
