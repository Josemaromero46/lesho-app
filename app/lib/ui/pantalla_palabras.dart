import 'dart:async';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:lesho_app/captura/controlador_camara.dart';
import 'package:lesho_app/control/reconocedor_palabra.dart';
import 'package:lesho_app/core/constantes.dart';
import 'package:lesho_app/core/marco.dart';
import 'package:lesho_app/core/normalizacion.dart';
import 'package:lesho_app/inferencia/cargador_modelos.dart';
import 'package:lesho_app/inferencia/modelo_b.dart';
import 'package:lesho_app/landmarks/detector_manos.dart';

/// Pantalla de PRUEBA del Modelo B (señas dinámicas). Graba una seña con un botón
/// y muestra la palabra reconocida. Es un banco de pruebas, no la UI final.
class PantallaPalabras extends StatefulWidget {
  const PantallaPalabras({super.key});

  @override
  State<PantallaPalabras> createState() => _EstadoPantallaPalabras();
}

class _EstadoPantallaPalabras extends State<PantallaPalabras>
    with WidgetsBindingObserver {
  final _camara = ControladorCamara();
  final _detector = DetectorManos();
  final _cargador = CargadorModelos();
  ReconocedorPalabra? _recon;

  bool _cargando = true;
  // Falso hasta que MediaPipe termina de inicializar (tarda en gama baja). El
  // preview ya se ve mientras tanto.
  bool _reconocimientoListo = false;
  String? _error;
  bool _procesando = false;
  bool _reconociendo = false; // procesando la secuencia tras soltar el botón
  Timer? _timer;

  List<Punto>? _izq;
  List<Punto>? _der;
  List<Punto3D>? _pose;
  ResultadoPalabra? _resultado;
  int _contadorFrame = 0;

  // Corte automático de grabación. Amplio a propósito: en gama baja (~5 fps)
  // conviene hacer la seña más lento para que caiga en más fotogramas.
  static const int _maxMs = 10000;
  double _inicioGrabMs = 0;

  // Cola para serializar el soltar/reabrir de la cámara en los cambios de ciclo de
  // vida (evita que se pisen si el usuario sale y entra muy rápido).
  Future<void> _colaCamara = Future.value();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
    _inicializar();
  }

  // La cámara la libera el sistema cuando la app pasa a segundo plano. Hay que
  // soltarla al salir y reabrirla al volver; si no, el preview queda congelado en
  // el último fotograma.
  @override
  void didChangeAppLifecycleState(AppLifecycleState estado) {
    if (estado == AppLifecycleState.resumed) {
      _encolarCamara(reanudar: true);
    } else if (estado == AppLifecycleState.inactive ||
        estado == AppLifecycleState.paused ||
        estado == AppLifecycleState.hidden) {
      _encolarCamara(reanudar: false);
    }
  }

  void _encolarCamara({required bool reanudar}) {
    _colaCamara =
        _colaCamara.then((_) => reanudar ? _reabrirCamara() : _soltarCamara());
  }

  Future<void> _soltarCamara() async {
    if (!_camara.estaInicializado) return;
    await _camara.liberar();
    if (mounted) setState(() {});
  }

  Future<void> _reabrirCamara() async {
    if (_cargando || _error != null || _camara.estaInicializado) return;
    try {
      await _camara.inicializar();
      _detector.rotacion = _camara.orientacionSensor;
      await _arrancarDeteccionSiListo();
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    }
    if (mounted) setState(() {});
  }

  // Arranca la detección en vivo si la cámara y el detector están listos y el flujo
  // no corre ya (al terminar la carga o al reabrir la cámara tras segundo plano).
  Future<void> _arrancarDeteccionSiListo() async {
    if (_camara.estaInicializado &&
        _detector.estaInicializado &&
        !_camara.estaFluyendo) {
      await _camara.iniciarFlujo(_procesarFrame);
    }
    if (mounted) setState(() => _reconocimientoListo = _detector.estaInicializado);
  }

  Future<void> _inicializar() async {
    try {
      final permiso = await Permission.camera.request();
      if (!permiso.isGranted) throw StateError('Permiso de cámara denegado.');

      // La cámara primero, y mostrar el preview de una vez; MediaPipe (la Pose con
      // GPU) tarda en gama baja y se prepara en segundo plano.
      await _camara.inicializar();
      _detector.rotacion = _camara.orientacionSensor;
      if (mounted) setState(() => _cargando = false);

      await _cargador.cargar();
      if (!_cargador.tieneModeloB) {
        throw StateError('Falta modelo_b.tflite en assets/models/.');
      }
      _recon = ReconocedorPalabra(ModeloB(_cargador));
      _recon!.onDeteccionVivo = (d) {
        _izq = d.manoIzquierda;
        _der = d.manoDerecha;
        if (d.pose != null) _pose = d.pose;
      };
      // Con Pose (Modelo B) y dos manos.
      await _detector.inicializar(numManos: 2, conPose: true);

      await _arrancarDeteccionSiListo();

      _timer = Timer.periodic(const Duration(milliseconds: 120), (_) {
        if (!mounted) return;
        // Corte automático si se pasa del tope.
        if ((_recon?.grabando ?? false) &&
            !_reconociendo &&
            DateTime.now().millisecondsSinceEpoch - _inicioGrabMs > _maxMs) {
          _detenerGrabacion();
        }
        setState(() {});
      });
    } catch (e) {
      if (mounted) setState(() {
        _cargando = false;
        _error = e.toString();
      });
    }
  }

  Future<void> _procesarFrame(CameraImage imagen) async {
    if (_recon == null) return;
    // Mientras se procesa la secuencia tras soltar, no correr detección en vivo
    // (competiría con el procesamiento fuera de línea por el detector).
    if (_reconociendo) return;

    // Durante la grabación, el reconocedor muestrea los fotogramas que necesita y
    // los procesa EN PARALELO (ver ReconocedorPalabra). Solo se copian los bytes
    // cuando los pide. El esqueleto se dibuja con esas mismas detecciones
    // (onDeteccionVivo), sin correr detecciones extra que compitan por el detector.
    if (_recon!.grabando) {
      if (_recon!.necesitaFrame()) {
        final bytes = Uint8List.fromList(imagen.planes.first.bytes);
        _recon!.agregarFrameCrudo(bytes, imagen.width, imagen.height);
      }
      return;
    }

    // Fuera de grabación: detección en vivo para el esqueleto y el HUD.
    if (_procesando) return;
    _procesando = true;
    try {
      _contadorFrame++;
      final pedirPose = _contadorFrame % 3 == 0;
      final d = await _detector.procesar(imagen, conPose: pedirPose);
      _izq = d.manoIzquierda;
      _der = d.manoDerecha;
      if (d.pose != null) _pose = d.pose;
    } finally {
      _procesando = false;
    }
  }

  void _alternarGrabacion() {
    if (_recon == null || _reconociendo) return;
    if (_recon!.grabando) {
      _detenerGrabacion();
    } else {
      _recon!.iniciar(_detector);
      _inicioGrabMs = DateTime.now().millisecondsSinceEpoch.toDouble();
      setState(() => _resultado = null);
    }
  }

  Future<void> _detenerGrabacion() async {
    if (_reconociendo) return;
    setState(() => _reconociendo = true);
    final r = await _recon!.detener();
    if (mounted) {
      setState(() {
        _resultado = r;
        _reconociendo = false;
      });
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _timer?.cancel();
    _camara.liberar();
    _detector.liberar();
    _cargador.liberar();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final colores = Theme.of(context).colorScheme;
    final grabando = _recon?.grabando ?? false;

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        foregroundColor: colores.onSurface,
        elevation: 0,
        title: const Text('Señas (prueba)'),
      ),
      body: Column(
        children: [
          Expanded(
            child: Stack(
              children: [
                Positioned.fill(child: _areaCamara()),
                if (_error == null && !_cargando)
                  Positioned.fill(
                    child: CustomPaint(painter: _PintorCuerpo(_izq, _der, _pose)),
                  ),
                if (_error == null && !_cargando)
                  Positioned(top: 8, left: 8, child: _hud(grabando)),
                if (!_reconocimientoListo && _error == null && !_cargando)
                  const Positioned.fill(
                    child: Center(child: _ChipPreparando()),
                  ),
              ],
            ),
          ),
          _panelInferior(grabando, colores),
        ],
      ),
    );
  }

  Widget _areaCamara() {
    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Text(_error!, textAlign: TextAlign.center),
        ),
      );
    }
    if (_cargando || !_camara.estaInicializado) {
      return const Center(child: CircularProgressIndicator());
    }
    return ClipRRect(child: CameraPreview(_camara.controlador!));
  }

  Widget _hud(bool grabando) {
    final hayPose = _pose != null;
    final manos = (_izq != null ? 1 : 0) + (_der != null ? 1 : 0);
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
          Text('Manos: $manos',
              style: const TextStyle(color: Colors.white, fontSize: 13)),
          Text('Cuerpo: ${hayPose ? "sí" : "no"}',
              style: TextStyle(
                  color: hayPose ? Colors.greenAccent : Colors.white54,
                  fontSize: 13)),
          if (grabando)
            Text('Grabando: ${_recon!.framesCrudos} fotogramas',
                style: const TextStyle(
                    color: Colors.redAccent,
                    fontSize: 13,
                    fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }

  Widget _panelInferior(bool grabando, ColorScheme colores) {
    return Container(
      color: Colors.white,
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (_resultado != null)
            Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: Column(
                children: [
                  Text(
                    '${_resultado!.sena}  (${(_resultado!.confianza * 100).toStringAsFixed(0)}%)',
                    style: Theme.of(context).textTheme.displaySmall?.copyWith(
                          color: colores.tertiary,
                          fontWeight: FontWeight.bold,
                        ),
                  ),
                  Text(
                    '${_recon!.frames} fotogramas · ${_recon!.framesConMarco} con cuerpo',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: colores.onSurface.withValues(alpha: 0.5),
                        ),
                  ),
                ],
              ),
            )
          else
            Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: Text('Presioná para grabar una seña',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: colores.onSurface.withValues(alpha: 0.5),
                      )),
            ),
          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              style: FilledButton.styleFrom(
                backgroundColor: grabando ? colores.error : colores.primary,
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
              onPressed: (!_reconocimientoListo || _error != null || _reconociendo)
                  ? null
                  : _alternarGrabacion,
              icon: _reconociendo
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: Colors.white),
                    )
                  : Icon(grabando
                      ? Icons.stop_rounded
                      : Icons.fiber_manual_record),
              label: Text(_reconociendo
                  ? 'Reconociendo...'
                  : (grabando ? 'Detener y reconocer' : 'Grabar seña')),
            ),
          ),
        ],
      ),
    );
  }
}

/// Aviso centrado mientras MediaPipe inicializa (el preview ya se ve detrás).
class _ChipPreparando extends StatelessWidget {
  const _ChipPreparando();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.6),
        borderRadius: BorderRadius.circular(12),
      ),
      child: const Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: 18,
            height: 18,
            child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
          ),
          SizedBox(width: 10),
          Text(
            'Preparando reconocimiento...',
            style: TextStyle(
                color: Colors.white, fontSize: 14, fontWeight: FontWeight.w600),
          ),
        ],
      ),
    );
  }
}

/// Dibuja las manos (esqueleto) y los puntos de la pose sobre el preview.
class _PintorCuerpo extends CustomPainter {
  final List<Punto>? izq;
  final List<Punto>? der;
  final List<Punto3D>? pose;

  _PintorCuerpo(this.izq, this.der, this.pose);

  static const List<List<int>> _huesos = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17],
  ];

  // Solo los puntos de pose que el modelo REALMENTE usa para la ubicación:
  // hombros (origen del marco), nariz, ojos, orejas y boca (anclas de la cara).
  static const List<int> _poseUsados = [
    Constantes.poseHombroIzq, Constantes.poseHombroDer,
    Constantes.poseNariz,
    Constantes.poseOjoIzq, Constantes.poseOjoDer,
    Constantes.poseOrejaIzq, Constantes.poseOrejaDer,
    Constantes.poseBocaIzq, Constantes.poseBocaDer,
  ];

  @override
  void paint(Canvas canvas, Size size) {
    final p = pose;
    if (p != null && p.length == Constantes.numLandmarksPose) {
      final pincel = Paint()..color = const Color(0xCCFFD166);
      // Línea entre hombros (base del marco del cuerpo).
      final hi = p[Constantes.poseHombroIzq];
      final hd = p[Constantes.poseHombroDer];
      canvas.drawLine(
        Offset(hi.x * size.width, hi.y * size.height),
        Offset(hd.x * size.width, hd.y * size.height),
        Paint()
          ..color = const Color(0x99FFD166)
          ..strokeWidth = 3,
      );
      for (final i in _poseUsados) {
        final punto = p[i];
        if (punto.visibilidad >= Constantes.visibilidadMinimaPose) {
          canvas.drawCircle(
              Offset(punto.x * size.width, punto.y * size.height), 5, pincel);
        }
      }
    }
    for (final mano in [izq, der]) {
      if (mano == null || mano.length != Constantes.numLandmarks) continue;
      _mano(canvas, size, mano);
    }
  }

  void _mano(Canvas canvas, Size size, List<Punto> mano) {
    final lineas = Paint()
      ..color = const Color(0xCC00E5A0)
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;
    final puntos = Paint()..color = const Color(0xFFFFFFFF);
    Offset a(Punto pt) => Offset(pt.x * size.width, pt.y * size.height);
    for (final h in _huesos) {
      canvas.drawLine(a(mano[h[0]]), a(mano[h[1]]), lineas);
    }
    for (final pt in mano) {
      canvas.drawCircle(a(pt), 4, puntos);
    }
  }

  @override
  bool shouldRepaint(_PintorCuerpo old) =>
      old.izq != izq || old.der != der || old.pose != pose;
}
