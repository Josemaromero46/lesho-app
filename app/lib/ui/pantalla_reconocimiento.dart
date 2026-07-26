import 'dart:async';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:lesho_app/core/constantes.dart';
import 'package:lesho_app/core/etiquetas_legibles.dart';
import 'package:lesho_app/core/normalizacion.dart';
import 'package:lesho_app/captura/controlador_camara.dart';
import 'package:lesho_app/control/maquina_estados.dart';
import 'package:lesho_app/control/reconocedor_palabra.dart';
import 'package:lesho_app/inferencia/cargador_modelos.dart';
import 'package:lesho_app/inferencia/modelo_a.dart';
import 'package:lesho_app/inferencia/modelo_b.dart';
import 'package:lesho_app/landmarks/detector_manos.dart';

/// Pantalla del niño (Dirección 1: LESHO -> español). Une el DELETREO (Modelo A,
/// letra por letra) con las PALABRAS (Modelo B, señas dinámicas).
///
/// - Deletreo: por defecto, cada letra que se firma se agrega al texto.
/// - Palabra: al tocar "Grabar palabra" se abre una grabación; los fotogramas se
///   guardan y, al tocar "Detener", se procesan con el Modelo B y la palabra se
///   agrega al texto. (Las señas INICIO/FIN se quitaron por falsos positivos.)
/// - Botones: espacio (separa palabras), borrar y limpiar (en la barra superior).
class PantallaReconocimiento extends StatefulWidget {
  const PantallaReconocimiento({super.key});

  @override
  State<PantallaReconocimiento> createState() => _EstadoPantallaReconocimiento();
}

class _EstadoPantallaReconocimiento extends State<PantallaReconocimiento>
    with WidgetsBindingObserver {
  final _camara = ControladorCamara();
  final _detector = DetectorManos();
  final _cargador = CargadorModelos();
  MaquinaEstados? _maquina;
  ReconocedorPalabra? _recon;

  bool _cargando = true;
  // Falso hasta que MediaPipe termina de inicializar (en gama baja tarda varios
  // segundos). Mientras tanto el preview YA se ve, pero la detección aún no corre.
  bool _reconocimientoListo = false;
  String? _errorCarga;
  bool _procesandoFrame = false;
  // Verdadero mientras se procesa la secuencia de una palabra tras cerrarla.
  bool _reconociendoPalabra = false;
  Timer? _timerHud;
  // Cola para serializar el soltar/reabrir de la cámara en los cambios de ciclo de
  // vida (evita que se pisen si el usuario sale y entra muy rápido).
  Future<void> _colaCamara = Future.value();

  // Últimas manos detectadas, para dibujar el esqueleto sobre el preview.
  List<Punto>? _manoIzq;
  List<Punto>? _manoDer;

  // Recuadro de texto: control de scroll para bajar al final cuando entra texto.
  final ScrollController _scrollTexto = ScrollController();
  static const double _fontTexto = 26;
  static const double _lineaTexto = 1.35;
  static const double _altoTresLineas = _fontTexto * _lineaTexto * 3;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    // La rotación del fotograma asume orientación vertical.
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
    if (_cargando || _errorCarga != null || _camara.estaInicializado) return;
    try {
      await _camara.inicializar();
      _detector.rotacion = _camara.orientacionSensor;
      await _arrancarDeteccionSiListo();
    } catch (e) {
      if (mounted) setState(() => _errorCarga = e.toString());
    }
    if (mounted) setState(() {});
  }

  // Arranca la detección en vivo si la cámara y el detector están listos y el flujo
  // no corre ya. Sirve tanto al terminar la carga como al reabrir la cámara tras
  // volver de segundo plano (en cualquier orden que hayan quedado listos).
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
      if (!permiso.isGranted) {
        throw StateError('Permiso de cámara denegado.');
      }

      // 1) La CÁMARA primero, y mostrar el preview de una vez. MediaPipe (sobre
      //    todo la Pose con GPU) tarda varios segundos en inicializar en gama baja;
      //    no hay que hacer esperar al usuario mirando una pantalla de carga. Se ve
      //    la cámara y se ubica mientras el reconocimiento se prepara en segundo
      //    plano. El preview de la cámara no necesita el flujo de fotogramas.
      await _camara.inicializar();
      _detector.rotacion = _camara.orientacionSensor;
      if (mounted) setState(() => _cargando = false);

      // 2) Modelos (rápido) y detector MediaPipe (lento) en segundo plano.
      await _cargador.cargar();
      _maquina = MaquinaEstados(modeloA: ModeloA(_cargador));
      _maquina!.onTextoCambiado = _alTextoCambiado;
      // El Modelo B (palabras) es opcional: si falta, solo funciona el deletreo.
      if (_cargador.tieneModeloB) {
        _recon = ReconocedorPalabra(ModeloB(_cargador));
        // El esqueleto durante la grabación se dibuja con las detecciones que el
        // reconocedor ya hace, sin correr detecciones aparte.
        _recon!.onDeteccionVivo = (d) {
          _manoIzq = d.manoIzquierda;
          _manoDer = d.manoDerecha;
        };
        // Las señas INICIO/FIN se quitaron: daban falsos positivos (algunas señas
        // que juntan las puntas de los dedos se confundían con FIN y cerraban la
        // palabra a media seña). La palabra ahora se abre y se cierra SOLO con el
        // botón "Grabar palabra" / "Detener".
      }
      // Dos manos (para INICIO/FIN y las señas bimanuales) y Pose (para el Modelo
      // B, que la usa al procesar la palabra al soltar).
      await _detector.inicializar(numManos: 2, conPose: _recon != null);

      // 3) Ya está todo listo: arranca la detección en vivo (si la cámara sigue
      //    abierta; si se fue a segundo plano durante la carga, la reabre el ciclo
      //    de vida y arranca ahí).
      await _arrancarDeteccionSiListo();

      // Refresca el HUD de diagnóstico a ~7 Hz, sin atarlo al ritmo de frames.
      _timerHud = Timer.periodic(const Duration(milliseconds: 150), (_) {
        if (mounted) setState(() {});
      });
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

  // Baja el recuadro de texto al final para mostrar lo último escrito.
  void _alTextoCambiado() {
    if (mounted) setState(() {});
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollTexto.hasClients) {
        _scrollTexto.jumpTo(_scrollTexto.position.maxScrollExtent);
      }
    });
  }

  Future<void> _procesarFrame(CameraImage imagen) async {
    if (_maquina == null) return;
    // Mientras se procesa la palabra tras soltar, no correr detección en vivo.
    if (_reconociendoPalabra) return;

    // Durante la grabación de una palabra: el reconocedor muestrea los fotogramas
    // que necesita y los va procesando EN PARALELO (ver ReconocedorPalabra). Aquí
    // solo se le entrega el fotograma cuando lo pide, y la copia de bytes se hace
    // únicamente en ese caso (la cámara reutiliza su buffer y copiar cuesta). El
    // Modelo A no corre: la palabra se cierra con el botón "Detener". El esqueleto
    // se dibuja con las detecciones que el propio reconocedor va produciendo
    // (onDeteccionVivo), sin gastar detecciones extra.
    if (_recon?.grabando ?? false) {
      if (_recon!.necesitaFrame()) {
        final bytes = Uint8List.fromList(imagen.planes.first.bytes);
        _recon!.agregarFrameCrudo(bytes, imagen.width, imagen.height);
      }
      return;
    }

    // Fuera de grabación: la detección en vivo (Modelo A: deletreo, INICIO,
    // esqueleto) corre cuando no está ocupada; si lo está, se saltea esa detección.
    if (_procesandoFrame) return;
    _procesandoFrame = true;
    try {
      final deteccion = await _detector.procesar(imagen, conPose: false);
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

  // Abre la grabación de una palabra (por el botón "Grabar palabra").
  void _iniciarPalabra() {
    if (_recon == null || _recon!.grabando || _reconociendoPalabra) return;
    _recon!.iniciar(_detector);
    _maquina!.modoPalabra = true;
    setState(() {});
  }

  // Cierra la grabación, corre el Modelo B y agrega la palabra al texto.
  Future<void> _terminarPalabra() async {
    if (_recon == null || !_recon!.grabando) return;
    _maquina!.modoPalabra = false;
    setState(() => _reconociendoPalabra = true);
    final resultado = await _recon!.detener();
    if (!mounted) return;
    if (resultado != null) {
      // La clase del modelo va sin tildes por contrato; en pantalla se escribe
      // con la ortografía correcta (MAMA -> MAMÁ, POR_FAVOR -> POR FAVOR).
      _maquina!.agregarPalabra(textoLegible(resultado.sena));
    }
    setState(() => _reconociendoPalabra = false);
  }

  void _alternarPalabra() {
    if (_recon == null || _reconociendoPalabra) return;
    if (_recon!.grabando) {
      _terminarPalabra();
    } else {
      _iniciarPalabra();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _timerHud?.cancel();
    _scrollTexto.dispose();
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
        title: const Text('El niño firma'),
        actions: [
          if (_maquina != null)
            IconButton(
              icon: const Icon(Icons.backspace_rounded),
              tooltip: 'Borrar último',
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
                // Aviso de grabación de palabra sobre la cámara.
                if (grabando || _reconociendoPalabra)
                  Positioned(
                    top: 8,
                    right: 8,
                    child: _AvisoPalabra(
                      reconociendo: _reconociendoPalabra,
                    ),
                  ),
                // Mientras MediaPipe inicializa: el preview ya se ve, pero avisamos
                // que el reconocimiento se está preparando.
                if (!_reconocimientoListo && _errorCarga == null && !_cargando)
                  const Positioned.fill(
                    child: Center(child: _ChipPreparando()),
                  ),
              ],
            ),
          ),
          _panelInferior(colores, grabando),
        ],
      ),
    );
  }

  Widget _panelInferior(ColorScheme colores, bool grabando) {
    final texto = _maquina?.texto ?? '';
    final hayModeloB = _recon != null;

    return Container(
      color: Colors.white,
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Recuadro de texto: crece según la necesidad hasta 3 líneas, y de ahí
          // se puede subir/bajar con scroll (baja solo cuando entra texto nuevo).
          ConstrainedBox(
            constraints: const BoxConstraints(maxHeight: _altoTresLineas),
            child: SingleChildScrollView(
              controller: _scrollTexto,
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  texto.isEmpty ? 'Firma para escribir...' : texto,
                  style: TextStyle(
                    fontSize: _fontTexto,
                    height: _lineaTexto,
                    letterSpacing: 2,
                    fontWeight: FontWeight.w600,
                    color: texto.isEmpty
                        ? colores.onSurface.withValues(alpha: 0.25)
                        : colores.onSurface,
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              // Botón de espacio.
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: !_reconocimientoListo || _errorCarga != null
                      ? null
                      : () => _maquina?.agregarEspacio(),
                  icon: const Icon(Icons.space_bar_rounded),
                  label: const Text('Espacio'),
                  style: OutlinedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 14),
                  ),
                ),
              ),
              const SizedBox(width: 12),
              // Botón de grabar/detener palabra (Modelo B).
              Expanded(
                flex: 2,
                child: FilledButton.icon(
                  style: FilledButton.styleFrom(
                    backgroundColor: grabando ? colores.error : colores.primary,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                  ),
                  onPressed: (!hayModeloB || _reconociendoPalabra ||
                          !_reconocimientoListo || _errorCarga != null)
                      ? null
                      : _alternarPalabra,
                  icon: _reconociendoPalabra
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(
                              strokeWidth: 2, color: Colors.white),
                        )
                      : Icon(grabando
                          ? Icons.stop_rounded
                          : Icons.fiber_manual_record),
                  label: Text(_reconociendoPalabra
                      ? 'Reconociendo...'
                      : (grabando ? 'Detener' : 'Grabar palabra')),
                ),
              ),
            ],
          ),
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

/// Aviso sobre la cámara mientras se graba o se reconoce una palabra.
class _AvisoPalabra extends StatelessWidget {
  final bool reconociendo;

  const _AvisoPalabra({required this.reconociendo});

  @override
  Widget build(BuildContext context) {
    final rojo = reconociendo ? Colors.amber : Colors.redAccent;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.6),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(reconociendo ? Icons.hourglass_top_rounded : Icons.circle,
              size: 12, color: rojo),
          const SizedBox(width: 6),
          Text(
            reconociendo ? 'Reconociendo' : 'Grabando palabra',
            style: TextStyle(color: rojo, fontSize: 13, fontWeight: FontWeight.w600),
          ),
        ],
      ),
    );
  }
}

/// Aviso centrado mientras MediaPipe inicializa. El preview de la cámara ya se ve
/// detrás; esto solo indica que la detección se está preparando.
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

/// Dibuja el esqueleto de la(s) mano(s) sobre el preview, con los landmarks
/// normalizados [0,1] que devuelve MediaPipe.
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

/// HUD de diagnóstico para la prueba en el teléfono: manos, letra candidata,
/// movimiento y confianza. Se quita en la versión final para niños.
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
