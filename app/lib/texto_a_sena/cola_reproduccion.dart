import 'package:flutter/foundation.dart';
import 'package:lesho_app/texto_a_sena/diccionario_videos.dart';
import 'package:lesho_app/texto_a_sena/tokenizador.dart';
import 'package:video_player/video_player.dart';

/// Arma y reproduce la cola de videos correspondiente a una frase completa.
///
/// Los videos se reproducen en secuencia: una palabra (o sus letras en
/// fallback) antes de pasar a la siguiente.
class ColaReproduccion extends ChangeNotifier {
  final _tokenizador = Tokenizador();
  final _diccionario = DiccionarioVideos();

  VideoPlayerController? _controlador;
  List<String> _cola = [];
  int _indiceActual = 0;
  bool _reproduciendo = false;

  VideoPlayerController? get controlador => _controlador;
  bool get reproduciendo => _reproduciendo;
  bool get terminada => !_reproduciendo && _cola.isNotEmpty;
  int get totalVideos => _cola.length;
  int get videoActual => _indiceActual.clamp(0, _cola.length);

  /// Tokeniza [frase], arma la cola de rutas de video y empieza la reproducción.
  Future<void> reproducirFrase(String frase) async {
    await _detenerYLimpiar();

    _cola = [];
    for (final palabra in _tokenizador.tokenizar(frase)) {
      _cola.addAll(_diccionario.resolver(palabra));
    }
    if (_cola.isEmpty) return;

    _indiceActual = 0;
    _reproduciendo = true;
    notifyListeners();

    await _reproducirIndiceActual();
  }

  /// Detiene la reproducción en curso.
  Future<void> detener() async => _detenerYLimpiar();

  Future<void> _reproducirIndiceActual() async {
    if (_indiceActual >= _cola.length) {
      await _detenerYLimpiar();
      return;
    }

    final ruta = _cola[_indiceActual];

    await _controlador?.dispose();
    final nuevo = VideoPlayerController.asset(ruta);
    _controlador = nuevo;
    notifyListeners();

    await nuevo.initialize();
    await nuevo.setLooping(false);

    nuevo.addListener(() {
      if (!nuevo.value.isPlaying &&
          nuevo.value.position >= nuevo.value.duration &&
          nuevo.value.duration > Duration.zero) {
        _indiceActual++;
        _reproducirIndiceActual();
      }
    });

    await nuevo.play();
  }

  Future<void> _detenerYLimpiar() async {
    await _controlador?.dispose();
    _controlador = null;
    _reproduciendo = false;
    notifyListeners();
  }

  @override
  void dispose() {
    _controlador?.dispose();
    super.dispose();
  }
}
