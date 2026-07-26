import 'package:lesho_app/core/constantes.dart';
import 'package:lesho_app/core/normalizacion.dart';
import 'package:lesho_app/core/suavizado.dart';
import 'package:lesho_app/inferencia/modelo_a.dart';

/// Reconocimiento del alfabeto (Modelo A) en tiempo real.
///
/// CONTRATO: replica la lógica de `demo_deletreo.py` (la referencia validada del
/// deletreo), para que la app se comporte igual que la demo de escritorio. Por
/// cada fotograma con manos:
///   1. Traslación respecto a la muñeca (vector de 126).
///   2. Suavizado One Euro del vector (igual que en el preprocesamiento).
///   3. Escala (invariante a la distancia) y a la ventana rodante de 20.
///   4. Con la ventana llena, corre el Modelo A y aplica la COMPUERTA DE
///      MOVIMIENTO (bloquea letras de movimiento si la ventana está quieta, y las
///      estáticas gemelas si se mueve claro).
///   5. Persistencia (5 ventanas) + cooldown (1200 ms) para confirmar.
///
/// Al confirmar: una letra se agrega al texto. INICIO (dos palmas abiertas) y FIN
/// (dos puños) NO editan el texto: disparan callbacks (onInicioSena / onFinSena)
/// para que la pantalla abra y cierre la grabación de una palabra dinámica (Modelo
/// B). El espacio y el borrado se manejan con botones. REPOSO no hace nada.
class MaquinaEstados {
  final ModeloA _modeloA;

  final FiltroUnEuro _filtroVector = FiltroUnEuro(
    fps: Constantes.fpsObjetivo.toDouble(),
    minCutoff: Constantes.suavizadoMinCutoff,
    beta: Constantes.suavizadoBeta,
    dCutoff: Constantes.suavizadoDCutoff,
  );

  // Buffer con marca de tiempo: cada muestra es (tMs, vector escalado). En vez de
  // usar los últimos 20 fotogramas crudos (que a bajo fps cubren demasiado tiempo
  // y arrastran), se toma el último tramo de _duracionVentanaMs y se remuestrea a
  // 20 fotogramas, para que el modelo vea siempre ~0.67s como en el entrenamiento.
  final List<(double, List<double>)> _buffer = [];
  // Duración de la ventana que vio el modelo al entrenar: 20 fotogramas a 30 fps.
  static const double _duracionVentanaMs =
      Constantes.tamanoVentanaA / Constantes.fpsObjetivo * 1000.0;

  final List<String> _texto = [];

  // Persistencia + cooldown (equivalente a la clase Deletreador de la demo).
  String? _candidato;
  int _contador = 0;
  double _finCooldownMs = 0;

  // Índices de clase para la compuerta de movimiento (se calculan una vez, al
  // tener las etiquetas del modelo cargadas).
  List<int> _idxMovimiento = const [];
  List<int> _idxGemelas = const [];
  bool _indicesListos = false;

  /// Se llama al confirmar una letra (no en FIN/INICIO/REPOSO).
  void Function(String letra)? onLetraAgregada;

  /// Se llama cada vez que el texto cambia (letra, espacio, palabra o borrado).
  void Function()? onTextoCambiado;

  /// Modo palabra: entre INICIO y FIN, los fotogramas son para el Modelo B y no se
  /// confirman letras. Lo prende y apaga la pantalla.
  bool modoPalabra = false;

  /// Se dispara al confirmar la seña INICIO (dos palmas abiertas): la pantalla abre
  /// la grabación de una palabra dinámica.
  void Function()? onInicioSena;

  /// Se dispara al confirmar la seña FIN (dos puños) estando en modo palabra: la
  /// pantalla cierra la grabación y corre el Modelo B.
  void Function()? onFinSena;

  // Diagnóstico en vivo (para el HUD de prueba, no forma parte del texto).
  int manosVisibles = 0;
  String candidato = '';
  double movimiento = 0.0;
  double confianza = 0.0;
  double fps = 0.0;
  int nivelVentana = 0;
  double _ultimoFrameMs = 0.0;
  // Tiempo real (segundos) entre fotogramas procesados. En el teléfono la
  // detección corre más lento que 30 fps; con este dt real se corrigen el
  // suavizado y la medida de movimiento para que se comporten como a 30 fps.
  static const double _dtNominal = 1.0 / 30.0;
  double _dtSeg = _dtNominal;

  MaquinaEstados({required ModeloA modeloA}) : _modeloA = modeloA;

  // Mide el ritmo real de fotogramas PROCESADOS (los que completan la detección),
  // con un suavizado exponencial. Es el fps efectivo del reconocimiento, y de
  // paso deja el dt real (acotado) para corregir suavizado y movimiento.
  void _marcarFrame() {
    final ahora = DateTime.now().millisecondsSinceEpoch.toDouble();
    if (_ultimoFrameMs > 0) {
      final dtMs = ahora - _ultimoFrameMs;
      if (dtMs > 0) {
        _dtSeg = (dtMs / 1000.0).clamp(1.0 / 60.0, 0.5);
        final instantaneo = 1000.0 / dtMs;
        fps = fps == 0.0 ? instantaneo : fps * 0.8 + instantaneo * 0.2;
      }
    }
    _ultimoFrameMs = ahora;
  }

  /// Texto deletreado hasta ahora.
  String get texto => _texto.join();

  // Factor de corrección de aspecto vigente (se recibe por fotograma, calculado
  // en vivo con las dimensiones reales de la cámara del teléfono).
  double _correccionAspecto = Constantes.correccionAspectoX;

  // Lleva los landmarks del aspecto del teléfono al del entrenamiento (16:9)
  // multiplicando X y Z por el factor de corrección. Devuelve una copia nueva.
  List<Punto>? _corregirAspecto(List<Punto>? mano) {
    if (mano == null) return null;
    final f = _correccionAspecto;
    return [for (final p in mano) Punto(p.x * f, p.y, p.z * f)];
  }

  void _prepararIndices() {
    final etiquetas = _modeloA.etiquetas;
    List<int> indicesDe(List<String> nombres) => nombres
        .map(etiquetas.indexOf)
        .where((i) => i >= 0)
        .toList(growable: false);
    _idxMovimiento = indicesDe(Constantes.letrasConMovimiento);
    _idxGemelas = indicesDe(Constantes.estaticasGemelas);
    _indicesListos = true;
  }

  /// Procesa un fotograma con manos ya asignadas a izquierda/derecha.
  ///
  /// [correccionAspecto] es el factor (calculado en vivo por el detector) para
  /// llevar los landmarks del aspecto de la cámara al del entrenamiento.
  void procesarManos(List<Punto>? izquierda, List<Punto>? derecha,
      {double correccionAspecto = Constantes.correccionAspectoX}) {
    _marcarFrame();
    _correccionAspecto = correccionAspecto;
    if (!_indicesListos) _prepararIndices();
    manosVisibles = (izquierda != null ? 1 : 0) + (derecha != null ? 1 : 0);

    // Corrección de aspecto: los landmarks del teléfono (3:4) se llevan al aspecto
    // del entrenamiento (16:9) antes de todo, o el modelo ve una mano deformada.
    // Se corrigen solo para el reconocimiento; el esqueleto usa los crudos.
    final izqC = _corregirAspecto(izquierda);
    final derC = _corregirAspecto(derecha);

    // Traslación -> suavizado (One Euro con el dt real) -> escala. Igual que en la
    // demo, pero corrigiendo por la tasa de fotogramas real del teléfono.
    var trasladado = componerVectorDosManos(izqC, derC);
    trasladado = _filtroVector.filtrar(trasladado, dt: _dtSeg);
    final escalado = escalarVector(trasladado);

    final ahoraMs = DateTime.now().millisecondsSinceEpoch.toDouble();
    _buffer.add((ahoraMs, escalado));
    // Se descarta lo más viejo que 1.5 ventanas (no hace falta para remuestrear).
    final limiteViejo = ahoraMs - _duracionVentanaMs * 1.5;
    while (_buffer.isNotEmpty && _buffer.first.$1 < limiteViejo) {
      _buffer.removeAt(0);
    }

    // La ventana está lista cuando hay al menos _duracionVentanaMs de historia.
    final historiaMs = ahoraMs - _buffer.first.$1;
    nivelVentana = (historiaMs / _duracionVentanaMs * Constantes.tamanoVentanaA)
        .clamp(0, Constantes.tamanoVentanaA)
        .round();
    if (historiaMs < _duracionVentanaMs || _buffer.length < 2) return;

    // Remuestrea el último tramo de _duracionVentanaMs a 20 fotogramas: equivale a
    // 30 fps, así el modelo ve lo mismo que en el entrenamiento sin importar el
    // fps real, y el movimiento queda directamente comparable con sus umbrales.
    final ventana = _remuestrear(ahoraMs - _duracionVentanaMs, ahoraMs,
        Constantes.tamanoVentanaA);

    final probs = _modeloA.predecir(ventana);
    final mov = _velocidadMedia(ventana);

    // Compuerta de movimiento (dos sentidos), idéntica a la demo.
    if (mov < Constantes.umbralMovimientoAbs) {
      for (final i in _idxMovimiento) {
        probs[i] = 0.0;
      }
    } else if (mov > Constantes.umbralMovimientoMoviendo) {
      for (final i in _idxGemelas) {
        probs[i] = 0.0;
      }
    }

    var idx = 0;
    for (var i = 1; i < probs.length; i++) {
      if (probs[i] > probs[idx]) idx = i;
    }
    movimiento = mov;
    candidato = _modeloA.etiquetas[idx];
    confianza = probs[idx];
    _registrar(_modeloA.etiquetas[idx], probs[idx]);
  }

  /// La mano desapareció: se limpia la ventana, el candidato y el filtro, para
  /// que al volver la mano el suavizado arranque limpio (sin interpolar un salto).
  void sinManos() {
    _marcarFrame();
    _buffer.clear();
    _candidato = null;
    _contador = 0;
    _filtroVector.reiniciar();
    manosVisibles = 0;
    candidato = '';
    movimiento = 0.0;
    confianza = 0.0;
    nivelVentana = 0;
  }

  // Remuestrea el buffer a [n] fotogramas repartidos uniformemente en [tIni, tFin]
  // por interpolación lineal en el tiempo. Convierte una ventana grabada a un fps
  // cualquiera en una de 20 fotogramas a 30 fps efectivos.
  List<List<double>> _remuestrear(double tIni, double tFin, int n) {
    final salida = <List<double>>[];
    for (var i = 0; i < n; i++) {
      final t = tIni + (tFin - tIni) * (i / (n - 1));
      salida.add(_interpolar(t));
    }
    return salida;
  }

  List<double> _interpolar(double t) {
    if (t <= _buffer.first.$1) return _buffer.first.$2;
    if (t >= _buffer.last.$1) return _buffer.last.$2;
    for (var i = 1; i < _buffer.length; i++) {
      if (_buffer[i].$1 >= t) {
        final t0 = _buffer[i - 1].$1;
        final t1 = _buffer[i].$1;
        final v0 = _buffer[i - 1].$2;
        final v1 = _buffer[i].$2;
        final a = (t1 - t0) == 0 ? 0.0 : (t - t0) / (t1 - t0);
        final out = List<double>.filled(v0.length, 0.0);
        for (var k = 0; k < v0.length; k++) {
          out[k] = v0[k] * (1 - a) + v1[k] * a;
        }
        return out;
      }
    }
    return _buffer.last.$2;
  }

  void borrarUltima() {
    if (_texto.isNotEmpty) {
      _texto.removeLast();
      onTextoCambiado?.call();
    }
  }

  void limpiar() {
    if (_texto.isNotEmpty) {
      _texto.clear();
      onTextoCambiado?.call();
    }
  }

  // Movimiento medio de la ventana: |diferencia| media entre fotogramas
  // consecutivos sobre las coordenadas no-cero. Réplica de velocidad_media.
  double _velocidadMedia(List<List<double>> ventana) {
    if (ventana.length < 2) return 0.0;
    var suma = 0.0;
    var cuenta = 0;
    for (var t = 1; t < ventana.length; t++) {
      final actual = ventana[t];
      final previo = ventana[t - 1];
      for (var k = 0; k < actual.length; k++) {
        if (actual[k] != 0.0) {
          suma += (actual[k] - previo[k]).abs();
          cuenta++;
        }
      }
    }
    return cuenta == 0 ? 0.0 : suma / cuenta;
  }

  void _registrar(String clase, double confianza) {
    final ahora = DateTime.now().millisecondsSinceEpoch.toDouble();
    if (ahora < _finCooldownMs || confianza < Constantes.umbralConfianza) {
      _candidato = null;
      _contador = 0;
      return;
    }
    if (clase == _candidato) {
      _contador++;
    } else {
      _candidato = clase;
      _contador = 1;
    }
    if (_contador >= Constantes.fotogramasPersistencia) {
      _confirmar(clase);
      _candidato = null;
      _contador = 0;
      _finCooldownMs = ahora + Constantes.cooldownMs;
    }
  }

  void _confirmar(String clase) {
    if (clase == Constantes.claseReposo) return;
    // En modo palabra (entre INICIO y FIN) los fotogramas alimentan al Modelo B,
    // así que NO se confirman letras: solo se escucha FIN para cerrar la palabra.
    if (modoPalabra) {
      if (clase == Constantes.claseFin) onFinSena?.call();
      return;
    }
    if (clase == Constantes.claseInicio) {
      // Dos palmas abiertas: abre la grabación de una palabra dinámica (Modelo B).
      onInicioSena?.call();
    } else if (clase == Constantes.claseFin) {
      // FIN (dos puños) fuera de una palabra no hace nada: el espacio y el borrado
      // tienen sus propios botones.
    } else {
      _texto.add(clase);
      onLetraAgregada?.call(clase);
      onTextoCambiado?.call();
    }
  }

  /// Agrega un espacio al texto (botón de espacio), si el último no es ya un
  /// espacio. Sirve para separar palabras deletreadas.
  void agregarEspacio() {
    if (_texto.isEmpty || _texto.last != ' ') {
      _texto.add(' ');
      onTextoCambiado?.call();
    }
  }

  /// Agrega una palabra reconocida por el Modelo B, seguida de un espacio.
  void agregarPalabra(String palabra) {
    _texto.add(palabra);
    _texto.add(' ');
    onTextoCambiado?.call();
  }
}
