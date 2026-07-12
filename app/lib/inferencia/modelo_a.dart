import 'package:lesho_app/core/constantes.dart';
import 'package:lesho_app/inferencia/cargador_modelos.dart';

/// Resultado de la inferencia del Modelo A (clasificador de ventana temporal).
class ResultadoModeloA {
  final String clase;
  final double confianza;

  const ResultadoModeloA({required this.clase, required this.confianza});
}

/// Ejecuta el Modelo A (clasificador de ventana temporal) sobre una ventana de
/// landmarks.
///
/// Recibe una ventana de [Constantes.tamanoVentanaA] fotogramas, cada uno un
/// vector de 126 valores (dos manos normalizadas), y devuelve la clase de mayor
/// probabilidad junto con su confianza. Un solo modelo reconoce tanto las
/// letras estáticas como las de movimiento (J, Ñ, Z, LL, RR).
class ModeloA {
  final CargadorModelos _cargador;

  ModeloA(this._cargador);

  /// Etiquetas de las clases del Modelo A, en el orden de salida del modelo.
  List<String> get etiquetas => _cargador.etiquetasA;

  /// Corre el modelo sobre la [ventana] y devuelve las probabilidades CRUDAS de
  /// las 33 clases (sin argmax). La capa de control aplica sobre ellas la
  /// compuerta de movimiento (anula ciertas clases) antes de elegir la ganadora.
  List<double> predecir(List<List<double>> ventana) {
    assert(ventana.length == Constantes.tamanoVentanaA);
    final entrada = [ventana];
    final salida = [List.filled(Constantes.numClasesA, 0.0)];
    _cargador.interpreteA.run(entrada, salida);
    return salida[0];
  }

  /// Clasifica la [ventana] de N fotogramas de 126 landmarks normalizados.
  ///
  /// La ventana la mantiene el buffer rodante de la capa de control.
  ResultadoModeloA clasificar(List<List<double>> ventana) {
    assert(
      ventana.length == Constantes.tamanoVentanaA,
      'La ventana debe tener ${Constantes.tamanoVentanaA} fotogramas.',
    );
    assert(
      ventana.isEmpty || ventana.first.length == Constantes.tamanoVector,
      'Cada fotograma debe tener ${Constantes.tamanoVector} valores.',
    );

    // Forma [1, N, 126]: un lote de una ventana.
    final entrada = [ventana];
    final salida = [List.filled(Constantes.numClasesA, 0.0)];

    _cargador.interpreteA.run(entrada, salida);

    final probabilidades = salida[0];
    final indiceMax = _argmax(probabilidades);

    return ResultadoModeloA(
      clase: _cargador.etiquetasA[indiceMax],
      confianza: probabilidades[indiceMax],
    );
  }

  int _argmax(List<double> lista) {
    var mejorIndice = 0;
    for (var i = 1; i < lista.length; i++) {
      if (lista[i] > lista[mejorIndice]) mejorIndice = i;
    }
    return mejorIndice;
  }
}
