import 'package:lesho_app/core/constantes.dart';
import 'package:lesho_app/inferencia/cargador_modelos.dart';

/// Ejecuta el Modelo B (LSTM dinámico) sobre una secuencia ya preprocesada.
///
/// La secuencia debe venir con la forma [longitudFijaSecuencia, tamanoEntradaB]
/// (40 x 148), tal como la produce `procesarSecuenciaB`. Devuelve las
/// probabilidades crudas de las 50 clases; la capa de arriba elige la ganadora
/// (enmascarando a las clases realmente entrenadas cuando el modelo es parcial).
class ModeloB {
  final CargadorModelos _cargador;

  ModeloB(this._cargador);

  /// Etiquetas de las 50 señas, en el orden de salida del modelo.
  List<String> get etiquetas => _cargador.etiquetasB;

  /// Corre el modelo y devuelve las 50 probabilidades crudas.
  List<double> predecir(List<List<double>> secuencia) {
    assert(secuencia.length == Constantes.longitudFijaSecuencia);
    final entrada = [secuencia];
    final salida = [List.filled(Constantes.numClasesB, 0.0)];
    _cargador.interpreteB.run(entrada, salida);
    return salida[0];
  }
}
