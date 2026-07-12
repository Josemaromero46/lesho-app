import 'package:flutter/foundation.dart';

/// Mantiene la cadena de letras que el usuario va deletreando seña por seña.
///
/// Notifica a sus oyentes cada vez que el texto cambia.
class BufferDeletreo extends ChangeNotifier {
  String _texto = '';

  String get texto => _texto;
  bool get estaVacio => _texto.isEmpty;

  void agregarLetra(String letra) {
    _texto += letra;
    notifyListeners();
  }

  void borrarUltima() {
    if (_texto.isNotEmpty) {
      _texto = _texto.substring(0, _texto.length - 1);
      notifyListeners();
    }
  }

  void limpiar() {
    if (_texto.isNotEmpty) {
      _texto = '';
      notifyListeners();
    }
  }
}
