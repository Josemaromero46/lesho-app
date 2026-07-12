import 'package:lesho_app/core/constantes.dart';

/// Mapea palabras en español a rutas de video en los assets.
///
/// Si la palabra existe en [_indice], retorna su video directamente.
/// Si no existe, retorna la lista de videos letra por letra (fallback).
///
/// Los videos se encuentran en [Constantes.directorioVideos].
class DiccionarioVideos {
  // Diccionario principal: palabra en minúsculas -> nombre de archivo .mp4
  // Se puebla conforme se graben los videos del diccionario visual (semana 6).
  static const Map<String, String> _indice = {
    // Señas de vocabulario común (poblar con los videos grabados):
    // 'hola':     'hola.mp4',
    // 'gracias':  'gracias.mp4',
    // 'agua':     'agua.mp4',
    // 'comida':   'comida.mp4',
    // 'ayuda':    'ayuda.mp4',
    // 'bano':     'bano.mp4',
    // 'casa':     'casa.mp4',
    // 'mama':     'mama.mp4',
    // 'papa':     'papa.mp4',
    // 'amigo':    'amigo.mp4',
    // etc...
  };

  // Las 30 letras del alfabeto (incluye los dígrafos CH, LL, RR) siempre
  // disponibles para el fallback de deletreo. Los videos se graban en la
  // semana 6 del cronograma.
  static const Map<String, String> _letras = {
    'a': 'letra_a.mp4', 'b': 'letra_b.mp4', 'c': 'letra_c.mp4',
    'ch': 'letra_ch.mp4',
    'd': 'letra_d.mp4', 'e': 'letra_e.mp4', 'f': 'letra_f.mp4',
    'g': 'letra_g.mp4', 'h': 'letra_h.mp4', 'i': 'letra_i.mp4',
    'j': 'letra_j.mp4', 'k': 'letra_k.mp4', 'l': 'letra_l.mp4',
    'll': 'letra_ll.mp4',
    'm': 'letra_m.mp4', 'n': 'letra_n.mp4', 'ñ': 'letra_nn.mp4',
    'o': 'letra_o.mp4', 'p': 'letra_p.mp4', 'q': 'letra_q.mp4',
    'r': 'letra_r.mp4', 'rr': 'letra_rr.mp4', 's': 'letra_s.mp4',
    't': 'letra_t.mp4', 'u': 'letra_u.mp4', 'v': 'letra_v.mp4',
    'w': 'letra_w.mp4', 'x': 'letra_x.mp4', 'y': 'letra_y.mp4',
    'z': 'letra_z.mp4',
  };

  // Dígrafos del español que en LESHO son una sola seña. El deletreo los
  // reconoce antes que las letras sueltas (por ejemplo, "calle" se deletrea
  // c, a, ll, e y no c, a, l, l, e).
  static const List<String> _digrafos = ['ch', 'll', 'rr'];

  /// Retorna la lista de rutas de video para una palabra.
  ///
  /// Si la palabra está en el diccionario, devuelve una lista con un solo video.
  /// Si no está, devuelve la lista de videos letra por letra (puede estar vacía
  /// si la palabra contiene caracteres sin video asignado).
  List<String> resolver(String palabra) {
    final clave = palabra.toLowerCase();
    if (_indice.containsKey(clave)) {
      return ['${Constantes.directorioVideos}${_indice[clave]}'];
    }
    return _deletrear(clave);
  }

  /// Indica si una palabra tiene video directo en el diccionario.
  bool tieneVideoDirecto(String palabra) =>
      _indice.containsKey(palabra.toLowerCase());

  List<String> _deletrear(String palabra) {
    final rutas = <String>[];
    var i = 0;
    while (i < palabra.length) {
      // Intenta primero un dígrafo (ch, ll, rr); si no, una sola letra.
      String? letra;
      if (i + 1 < palabra.length) {
        final par = palabra.substring(i, i + 2);
        if (_digrafos.contains(par)) letra = par;
      }
      letra ??= palabra[i];
      if (_letras.containsKey(letra)) {
        rutas.add('${Constantes.directorioVideos}${_letras[letra]}');
      }
      i += letra.length;
    }
    return rutas;
  }
}
