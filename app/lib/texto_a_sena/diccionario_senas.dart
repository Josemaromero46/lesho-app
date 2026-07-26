/// Diccionario de señas para el muñeco (Dirección 2: texto -> seña en clips).
///
/// Convierte la oración completa que escribe la persona oyente en la lista de
/// señas a reproducir, resolviendo tres problemas:
///
///  1. SEÑAS COMPUESTAS: "buenos días" es UNA seña en LESHO, no dos. El
///     diccionario admite claves de una o más palabras (BUENOS_DIAS) y el
///     recorrido usa emparejamiento voraz de la FRASE MAS LARGA en CADA
///     posición de la oración (el mismo principio del deletreo con dígrafos,
///     promovido del nivel de letras al nivel de palabras). La compuesta se
///     detecta esté donde esté: al inicio, en medio o al final.
///  2. NORMALIZACION: la búsqueda no distingue mayúsculas ni tildes de vocal
///     ("días" y "dias" encuentran la misma seña, porque en el teléfono se
///     escribe de ambas formas). La Ñ SI se conserva: "baño" busca BAÑO y una
///     palabra deletreada usa la seña de la Ñ. El texto que escribió el
///     usuario se muestra tal cual; la normalización es solo interna.
///  3. ALIAS: flexiones y sinónimos apuntan a la seña que existe (LLAMO ->
///     LLAMAR; AUTO -> CARRO). Los pares se validan con la asesoría LESHO;
///     no se hace gramática (fuera del alcance de la tesis).
///
/// Si ninguna clave ni alias resuelve una palabra, cae al deletreo letra por
/// letra con los clips del alfabeto, consciente de dígrafos ("calle" -> C, A,
/// LL, E) y de la eñe.
class DiccionarioSenas {
  /// Índice principal: clave normalizada -> asset del clip. Las claves siguen
  /// la convención de los identificadores de clips (MAYUSCULAS, sin tildes de
  /// vocal, con Ñ, espacios como guion bajo). Se puebla conforme se graben
  /// los clips con capture/captura_diccionario.py.
  static const Map<String, String> indicePorDefecto = {
    // Clips del piloto (se reemplazan por los definitivos al grabarlos):
    'HOLA': 'assets/clips/piloto/HOLA_rec.json',
    'CASA': 'assets/clips/piloto/CASA_rec.json',
    'NO': 'assets/clips/piloto/NO_rec.json',
    'HAMBRE': 'assets/clips/piloto/HAMBRE_rec.json',
    // Ejemplos de señas compuestas (grabar con: --palabras "BUENOS DIAS"):
    // 'BUENOS_DIAS': 'assets/clips/palabras/BUENOS_DIAS.json',
    // 'POR_FAVOR': 'assets/clips/palabras/POR_FAVOR.json',
  };

  /// Alias: variante normalizada -> clave del índice. Cubre flexiones y
  /// sinónimos de contexto (varias formas de escribir, una misma seña).
  /// Un alias solo actúa si su destino existe en el índice.
  static const Map<String, String> aliasPorDefecto = {
    // Flexiones frecuentes (validar pares con la asesoría LESHO):
    // 'LLAMO': 'LLAMAR',
    // 'NIÑOS': 'NIÑO',
    // Sinónimos de contexto:
    // 'AUTO': 'CARRO',
    // Frases equivalentes (también pueden ser compuestas):
    // 'BUEN_DIA': 'BUENOS_DIAS',
  };

  /// Clips del alfabeto para el fallback de deletreo (30 letras con dígrafos).
  static const Map<String, String> letrasPorDefecto = {
    'A': 'assets/clips/letras/A.json',
    'B': 'assets/clips/letras/B.json',
    'C': 'assets/clips/letras/C.json',
    'CH': 'assets/clips/letras/CH.json',
    'D': 'assets/clips/letras/D.json',
    'E': 'assets/clips/letras/E.json',
    'F': 'assets/clips/letras/F.json',
    'G': 'assets/clips/letras/G.json',
    'H': 'assets/clips/letras/H.json',
    'I': 'assets/clips/letras/I.json',
    'J': 'assets/clips/letras/J.json',
    'K': 'assets/clips/letras/K.json',
    'L': 'assets/clips/letras/L.json',
    'LL': 'assets/clips/letras/LL.json',
    'M': 'assets/clips/letras/M.json',
    'N': 'assets/clips/letras/N.json',
    'Ñ': 'assets/clips/letras/Ñ.json',
    'O': 'assets/clips/letras/O.json',
    'P': 'assets/clips/letras/P.json',
    'Q': 'assets/clips/letras/Q.json',
    'R': 'assets/clips/letras/R.json',
    'RR': 'assets/clips/letras/RR.json',
    'S': 'assets/clips/letras/S.json',
    'T': 'assets/clips/letras/T.json',
    'U': 'assets/clips/letras/U.json',
    'V': 'assets/clips/letras/V.json',
    'W': 'assets/clips/letras/W.json',
    'X': 'assets/clips/letras/X.json',
    'Y': 'assets/clips/letras/Y.json',
    'Z': 'assets/clips/letras/Z.json',
  };

  static const List<String> _digrafos = ['CH', 'LL', 'RR'];

  final Map<String, String> _indice;
  final Map<String, String> _alias;
  final Map<String, String> _letras;

  /// Largo máximo (en palabras) de una clave o alias. Acota cuántas ventanas
  /// se prueban en cada posición de la oración.
  final int _largoMaximo;

  DiccionarioSenas({
    Map<String, String>? indice,
    Map<String, String>? alias,
    Map<String, String>? letras,
  })  : _indice = indice ?? indicePorDefecto,
        _alias = alias ?? aliasPorDefecto,
        _letras = letras ?? letrasPorDefecto,
        _largoMaximo = _calcularLargoMaximo(
          indice ?? indicePorDefecto,
          alias ?? aliasPorDefecto,
        );

  static int _calcularLargoMaximo(
      Map<String, String> indice, Map<String, String> alias) {
    var maximo = 1;
    for (final clave in [...indice.keys, ...alias.keys]) {
      final palabras = clave.split('_').length;
      if (palabras > maximo) maximo = palabras;
    }
    return maximo;
  }

  // -- Normalización ----------------------------------------------------------

  static final RegExp _separadores = RegExp(r'[\s,.:;!?¿¡"()\-]+');

  /// Normaliza una palabra SOLO para buscarla: mayúsculas y sin tildes de
  /// vocal. La Ñ se conserva. (El texto original del usuario no se altera.)
  static String normalizar(String palabra) {
    const tildes = {
      'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ü': 'U',
    };
    final mayuscula = palabra.toUpperCase();
    final sb = StringBuffer();
    for (final ch in mayuscula.split('')) {
      sb.write(tildes[ch] ?? ch);
    }
    return sb.toString();
  }

  /// Separa la oración en palabras tal como las escribió el usuario
  /// (la puntuación actúa como separador).
  static List<String> tokenizar(String frase) => frase
      .trim()
      .split(_separadores)
      .where((t) => t.trim().isNotEmpty)
      .map((t) => t.trim())
      .toList();

  // -- Traducción ---------------------------------------------------------------

  /// Convierte la oración completa en la secuencia de señas a reproducir.
  ///
  /// Recorre la oración posición por posición y en CADA posición prueba
  /// primero la ventana más larga de palabras contra el índice (y los alias);
  /// la primera que existe se consume como UNA seña. Si ni la palabra sola
  /// existe, se deletrea. Así una seña compuesta se detecta al inicio, en
  /// medio o al final de la oración.
  List<UnidadSena> traducir(String frase) {
    final originales = tokenizar(frase);
    final normalizadas = originales.map(normalizar).toList();
    final unidades = <UnidadSena>[];

    var i = 0;
    while (i < originales.length) {
      final restantes = originales.length - i;
      final tope = restantes < _largoMaximo ? restantes : _largoMaximo;

      String? claveEncontrada;
      var consumidas = 1;
      for (var ventana = tope; ventana >= 1; ventana--) {
        final clave = normalizadas.sublist(i, i + ventana).join('_');
        final resuelta = _resolverClave(clave);
        if (resuelta != null) {
          claveEncontrada = resuelta;
          consumidas = ventana;
          break;
        }
      }

      final texto = originales.sublist(i, i + consumidas).join(' ');
      if (claveEncontrada != null) {
        unidades.add(UnidadSena.clip(
          texto: texto,
          clave: claveEncontrada,
          ruta: _indice[claveEncontrada]!,
        ));
      } else {
        unidades.add(UnidadSena.deletreo(
          texto: texto,
          letras: _deletrear(normalizadas[i]),
        ));
      }
      i += consumidas;
    }
    return unidades;
  }

  /// Busca la clave en el índice; si no está, prueba el alias (solo si el
  /// destino del alias existe en el índice). Devuelve la clave FINAL o null.
  String? _resolverClave(String clave) {
    if (_indice.containsKey(clave)) return clave;
    final destino = _alias[clave];
    if (destino != null && _indice.containsKey(destino)) return destino;
    return null;
  }

  /// Deletreo consciente de dígrafos: "CALLE" -> C, A, LL, E. Devuelve las
  /// rutas de los clips de letras; los caracteres sin seña (números, símbolos)
  /// se omiten, como en el diccionario de videos original.
  List<String> _deletrear(String palabra) {
    final rutas = <String>[];
    var i = 0;
    while (i < palabra.length) {
      String? letra;
      if (i + 1 < palabra.length) {
        final par = palabra.substring(i, i + 2);
        if (_digrafos.contains(par)) letra = par;
      }
      letra ??= palabra[i];
      final ruta = _letras[letra];
      if (ruta != null) rutas.add(ruta);
      i += letra.length;
    }
    return rutas;
  }
}

/// Una unidad de la traducción: una seña de diccionario o un deletreo.
class UnidadSena {
  /// El texto tal como lo escribió el usuario (para mostrarlo en pantalla
  /// mientras se reproduce su seña), por ejemplo "Buenos días".
  final String texto;

  /// Clave del diccionario que se resolvió (BUENOS_DIAS), o '' si es deletreo.
  final String clave;

  /// Clips a reproducir en orden: uno solo si es seña de diccionario, o la
  /// lista de clips de letras si es deletreo.
  final List<String> clips;

  final bool esDeletreo;

  UnidadSena.clip(
      {required this.texto, required this.clave, required String ruta})
      : clips = [ruta],
        esDeletreo = false;

  UnidadSena.deletreo({required this.texto, required List<String> letras})
      : clave = '',
        clips = letras,
        esDeletreo = true;
}
