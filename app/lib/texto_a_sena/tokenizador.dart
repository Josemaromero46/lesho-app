/// Divide una frase en español en palabras normalizadas para buscarlas en
/// el diccionario de videos.
class Tokenizador {
  // Separa por espacios y signos de puntuación comunes.
  static final RegExp _separadores = RegExp(r'[\s,.:;!?¿¡"()\-]+');

  /// Retorna la lista de palabras en minúsculas, sin puntuación.
  List<String> tokenizar(String frase) {
    return frase
        .trim()
        .split(_separadores)
        .map((t) => t.trim().toLowerCase())
        .where((t) => t.isNotEmpty)
        .toList();
  }
}
