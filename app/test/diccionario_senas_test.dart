import 'package:flutter_test/flutter_test.dart';
import 'package:lesho_app/texto_a_sena/diccionario_senas.dart';

void main() {
  // Diccionario de prueba con señas simples, compuestas y alias.
  final diccionario = DiccionarioSenas(
    indice: const {
      'HOLA': 'clips/HOLA.json',
      'BUENOS_DIAS': 'clips/BUENOS_DIAS.json',
      'QUERER': 'clips/QUERER.json',
      'AGUA': 'clips/AGUA.json',
      'BAÑO': 'clips/BAÑO.json',
      'CARRO': 'clips/CARRO.json',
      'COMO_ESTAS_TU': 'clips/COMO_ESTAS_TU.json',
    },
    alias: const {
      'QUIERO': 'QUERER',
      'AUTO': 'CARRO',
      'BUEN_DIA': 'BUENOS_DIAS',
      'ROTO': 'NO_EXISTE', // alias con destino inexistente: debe ignorarse
    },
  );

  test('seña compuesta EN MEDIO de la oración completa', () {
    final unidades = diccionario.traducir('Hola muy buenos días señor');
    expect(unidades.map((u) => u.esDeletreo ? 'deletreo' : u.clave).toList(),
        ['HOLA', 'deletreo', 'BUENOS_DIAS', 'deletreo']);
    // "buenos días" se consumió como UNA seña, no como dos.
    expect(unidades[2].texto, 'buenos días');
    expect(unidades[2].clips, ['clips/BUENOS_DIAS.json']);
    // "muy" y "señor" no están en el diccionario: deletreo.
    expect(unidades[1].texto, 'muy');
    expect(unidades[3].texto, 'señor');
  });

  test('compuesta al inicio y al final tambien se detecta', () {
    final inicio = diccionario.traducir('buenos días señor');
    expect(inicio.first.clave, 'BUENOS_DIAS');
    final fin = diccionario.traducir('hola buenos días');
    expect(fin.last.clave, 'BUENOS_DIAS');
  });

  test('la mas larga gana: tres palabras antes que dos o una', () {
    final unidades = diccionario.traducir('como estas tu');
    expect(unidades.length, 1);
    expect(unidades.single.clave, 'COMO_ESTAS_TU');
  });

  test('tildes: "días" y "dias" encuentran la misma seña', () {
    final conTilde = diccionario.traducir('buenos días');
    final sinTilde = diccionario.traducir('buenos dias');
    expect(conTilde.single.clave, 'BUENOS_DIAS');
    expect(sinTilde.single.clave, 'BUENOS_DIAS');
    // El texto del usuario se conserva tal como lo escribió.
    expect(conTilde.single.texto, 'buenos días');
  });

  test('la eñe se conserva: "baño" encuentra BAÑO', () {
    final unidades = diccionario.traducir('baño');
    expect(unidades.single.clave, 'BAÑO');
  });

  test('deletreo con eñe: "señor" usa la seña de la Ñ', () {
    final unidades = diccionario.traducir('señor');
    expect(unidades.single.esDeletreo, isTrue);
    expect(unidades.single.clips, [
      'assets/clips/letras/S.json',
      'assets/clips/letras/E.json',
      'assets/clips/letras/Ñ.json',
      'assets/clips/letras/O.json',
      'assets/clips/letras/R.json',
    ]);
  });

  test('deletreo con dígrafo: "calle" es C, A, LL, E', () {
    final unidades = diccionario.traducir('calle');
    expect(unidades.single.clips, [
      'assets/clips/letras/C.json',
      'assets/clips/letras/A.json',
      'assets/clips/letras/LL.json',
      'assets/clips/letras/E.json',
    ]);
  });

  test('alias de flexión: "quiero agua" muestra QUERER y AGUA', () {
    final unidades = diccionario.traducir('quiero agua');
    expect(unidades.map((u) => u.clave).toList(), ['QUERER', 'AGUA']);
    // El texto original se conserva para mostrarlo junto a la seña.
    expect(unidades.first.texto, 'quiero');
  });

  test('alias de sinónimo: "auto" muestra la seña de CARRO', () {
    final unidades = diccionario.traducir('auto');
    expect(unidades.single.clave, 'CARRO');
  });

  test('alias de frase equivalente: "buen día" -> BUENOS_DIAS', () {
    final unidades = diccionario.traducir('Buen día');
    expect(unidades.single.clave, 'BUENOS_DIAS');
    expect(unidades.single.texto, 'Buen día');
  });

  test('alias con destino inexistente se ignora (cae a deletreo)', () {
    final unidades = diccionario.traducir('roto');
    expect(unidades.single.esDeletreo, isTrue);
  });

  test('puntuación y mayúsculas no estorban', () {
    final unidades = diccionario.traducir('¡HOLA! ¿Buenos días?');
    expect(unidades.map((u) => u.clave).toList(), ['HOLA', 'BUENOS_DIAS']);
  });

  test('oración vacía produce lista vacía', () {
    expect(diccionario.traducir('   '), isEmpty);
  });
}
