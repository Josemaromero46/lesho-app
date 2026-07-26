/// Traducción del identificador de clase del modelo al texto que ve el usuario.
///
/// Los identificadores de las 50 señas dinámicas se escriben en mayúscula, SIN
/// tildes de vocal y con guion bajo en lugar de espacio, por contrato con el
/// pipeline de entrenamiento (ver `training/comun/definiciones.py`). Ese contrato
/// no se toca: cambiarlo obligaría a reentrenar y a regenerar las etiquetas.
///
/// Esta tabla es SOLO de presentación. Convierte, por ejemplo, MAMA en MAMÁ y
/// POR_FAVOR en POR FAVOR al momento de escribir la palabra en pantalla, para que
/// el texto en español quede bien escrito.
const Map<String, String> _conTilde = {
  'ADIOS': 'ADIÓS',
  'PERDON': 'PERDÓN',
  'SI': 'SÍ',
  'MAMA': 'MAMÁ',
  'PAPA': 'PAPÁ',
  'DIA': 'DÍA',
};

/// Devuelve el texto legible de una clase del Modelo B.
///
/// Aplica la tilde cuando corresponde y cambia el guion bajo por espacio. Las
/// clases que ya se escriben bien (BAÑO, NIÑO, MAÑANA, y todas las demás) pasan
/// sin cambios.
String textoLegible(String clase) =>
    _conTilde[clase] ?? clase.replaceAll('_', ' ');
