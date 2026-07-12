import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:lesho_app/ui/pantalla_inicio.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();

  // La app es solo vertical: el usuario necesita ambas manos libres y el
  // dispositivo apoyado en una superficie, siempre en modo retrato.
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);

  runApp(const AppLesho());
}

class AppLesho extends StatelessWidget {
  const AppLesho({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'LESHO',
      debugShowCheckedModeBanner: false,
      theme: _tema(),
      home: const PantallaInicio(),
    );
  }

  ThemeData _tema() {
    const primario = Color(0xFFC8571B); // terracota cálido
    const secundario = Color(0xFFB5832C); // ámbar dorado
    const terciario = Color(0xFF4C7F61); // verde bosque
    const fondo = Color(0xFFFFF8EE); // crema cálida
    const textoOscuro = Color(0xFF22120A); // marrón muy oscuro

    return ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.fromSeed(
        seedColor: primario,
        brightness: Brightness.light,
        primary: primario,
        onPrimary: Colors.white,
        primaryContainer: const Color(0xFFFFDBC8),
        secondary: secundario,
        onSecondary: Colors.white,
        secondaryContainer: const Color(0xFFFFDDB3),
        tertiary: terciario,
        onTertiary: Colors.white,
        tertiaryContainer: const Color(0xFFC0E8D0),
        surface: Colors.white,
        onSurface: textoOscuro,
        surfaceContainerHighest: const Color(0xFFF1DFD4),
        error: const Color(0xFFBA1A1A),
      ),
      scaffoldBackgroundColor: fondo,
      fontFamily: 'Roboto',
      textTheme: const TextTheme(
        displayLarge: TextStyle(
          fontSize: 48,
          fontWeight: FontWeight.w700,
          color: textoOscuro,
        ),
        displayMedium: TextStyle(
          fontSize: 36,
          fontWeight: FontWeight.w700,
          color: textoOscuro,
        ),
        titleLarge: TextStyle(
          fontSize: 22,
          fontWeight: FontWeight.w600,
          color: textoOscuro,
        ),
        bodyLarge: TextStyle(
          fontSize: 18,
          color: textoOscuro,
        ),
        bodyMedium: TextStyle(
          fontSize: 15,
          color: Color(0xFF5A3020),
        ),
        labelLarge: TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.3,
        ),
      ),
      cardTheme: CardThemeData(
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
        ),
        color: Colors.white,
        surfaceTintColor: Colors.transparent,
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          minimumSize: const Size(double.infinity, 54),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
          textStyle: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: Colors.white,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: Color(0xFFD4B8A8)),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: Color(0xFFD4B8A8)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: primario, width: 2),
        ),
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      ),
    );
  }
}
