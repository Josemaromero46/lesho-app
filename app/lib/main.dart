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

  /// Paleta "añil y arena". El añil da el registro serio e institucional; el
  /// ámbar devuelve la calidez que el azul solo no tiene. El fondo es un arena
  /// cálido, nunca blanco puro ni gris frío: ahí está la calidez, no en colores
  /// saturados (que es lo que vuelve aniñada una interfaz).
  ThemeData _tema() {
    const anil = Color(0xFF1F3A5F); // añil profundo, acción principal
    const ambar = Color(0xFFE8A13A); // ámbar cálido, segunda dirección
    const arena = Color(0xFFF7F4EE); // fondo cálido
    const tinta = Color(0xFF1A1D24); // texto principal
    const tintaSuave = Color(0xFF5C6472); // texto secundario
    const borde = Color(0xFFE3DDD2); // hairline sobre arena

    return ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.fromSeed(
        seedColor: anil,
        brightness: Brightness.light,
        primary: anil,
        onPrimary: Colors.white,
        primaryContainer: const Color(0xFFDDE6F1),
        onPrimaryContainer: anil,
        secondary: ambar,
        onSecondary: const Color(0xFF3A2708),
        secondaryContainer: const Color(0xFFFBEBD2),
        onSecondaryContainer: const Color(0xFF6B4A12),
        tertiary: ambar,
        onTertiary: const Color(0xFF3A2708),
        tertiaryContainer: const Color(0xFFFBEBD2),
        surface: Colors.white,
        onSurface: tinta,
        onSurfaceVariant: tintaSuave,
        outlineVariant: borde,
        surfaceContainerHighest: const Color(0xFFEFEAE0),
        error: const Color(0xFFB3261E),
      ),
      scaffoldBackgroundColor: arena,
      fontFamily: 'Roboto',
      textTheme: const TextTheme(
        displayLarge: TextStyle(
          fontSize: 44,
          fontWeight: FontWeight.w600,
          color: tinta,
          height: 1.1,
        ),
        displayMedium: TextStyle(
          fontSize: 32,
          fontWeight: FontWeight.w600,
          color: tinta,
          height: 1.15,
        ),
        titleLarge: TextStyle(
          fontSize: 22,
          fontWeight: FontWeight.w600,
          color: tinta,
          height: 1.25,
        ),
        bodyLarge: TextStyle(
          fontSize: 16,
          color: tinta,
          height: 1.45,
        ),
        bodyMedium: TextStyle(
          fontSize: 14.5,
          color: tintaSuave,
          height: 1.45,
        ),
        labelLarge: TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.2,
        ),
      ),
      cardTheme: CardThemeData(
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        color: Colors.white,
        surfaceTintColor: Colors.transparent,
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          minimumSize: const Size(double.infinity, 54),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
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
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: borde),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: borde),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: anil, width: 2),
        ),
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      ),
    );
  }
}
