import 'package:flutter/material.dart';
import 'package:lesho_app/ui/pantalla_palabras.dart';
import 'package:lesho_app/ui/pantalla_prueba_muneco.dart';
import 'package:lesho_app/ui/pantalla_reconocimiento.dart';
import 'package:lesho_app/ui/pantalla_texto_a_sena.dart';

/// Pantalla de inicio: selección entre las dos direcciones de comunicación.
///
/// Diseñada para ser usada tanto por niños como por personas oyentes.
/// Botones grandes y texto claro para facilitar la elección.
class PantallaInicio extends StatelessWidget {
  const PantallaInicio({super.key});

  @override
  Widget build(BuildContext context) {
    final colores = Theme.of(context).colorScheme;

    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 32),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(height: 16),
              _Encabezado(),
              const SizedBox(height: 48),
              Expanded(
                child: Column(
                  children: [
                    Expanded(
                      child: _TarjetaDireccion(
                        titulo: 'El niño firma',
                        descripcion:
                            'Coloca el teléfono frente a ti y empieza a hacer señas. La app convierte las señas en texto.',
                        icono: Icons.sign_language_rounded,
                        color: colores.primary,
                        colorContenedor: colores.primaryContainer,
                        onTap: () => Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (_) => const PantallaReconocimiento(),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 20),
                    Expanded(
                      child: _TarjetaDireccion(
                        titulo: 'La persona oyente escribe',
                        descripcion:
                            'Escribe lo que quieres decir y la app muestra la seña correspondiente en pantalla.',
                        icono: Icons.keyboard_rounded,
                        color: colores.tertiary,
                        colorContenedor: colores.tertiaryContainer,
                        onTap: () => Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (_) => const PantallaTextoASena(),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 4),
              TextButton.icon(
                onPressed: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const PantallaPalabras()),
                ),
                icon: const Icon(Icons.science_outlined, size: 18),
                label: const Text('Probar señas dinámicas (prueba)'),
              ),
              TextButton.icon(
                onPressed: () => Navigator.push(
                  context,
                  MaterialPageRoute(
                      builder: (_) => const PantallaPruebaMuneco()),
                ),
                icon: const Icon(Icons.accessibility_new_rounded, size: 18),
                label: const Text('Probar muñeco de señas (prueba)'),
              ),
              const SizedBox(height: 8),
              _PieVersion(),
            ],
          ),
        ),
      ),
    );
  }
}

class _Encabezado extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'LESHO',
          style: Theme.of(context).textTheme.displayMedium?.copyWith(
                color: Theme.of(context).colorScheme.primary,
                letterSpacing: 2,
              ),
        ),
        const SizedBox(height: 4),
        Text(
          'Lenguaje de Señas Hondureño',
          style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                color: Theme.of(context).colorScheme.onSurface.withValues(alpha:0.6),
              ),
        ),
        const SizedBox(height: 16),
        Text(
          'Elige cómo quieres comunicarte:',
          style: Theme.of(context).textTheme.titleLarge,
        ),
      ],
    );
  }
}

class _TarjetaDireccion extends StatelessWidget {
  final String titulo;
  final String descripcion;
  final IconData icono;
  final Color color;
  final Color colorContenedor;
  final VoidCallback onTap;

  const _TarjetaDireccion({
    required this.titulo,
    required this.descripcion,
    required this.icono,
    required this.color,
    required this.colorContenedor,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 120),
        decoration: BoxDecoration(
          color: colorContenedor,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: color.withValues(alpha:0.3), width: 1.5),
        ),
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Row(
            children: [
              Container(
                width: 64,
                height: 64,
                decoration: BoxDecoration(
                  color: color,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Icon(icono, color: Colors.white, size: 34),
              ),
              const SizedBox(width: 20),
              Expanded(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      titulo,
                      style: Theme.of(context).textTheme.titleLarge?.copyWith(
                            color: color,
                          ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      descripcion,
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              Icon(
                Icons.arrow_forward_ios_rounded,
                color: color.withValues(alpha:0.6),
                size: 18,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _PieVersion extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Text(
      'UNAH · Campus Comayagua · 2026',
      textAlign: TextAlign.center,
      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
            color: Theme.of(context).colorScheme.onSurface.withValues(alpha:0.35),
            fontSize: 12,
          ),
    );
  }
}
