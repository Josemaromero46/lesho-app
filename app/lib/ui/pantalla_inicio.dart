import 'package:flutter/material.dart';
import 'package:lesho_app/ui/pantalla_reconocimiento.dart';
import 'package:lesho_app/ui/pantalla_texto_a_sena.dart';

/// Pantalla de inicio: la elección entre las dos direcciones de comunicación.
///
/// Es la primera decisión de la app, y la toman dos personas muy distintas: un
/// niño sordo y una persona oyente. Por eso el peso visual está en el ICONO y en
/// el color de cada tarjeta, no en el texto: un niño sordo puede no leer español
/// con fluidez, así que debe poder elegir sin leer. El texto queda como apoyo
/// para la persona oyente.
class PantallaInicio extends StatelessWidget {
  const PantallaInicio({super.key});

  @override
  Widget build(BuildContext context) {
    final colores = Theme.of(context).colorScheme;

    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(24, 28, 24, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const _Marca(),
              const SizedBox(height: 32),
              // Las tarjetas ocupan todo el alto disponible (área táctil grande,
              // que en una app usada por niños es una ventaja concreta) y su
              // contenido va CENTRADO dentro de cada una. Estirarlas empujando el
              // texto al fondo dejaba un hueco muerto enorme en el medio.
              Expanded(
                child: Column(
                  children: [
                    Expanded(
                      child: _TarjetaDireccion(
                        titulo: 'Traducir señas a texto',
                        descripcion:
                            'El niño hace las señas frente a la cámara y aparecen escritas.',
                        icono: Icons.sign_language_rounded,
                        acento: colores.primary,
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
                        titulo: 'Escribir para mostrar señas',
                        descripcion:
                            'Escribe una frase y el muñeco la hace en señas para el niño.',
                        icono: Icons.keyboard_rounded,
                        acento: colores.secondary,
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
            ],
          ),
        ),
      ),
    );
  }
}

/// Marca de la app. Sobria a propósito: nombre y qué es, nada más.
class _Marca extends StatelessWidget {
  const _Marca();

  @override
  Widget build(BuildContext context) {
    final colores = Theme.of(context).colorScheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'LESHO',
          style: Theme.of(context).textTheme.displayMedium?.copyWith(
                color: colores.primary,
                letterSpacing: 3,
              ),
        ),
        const SizedBox(height: 6),
        Text(
          'Lengua de Señas Hondureña',
          style: Theme.of(context).textTheme.bodyMedium,
        ),
      ],
    );
  }
}

/// Tarjeta de una dirección de comunicación.
///
/// Superficie blanca sobre el fondo arena, con un filete fino: el color entra
/// solo por el icono, que es la pieza grande y reconocible. Así la pantalla se
/// lee tranquila y aun así cada opción se distingue de un vistazo.
class _TarjetaDireccion extends StatelessWidget {
  final String titulo;
  final String descripcion;
  final IconData icono;
  final Color acento;
  final VoidCallback onTap;

  const _TarjetaDireccion({
    required this.titulo,
    required this.descripcion,
    required this.icono,
    required this.acento,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final colores = Theme.of(context).colorScheme;

    return Material(
      color: colores.surface,
      borderRadius: BorderRadius.circular(16),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
        child: Ink(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: colores.outlineVariant),
          ),
          child: Padding(
            padding: const EdgeInsets.all(28),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  width: 80,
                  height: 80,
                  decoration: BoxDecoration(
                    color: acento,
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Icon(icono, color: Colors.white, size: 44),
                ),
                const SizedBox(height: 24),
                Text(
                  titulo,
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                const SizedBox(height: 6),
                Row(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Expanded(
                      child: Text(
                        descripcion,
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Icon(
                      Icons.arrow_forward_rounded,
                      color: acento,
                      size: 24,
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
