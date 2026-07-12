import 'package:flutter/material.dart';
import 'package:lesho_app/texto_a_sena/cola_reproduccion.dart';
import 'package:video_player/video_player.dart';

/// Pantalla de texto a seña (Dirección 2: persona oyente escribe -> video).
///
/// La persona oyente escribe una frase. La app tokeniza el texto, busca
/// cada palabra en el diccionario visual y reproduce los videos en secuencia.
/// Si una palabra no tiene video, hace el fallback de deletreo letra por letra.
class PantallaTextoASena extends StatefulWidget {
  const PantallaTextoASena({super.key});

  @override
  State<PantallaTextoASena> createState() => _EstadoPantallaTextoASena();
}

class _EstadoPantallaTextoASena extends State<PantallaTextoASena> {
  final _cola = ColaReproduccion();
  final _controladorTexto = TextEditingController();
  final _focoTexto = FocusNode();

  @override
  void initState() {
    super.initState();
    _cola.addListener(() => setState(() {}));
  }

  @override
  void dispose() {
    _cola.dispose();
    _controladorTexto.dispose();
    _focoTexto.dispose();
    super.dispose();
  }

  Future<void> _mostrarSena() async {
    final texto = _controladorTexto.text.trim();
    if (texto.isEmpty) return;
    _focoTexto.unfocus();
    await _cola.reproducirFrase(texto);
  }

  @override
  Widget build(BuildContext context) {
    final colores = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        foregroundColor: colores.onSurface,
        elevation: 0,
        title: const Text('Mostrar seña'),
      ),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: _AreaVideo(cola: _cola),
            ),
            _PanelEntrada(
              controlador: _controladorTexto,
              foco: _focoTexto,
              cola: _cola,
              onMostrar: _mostrarSena,
            ),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Widgets internos
// ---------------------------------------------------------------------------

class _AreaVideo extends StatelessWidget {
  final ColaReproduccion cola;

  const _AreaVideo({required this.cola});

  @override
  Widget build(BuildContext context) {
    final colores = Theme.of(context).colorScheme;

    if (!cola.reproduciendo && cola.controlador == null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(40),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                Icons.play_circle_outline_rounded,
                size: 80,
                color: colores.tertiary.withValues(alpha:0.4),
              ),
              const SizedBox(height: 20),
              Text(
                'Escribe algo abajo y presiona\n"Mostrar seña"',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                      color: colores.onSurface.withValues(alpha:0.45),
                    ),
              ),
            ],
          ),
        ),
      );
    }

    final controlador = cola.controlador;

    if (controlador == null || !controlador.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }

    return Column(
      children: [
        Expanded(
          child: Center(
            child: AspectRatio(
              aspectRatio: controlador.value.aspectRatio,
              child: VideoPlayer(controlador),
            ),
          ),
        ),
        if (cola.totalVideos > 1)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 8),
            child: Row(
              children: [
                Text(
                  'Video ${cola.videoActual} de ${cola.totalVideos}',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: colores.onSurface.withValues(alpha:0.5),
                      ),
                ),
              ],
            ),
          ),
      ],
    );
  }
}

class _PanelEntrada extends StatelessWidget {
  final TextEditingController controlador;
  final FocusNode foco;
  final ColaReproduccion cola;
  final VoidCallback onMostrar;

  const _PanelEntrada({
    required this.controlador,
    required this.foco,
    required this.cola,
    required this.onMostrar,
  });

  @override
  Widget build(BuildContext context) {
    final colores = Theme.of(context).colorScheme;

    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha:0.06),
            blurRadius: 12,
            offset: const Offset(0, -3),
          ),
        ],
      ),
      padding: const EdgeInsets.fromLTRB(20, 20, 20, 24),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          TextField(
            controller: controlador,
            focusNode: foco,
            maxLines: 3,
            minLines: 1,
            textCapitalization: TextCapitalization.sentences,
            decoration: const InputDecoration(
              hintText: 'Escribe lo que quieres decir...',
              prefixIcon: Icon(Icons.edit_rounded),
            ),
            onSubmitted: (_) => onMostrar(),
          ),
          const SizedBox(height: 14),
          FilledButton.icon(
            onPressed: cola.reproduciendo ? null : onMostrar,
            icon: cola.reproduciendo
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: Colors.white,
                    ),
                  )
                : const Icon(Icons.play_arrow_rounded),
            label: Text(cola.reproduciendo ? 'Reproduciendo...' : 'Mostrar seña'),
            style: FilledButton.styleFrom(
              backgroundColor: colores.tertiary,
            ),
          ),
        ],
      ),
    );
  }
}
