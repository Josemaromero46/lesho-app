import 'package:flutter/material.dart';
import 'package:lesho_app/texto_a_sena/clip_sena.dart';
import 'package:lesho_app/texto_a_sena/reproductor_sena.dart';

/// Pantalla de prueba del muñeco de cápsulas (Fase 0 del piloto).
///
/// Reproduce los clips empaquetados en assets/clips/piloto/ para validar la
/// legibilidad del muñeco en el teléfono real: velocidad, espejo y cambio de
/// seña. Es un banco de pruebas; la integración final va en la pantalla de
/// texto a seña, con el tokenizador y el fallback de deletreo existentes.
class PantallaPruebaMuneco extends StatefulWidget {
  const PantallaPruebaMuneco({super.key});

  @override
  State<PantallaPruebaMuneco> createState() => _PantallaPruebaMunecoState();
}

class _PantallaPruebaMunecoState extends State<PantallaPruebaMuneco> {
  /// Clips del piloto empaquetados como assets (se reemplazan al grabar los
  /// definitivos con capture/captura_diccionario.py).
  static const _rutas = [
    'assets/clips/piloto/HOLA_rec.json',
    'assets/clips/piloto/CASA_rec.json',
    'assets/clips/piloto/NO_rec.json',
    'assets/clips/piloto/HAMBRE_rec.json',
  ];

  final Map<String, ClipSena> _clips = {};
  String? _error;
  ClipSena? _actual;
  double _velocidad = 1.0;

  @override
  void initState() {
    super.initState();
    _cargar();
  }

  Future<void> _cargar() async {
    try {
      for (final ruta in _rutas) {
        final clip = await ClipSena.desdeAsset(ruta);
        _clips[clip.palabra] = clip;
      }
      setState(() => _actual = _clips.values.first);
    } catch (e) {
      setState(() => _error = 'No se pudieron cargar los clips: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    final colores = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(title: const Text('Muñeco de señas (prueba)')),
      body: SafeArea(
        child: _error != null
            ? Center(
                child: Padding(
                  padding: const EdgeInsets.all(24),
                  child: Text(_error!, textAlign: TextAlign.center),
                ),
              )
            : _actual == null
                ? const Center(child: CircularProgressIndicator())
                : Column(
                    children: [
                      Expanded(
                        child: Container(
                          margin: const EdgeInsets.fromLTRB(16, 12, 16, 4),
                          decoration: BoxDecoration(
                            color: colores.surface,
                            borderRadius: BorderRadius.circular(20),
                            border: Border.all(
                              color: colores.primary.withValues(alpha: 0.25),
                              width: 1.5,
                            ),
                          ),
                          clipBehavior: Clip.antiAlias,
                          child: ReproductorSena(
                            clip: _actual!,
                            velocidad: _velocidad,
                          ),
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            Wrap(
                              spacing: 8,
                              alignment: WrapAlignment.center,
                              children: [
                                for (final palabra in _clips.keys)
                                  ChoiceChip(
                                    label: Text(palabra),
                                    selected: _actual?.palabra == palabra,
                                    onSelected: (_) => setState(
                                        () => _actual = _clips[palabra]),
                                  ),
                              ],
                            ),
                            const SizedBox(height: 4),
                            Row(
                              children: [
                                const Text('Velocidad'),
                                Expanded(
                                  child: Slider(
                                    value: _velocidad,
                                    min: 0.25,
                                    max: 1.0,
                                    divisions: 3,
                                    label: '${_velocidad}x',
                                    onChanged: (v) =>
                                        setState(() => _velocidad = v),
                                  ),
                                ),
                                Text('${_velocidad}x'),
                              ],
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
      ),
    );
  }
}
