import 'package:flutter/material.dart';
import 'package:lesho_app/texto_a_sena/clip_sena.dart';
import 'package:lesho_app/texto_a_sena/diccionario_senas.dart';
import 'package:lesho_app/texto_a_sena/reproductor_sena.dart';

/// Pantalla de texto a seña (Dirección 2: la persona oyente escribe, el niño ve).
///
/// La persona escribe una frase, el diccionario la descompone en unidades (una
/// seña propia por palabra, o el deletreo letra por letra cuando la palabra no
/// tiene seña registrada) y el muñeco las reproduce en orden.
///
/// Los clips que aún no se han grabado se saltan sin romper la reproducción: se
/// avisa al final qué palabras no se pudieron mostrar, en vez de dejar la
/// pantalla colgada.
class PantallaTextoASena extends StatefulWidget {
  const PantallaTextoASena({super.key});

  @override
  State<PantallaTextoASena> createState() => _EstadoPantallaTextoASena();
}

/// Un paso de la reproducción: el clip ya cargado y la palabra que representa.
class _Paso {
  final ClipSena clip;
  final String etiqueta;
  const _Paso(this.clip, this.etiqueta);
}

class _EstadoPantallaTextoASena extends State<PantallaTextoASena> {
  final _diccionario = DiccionarioSenas();
  final _controladorTexto = TextEditingController();
  final _focoTexto = FocusNode();

  List<_Paso> _pasos = [];
  int _indice = 0;
  bool _preparando = false;
  // La frase llegó al final: el muñeco se queda en la última seña, en vez de
  // desaparecer de golpe, y se ofrece repetirla.
  bool _finalizado = false;
  // Cuenta las veces que se reprodujo la frase. Entra en la clave del
  // reproductor para que al repetir se cree uno nuevo: si la frase tiene una
  // sola seña, el índice no cambia y sin esto el clip no volvería a arrancar.
  int _pase = 0;
  List<String> _sinClip = [];

  bool get _hayEscena => _pasos.isNotEmpty;
  bool get _reproduciendo => _hayEscena && !_finalizado;

  @override
  void dispose() {
    _controladorTexto.dispose();
    _focoTexto.dispose();
    super.dispose();
  }

  /// Traduce la frase y carga los clips necesarios. Los que falten se anotan
  /// para avisar al final, y la reproducción sigue con los que sí están.
  Future<void> _mostrar() async {
    final frase = _controladorTexto.text.trim();
    if (frase.isEmpty) return;
    _focoTexto.unfocus();

    setState(() {
      _preparando = true;
      _pasos = [];
      _indice = 0;
      _finalizado = false;
      _sinClip = [];
    });

    final unidades = _diccionario.traducir(frase);
    final pasos = <_Paso>[];
    final faltantes = <String>[];

    for (final unidad in unidades) {
      for (final ruta in unidad.clips) {
        try {
          pasos.add(_Paso(await ClipSena.desdeAsset(ruta), unidad.texto));
        } catch (_) {
          if (!faltantes.contains(unidad.texto)) faltantes.add(unidad.texto);
        }
      }
    }

    if (!mounted) return;
    setState(() {
      _preparando = false;
      _pasos = pasos;
      _sinClip = faltantes;
    });
  }

  void _siguiente() {
    if (!mounted) return;
    if (_indice + 1 < _pasos.length) {
      setState(() => _indice++);
    } else {
      setState(() => _finalizado = true);
    }
  }

  /// Vuelve a reproducir la frase desde la primera seña.
  void _repetir() {
    setState(() {
      _indice = 0;
      _finalizado = false;
      _pase++;
    });
  }

  void _detener() {
    setState(() {
      _pasos = [];
      _indice = 0;
      _finalizado = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final colores = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        foregroundColor: colores.onSurface,
        elevation: 0,
        title: const Text('Escribir para mostrar señas'),
      ),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(child: _escena(colores)),
            _panelEntrada(colores),
          ],
        ),
      ),
    );
  }

  /// El muñeco, o el estado inicial cuando todavía no hay nada que mostrar.
  Widget _escena(ColorScheme colores) {
    if (_preparando) {
      return const Center(child: CircularProgressIndicator());
    }

    if (!_hayEscena) {
      return _EstadoInicial(sinClip: _sinClip);
    }

    final paso = _pasos[_indice];
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 4, 20, 12),
      child: Column(
        children: [
          Expanded(
            child: Container(
              decoration: BoxDecoration(
                color: colores.surface,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: colores.outlineVariant),
              ),
              clipBehavior: Clip.antiAlias,
              child: ReproductorSena(
                key: ValueKey('$_pase-$_indice'),
                clip: paso.clip,
                // Sin bucle: la frase avanza al siguiente clip cuando este
                // termina. Con `repetir` en true (su valor por defecto) el clip
                // se repetiría para siempre y nunca avisaría que acabó.
                repetir: false,
                alTerminar: _siguiente,
              ),
            ),
          ),
          const SizedBox(height: 14),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                paso.etiqueta.toUpperCase(),
                style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      color: colores.primary,
                      letterSpacing: 1.5,
                    ),
              ),
              if (_pasos.length > 1) ...[
                const SizedBox(width: 12),
                Text(
                  '${_indice + 1} de ${_pasos.length}',
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ],
            ],
          ),
        ],
      ),
    );
  }

  Widget _panelEntrada(ColorScheme colores) {
    return Container(
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      padding: const EdgeInsets.fromLTRB(20, 18, 20, 20),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          TextField(
            controller: _controladorTexto,
            focusNode: _focoTexto,
            maxLines: 2,
            minLines: 1,
            textCapitalization: TextCapitalization.sentences,
            textInputAction: TextInputAction.send,
            decoration: const InputDecoration(
              hintText: 'Escribe lo que quieres decir',
            ),
            onSubmitted: (_) => _mostrar(),
          ),
          const SizedBox(height: 14),
          FilledButton.icon(
            onPressed: _preparando
                ? null
                : _reproduciendo
                    ? _detener
                    : (_finalizado ? _repetir : _mostrar),
            icon: Icon(_reproduciendo
                ? Icons.stop_rounded
                : (_finalizado
                    ? Icons.replay_rounded
                    : Icons.play_arrow_rounded)),
            label: Text(_reproduciendo
                ? 'Detener'
                : (_finalizado ? 'Repetir' : 'Mostrar en señas')),
            style: FilledButton.styleFrom(
              backgroundColor:
                  _reproduciendo ? colores.error : colores.secondary,
              foregroundColor:
                  _reproduciendo ? Colors.white : colores.onSecondary,
              padding: const EdgeInsets.symmetric(vertical: 16),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// Estado inicial: invita a escribir y, si la última frase tuvo palabras sin
/// clip grabado, lo dice con claridad en vez de fallar en silencio.
class _EstadoInicial extends StatelessWidget {
  final List<String> sinClip;

  const _EstadoInicial({required this.sinClip});

  @override
  Widget build(BuildContext context) {
    final colores = Theme.of(context).colorScheme;

    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 40),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 96,
              height: 96,
              decoration: BoxDecoration(
                color: colores.secondaryContainer,
                borderRadius: BorderRadius.circular(24),
              ),
              child: Icon(
                Icons.accessibility_new_rounded,
                size: 52,
                color: colores.secondary,
              ),
            ),
            const SizedBox(height: 24),
            Text(
              'Escribe una frase y el muñeco la hará en señas.',
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                    color: colores.onSurfaceVariant,
                  ),
            ),
            if (sinClip.isNotEmpty) ...[
              const SizedBox(height: 24),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                decoration: BoxDecoration(
                  color: colores.secondaryContainer,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  'Todavía no hay seña grabada para: ${sinClip.join(", ")}',
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: colores.onSecondaryContainer,
                      ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
