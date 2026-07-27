# COMANDOS_DIRECCION2.md — Flujo de la Dirección 2 (texto → seña, muñeco)

Guía rápida de comandos del muñeco de cápsulas. Todo se corre desde la carpeta
`training/` con el Python del entorno virtual:

```
cd C:\Users\josem\Downloads\TESIS\training
..\.venv\Scripts\python.exe <comando>
```

---

## 1. Grabar clips (captura con revisión en vivo)

Cada palabra se graba como un clip de landmarks (JSON), sin guardar video.
Tras cada toma, el clip se reproduce DE INMEDIATO sobre el muñeco real:
ENTER lo guarda, R lo repite. Así se sale de la sesión con clips validados.

```
# MODO LIBRE (por defecto, recomendado): se escribe la palabra en la ventana,
# ENTER la graba, se revisa en el muneco, y al guardar vuelve a pedir palabra.
..\.venv\Scripts\python.exe capture\captura_diccionario.py

# MODO LISTA: recorre una lista predefinida.
..\.venv\Scripts\python.exe capture\captura_diccionario.py --archivo capture\palabras\piloto.txt
..\.venv\Scripts\python.exe capture\captura_diccionario.py --palabras "HOLA,BUENOS DIAS,AGUA"

# Opciones utiles:
#   --tomas 1          (modo lista) una toma por palabra
#   --carpeta piloto   subcarpeta de salida en training/clips/ (default piloto)
#   --sin-revision     grabar de corrido, sin muneco tras cada toma
#   --persona nombre   metadato de control de calidad
```

En el MODO LIBRE: escribir la palabra (el ESPACIO arma frases compuestas, que
se guardan como UNA sena: "BUENOS DIAS" -> BUENOS_DIAS_t01.json), ENTER graba
con cuenta de 3 segundos, el muneco reproduce la toma, ENTER la guarda o R la
repite (con otra cuenta de 3 segundos), y vuelve a pedir la palabra. ESC
termina y muestra el resumen. La enie se escribe normal; en un teclado en
ingles la tecla ; la produce.

ENCUADRE: deben verse la cara, los hombros y los CODOS (el muñeco dibuja los
brazos). Las caderas ya NO se exigen: obligaban a alejarse y la mano quedaba
con pocos píxeles, que es justo lo que impedía separar los dedos cuando van
juntos. Sentarse a ~70 cm da cerca de 45% más de detalle en los dedos. Si sale
"Codos fuera de cuadro", alejarse un poco y repetir.

LUZ (lo que más ayuda con los dedos juntos): usar luz LATERAL, no frontal. La
luz de costado marca sombras ENTRE los dedos, y esas sombras son la pista que
MediaPipe usa para separarlos; la luz plana de frente los funde. Conviene
además un fondo liso que contraste con la piel, y girar un poco la mano en las
señas de dedos juntos en vez de mostrarla de canto.

Teclas durante la grabación: ESPACIO pausa (solo en modo lista), R repetir
toma, Q salir. Durante la revisión: ENTER guardar, R repetir (3 s de cuenta).

Salida: `training/clips/<carpeta>/<PALABRA>_tNN.json` (30-60 KB por clip).

## 2. Re-revisar clips guardados (visor)

```
..\.venv\Scripts\python.exe demo\visor_clips.py                # todos los del piloto
..\.venv\Scripts\python.exe demo\visor_clips.py clips\piloto\HOLA_t01.json
```

Teclas: ESPACIO pausa, A/D fotograma, V velocidad (1x/0.75x/0.5x/0.25x),
M espejo, N/P clip siguiente/anterior, G guardar PNG, Q salir.
Con `--exportar carpeta --cada 5` guarda PNGs sin abrir ventana.

### Arreglar un clip con temblor o dedos perdidos (modo edicion)

Dentro del visor, la tecla **E** entra al MODO EDICION. Ahi se corrige un tramo
malo (mano temblorosa o perdida) SIN colocar los dedos a mano: el tramo se
regenera desde los frames buenos de los lados.

  - A/D : moverse fotograma a fotograma hasta el tramo malo
  - I   : marcar el INICIO del tramo malo (fotograma actual)
  - O   : marcar el FIN del tramo malo
  - F   : arreglar (regenera ese tramo interpolando desde los frames buenos)
  - Z   : deshacer el ultimo arreglo
  - S   : guardar el clip corregido (el original queda respaldado en un .bak)
  - C   : limpiar la marca    E : salir del modo edicion

Los huecos CORTOS de mano (hasta 0.6 s, por ejemplo al pasar la mano por la
cara) se rellenan solos al cargar el clip; el modo edicion es para los tramos
mas largos o el temblor.

Este es el paso de VALIDACIÓN con la persona que sabe LESHO: ¿se entiende la
seña? ¿la orientación de la palma se lee? (Fase 0, PLAN_DIRECCION2.md).

## 3. Pasar los clips a la app

```
# Copiar los clips elegidos (la mejor toma de cada palabra) a los assets:
copy training\clips\piloto\HOLA_t01.json app\assets\clips\piloto\
```

Los assets de `app/assets/clips/piloto/` ya están declarados en pubspec.yaml.
La pantalla de prueba (inicio -> "Probar muñeco de señas (prueba)") lista los
clips en `pantalla_prueba_muneco.dart` (`_rutas`): agregar ahí la ruta del
clip nuevo.

## 4. Compilar e instalar en el A13 (32 bits)

```
cd C:\Users\josem\Downloads\TESIS\app
flutter build apk --release --target-platform android-arm
C:\Users\josem\AppData\Local\Android\sdk\platform-tools\adb.exe install -r build\app\outputs\flutter-apk\app-release.apk
```

(El S23 de 64 bits solo cuando se pida: `--target-platform android-arm64`.)

---

Estado y decisiones de diseño: PLAN_DIRECCION2.md (contrato del clip, fases,
decisiones abiertas con la asesoría LESHO: vista espejo predeterminada y color
del muñeco).
