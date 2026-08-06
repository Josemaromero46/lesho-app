# Guión de video técnico · recorrido por el código

**Aplicación móvil de traducción bidireccional del LESHO**
José Manuel Romero Martínez

---

**Cómo usar este guión.** Cada bloque tiene tres partes: **qué mostrar en pantalla**
(archivo y líneas), **el código** que va a quedar visible, y **qué decir** mientras se ve.
Lo que va *en cursiva* son notas para vos, no se dice.

Duración estimada: **15 minutos**. Los bloques son independientes, así que si te pasás
de tiempo podés cortar el 6 o el 8 sin romper el hilo.

*(Consejo de grabación: aumentá el tamaño de fuente del editor antes de grabar. Lo que
se lee cómodo en tu monitor no se lee en un video comprimido.)*

---

## 0 · Estructura del proyecto — 60 s

**Mostrar:** el árbol de `app/lib/` en el explorador de archivos del editor.

**Decir:**

> El proyecto tiene dos partes. La aplicación en Flutter, dentro de `app/`, y el
> pipeline de entrenamiento en Python, dentro de `training/`. Son sistemas separados:
> Python produce los modelos, la aplicación los consume.
>
> Dentro de `lib/` el código está organizado en capas. `ui` son las pantallas,
> `control` es la lógica de sesión, `inferencia` ejecuta los modelos, `landmarks`
> habla con MediaPipe, y `core` guarda los contratos numéricos que se comparten con
> Python.
>
> Son unas cuatro mil líneas de Dart y unas trescientas ochenta de Kotlin. Voy a
> recorrer las seis piezas donde está la ingeniería de verdad.

---

## 1 · La frontera nativa — 150 s

### 1.1 El hilo aparte

**Mostrar:** `android/app/src/main/kotlin/com/josemrm/lesho_app/MainActivity.kt`, líneas 52 a 55

```kotlin
private val hiloDeteccion = HandlerThread("mediapipe").apply { start() }
private val handlerDeteccion = Handler(hiloDeteccion.looper)
private val handlerPrincipal = Handler(Looper.getMainLooper())
```

**Decir:**

> Esta es la primera decisión de diseño del sistema. MediaPipe no puede correr en el
> hilo de la interfaz, por dos razones: su API es de Java, y tarda unos 250
> milisegundos por fotograma. En el hilo de interfaz, la aplicación se congelaría en
> cada seña.
>
> Así que vive en un `HandlerThread` aparte, con su propia cola de mensajes. Y fíjense
> que hay dos handlers: uno para trabajar y otro para el hilo principal. Ya van a ver
> por qué hacen falta los dos.

### 1.2 El canal

**Mostrar:** líneas 57 a 60, y después 74 a 94

```kotlin
override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
    super.configureFlutterEngine(flutterEngine)
    MethodChannel(flutterEngine.dartExecutor.binaryMessenger, canal)
        .setMethodCallHandler { call, result ->
```

```kotlin
"detectar" -> {
    val bytes = call.argument<ByteArray>("bytes")
    val w = call.argument<Int>("width")
    val h = call.argument<Int>("height")
    val rot = call.argument<Int>("rotation") ?: 0
    val conPoseFrame = call.argument<Boolean>("conPose") ?: true
    ...
    handlerDeteccion.post {
        try {
            val salida = detectar(bytes, w, h, rot, conPoseFrame)
            handlerPrincipal.post { result.success(salida) }
```

**Decir:**

> Este es el único punto donde Dart y Kotlin se hablan: un canal de métodos llamado
> `lesho/manos`. Dart manda el fotograma crudo en formato NV21, más el ancho, el alto,
> la rotación, y si en este fotograma hay que correr también la detección de cuerpo.
>
> El trabajo se postea al hilo de MediaPipe, pero miren la línea de la respuesta:
> `handlerPrincipal.post`. La respuesta tiene que devolverse desde el hilo principal
> de Android. Es un requisito de Flutter, y si no se respeta la aplicación revienta.
> Ese detalle me costó una tarde.

### 1.3 Los delegados

**Mostrar:** líneas 140 a 160

```kotlin
hands = try {
    crearHands(numManos, Delegate.GPU)
} catch (e: Exception) {
    crearHands(numManos, Delegate.CPU)
}
...
pose = try {
    crearPose(Delegate.CPU)
} catch (e: Exception) {
    crearPose(Delegate.GPU)
}
```

**Decir:**

> Acá hay algo que parece un error y no lo es. Las manos van en la tarjeta gráfica y
> el cuerpo en el procesador. Al revés de lo que uno esperaría, porque la GPU es más
> rápida.
>
> La razón es que no se usan igual. Las manos corren en vivo, en cada fotograma, así
> que lo que importa es el tiempo por fotograma: medí 250 milisegundos en GPU contra
> 546 en procesador. El cuerpo solo corre al procesar una palabra ya grabada, en
> lote, así que lo que se nota no es la velocidad sino el arranque: la GPU tarda 16
> segundos en inicializar porque tiene que compilar los shaders, contra 2.8 el
> procesador.
>
> Cada una está donde le conviene, y noten el `try/catch`: si un teléfono no soporta
> el delegado, cae al otro en vez de fallar.

### 1.4 La conversión de imagen

**Mostrar:** líneas 208 a 216

```kotlin
private fun nv21ABitmap(nv21: ByteArray, width: Int, height: Int): Bitmap {
    val frameSize = width * height
    if (bufferArgb.size != frameSize) bufferArgb = IntArray(frameSize)
    val argb = bufferArgb
```

**Decir:**

> Y esto es puro rendimiento. La primera versión pasaba de NV21 a JPEG y de JPEG a
> Bitmap, usando las utilidades de Android. Funcionaba, pero comprimía y
> descomprimía en cada fotograma.
>
> Ahora la conversión es directa, con la fórmula BT.601 escrita a mano, y el buffer
> se reutiliza entre fotogramas en vez de reservar memoria nueva cada vez. En un
> teléfono de gama baja eso se nota.

---

## 2 · El cliente del canal y el bug más caro — 120 s

**Mostrar:** `lib/landmarks/detector_manos.dart`, líneas 41 y 84 a 89

```dart
static const MethodChannel _canal = MethodChannel('lesho/manos');
```

```dart
final rotado = rotacion % 180 == 90;
final aspectoImagen = rotado ? height / width : width / height;
factorAspecto = aspectoImagen / Constantes.aspectoEntrenamiento;
```

**Decir:**

> Del lado de Dart, esta clase es la que arma el mensaje y traduce la respuesta.
>
> Y estas tres líneas son la corrección del error más caro del proyecto. MediaPipe
> devuelve coordenadas normalizadas de cero a uno, pero normaliza la equis por el
> ancho y la ye por el alto, por separado. Yo entrené con video de 1280 por 720, o
> sea dieciséis a nueve, y el teléfono entrega tres a cuatro.
>
> El resultado es que la misma mano llega deformada al modelo, y confundía casi todas
> las letras. Lo verifiqué en Python: datos reales daban treinta de treinta, al
> deformarlos al aspecto del teléfono caían a quince, y con los mismos errores que
> yo veía en vivo.
>
> La corrección calcula el aspecto real en vivo, así que sirve en cualquier teléfono
> sin importar la proporción de su cámara.

---

## 3 · El contrato numérico — 120 s

**Mostrar:** `lib/core/normalizacion.dart`, líneas 38 a 45

```dart
final muneca = landmarksCrudos[0]; // landmark 0 = muñeca
final vector = <double>[];

for (final punto in landmarksCrudos) {
  vector.add(punto.x - muneca.x);
  vector.add(punto.y - muneca.y);
  vector.add(punto.z - muneca.z);
}
```

**Mostrar después:** líneas 80 a 89 y el comentario de la línea 92

```dart
double _tamanoMano(List<double> mano) {
  var total = 0.0;
  for (final k in _nudillos) {
    final x = mano[k * 3];
    final y = mano[k * 3 + 1];
    total += sqrt(x * x + y * y);
  }
  return total / _nudillos.length;
}
```

**Decir:**

> La normalización tiene dos pasos y cada uno quita una variable del problema.
>
> El primero traslada todos los puntos respecto a la muñeca. Con eso el vector deja
> de depender de dónde está la mano en el encuadre: la misma seña arriba o abajo
> produce los mismos números.
>
> El segundo divide por el tamaño de la mano, que se calcula como la distancia media
> de la muñeca a los cuatro nudillos. Con eso deja de depender de la distancia a la
> cámara.
>
> Lo que queda es la forma pura de la mano. Veintiún puntos por tres coordenadas son
> sesenta y tres valores, y por las dos manos, ciento veintiséis.
>
> *(Señalar el comentario del código)* Y quiero destacar esto: el comentario dice
> "replica exactamente `escalar_vector` de `training/comun/normalizacion.py`". Ese es
> el contrato del proyecto. Si Python y Dart no calculan idéntico, el modelo recibe
> algo distinto a lo que entrenó y falla en silencio. Es el tipo de error que no
> lanza excepción, solo da resultados malos.

---

## 4 · El deletreo: ventana, compuerta y persistencia — 180 s

### 4.1 La ventana rodante

**Mostrar:** `lib/core/constantes.dart`, línea 57, y `lib/control/maquina_estados.dart`

```dart
static const int tamanoVentanaA = 20;
```

**Decir:**

> El Modelo A no clasifica un fotograma suelto, clasifica una ventana de veinte.
> La razón son cinco letras del alfabeto: la jota, la eñe, la zeta, la elle y la
> erre, que se hacen con movimiento. Una sola pose no las representa.
>
> Con una ventana, una letra quieta produce veinte fotogramas casi idénticos y una
> letra con movimiento produce su trayectoria. Una misma arquitectura reconoce las
> dos, porque una pose fija es un caso particular de secuencia.

### 4.2 La compuerta de movimiento

**Mostrar:** `lib/control/maquina_estados.dart`, líneas 176 a 187

```dart
final mov = _velocidadMedia(ventana);

// Compuerta de movimiento (dos sentidos), idéntica a la demo.
if (mov < Constantes.umbralMovimientoAbs) {
  for (final i in _idxMovimiento) {
    probs[i] = 0.0;
  }
} else if (mov > Constantes.umbralMovimientoMoviendo) {
  for (final i in _idxGemelas) {
    probs[i] = 0.0;
  }
}
```

**Decir:**

> Este es mi truco favorito del proyecto, y resuelve un problema concreto.
>
> La ene y la eñe tienen la misma forma de mano. La ele y la elle también. La erre y
> la doble erre igual. Lo único que las distingue es que una se mueve y la otra no.
> El modelo se confundía entre ellas todo el tiempo.
>
> La solución no fue tocar el modelo, fue una compuerta que trabaja en los dos
> sentidos. Si la mano está prácticamente quieta, por debajo de 0.0022, se anulan las
> probabilidades de las letras que llevan movimiento: si no te movés, no podés estar
> haciendo una eñe. Y si la mano se mueve claro, por encima de 0.006, se anulan las
> gemelas estáticas: si te estás moviendo, no es una ene.
>
> Es conocimiento del dominio metido como código, y arregló en diez líneas algo que
> con más datos habría costado semanas.

### 4.3 El filtro de persistencia

**Mostrar:** `lib/control/filtro_persistencia.dart`, líneas 31 a 52

```dart
if (confianza < Constantes.umbralConfianza) {
  _reiniciarContador();
  return null;
}

if (clase == _claseActual) {
  _contadorFrames++;
} else {
  _claseActual = clase;
  _contadorFrames = 1;
}

if (_contadorFrames >= Constantes.fotogramasPersistencia) {
  ...
  _activarCooldown();
```

**Decir:**

> Sin este filtro la pantalla sería un desastre. Mientras la mano viaja de una letra
> a la otra pasa por formas intermedias, y el modelo clasifica cada una de ellas.
>
> Así que una letra solo se acepta si el modelo la predice cinco veces seguidas con
> al menos sesenta por ciento de confianza. Cualquier cambio reinicia el contador.
>
> Y una vez aceptada se activa un enfriamiento de 1200 milisegundos, para que no se
> escriba la misma letra cinco veces por sostener la mano.

---

## 5 · Las palabras: el preprocesado del Modelo B — 120 s

**Mostrar:** `lib/core/constantes.dart`, líneas 30 a 34, y `lib/control/secuencias_b.dart`, líneas 138 a 153

```dart
static const int tamanoUbicacion = 18;
static const int tamanoVectorB = 126 + tamanoUbicacion; // 144
static const int tamanoRelativo = 8;
static const int tamanoEntradaB = tamanoVectorB + tamanoRelativo; // 152
static const double gananciaUbicacion = 2.5; // pesa el lugar frente a la forma
```

```dart
seq = _recortarQuietos(seq);
// 3-4. Config escalada (126) + [ubicación(14) + relativos(8)] x ganancia.
...
for (final v in [...ubicCruda, ...relativos]) v * Constantes.gananciaUbicacion
...
return _remuestrear(procesada, longitud);
```

**Decir:**

> El Modelo B es distinto, y la diferencia está en lo que ve.
>
> Muchas señas del LESHO se distinguen por dónde se hacen, no por la forma de la
> mano. HOLA se hace en la frente y AGUA en la boca, con una configuración parecida.
> Pero la normalización respecto a la muñeca, la que acabo de mostrar, borra
> justamente esa información: la vuelve invariante a la posición.
>
> Así que para el Modelo B se agrega un marco de referencia del cuerpo, con
> MediaPipe Pose. Cada fotograma lleva la configuración de las manos, ciento
> veintiséis valores, más la ubicación respecto al cuerpo, dieciocho más.
>
> Y esta constante es una decisión importante: la ganancia de 2.5. La ubicación se
> multiplica por dos y medio antes de entrar al modelo, para que pese más que la
> forma. Sin eso el modelo se apoyaba demasiado en la configuración de la mano y
> confundía las señas que difieren solo por el lugar.
>
> El preprocesado además recorta los tramos donde la mano está quieta, al principio y
> al final, y remuestrea todo a cuarenta fotogramas, para que una seña lenta y una
> rápida lleguen con la misma longitud.

---

## 6 · Dirección 2: el tokenizador voraz — 120 s

**Mostrar:** `lib/texto_a_sena/diccionario_senas.dart`, líneas 156 a 192

```dart
var i = 0;
while (i < originales.length) {
  final restantes = originales.length - i;
  final tope = restantes < _largoMaximo ? restantes : _largoMaximo;

  String? claveEncontrada;
  var consumidas = 1;
  for (var ventana = tope; ventana >= 1; ventana--) {
    final clave = normalizadas.sublist(i, i + ventana).join('_');
    final resuelta = _resolverClave(clave);
    if (resuelta != null) {
      claveEncontrada = resuelta;
      consumidas = ventana;
      break;
    }
  }
  ...
  } else {
    unidades.add(UnidadSena.deletreo(
      texto: texto,
      letras: _deletrear(normalizadas[i]),
    ));
  }
```

**Decir:**

> La otra dirección tiene su propio problema: hay señas que son una sola pero se
> escriben con varias palabras, como BUENOS DÍAS o POR FAVOR.
>
> La solución es un emparejamiento voraz de la frase más larga. En cada posición de
> la oración prueba primero la ventana más larga contra el diccionario y va bajando.
> La primera que existe se consume como una sola seña. Así una seña compuesta se
> detecta esté al inicio, en medio o al final.
>
> Y miren el `else`: si ninguna ventana coincide, ni siquiera la palabra sola, cae al
> deletreo letra por letra. Ese es el respaldo, y es lo que hace que la aplicación
> nunca se quede callada. Un nombre propio, una palabra que no está en el
> vocabulario, cualquier cosa: siempre hay algo que mostrar.
>
> La normalización de la búsqueda quita tildes y mayúsculas, pero conserva la eñe,
> porque el alfabeto dactilológico sí tiene seña para la eñe.

---

## 7 · Cierre — 40 s

**Mostrar:** el árbol de `lib/` otra vez, o el diagrama de arquitectura.

**Decir:**

> Para cerrar, tres ideas que atraviesan todo el código.
>
> La primera es que el contrato numérico entre Python y Dart es sagrado. Cada función
> de `core` tiene un comentario que dice a qué función de Python replica, porque si
> se separan, el modelo falla en silencio.
>
> La segunda es que varias de las mejoras que más sirvieron no fueron de arquitectura
> ni de modelo, sino conocimiento del dominio metido como código: la compuerta de
> movimiento, la ganancia de la ubicación, la corrección de aspecto.
>
> Y la tercera es que todo esto corre dentro del teléfono. No hay servidor, no hay
> red, no hay una sola llamada saliente. Y eso no es una configuración: es una
> propiedad estructural del código que acabo de mostrar.

---

# Notas de grabación

**Antes de empezar**
- Subí el tamaño de fuente del editor a algo que se lea en video comprimido.
- Cerrá paneles que no uses: terminal, control de versiones, problemas.
- Tené los archivos ya abiertos en pestañas, en el orden del guión.

**Al grabar**
- No leas el código en voz alta línea por línea. Mostralo y explicá qué resuelve.
- Cuando cites un número, que esté visible en pantalla en ese momento.
- Los bloques 6 y 8 son los que podés cortar si te pasás de tiempo.

**Lo que más suma**
Los tres momentos donde contás un problema real y su solución: el bug del aspecto,
la compuerta de movimiento y los delegados híbridos. Ahí es donde se ve que hubo
ingeniería y no solo integración de librerías. Si tenés que elegir qué contar bien,
elegí esos.
