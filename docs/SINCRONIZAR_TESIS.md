# Sincronización de la tesis con el sistema implementado

Este documento lista los cambios técnicos que se hicieron en el código respecto
al diseño original. La tesis (archivos `.tex`) se escribió con las decisiones
viejas, así que hay que revisarla contra esta lista. Es el checklist de la
Fase 8 (sincronizar la tesis).

Estado: el código de todo el sistema del alfabeto está construido y probado.
Falta reconciliar el documento de tesis.

---

## Resumen de cambios (viejo → nuevo)

| # | Tema | Antes (en la tesis) | Ahora (en el código) |
|---|------|---------------------|----------------------|
| 1 | Letras del alfabeto | 27 letras (A..Z, Ñ) | **30 letras**, se agregan los dígrafos **CH, LL, RR** |
| 2 | Clases del Modelo A | 30 clases | **33 clases** (30 letras + INICIO + FIN + REPOSO) |
| 3 | Arquitectura del Modelo A | MLP (perceptrón multicapa), estático frame por frame | **Red convolucional 1D sobre una ventana temporal** |
| 4 | Entrada del Modelo A | 1 fotograma de 63 valores | **Ventana de 20 fotogramas de 126 valores**, forma `[20, 126]` |
| 5 | Vector de landmarks | 63 valores (una mano) | **126 valores** (dos manos: izquierda 63 + derecha 63, ceros si falta) |
| 6 | Letras con movimiento | No contempladas | **J, Ñ, Z, LL, RR** se reconocen por la ventana temporal; CH y R son estáticas |
| 7 | Reconocimiento del alfabeto | Pose única por frame | **Buffer rodante** de 20 fotogramas, sin INICIO/FIN para el deletreo |
| 8 | Tamaño del Modelo A en TFLite | menos de 2 MB (estimado) | **~34 KB** con cuantización de rango dinámico |

---

## Detalle por tema

### 1 y 2. Alfabeto completo y número de clases

El alfabeto pasó de 27 a 30 letras al incluir los dígrafos CH, LL y RR como
señas propias, no como deletreo de dos letras. Orden de contrato (define los
índices de salida del modelo):

```
A B C CH D E F G H I J K L LL M N Ñ O P Q R RR S T U V W X Y Z INICIO FIN REPOSO
```

El Modelo A tiene 33 salidas.

### 3, 4 y 7. Ventana temporal en vez de MLP estático

El Modelo A ya no es un MLP que clasifica un fotograma aislado. Es una red
convolucional 1D que clasifica una ventana de los últimos 20 fotogramas
(`TAMANO_VENTANA_A = 20`, cerca de 0.7 s a 30 fps). En tiempo real la app
mantiene un buffer rodante: agrega cada fotograma, descarta el más viejo, y
clasifica cuando la ventana está llena. Una letra estática produce una ventana
casi quieta; una letra con movimiento produce una ventana con su trayectoria.
Se eligió convolución 1D en lugar de LSTM/GRU por optimización para teléfonos de
gama baja o media.

### 5. Vector de dos manos y normalización por escala

El vector de entrada por fotograma es de 126 valores (dos manos de 63). La
primera mitad es la mano izquierda, la segunda la derecha; la mitad de una mano
ausente va en ceros. Cada mano se normaliza en dos pasos: primero por
**traslación** (se resta la muñeca, para que no importe dónde está la mano en el
cuadro) y luego por **escala** (se divide por el tamaño de la mano, la distancia
media de la muñeca a los nudillos, para que no importe la distancia a la cámara).
Sin el paso de escala, el modelo confundía letras de la misma forma de mano
(N con Ñ, L con LL) según qué tan cerca estaba la mano. Si la tesis describe la
normalización solo como traslación respecto a la muñeca, hay que agregar el paso
de escala.

### 6. Letras con movimiento

Cinco letras se ejecutan con movimiento: J, Ñ, Z, LL, RR. Se graban con una
captura de secuencia aparte y se reconocen por la ventana temporal, sin usar las
señas de INICIO y FIN (esas siguen reservadas para las palabras del Modelo B).
CH y R son estáticas.

### 8. Optimización

Arquitectura ligera (unos 25 mil parámetros). El modelo TFLite con cuantización
de rango dinámico pesa alrededor de 34 KB, muy por debajo de los 2 MB. La entrada
y la salida siguen en float para que la app no tenga que cuantizar los landmarks.

### Dirección 2 (texto a seña)

El fallback de deletreo ahora reconoce los dígrafos: "calle" se deletrea c, a,
ll, e y no c, a, l, l, e. El diccionario de letras para el fallback tiene las 30
letras.

---

## Términos y números a buscar en los `.tex`

Al revisar la tesis, buscar y corregir donde aparezcan:

- "27 letras" → 30 letras
- "30 clases" (del Modelo A) → 33 clases
- "MLP", "perceptrón multicapa" (para el Modelo A) → red convolucional 1D de ventana temporal
- "63 valores", "63 landmarks", "21 × 3 = 63" → 126 valores (dos manos)
- "normaliza respecto a la muñeca" (solo traslación) → agregar el paso de normalización por escala (dividir por el tamaño de la mano)
- "frame por frame", "una sola pose", "clasificador estático" (del Modelo A) → ventana temporal
- La lista del alfabeto sin CH, LL, RR → incluirlos
- Diagramas del Capítulo de Implementación que muestren el Modelo A como estático o con entrada de 63

Sobre el Modelo B (50 señas dinámicas): sigue siendo LSTM/GRU de secuencia, pero
se le agregó la **ubicación respecto al cuerpo** (lugar de articulación). Muchas
señas se distinguen por la zona del cuerpo donde se hacen, y la normalización a la
muñeca borra eso. La solución es agregar MediaPipe Pose en la ruta dinámica y
describir cada seña como configuración de la mano (relativa a la muñeca) más
ubicación de la mano relativa al cuerpo (centro de los hombros, escalada por el
ancho de los hombros). Si la tesis describe el Modelo B solo con landmarks de la
mano, hay que agregar esta parte. La definición de las 50 clases sigue pendiente de
la asesoría LESHO. Esto no afecta al Modelo A.

---

## Qué NO cambió

- El Modelo B sigue siendo un clasificador de secuencia entre INICIO y FIN (pero ahora con la ubicación respecto al cuerpo agregada, ver arriba).
- Las señas INICIO, FIN y REPOSO siguen existiendo con la misma función.
- La arquitectura on-device, sin backend, sin nube.
- El flujo de las dos direcciones de comunicación.
