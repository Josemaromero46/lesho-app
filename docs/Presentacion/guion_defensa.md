# Guión de defensa · 15 minutos

**Aplicación móvil de traducción bidireccional del Lenguaje de Señas Hondureño
mediante inteligencia artificial**

José Manuel Romero Martínez · UNAH Campus Comayagua

---

**Cómo usar este guión.** Los tiempos son acumulativos: indican el minuto en que
deberías estar terminando cada diapositiva. El guión completo suma unos 14 minutos
y medio, así que hay margen. Lo que va *en cursiva* son indicaciones para vos, no
se dice en voz alta.

---

## 1 · Portada — 60 s · *(termina en 1:00)*

Buenos días. Mi nombre es José Manuel Romero Martínez, estudiante de la carrera de
Ingeniería en Sistemas Computacionales de la Universidad Nacional Autónoma de
Honduras, Campus Comayagua. Agradezco al jurado su tiempo, y a mi asesor, el
Dr. Óscar Guillermo Hernández Ramírez, su acompañamiento en este trabajo.

*(Pausá. Cambiá el tono: acá empieza el contenido.)*

Quiero empezar por algo concreto.

Un niño sordo que crece en un hogar oyente puede señalar. Si tiene hambre, señala
la comida. Si le duele algo, señala dónde.

Señalar alcanza para pedir cosas.

Lo que no se puede señalar es una pregunta. Ni una explicación. Ni el miedo.

Un niño de siete años pregunta *por qué* decenas de veces al día. Ese niño no
tiene cómo.

*(Pausá otra vez, y conectá con el tema.)*

De ahí nace esta tesis: una aplicación móvil de traducción bidireccional del
Lenguaje de Señas Hondureño mediante inteligencia artificial.

---

## 2 · Agenda — 15 s · *(1:15)*

Voy a recorrer siete puntos: el planteamiento del problema, los objetivos, su
relación con los Objetivos de Desarrollo Sostenible, la metodología, la
arquitectura del sistema, los resultados, y las conclusiones.

En el camino les voy a mostrar la aplicación funcionando.

---

## 3 · Planteamiento del problema — 80 s · *(2:35)*

Pongamos números a esa situación.

En Honduras hay alrededor de cien mil personas con discapacidad auditiva. De todas
ellas, apenas cerca del uno por ciento tiene un carné oficial de identificación.

Ese uno por ciento no es un dato administrativo. Es la medida de cuánto sabe el
Estado de su propia gente.

El LESHO es su lengua natural, reconocida por ley desde 2013. Pero la mayoría de
los niños sordos nacen en hogares oyentes, en familias que no conocen la lengua. Y
fuera de las escuelas especializadas, sobre todo en el interior del país, casi no
hay intérpretes.

El acceso tardío a una lengua completa afecta el desarrollo del niño. No es un
problema de comodidad, es un problema de desarrollo.

Y del lado de la tecnología el vacío es igual de claro. Existen aplicaciones para
la lengua de señas americana, para la brasileña, para varias otras. Para el LESHO
no encontré ninguna que funcione en tiempo real.

*(Transición)* De ese problema salieron los objetivos.

---

## 4 · Objetivos — 50 s · *(3:25)*

El objetivo general fue desarrollar una aplicación de inteligencia artificial para
la comunicación entre niños sordos y personas oyentes, que funcione **por completo
en el dispositivo**.

Esa última parte no es un detalle técnico. Si la aplicación dependiera de internet,
fallaría justo donde más se necesita.

Para lograrlo me planteé cuatro objetivos específicos: construir un conjunto de
datos propio del LESHO, diseñar dos modelos de clasificación aptos para teléfonos,
desarrollar la aplicación con las dos direcciones de comunicación, y evaluar el
sistema.

---

## 5 · Objetivos de Desarrollo Sostenible — 35 s · *(4:00)*

El trabajo se conecta con tres Objetivos de Desarrollo Sostenible.

Con el cuatro, educación de calidad, porque la barrera del idioma limita el
aprendizaje del niño sordo. Con el nueve, innovación, porque desarrolla tecnología
propia donde no existía ninguna. Y con el diez, reducción de las desigualdades,
porque atiende a una población que el propio Estado apenas registra.

---

## 6 · Metodología — 60 s · *(5:00)*

La investigación es aplicada, porque no busca conocimiento teórico nuevo sino
resolver un problema concreto. El enfoque es mixto: las métricas miden el
desempeño técnico y la observación con usuarios evalúa si la aplicación de verdad
sirve. Y su alcance es exploratorio y descriptivo, porque no había antecedentes de
reconocimiento del LESHO con aprendizaje profundo.

La muestra son 30 personas seleccionadas de forma intencional, con criterios de
inclusión definidos, distribuidas en tres grupos: personas sordas, personas con
dominio del LESHO y personas oyentes. Hacen falta los tres porque la aplicación
tiene dos direcciones y cada grupo evalúa una parte distinta.

El método tuvo cuatro etapas, en orden.

Primero, definir el vocabulario con asesoría en LESHO, porque yo no soy quien
decide cómo se hace una seña.

Segundo, capturar los datos con un protocolo documentado y repetible.

Tercero, entrenar los modelos y llevarlos al teléfono.

Y cuarto, evaluar: exactitud, puntaje F1, el costo de recursos en el dispositivo, y
la usabilidad percibida.

---

## 7 · Herramientas utilizadas — 30 s · *(5:30)*

En software: Flutter para la aplicación, MediaPipe para extraer los puntos de las
manos y del cuerpo, y TensorFlow Lite para ejecutar los modelos dentro del
teléfono.

En hardware, y esto importa: todas las pruebas las hice en un Samsung Galaxy A13,
un teléfono de gama baja. Lo elegí a propósito. Si funciona ahí, funciona en el
teléfono que una familia hondureña realmente tiene.

---

## 8 · Arquitectura general del sistema — 60 s · *(6:30)*

Este es el sistema por dentro.

Lo primero, la línea punteada de afuera. Todo lo que está adentro corre en el
teléfono: no hay servidor, no hay nube, no hay conexión. Y eso no es una
configuración que se pueda cambiar: ningún componente abre una conexión de red. La
privacidad y el que funcione sin internet son consecuencia de esa decisión
estructural.

Adentro, la aplicación está en capas, y las flechas del costado van en un solo
sentido: ninguna capa de abajo conoce a las de arriba. Por eso pude rediseñar toda
la interfaz sin tocar un solo modelo.

*(Señalar la caja de la derecha)* Y acá está la decisión de diseño más importante.
MediaPipe tarda unos 250 milisegundos por fotograma. En el hilo de la interfaz, la
aplicación se congelaría. Por eso vive en un hilo nativo aparte, y los dos se
comunican por un canal de métodos.

---

## 9 · Vista funcional — 40 s · *(7:10)*

Visto desde quien la usa, es más simple.

El niño firma y el texto aparece en pantalla. La persona oyente escribe y el muñeco
hace la seña.

Y si escribe una palabra que todavía no tiene seña registrada, el sistema la
deletrea con el alfabeto. La aplicación nunca se queda callada.

*(Transición, con energía)* Y esto no es una maqueta. Se los voy a mostrar.

---

## 10 · Demostración de la aplicación — 90 s · *(8:40)*

*(Este es el momento de la defensa. No lo apures ni lo narres encima. Dejá que se
vea.)*

Esto es la aplicación real, corriendo en el teléfono de gama baja del que les
hablé.

Primero el deletreo.

*(Silencio. Dejá que se reconozcan las letras.)*

Ahora una seña completa.

*(Silencio.)*

Ahí la aplicación detectó las dos manos, su posición respecto al cuerpo, procesó el
movimiento y entregó la palabra. Todo eso ocurrió dentro del teléfono, sin
conexión.

*(Si algo falla: no te disculpes ni pelees con el equipo. Decí "les muestro la
grabación que preparé" y seguí. Perder treinta segundos ahí cuesta más que el
video.)*

---

## 11 · Resultados: Modelo A (alfabeto) — 60 s · *(9:40)*

Detrás de esa demostración hay dos modelos. El primero reconoce el alfabeto:
99.54 % de exactitud, en 33.9 kilobytes.

Pero el número interesante no es ese. Es este: cinco letras del LESHO se hacen
**con movimiento**. Y hay pares que comparten exactamente la misma forma de mano.
La L y la LL. La N y la Ñ. La R y la RR. Lo único que las distingue es que una se
mueve.

Eso significa que una fotografía no basta. Por eso el modelo no clasifica una
imagen, clasifica una **ventana de veinte fotogramas**. Ve la trayectoria, no la
pose.

---

## 12 · Resultados: Modelo B (señas dinámicas) — 70 s · *(10:50)*

El segundo modelo reconoce cincuenta señas dinámicas: 98.80 % de exactitud, en 78
kilobytes.

Y acá me encontré con el problema más interesante de todo el trabajo.

Muchas señas se distinguen por **dónde** se hacen. HOLA se hace cerca de la frente.
AGUA se hace cerca de la boca. La mano hace casi lo mismo; lo que cambia es el
lugar del cuerpo.

Si el modelo solo mirara la forma de la mano, las confundiría. Y las confundía.

La solución fue darle la ubicación de la mano respecto al cuerpo, y en particular
**hacia dónde apunta el índice**, porque en muchas señas la muñeca queda al costado
mientras el dedo señala la zona. Cuando incorporé eso, esas confusiones se
resolvieron.

---

## 13 · Resultados del dataset — 50 s · *(11:40)*

Ahora, si algo de este trabajo sobrevive al trabajo mismo, es esto.

Construí un conjunto de datos propio del LESHO: el alfabeto completo y cincuenta
señas dinámicas.

Y fíjense en el tercer número: **cero videos guardados**. Cada seña se procesa en
el momento, se extraen las coordenadas, y el video se descarta ahí mismo. Lo único
que queda son números.

Eso protege a quienes grabaron y hace que el conjunto se pueda publicar sin exponer
a nadie. En mi revisión no encontré antecedentes de un recurso así para el LESHO.

---

## 14 · Resultados: desempeño en el dispositivo — 45 s · *(12:25)*

Un modelo preciso que no corre en un teléfono real no sirve de nada. Así que lo
medí directamente sobre el Galaxy A13, mientras la aplicación reconocía señas.

Alrededor del 78 % del procesador, unos 210 megabytes de memoria, y cerca del 2 %
de batería cada diez minutos.

Son cifras de un teléfono modesto exigido al máximo, detectando dos manos y un
cuerpo en cada fotograma. Y aun así, funciona.

---

## 15 · Resumen de resultados — 30 s · *(12:55)*

*(Rápido. No releas la diapositiva.)*

En resumen: dos modelos, 99.54 % y 98.80 %, ambos por debajo de cien kilobytes,
funcionando sin internet en un teléfono de gama baja.

---

## 16 · Conclusiones — 65 s · *(14:00)*

Cierro con tres cosas.

**La aplicación** traduce en los dos sentidos y funciona por completo en el
teléfono, sin internet.

**Los datos** son un conjunto propio del LESHO, guardado solo como coordenadas, sin
antecedentes hallados. Es el aporte que queda disponible para que otros lo amplíen.

**Los modelos** alcanzan esa precisión ocupando menos de cien kilobytes cada uno.

Y quiero decir algo con claridad: la exactitud es alta sobre grabaciones que el
modelo nunca vio, pero el conjunto de personas que aportaron datos todavía es
reducido. Para afirmar que funciona con cualquier persona hace falta ampliar el
corpus. Esa es la recomendación principal de este trabajo, y no la escondo porque
es lo que dice el dato.

---

## 17 · ¡Gracias! — 30 s · *(14:30)*

*(Bajá el ritmo. Volvé al principio.)*

Al inicio les dije que un niño sordo en un hogar oyente puede señalar lo que
necesita, pero no tiene cómo preguntar por qué.

Esta aplicación no resuelve eso. Pero pone una herramienta más en esa casa, y
funciona en el teléfono que esa familia ya tiene.

Gracias a mi asesor, a la fundación que colaboró, y a la comunidad sorda que hizo
posible este trabajo. Quedo atento a sus preguntas.

---

## 18 · Bibliografía

*(No se lee. Queda proyectada durante las preguntas.)*

---

# Notas de entrega

## Lo que conviene memorizar

Cuatro momentos, el resto se improvisa:

1. La apertura completa, hasta "ese niño no tiene cómo".
2. La frase del uno por ciento.
3. La explicación de HOLA contra AGUA.
4. El cierre.

Son los cuatro puntos que deciden cómo te recuerdan.

## Ritmo

El guión suma unos 14 minutos y medio con las pausas. Si vas atrasado, la única
parte que podés comprimir es Conclusiones, porque los números ya los dijiste. Nunca
apures la demostración ni el cierre.

## Tres preguntas que van a salir

**"¿Cuántas personas participaron?"**
Decí el número real. Tu fortaleza no es el tamaño del corpus, es que vos mismo
señalás el límite antes de que te lo señalen.

**"¿Y los resultados de usabilidad?"**
"Hice sesiones de uso con las treinta personas y recogí la retroalimentación de
forma cualitativa, que está reportada en el capítulo de resultados. Lo que no
alcancé a aplicar dentro del plazo son los dos instrumentos estructurados, así que
esa valoración favorable no está cuantificada." Decilo en ese orden: primero lo que
sí hiciste, después el límite. Nunca insinúes que tenés datos en escala.

**"¿La latencia dónde la midió?"**
"En la computadora de desarrollo, como referencia del costo del modelo. El
desempeño en el teléfono lo medí aparte: procesador, memoria, batería y cuadros por
segundo."

**"¿Por qué no usó un muestreo aleatorio?"**
Porque no existe un marco muestral del cual sortear: no hay un registro de las
personas sordas de Honduras. El propio dato del uno por ciento con carné oficial lo
demuestra. Por eso el muestreo es intencional, con criterios de inclusión
definidos: no se tomó a quien estuviera a mano, se eligió a quien cumple el perfil.

**"¿Por qué 30 personas?"**
Por tres razones: representar los tres perfiles del problema, tener suficiencia
para un análisis descriptivo con frecuencias y porcentajes, y la viabilidad real de
acceso, porque la población con dominio del LESHO y afiliación institucional
verificable en Honduras es reducida.

## Sobre la arquitectura

Es la diapositiva que más va a interrogar un ingeniero de sistemas. Estas cuatro
cubren casi todo.

**"¿Qué patrón arquitectónico siguió?"**
Un monolito en capas, ejecutado sobre dos contextos de concurrencia. En capas
porque la dependencia va en un solo sentido, de la interfaz hacia la percepción, y
eso permite sustituir una capa sin tocar las de abajo. Monolito porque el sistema
se despliega como una sola unidad: no hay nada distribuido que justifique separarlo.

**"¿Por qué un monolito y no servicios separados?"**
Porque no hay red. Separar en servicios tiene sentido cuando hay que escalar partes
por separado o desplegarlas en máquinas distintas, y acá el requisito es el
contrario: todo tiene que caber y correr en un teléfono de gama baja, sin conexión.
Un monolito en capas da la separación de responsabilidades sin pagar el costo de
comunicación entre procesos.

**"¿Cómo se comunican Dart y Kotlin?"**
Por un canal de métodos de Flutter, que es una interfaz asíncrona de mensajes
serializados. Dart envía el fotograma y Kotlin devuelve las coordenadas de los
puntos. Es la única frontera entre los dos lenguajes, y está en un solo lugar del
código.

**"¿Por qué MediaPipe en la GPU y Pose en el procesador?"**
Porque los medí. Con el procesador, las manos tardaban 546 milisegundos por
fotograma contra 250 en la tarjeta gráfica, y las manos corren en vivo. La pose, en
cambio, solo corre fuera de línea, pero en la tarjeta gráfica tardaba 16 segundos
en inicializarse contra 2.8 en el procesador. Cada una está donde le conviene, y la
decisión salió de la medición, no de la intuición.

## Otras preguntas probables

**"¿Por qué no usó un Transformer?"**
Por la relación entre datos y recursos. Un Transformer necesita muchos más datos
para no sobreajustar, y la meta era que corriera en un teléfono de gama baja
ocupando menos de cien kilobytes.

**"MediaPipe ya es de Google. ¿Cuál es su aporte?"**
MediaPipe ubica los puntos de la mano, pero no sabe nada de lengua de señas. El
aporte está en todo lo que va después: los dos modelos, la incorporación del lugar
de articulación, y sobre todo el conjunto de datos, que no existía.

**"¿Y las expresiones faciales?"**
Forman parte de la gramática del LESHO y el sistema actual no las considera. Está
declarado como limitación y como trabajo futuro.

## Lo más importante

No hables de tu trabajo como si te disculparas por lo que le falta. Hiciste algo
que no existía para tu país. Eso se para solo.
