# ESTRUCTURA_TESIS.md

Esqueleto general de capítulos del documento de tesis. Este archivo es referencia obligatoria al redactar cualquier sección del documento.

**Título:** Aplicación de inteligencia artificial para la comunicación entre niños sordos y oyentes mediante reconocimiento de señas LESHO

---

## CAPÍTULO 1. PRESENTACIÓN

1.1. Planteamiento del problema
&nbsp;&nbsp;&nbsp;&nbsp;1.1.1. Contexto de la comunidad sorda en Honduras
&nbsp;&nbsp;&nbsp;&nbsp;1.1.2. Brecha de comunicación entre niños sordos y personas oyentes
&nbsp;&nbsp;&nbsp;&nbsp;1.1.3. Ausencia de herramientas tecnológicas accesibles para LESHO
1.2. Preguntas de investigación
1.3. Justificación del estudio
1.4. Objetivo general
1.5. Objetivos específicos
1.6. Alcance del estudio
1.7. Limitaciones del estudio
1.8. Viabilidad del estudio

---

## CAPÍTULO 2. INVESTIGACIONES PREVIAS RELACIONADAS

(Ver detalle en ESTRUCTURA_MARCO_TEORICO.md)

---

## CAPÍTULO 3. DISCAPACIDAD AUDITIVA Y COMUNIDAD SORDA

(Ver detalle en ESTRUCTURA_MARCO_TEORICO.md)

---

## CAPÍTULO 4. LENGUA DE SEÑAS HONDUREÑA (LESHO)

(Ver detalle en ESTRUCTURA_MARCO_TEORICO.md)

---

## CAPÍTULO 5. VISIÓN POR COMPUTADORA APLICADA AL RECONOCIMIENTO DE GESTOS

(Ver detalle en ESTRUCTURA_MARCO_TEORICO.md)

---

## CAPÍTULO 6. APRENDIZAJE PROFUNDO PARA CLASIFICACIÓN DE GESTOS

(Ver detalle en ESTRUCTURA_MARCO_TEORICO.md)

---

## CAPÍTULO 7. DESARROLLO DE APLICACIONES MÓVILES INCLUSIVAS

(Ver detalle en ESTRUCTURA_MARCO_TEORICO.md)

---

## CAPÍTULO 8. MARCO CONCEPTUAL

8.1. Glosario conceptual del estudio

---

## CAPÍTULO 9. MARCO LEGAL

9.1. Marco normativo internacional
&nbsp;&nbsp;&nbsp;&nbsp;9.1.1. Convención sobre los Derechos de las Personas con Discapacidad
&nbsp;&nbsp;&nbsp;&nbsp;9.1.2. Objetivos de Desarrollo Sostenible aplicables
9.2. Marco normativo nacional
&nbsp;&nbsp;&nbsp;&nbsp;9.2.1. Constitución de la República de Honduras
&nbsp;&nbsp;&nbsp;&nbsp;9.2.2. Ley de Equidad y Desarrollo Integral para las Personas con Discapacidad
9.3. Consideraciones éticas y de protección de datos

---

## CAPÍTULO 10. METODOLOGÍA

10.1. Tipo y enfoque de la investigación
10.2. Diseño metodológico
10.3. Población y muestra
&nbsp;&nbsp;&nbsp;&nbsp;10.3.1. Colaboradores para la construcción del corpus
&nbsp;&nbsp;&nbsp;&nbsp;10.3.2. Usuarios para la evaluación del sistema
10.4. Construcción del corpus de señas
&nbsp;&nbsp;&nbsp;&nbsp;10.4.1. Selección del vocabulario inicial
&nbsp;&nbsp;&nbsp;&nbsp;10.4.2. Definición de las señas de control (INICIO y FIN)
&nbsp;&nbsp;&nbsp;&nbsp;10.4.3. Protocolo y condiciones de grabación
&nbsp;&nbsp;&nbsp;&nbsp;10.4.4. Estructura y formato del dataset
10.5. Procesamiento y preparación de datos
&nbsp;&nbsp;&nbsp;&nbsp;10.5.1. Extracción de landmarks con MediaPipe
&nbsp;&nbsp;&nbsp;&nbsp;10.5.2. Normalización respecto a la muñeca
&nbsp;&nbsp;&nbsp;&nbsp;10.5.3. División en conjuntos de entrenamiento, validación y prueba
10.6. Diseño y entrenamiento de los modelos
&nbsp;&nbsp;&nbsp;&nbsp;10.6.1. Hiperparámetros y criterios de selección
&nbsp;&nbsp;&nbsp;&nbsp;10.6.2. Métricas de evaluación
10.7. Instrumentos para la evaluación con usuarios

---

## CAPÍTULO 11. IMPLEMENTACIÓN

11.1. Arquitectura general del sistema
&nbsp;&nbsp;&nbsp;&nbsp;11.1.1. Diagrama de arquitectura
&nbsp;&nbsp;&nbsp;&nbsp;11.1.2. Componentes y flujos de datos
&nbsp;&nbsp;&nbsp;&nbsp;11.1.3. Justificación del enfoque on-device
11.2. Construcción del dataset
&nbsp;&nbsp;&nbsp;&nbsp;11.2.1. Script de captura guiada
&nbsp;&nbsp;&nbsp;&nbsp;11.2.2. Sesiones de grabación con colaboradores
&nbsp;&nbsp;&nbsp;&nbsp;11.2.3. Consolidación del dataset final
11.3. Implementación del modelo estático
&nbsp;&nbsp;&nbsp;&nbsp;11.3.1. Arquitectura de la red neuronal
&nbsp;&nbsp;&nbsp;&nbsp;11.3.2. Proceso de entrenamiento
&nbsp;&nbsp;&nbsp;&nbsp;11.3.3. Exportación a TensorFlow Lite
11.4. Implementación del modelo dinámico
&nbsp;&nbsp;&nbsp;&nbsp;11.4.1. Arquitectura de la red LSTM
&nbsp;&nbsp;&nbsp;&nbsp;11.4.2. Manejo de secuencias temporales
&nbsp;&nbsp;&nbsp;&nbsp;11.4.3. Exportación a TensorFlow Lite
11.5. Pipeline de detección en tiempo real
&nbsp;&nbsp;&nbsp;&nbsp;11.5.1. Captura de frames y extracción de landmarks
&nbsp;&nbsp;&nbsp;&nbsp;11.5.2. Filtros de persistencia temporal y cooldown
&nbsp;&nbsp;&nbsp;&nbsp;11.5.3. Lógica de transición entre modelo estático y dinámico
11.6. Construcción del diccionario visual
&nbsp;&nbsp;&nbsp;&nbsp;11.6.1. Selección y grabación de señas de referencia
&nbsp;&nbsp;&nbsp;&nbsp;11.6.2. Organización y empaquetado en la aplicación
11.7. Desarrollo de la aplicación móvil
&nbsp;&nbsp;&nbsp;&nbsp;11.7.1. Módulo de reconocimiento (niño firma a cámara)
&nbsp;&nbsp;&nbsp;&nbsp;11.7.2. Módulo de representación visual (oyente escribe)
&nbsp;&nbsp;&nbsp;&nbsp;11.7.3. Mecanismo de fallback con deletreo
&nbsp;&nbsp;&nbsp;&nbsp;11.7.4. Diseño de interfaz pensada para niños

---

## CAPÍTULO 12. RESULTADOS Y ANÁLISIS

12.1. Resultados del modelo estático
&nbsp;&nbsp;&nbsp;&nbsp;12.1.1. Precisión global y por clase
&nbsp;&nbsp;&nbsp;&nbsp;12.1.2. Matriz de confusión y análisis de errores
12.2. Resultados del modelo dinámico
&nbsp;&nbsp;&nbsp;&nbsp;12.2.1. Precisión global y por clase
&nbsp;&nbsp;&nbsp;&nbsp;12.2.2. Matriz de confusión y análisis de errores
12.3. Evaluación del sistema en tiempo real
&nbsp;&nbsp;&nbsp;&nbsp;12.3.1. Tiempo de respuesta y latencia
&nbsp;&nbsp;&nbsp;&nbsp;12.3.2. Consumo de recursos en el dispositivo
&nbsp;&nbsp;&nbsp;&nbsp;12.3.3. Robustez ante condiciones variables
12.4. Evaluación con usuarios
&nbsp;&nbsp;&nbsp;&nbsp;12.4.1. Perfil de los participantes
&nbsp;&nbsp;&nbsp;&nbsp;12.4.2. Resultados de usabilidad
&nbsp;&nbsp;&nbsp;&nbsp;12.4.3. Resultados de claridad y comprensión
&nbsp;&nbsp;&nbsp;&nbsp;12.4.4. Análisis de retroalimentación cualitativa
12.5. Discusión y análisis comparativo con trabajos previos

---

## CAPÍTULO 13. CONCLUSIONES Y RECOMENDACIONES

13.1. Conclusiones
13.2. Cumplimiento de los objetivos planteados
13.3. Limitaciones identificadas durante el desarrollo
13.4. Recomendaciones
13.5. Trabajo futuro

---

## SECCIONES FINALES

Referencias bibliográficas (formato IEEE)
Anexos
&nbsp;&nbsp;&nbsp;&nbsp;Anexo A. Glosario de términos
&nbsp;&nbsp;&nbsp;&nbsp;Anexo B. Vocabulario completo del sistema
&nbsp;&nbsp;&nbsp;&nbsp;Anexo C. Instrumentos de evaluación con usuarios
&nbsp;&nbsp;&nbsp;&nbsp;Anexo D. Manual de usuario de la aplicación
&nbsp;&nbsp;&nbsp;&nbsp;Anexo E. Fragmentos de código relevantes
&nbsp;&nbsp;&nbsp;&nbsp;Anexo F. Material fotográfico del proceso
&nbsp;&nbsp;&nbsp;&nbsp;Anexo G. Documentos de consentimiento informado
