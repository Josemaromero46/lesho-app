# Tesis LESHO — código fuente en LaTeX

Aplicación móvil para la comunicación entre niños sordos y personas oyentes mediante
reconocimiento del Lenguaje de Señas Hondureño.

Autor: José Manuel Romero Martínez
Universidad Nacional Autónoma de Honduras, Campus Comayagua
Ingeniería en Sistemas

El documento se compone a partir de `main.tex`. El resultado es `main.pdf`, de 147 páginas.

## Cómo compilar

```
latexmk -xelatex main.tex
```

El archivo `.latexmkrc` incluido ya configura el motor, la bibliografía y los glosarios,
de modo que ese único comando ejecuta todas las pasadas necesarias. Para borrar los
archivos intermedios, `latexmk -c`.

## Requisitos

| Requisito | Para qué |
|---|---|
| Distribución de TeX con **XeLaTeX** (MiKTeX o TeX Live) | Motor de composición. El documento usa fuentes OpenType, así que no compila con pdfLaTeX |
| **Biber** | Bibliografía, en formato IEEE mediante `biblatex` |
| **makeglossaries** | Glosario, acrónimos y símbolos |
| **Python con Pygments** (`pip install Pygments`) | El paquete `minted` colorea los fragmentos de código del anexo. Sin Pygments la compilación falla |

`minted` necesita además la opción `-shell-escape`, que el `.latexmkrc` ya activa. Si se
compila a mano sin latexmk, hay que agregarla: `xelatex -shell-escape main.tex`.

## Estructura

| Carpeta | Contenido |
|---|---|
| `Chapters/` | Los capítulos, numerados igual que en el documento |
| `Matter/` | Páginas preliminares y finales: portada, dedicatoria, resumen, glosarios |
| `Figures/` | Figuras, organizadas por capítulo |
| `Configurations/` | Estilos de la plantilla: márgenes, fuentes, encabezados, tablas |
| `Bibliography/` | `Bibliography.bib`, referencias en formato IEEE |
| `Metadata/` | Título, autor, asesor y demás datos del documento |
| `ThesisUNAH.cls` | Clase del documento |

Las figuras propias del trabajo están en `Figures/Teoria/`, `Figures/Implementacion/`,
`Figures/Resultados/` y `Figures/Anexos/`. Las que se construyeron para este documento se
conservan también en formato SVG editable junto al PDF que se inserta.

## Opciones del documento

Se definen en las opciones de clase, en `main.tex`. Las relevantes son el idioma
(`spanish`), el medio (`screen` o `paper`, que controla las páginas en blanco de la
edición impresa) y `aiack`, que activa o desactiva la página de declaración sobre el uso
de inteligencia artificial.

## Créditos de la plantilla

La plantilla base es obra de José Areia (jose.apareia@gmail.com), adaptada para la UNAH.
