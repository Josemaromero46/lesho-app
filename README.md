# Instrucciones para Compilar y Visualizar el Documento LaTeX

Este repositorio contiene el código fuente de un documento escrito en **LaTeX**. A continuación se describen los pasos necesarios para compilarlo desde la consola y visualizar correctamente el PDF resultante.

---

## 1. Requisitos Previos

Antes de compilar el documento, asegúrate de tener instalado lo siguiente:

> MiKTeX

### Fuentes tipográficas necesarias

- Crimson Pro
- Lato
- EB Garamond
- TeX Gyre Pagella

### Motor Recomendado

- xelatex
- biber (para la bibliografía)
- makeglossaries (para la generación de índices)

## 2. Compilar el documento

Ubícate en la carpeta donde está el archivo principal

### Compilación básica

```bash
xelatex -shell-escape .\main.tex
makeglossaries main
biber main
xelatex -shell-escape .\main.tex
xelatex -shell-escape .\main.tex
```

## 3. Ubicación del documento generado

El archivo PDF generado se guardará en la misma carpeta del archivo .tex, por ejemplo:

```bash
main.tex
```
