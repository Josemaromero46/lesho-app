"""
Genera las figuras del capitulo de Resultados y Analisis de la tesis.

Reentrena de forma HONESTA (division por toma: ninguna toma cruza de entrenamiento
a prueba) los dos modelos, los evalua en la particion de prueba y produce las
figuras en PDF vectorial con una paleta formal (azules), listas para incluir con
\\includegraphics en la plantilla LaTeX.

Las figuras que SI tienen datos reales se marcan como tales. Las que aun no tienen
datos (costo de recursos del dispositivo, usabilidad) se generan con DATOS DE
EJEMPLO, rotulados de forma visible, para reemplazar cuando existan las mediciones.

Uso (desde la carpeta training/):

    ../.venv/Scripts/python.exe evaluation/generar_resultados.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

import config  # noqa: E402
import tensorflow as tf  # noqa: E402
from comun.definiciones import (  # noqa: E402
    CLASES_DINAMICAS, NUM_CLASES_A, NUM_CLASES_B,
)
from models.arquitectura_a import compilar_modelo_a, construir_modelo_a  # noqa: E402
from models.arquitectura_b import compilar_modelo_b, construir_modelo_b  # noqa: E402
from models.datos import cargar_dataset, pesos_de_clase  # noqa: E402

# ---------------------------------------------------------------------------
# Paleta formal (azules como color principal; acentos sobrios; nada de gris puro)
# ---------------------------------------------------------------------------
AZUL_OSCURO = "#1F4E79"
AZUL = "#2E75B6"
AZUL_CLARO = "#9DC3E6"
VERDE = "#2C7C6C"
AMBAR = "#C08A2E"
ROJO = "#B4433B"
TEXTO = "#22303C"
CMAP_AZUL = LinearSegmentedColormap.from_list(
    "azules_tesis", ["#F3F8FD", "#9DC3E6", "#2E75B6", "#1F4E79"]
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": TEXTO,
    "axes.labelcolor": TEXTO,
    "text.color": TEXTO,
    "xtick.color": TEXTO,
    "ytick.color": TEXTO,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})

SALIDA = (Path(__file__).resolve().parents[2]
          / "docs" / "Plantilla-Tesis-IS-main" / "Figures" / "Resultados")
SALIDA.mkdir(parents=True, exist_ok=True)

# Grupos semanticos de las 50 senas dinamicas (para el grafico de arana).
GRUPOS_SENAS = {
    "Cortesía": ["HOLA", "ADIOS", "GRACIAS", "POR_FAVOR", "PERDON"],
    "Respuestas": ["SI", "NO", "BIEN"],
    "Necesidades": ["AGUA", "COMER", "BAÑO", "DORMIR", "AYUDA", "DOLOR",
                    "HAMBRE", "ENFERMO"],
    "Familia": ["MAMA", "PAPA", "FAMILIA", "AMIGO", "NIÑO", "MAESTRO", "DOCTOR"],
    "Verbos": ["LLAMAR", "NECESITAR", "TENER", "DAR", "PERDER", "VENIR", "JUGAR",
               "ESTUDIAR", "APRENDER", "COMPRAR", "ESPERAR"],
    "Lugares": ["CASA", "ESCUELA", "HOSPITAL", "TIENDA", "CALLE", "TRABAJO"],
    "Emociones": ["FELIZ", "TRISTE", "ENOJADO", "CANSADO", "MIEDO"],
    "Tiempo": ["HOY", "MAÑANA", "AYER", "DIA", "NOCHE"],
}


def _guardar(fig, nombre):
    ruta = SALIDA / nombre
    fig.savefig(ruta, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  figura: {ruta.name}")


def _rotulo_ejemplo(ax):
    """Marca visible de que la figura usa datos de ejemplo, no reales."""
    ax.text(0.5, 0.5, "DATOS DE EJEMPLO", transform=ax.transAxes,
            fontsize=26, color="#C08A2E", alpha=0.18, rotation=25,
            ha="center", va="center", fontweight="bold", zorder=0)


def cargar_etiquetas(ruta):
    return [l.strip() for l in open(ruta, encoding="utf-8") if l.strip()]


# ---------------------------------------------------------------------------
# Entrenamiento + evaluacion honesta de un modelo
# ---------------------------------------------------------------------------
def evaluar_modelo(nombre, ruta_dataset, num_clases, construir, compilar,
                   epocas, paciencia):
    print(f"\n== {nombre}: cargando y entrenando (division por toma) ==")
    X, y, personas, grupos = cargar_dataset(ruta_dataset)
    # Division estratificada por clase y agrupada por toma: ninguna toma cruza de
    # particion (sin fuga) y toda clase con datos aparece en las tres particiones,
    # asi la matriz de confusion incluye todas las clases evaluables.
    sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True,
                                random_state=config.SEMILLA)
    folds = [te for _, te in sgkf.split(X, y, groups=grupos)]
    idx_te, idx_val = folds[0], folds[1]
    idx_tr = np.setdiff1d(np.arange(len(y)), np.concatenate([idx_te, idx_val]))
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_val, y_val = X[idx_val], y[idx_val]
    X_te, y_te = X[idx_te], y[idx_te]
    pesos = pesos_de_clase(y_tr, num_clases)

    modelo = compilar(construir())
    parada = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=paciencia, restore_best_weights=True)
    hist = modelo.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                      epochs=epocas, batch_size=32, class_weight=pesos,
                      callbacks=[parada], verbose=2)

    prob = modelo.predict(X_te, verbose=0)
    pred = np.argmax(prob, axis=1)
    exactitud = float((pred == y_te).mean())
    f1_macro = float(f1_score(y_te, pred, average="macro", zero_division=0))
    print(f"  exactitud prueba: {exactitud:.4f}   F1 macro: {f1_macro:.4f}")

    return {
        "nombre": nombre, "modelo": modelo, "historia": hist.history,
        "y_true": y_te, "y_pred": pred, "num_clases": num_clases,
        "exactitud": exactitud, "f1_macro": f1_macro,
        "n_muestras": int(len(y)), "n_test": int(len(y_te)),
        "personas": sorted({str(p) for p in personas}),
        "params": int(modelo.count_params()),
    }


# ---------------------------------------------------------------------------
# Figuras REALES
# ---------------------------------------------------------------------------
def _suavizar(v, w=3):
    """Promedio movil centrado (ventana w) para leer la tendencia sin perder los
    valores reales, que se dibujan tenues debajo."""
    v = np.asarray(v, dtype=float)
    salida = np.empty_like(v)
    for i in range(len(v)):
        lo, hi = max(0, i - w // 2), min(len(v), i + w // 2 + 1)
        salida[i] = v[lo:hi].mean()
    return salida


def fig_curvas(res, nombre_archivo, titulo):
    h = res["historia"]
    ep = np.arange(1, len(h["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))

    def panel(ax, tr, va, titulo_ax, ylabel, cap=False):
        # Valor por epoca (tenue) + promedio movil (marcado).
        s_tr, s_va = _suavizar(tr), _suavizar(va)
        ax.plot(ep, tr, color=AZUL, alpha=0.18, lw=1)
        ax.plot(ep, va, color=AMBAR, alpha=0.18, lw=1)
        ax.plot(ep, s_tr, color=AZUL, lw=2.2, label="Entrenamiento")
        ax.plot(ep, s_va, color=AMBAR, lw=2.2, label="Validación")
        # En la perdida, limitar el eje al rango informativo para que un pico
        # transitorio de validacion no aplaste el resto de la curva.
        if cap:
            ax.set_ylim(0, 1.2 * max(float(s_tr.max()), float(s_va.max())))
        ax.set_title(titulo_ax); ax.set_xlabel("Época"); ax.set_ylabel(ylabel)
        ax.legend(frameon=False); ax.grid(alpha=0.25)

    panel(ax1, h["accuracy"], h["val_accuracy"], "Exactitud", "Exactitud")
    panel(ax2, h["loss"], h["val_loss"], "Pérdida", "Pérdida", cap=True)
    fig.suptitle(titulo, y=1.02, fontsize=12, fontweight="bold")
    _guardar(fig, nombre_archivo)


def fig_matriz_confusion(res, etiquetas, nombre_archivo, titulo,
                         anotar_umbral=0.01, fuente=6):
    cm = confusion_matrix(res["y_true"], res["y_pred"],
                          labels=list(range(res["num_clases"])))
    presentes = np.where(cm.sum(axis=1) > 0)[0]
    cm = cm[np.ix_(presentes, presentes)]
    nombres = [etiquetas[i] for i in presentes]
    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    lado = max(5.5, 0.32 * len(nombres))
    fig, ax = plt.subplots(figsize=(lado, lado * 0.94))
    im = ax.imshow(cmn, cmap=CMAP_AZUL, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(nombres))); ax.set_yticks(range(len(nombres)))
    ax.set_xticklabels(nombres, rotation=90, fontsize=7)
    ax.set_yticklabels(nombres, fontsize=7)
    ax.set_xlabel("Clase predicha"); ax.set_ylabel("Clase verdadera")
    ax.set_title(titulo)
    # Numero (porcentaje por fila) en cada celda con valor sobre el umbral. En la
    # matriz grande (Modelo B) el umbral deja el numero solo donde hay senal.
    for i in range(len(nombres)):
        for j in range(len(nombres)):
            v = cmn[i, j]
            if v >= anotar_umbral:
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                        fontsize=fuente, color="white" if v > 0.5 else TEXTO)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Proporción por fila (recuperación)")
    _guardar(fig, nombre_archivo)


def fig_f1_por_clase(res, etiquetas, nombre_archivo, titulo):
    f1 = f1_score(res["y_true"], res["y_pred"],
                  labels=list(range(res["num_clases"])), average=None,
                  zero_division=0)
    presentes = np.unique(res["y_true"])
    pares = sorted([(etiquetas[i], f1[i]) for i in presentes], key=lambda t: t[1])
    nombres = [p[0] for p in pares]
    vals = [p[1] for p in pares]
    colores = [ROJO if v < 0.80 else (AMBAR if v < 0.90 else AZUL) for v in vals]

    alto = max(3.2, 0.22 * len(nombres))
    fig, ax = plt.subplots(figsize=(7.2, alto))
    ax.barh(range(len(nombres)), vals, color=colores)
    ax.set_yticks(range(len(nombres))); ax.set_yticklabels(nombres, fontsize=7)
    ax.set_xlim(0, 1.0); ax.set_xlabel("Puntaje F1")
    ax.set_title(titulo); ax.grid(axis="x", alpha=0.25)
    leyenda = [Patch(facecolor=AZUL, label="F1 ≥ 0.90"),
               Patch(facecolor=AMBAR, label="0.80 ≤ F1 < 0.90"),
               Patch(facecolor=ROJO, label="F1 < 0.80")]
    ax.legend(handles=leyenda, frameon=False, fontsize=8, loc="lower right")
    _guardar(fig, nombre_archivo)


def _radar_ejes(ax, categorias):
    n = len(categorias)
    angs = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angs += angs[:1]
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angs[:-1]); ax.set_xticklabels(categorias, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    return angs


def fig_radar_grupos(res, etiquetas, nombre_archivo):
    nombre_a_idx = {n: i for i, n in enumerate(etiquetas)}
    f1 = f1_score(res["y_true"], res["y_pred"],
                  labels=list(range(res["num_clases"])), average=None,
                  zero_division=0)
    presentes = set(np.unique(res["y_true"]).tolist())
    categorias, valores = [], []
    for grupo, senas in GRUPOS_SENAS.items():
        idxs = [nombre_a_idx[s] for s in senas
                if s in nombre_a_idx and nombre_a_idx[s] in presentes]
        if not idxs:
            continue
        categorias.append(grupo)
        valores.append(float(np.mean([f1[i] for i in idxs])))
    valores_c = valores + valores[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    angs = _radar_ejes(ax, categorias)
    ax.plot(angs, valores_c, color=AZUL_OSCURO, lw=2)
    ax.fill(angs, valores_c, color=AZUL, alpha=0.30)
    ax.set_title("F1 promedio por grupo semántico (Modelo B)",
                 y=1.10, fontsize=12, fontweight="bold")
    _guardar(fig, nombre_archivo)


def fig_radar_comparacion(res_a, res_b, nombre_archivo):
    tam_a = config.RUTA_MODELO_A_TFLITE.stat().st_size / 1e6
    tam_b = config.RUTA_MODELO_B_TFLITE.stat().st_size / 1e6
    categorias = ["Exactitud", "F1 macro", "Ligereza\n(<2 MB)",
                  "Cobertura\n(clases)", "Simplicidad\n(parámetros)"]

    def perfil(res, tam, clases):
        return [
            res["exactitud"],
            res["f1_macro"],
            float(np.clip(1 - tam / 2.0, 0, 1)),
            float(np.clip(clases / 50.0, 0, 1)),
            float(np.clip(1 - res["params"] / 100000.0, 0, 1)),
        ]

    va = perfil(res_a, tam_a, res_a["num_clases"])
    vb = perfil(res_b, tam_b, res_b["num_clases"])
    va += va[:1]; vb += vb[:1]

    fig, ax = plt.subplots(figsize=(6.2, 6.2), subplot_kw=dict(polar=True))
    angs = _radar_ejes(ax, categorias)
    ax.plot(angs, va, color=AZUL, lw=2, label="Modelo A (alfabeto)")
    ax.fill(angs, va, color=AZUL, alpha=0.22)
    ax.plot(angs, vb, color=AMBAR, lw=2, label="Modelo B (señas)")
    ax.fill(angs, vb, color=AMBAR, alpha=0.18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), frameon=False)
    ax.set_title("Comparación de los dos modelos", y=1.10,
                 fontsize=12, fontweight="bold")
    _guardar(fig, nombre_archivo)


def fig_velas_latencia(nombre_archivo):
    """Latencia de inferencia (ms) por muestra, medida en la computadora de
    desarrollo. Es un dato REAL de referencia (no del telefono)."""
    lat = {}
    for etq, ruta, forma in [
        ("Modelo A", config.RUTA_MODELO_A_TFLITE, (1, config.TAMANO_VENTANA_A, 126)),
        ("Modelo B", config.RUTA_MODELO_B_TFLITE, (1, config.LONGITUD_FIJA_SECUENCIA, 152)),
    ]:
        try:
            interp = tf.lite.Interpreter(model_path=str(ruta))
            interp.allocate_tensors()
            ent = interp.get_input_details()[0]
            sal = interp.get_output_details()[0]
            tiempos = []
            for _ in range(200):
                x = np.random.randn(*forma).astype(np.float32)
                interp.set_tensor(ent["index"], x)
                t0 = time.perf_counter()
                interp.invoke()
                interp.get_tensor(sal["index"])
                tiempos.append((time.perf_counter() - t0) * 1000)
            lat[etq] = np.array(tiempos[20:])  # descarta calentamiento
        except Exception as e:  # noqa: BLE001
            print(f"  (latencia {etq} no medible: {e})")

    fig, ax = plt.subplots(figsize=(5.6, 4))
    for i, (etq, t) in enumerate(lat.items()):
        q1, med, q3 = np.percentile(t, [25, 50, 75])
        lo, hi = t.min(), t.max()
        ax.plot([i, i], [lo, hi], color=AZUL_OSCURO, lw=1.2, zorder=1)
        ax.add_patch(plt.Rectangle((i - 0.2, q1), 0.4, max(q3 - q1, 1e-3),
                     facecolor=AZUL, edgecolor=AZUL_OSCURO, zorder=2))
        ax.plot([i - 0.2, i + 0.2], [med, med], color="white", lw=2, zorder=3)
    ax.set_xticks(range(len(lat))); ax.set_xticklabels(list(lat.keys()))
    ax.set_ylabel("Latencia por inferencia (ms)")
    ax.set_title("Distribución de la latencia de inferencia\n(computadora de "
                 "desarrollo, referencial)")
    ax.grid(axis="y", alpha=0.25)
    _guardar(fig, nombre_archivo)


# ---------------------------------------------------------------------------
# Figuras de EJEMPLO (rotuladas; reemplazar con mediciones reales)
# ---------------------------------------------------------------------------
def fig_recursos_ejemplo(nombre_archivo):
    metr = ["CPU (%)", "RAM (MB)", "Batería\n(%/10 min)", "FPS efectivo"]
    valores = [46, 320, 4, 8]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    _rotulo_ejemplo(ax)
    ax.bar(metr, valores, color=[AZUL, AZUL_OSCURO, AMBAR, VERDE], zorder=3)
    for i, v in enumerate(valores):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_title("Costo de recursos en el dispositivo (ejemplo)")
    ax.grid(axis="y", alpha=0.25)
    _guardar(fig, nombre_archivo)


def fig_correlacion_ejemplo(nombre_archivo):
    variables = ["FPS", "Latencia", "CPU", "RAM", "Batería"]
    rng = np.random.default_rng(3)
    datos = rng.normal(size=(60, len(variables)))
    datos[:, 1] = -datos[:, 0] + 0.3 * rng.normal(size=60)   # fps vs latencia
    datos[:, 2] = 0.7 * datos[:, 1] + 0.4 * rng.normal(size=60)
    datos[:, 4] = 0.6 * datos[:, 2] + 0.5 * rng.normal(size=60)
    corr = np.corrcoef(datos, rowvar=False)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(corr, cmap=CMAP_AZUL, vmin=-1, vmax=1)
    ax.set_xticks(range(len(variables))); ax.set_yticks(range(len(variables)))
    ax.set_xticklabels(variables, rotation=45, ha="right")
    ax.set_yticklabels(variables)
    for i in range(len(variables)):
        for j in range(len(variables)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(corr[i, j]) > 0.5 else TEXTO)
    _rotulo_ejemplo(ax)
    ax.set_title("Correlación entre métricas de recurso (ejemplo)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _guardar(fig, nombre_archivo)


def fig_usabilidad_ejemplo(nombre_archivo):
    items = ["Fácil de usar", "Aprendizaje\nrápido", "Botones\nclaros",
             "Respuesta\nentendible", "Volvería a\nusarla"]
    muy_acuerdo = [7, 6, 8, 5, 7]
    acuerdo = [3, 3, 2, 3, 2]
    neutro = [0, 1, 0, 1, 1]
    desacuerdo = [0, 0, 0, 1, 0]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    _rotulo_ejemplo(ax)
    izq = np.zeros(len(items))
    for datos, color, etq in [
        (muy_acuerdo, AZUL_OSCURO, "Muy de acuerdo"),
        (acuerdo, AZUL, "De acuerdo"),
        (neutro, AZUL_CLARO, "Neutral"),
        (desacuerdo, AMBAR, "En desacuerdo"),
    ]:
        ax.barh(items, datos, left=izq, color=color, label=etq, zorder=3)
        izq = izq + np.array(datos)
    ax.set_xlabel("Cantidad de participantes")
    ax.set_title("Usabilidad percibida por ítem (ejemplo)")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    _guardar(fig, nombre_archivo)


def main():
    # Semilla global para que el reentrenamiento (y por lo tanto las figuras y las
    # metricas de las tablas) sea reproducible entre corridas.
    tf.keras.utils.set_random_seed(config.SEMILLA)

    # Tasa de aprendizaje mas baja que la de produccion: converge un poco mas
    # lento pero da curvas de validacion mas estables para la figura.
    res_a = evaluar_modelo(
        "Modelo A", config.RUTA_DATASET_A, NUM_CLASES_A,
        lambda: construir_modelo_a(),
        lambda m: compilar_modelo_a(m, tasa_aprendizaje=5e-4),
        epocas=50, paciencia=10)
    res_b = evaluar_modelo(
        "Modelo B", config.RUTA_DATASET_B, NUM_CLASES_B,
        lambda: construir_modelo_b(longitud=config.LONGITUD_FIJA_SECUENCIA),
        lambda m: compilar_modelo_b(m, tasa_aprendizaje=5e-4),
        epocas=70, paciencia=12)

    etq_a = cargar_etiquetas(config.RUTA_ETIQUETAS_A)
    etq_b = list(CLASES_DINAMICAS)  # vocabulario vigente (incluye LLAMAR)

    print("\n== Generando figuras ==")
    fig_curvas(res_a, "curvas_entrenamiento_a.pdf",
               "Curvas de entrenamiento del Modelo A")
    fig_curvas(res_b, "curvas_entrenamiento_b.pdf",
               "Curvas de entrenamiento del Modelo B")
    fig_matriz_confusion(res_a, etq_a, "matriz_confusion_a.pdf",
                         "Matriz de confusión del Modelo A",
                         anotar_umbral=0.01, fuente=6)
    fig_matriz_confusion(res_b, etq_b, "matriz_confusion_b.pdf",
                         "Matriz de confusión del Modelo B",
                         anotar_umbral=0.02, fuente=4.5)
    fig_f1_por_clase(res_a, etq_a, "f1_por_clase_a.pdf",
                     "Puntaje F1 por clase (Modelo A)")
    fig_f1_por_clase(res_b, etq_b, "f1_por_clase_b.pdf",
                     "Puntaje F1 por clase (Modelo B)")
    fig_radar_grupos(res_b, etq_b, "radar_grupos_b.pdf")
    fig_radar_comparacion(res_a, res_b, "radar_comparacion.pdf")
    fig_velas_latencia("velas_latencia.pdf")
    fig_recursos_ejemplo("recursos_dispositivo.pdf")
    fig_correlacion_ejemplo("correlacion_recursos.pdf")
    fig_usabilidad_ejemplo("usabilidad_likert.pdf")

    resumen = {
        "modelo_a": {k: res_a[k] for k in
                     ["exactitud", "f1_macro", "num_clases", "n_muestras",
                      "n_test", "personas", "params"]},
        "modelo_b": {k: res_b[k] for k in
                     ["exactitud", "f1_macro", "num_clases", "n_muestras",
                      "n_test", "personas", "params"]},
        "tam_a_kb": round(config.RUTA_MODELO_A_TFLITE.stat().st_size / 1024, 1),
        "tam_b_kb": round(config.RUTA_MODELO_B_TFLITE.stat().st_size / 1024, 1),
    }
    (SALIDA / "metricas_resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n== Resumen de métricas reales ==")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
