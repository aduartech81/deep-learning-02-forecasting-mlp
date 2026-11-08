# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 9. Síntesis, conclusiones y reproducibilidad
#
# Este capítulo reúne los resultados de los capítulos 1 a 8 en una sola vista. Las cifras se leen de `results/`,
# así que siempre corresponden a la última ejecución.

# %%
import importlib.metadata as md
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))
import pandas as pd
from IPython.display import Markdown, display

from fcst.data import read_manifest
from fcst.nbutils import setup

cfg, RES, FIG = setup()
APP = cfg.path("app_dir")
man = read_manifest(cfg)
part = pd.read_csv(RES / "particion.csv")
best = pd.read_csv(RES / "mejor_arquitectura.csv")
cons = pd.read_csv(RES / "consolidado_lag_h.csv")
vs = pd.read_csv(RES / "mlp_vs_referencias.csv")
dep = pd.read_csv(RES / "modelos_desplegados.csv")
bds = pd.read_csv(RES / "bds_validacion.csv")
vh = pd.read_csv(RES / "volatilidad_holdout.csv")
hold16 = pd.read_csv(RES / "holdout_16_configuraciones.csv")

# %% [markdown]
# ## 9.1 Resultados clave

# %%
strat = cons.groupby("estrategia")[["MAPE", "RMSE"]].mean()
fc_mape = strat.loc["forwardChaining", "MAPE"]
naive_fc = vs.query("estrategia == 'forwardChaining' and modelo == 'naive'").MAPE.mean()
mlp_fc = vs.query("estrategia == 'forwardChaining' and modelo == 'mlp'").MAPE.mean()
deg = cons.query("estrategia == 'forwardChaining'").groupby("H").RMSE.mean()
gana_hold = int((dep["RMSE hold-out"] < dep["RMSE ingenuo hold-out"]).sum())
key = pd.DataFrame([
    ("Datos", f"{man.get('filas', '—')} velas diarias {cfg['data']['symbol']} ({man.get('primera_fecha', '')} → {man.get('ultima_fecha', '')})"),
    ("Partición", f"CV: {part.iloc[0]['días']} días · hold-out: {part.iloc[1]['días']} días (desde {part.iloc[1]['desde']})"),
    ("Búsqueda de arquitectura", f"{len(best)} configuraciones lag × H; capas de la red elegida: "
                                 + ", ".join(f"{k} capa(s): {v}" for k, v in best.capas.value_counts().sort_index().items())),
    ("MAPE medio de validación por estrategia", ", ".join(f"{s}: {v:.2f} %" for s, v in strat.MAPE.items())),
    ("MLP vs ingenuo (validación, Forward Chaining)", f"MLP {mlp_fc:.2f} % vs ingenuo {naive_fc:.2f} % (MAPE medio)"),
    ("Degradación con el horizonte (RMSE FC)", " → ".join(f"H{h}: {v:,.0f}" for h, v in deg.items())),
    ("Hold-out (modelos desplegados)", f"el MLP le gana al ingenuo en {gana_hold} de {len(dep)} horizontes"),
    ("Residuos (BDS h = 1, validación)", f"{100 * (bds.folds_iid.sum() / bds.folds.sum()):.0f} % de los folds no rechazan iid"),
    ("Volatilidad (hold-out)", f"el MLP le gana al ingenuo en {int(vh.MLP_mejor_que_ingenuo.sum())} de 16 configuraciones"),
], columns=["aspecto", "resultado"])
key.to_csv(RES / "resultados_clave.csv", index=False)
key.style.hide(axis="index")

# %% [markdown]
# ## 9.2 Respuestas a lo que pide la guía (con los resultados de esta ejecución)
#
# **1. El precio se comporta casi como un paseo aleatorio.** En nivel, el cierre no es estacionario (ADF p = 0,72);
# los retornos sí, pero con colas muy pesadas (curtosis en exceso 18,2) y volatilidad agrupada (Ljung-Box sobre r²
# p = 7,6·10⁻¹², ARCH-LM p = 2,4·10⁻⁹). Con esa estructura, la mejor predicción del cierre de mañana usando solo el
# pasado es, casi siempre, el cierre de hoy.
#
# **2. El MLP no le gana al pronóstico ingenuo en el precio.** En validación (Forward Chaining) el MAPE del MLP es
# 7,2 / 10,5 / 13,5 / 12,5 % para H = 7 / 14 / 21 / 28, frente a 3,7 / 5,1 / 6,3 / 7,2 % del ingenuo; incluso el
# MLPRegressor de referencia (5,3 → 12,5 %) queda mejor que el MLP de Keras en los horizontes cortos. En el hold-out
# (14 dic 2025 – 31 oct 2026) los cuatro modelos desplegados pierden contra el ingenuo (MAPE 6,0 / 5,1 / 6,2 / 7,4 %
# frente a 3,1 / 4,4 / 5,6 / 6,7 %). Es un resultado negativo, pero es el resultado correcto: sin información externa,
# una red que predice el **nivel** reproduce el último valor con ruido y, cuando el precio entra en un régimen nuevo,
# arrastra sesgo (residuo medio de h = 1 en el hold-out: −4 209 USD para H = 7 y +5 703 USD para H = 28).
#
# **3. ¿Qué estrategia de validación es más realista y cuál más optimista?** Forward Chaining es la única que entrena
# siempre con el pasado (verificado en sus 80 folds) y es la estrategia final. K-Fold valida exactamente las mismas
# ventanas que Forward Chaining pero entrena además con el futuro (~2 548 ventanas frente a ~1 998): en los bloques con
# cambio de régimen eso le da un error menor (para lag = 7 y H = 7, RMSE medio de 3 642 frente a 5 406 USD), es decir, es
# **optimista**. Su último bloque, que no tiene futuro disponible, coincide exactamente con el de Forward Chaining, lo que
# sirve de control del diseño. Group K-Fold resultó la peor estimación: usa pocas ventanas de entrenamiento (~374),
# reparte la validación sobre toda la serie (incluidos los años de precios bajos, que inflan el MAPE a 33 %) y en 30 de
# sus 80 folds no se pudo purgar el solape de objetivos.
#
# **4. ¿Cambia la arquitectura óptima con el horizonte?** La red elegida tiene 2 capas en 7 configuraciones y 3 en 9;
# las ganadoras son anchas (256 neuronas en al menos una capa) y casi siempre con Dropout en la última capa oculta. Al
# alargar el horizonte crece el error de validación de la mejor red (best_val_rmse de ≈ 0,24 a ≈ 0,50 en unidades z) y
# desaparecen los candidatos balanceados: en 5 de las 16 configuraciones, todas con H ≥ 21, ninguno cumplió
# gap_ratio < 0,15 y se aplicó el criterio alterno documentado. Advertencia metodológica: Keras mide el RMSE de
# entrenamiento con el Dropout activo, así que más Dropout infla el gap_ratio aunque no haya más sobreajuste. Las curvas de aprendizaje (capítulo 3.4) muestran el porqué: el RMSE de validación se dispara tras pocas épocas mientras el de entrenamiento sigue bajando, porque el tramo de validación (2024–2025) tiene precios por encima de todo lo visto en entrenamiento y una red con ReLU que predice el nivel no extrapola; de ahí que las épocas óptimas sean tempranas (3 a 12).
#
# **5. ¿Cuánto se degrada con el horizonte?** El MAPE de validación casi se duplica de H = 7 a H = 21; de 21 a 28 ya
# no empeora (13,5 → 12,5 %), porque a esos plazos todas las redes convergen hacia una tendencia suave.
#
# **6. ¿Los residuos son ruido blanco?** No. El BDS rechaza independencia en todos los folds de Forward Chaining y de
# K-Fold y en el hold-out de los cuatro modelos; solo 6 de 80 folds de Group K-Fold no la rechazan. Queda dependencia sin
# modelar, coherente con la volatilidad agrupada del capítulo 1.
#
# **7. La volatilidad sí tiene estructura aprovechable.** Con la misma arquitectura, el MLP supera al ingenuo en RMSE
# del hold-out en 12 de 16 configuraciones (todas las de H ≥ 14 con lags 7 y 14), aunque su MAPE es mayor: reduce los
# errores grandes en los episodios de alta volatilidad a cambio de errores relativos mayores en los periodos tranquilos.
#
# **Qué haría distinto en una siguiente versión:** modelar retornos o diferencias logarítmicas en lugar del nivel,
# añadir variables exógenas (volumen, otros activos, tasas), y reportar siempre la mejora relativa frente al ingenuo como
# métrica principal.

# %% [markdown]
# ## 9.3 Limitaciones
#
# * **Información:** el modelo usa solo el pasado del propio precio, como exige la guía; no ve volumen, noticias,
#   tasas de interés ni otros activos.
# * **No estacionariedad:** se modela el nivel del precio con escaladores ajustados en el pasado; cuando el precio
#   alcanza máximos nuevos, la red debe extrapolar, algo que los MLP con ReLU hacen mal.
# * **Un solo hold-out:** el tramo final (~10 %) es un único periodo de mercado; su resultado no generaliza a
#   todos los regímenes.
# * **Adaptaciones a tsxv:** la agrupación de folds en bloques y la purga de objetivos solapados (capítulo 2) son
#   decisiones metodológicas propias, documentadas y probadas en `tests/test_cv.py`.
# * **Volatilidad:** se reutilizó la arquitectura del precio; no se repitió la búsqueda para ese objetivo.
# * **Uso:** ejercicio académico; no constituye una recomendación de inversión.

# %% [markdown]
# ## 9.4 Reproducibilidad

# %%
def version(dist_names):
    for d in dist_names:
        try:
            return md.version(d)
        except md.PackageNotFoundError:
            continue
    return "—"


pk = {"tensorflow": ["tensorflow", "tensorflow-cpu"], "keras": ["keras"], "scikit-learn": ["scikit-learn"],
      "timeseries-cv": ["timeseries-cv"], "statsmodels": ["statsmodels"], "pandas": ["pandas"], "numpy": ["numpy"],
      "fastapi": ["fastapi"]}
env = pd.DataFrame([("python", platform.python_version())] + [(k, version(v)) for k, v in pk.items()],
                   columns=["paquete", "versión"])
env.to_csv(RES / "versiones.csv", index=False)
display(Markdown(f"Perfil de ejecución: **{cfg.profile}** · semilla {cfg['seed']} · datos SHA-256 "
                 f"`{man.get('sha256', '—')[:16]}…`"))
env.style.hide(axis="index")
