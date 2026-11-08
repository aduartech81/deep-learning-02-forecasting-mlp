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
# ## 9.2 Respuestas a lo que pide la guía
#
# **¿Qué estrategia de validación es más realista y cuál más optimista?** Forward Chaining es la única que
# respeta la causalidad (entrena siempre con el pasado; verificado fold a fold en el capítulo 2) y por eso es la
# estrategia final. K-Fold y Group K-Fold incluyen entrenamiento posterior a la validación. La comparación
# numérica (capítulo 4) debe leerse con el MAPE, porque las estrategias no validan sobre los mismos días.
#
# **¿Cambia la arquitectura óptima con el horizonte?** Ver la tabla del capítulo 3.6 y la lectura automática de
# ese capítulo: se reporta cuántas capas y parámetros tiene la red elegida para cada horizonte y si el Dropout
# reduce el gap entre entrenamiento y validación.
#
# **¿Cuánto se degrada el pronóstico al alargar el horizonte?** El RMSE de validación crece con H en todas las
# estrategias y lags (capítulo 4.5); el RMSE por paso muestra que el error se acumula paso a paso dentro de cada
# horizonte.
#
# **¿El MLP aporta sobre una referencia simple?** La comparación con el pronóstico ingenuo (capítulos 4.6 y 5) es
# la prueba más exigente y la más honesta: un precio que se comporta casi como un paseo aleatorio deja muy poco
# margen a cualquier modelo que use solo su propio pasado.
#
# **¿Los residuos son ruido blanco?** El BDS (capítulo 6) indica en qué proporción de folds los residuos de h = 1
# son compatibles con iid; la dependencia remanente es coherente con la volatilidad agrupada del capítulo 1.

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
