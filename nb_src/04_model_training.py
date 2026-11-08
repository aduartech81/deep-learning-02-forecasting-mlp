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
# # 4. Entrenamiento fold a fold y comparación de estrategias de validación
#
# **Objetivo del capítulo.** Entrenar el MLP multisalida con la mejor arquitectura de cada (lag, H) en cada fold
# de las tres estrategias de tsxv (Group K-Fold, Forward Chaining y K-Fold), medir MAPE, MAE, RMSE, MSE y el
# p-valor BDS de los residuos de h = 1, y comparar contra dos referencias:
#
# * **MLPRegressor (64, 32)** de scikit-learn, el baseline sugerido por la guía.
# * **Pronóstico ingenuo** (paseo aleatorio): repetir el último cierre durante los H días. En precios de activos es
#   la referencia que cualquier modelo debe superar para aportar algo.
#
# **Todas las métricas de este capítulo son de VALIDACIÓN**, en dólares (USD) salvo el MAPE (%). El número de
# épocas de cada red es la época óptima encontrada en la búsqueda (capítulo 3) para esa configuración.

# %%
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from fcst.cv import STRATEGY_LABEL
from fcst.experiments import fold_table, parse_arch, run_cv, save_preds
from fcst.features import holdout_split
from fcst.nbutils import LAG_COLOR, MODEL_COLOR, MUTED, STRAT_COLOR, plt, savefig, setup

cfg, RES, FIG = setup()
series = pd.read_csv(RES / "serie_modelado.csv", parse_dates=["Date"])
y = series.Close.values
sp = holdout_split(len(y), cfg["holdout_frac"], cfg["horizons"])
y_cv = y[:sp.n_cv]
LAGS, HS, STRATS = cfg["lags"], cfg["horizons"], cfg["cv"]["strategies"]
best = pd.read_csv(RES / "mejor_arquitectura.csv")
(RES / "preds").mkdir(exist_ok=True)
(RES / "tablas_validacion").mkdir(exist_ok=True)

# %% [markdown]
# ## 4.1 Entrenamiento (3 estrategias × 4 lags × 4 horizontes × 5 folds)

# %%
t0 = time.time()
per_fold, per_h = [], []
for st in STRATS:
    for lag in LAGS:
        for H in HS:
            b = best.query("lag == @lag and H == @H").iloc[0]
            epochs = int(min(max(b.best_epoch, 1), cfg["train"]["epochs_max"]))
            pf, ph, preds = run_cv(y_cv, st, lag, H, parse_arch(b.arquitectura), parse_arch(b.dropout_mask),
                                   epochs, cfg, seed=cfg["seed"])
            per_fold.append(pf); per_h.append(ph)
            save_preds(preds, RES / "preds" / f"{st}_lag{lag}_h{H}.npz")
            fold_table(pf).to_csv(RES / "tablas_validacion" / f"{st}_lag{lag}_h{H}.csv")
    print(f"{STRATEGY_LABEL[st]}: listo ({(time.time() - t0) / 60:.1f} min acumulados)")
per_fold = pd.concat(per_fold, ignore_index=True)
per_h = pd.concat(per_h, ignore_index=True)
per_fold.to_csv(RES / "cv_metricas_por_fold.csv", index=False)
per_h.to_csv(RES / "cv_metricas_por_h.csv", index=False)

# %% [markdown]
# ## 4.2 Tablas por tamaño de entrada y horizonte (Forward Chaining, validación)
#
# Una tabla por cada (lag, H): filas = folds, más media y desviación estándar; columnas = métricas promediadas
# sobre h = 1..H y p-valor BDS de h = 1. Las 48 tablas (3 estrategias × 16) están en `results/tablas_validacion/`.

# %%
fmt = {"MAPE": "{:.2f}", "MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MSE": "{:,.3e}", "BDS_pvalue_h1": "{:.3f}"}
for lag in LAGS:
    for H in HS:
        t = fold_table(per_fold.query("estrategia == 'forwardChaining' and lag == @lag and H == @H"))
        display(Markdown(f"**lag = {lag} · H = {H}** (validación, Forward Chaining)"))
        display(t.style.format(fmt))

# %% [markdown]
# ## 4.3 Tabla consolidada lag × horizonte

# %%
mlp = per_fold[per_fold.modelo == "mlp"]
cons = mlp.groupby(["estrategia", "lag", "H"])[["RMSE", "MAPE", "MAE", "MSE", "BDS_pvalue_h1"]].mean().reset_index()
cons.to_csv(RES / "consolidado_lag_h.csv", index=False)
for metric in ["RMSE", "MAPE"]:
    t = cons[cons.estrategia == "forwardChaining"].pivot(index="lag", columns="H", values=metric)
    t.columns = [f"H = {h}" for h in t.columns]
    display(Markdown(f"**{metric} medio de validación (Forward Chaining)** — filas: lag; columnas: horizonte"))
    display(t.style.format("{:,.0f}" if metric == "RMSE" else "{:.2f}").background_gradient(cmap="Blues", axis=None))

# %% [markdown]
# ## 4.4 Comparación entre estrategias de validación

# %%
comp = cons.copy()
comp["estrategia"] = comp.estrategia.map(STRATEGY_LABEL)
comp = comp[["estrategia", "lag", "H", "RMSE", "MAPE", "MAE", "MSE", "BDS_pvalue_h1"]]
comp.to_csv(RES / "comparacion_estrategias.csv", index=False)
agg = (cons.groupby("estrategia")[["RMSE", "MAPE", "BDS_pvalue_h1"]].mean()
       .rename(index=STRATEGY_LABEL).sort_values("RMSE"))
agg["RMSE relativo a Forward Chaining"] = agg.RMSE / agg.loc["Forward Chaining", "RMSE"]
display(agg.style.format({"RMSE": "{:,.0f}", "MAPE": "{:.2f}", "BDS_pvalue_h1": "{:.3f}",
                          "RMSE relativo a Forward Chaining": "{:.2f}"}))
comp.style.format({"RMSE": "{:,.0f}", "MAPE": "{:.2f}", "MAE": "{:,.0f}", "MSE": "{:.3e}",
                   "BDS_pvalue_h1": "{:.3f}"}).hide(axis="index")

# %% [markdown]
# **Cuidado al comparar estrategias.** Las tres no validan sobre los mismos días: Forward Chaining y K-Fold validan
# en la segunda mitad de la serie de CV (por bloques), mientras que Group K-Fold reparte sus ventanas de validación
# por toda la serie, incluidos los años de precios bajos. Como el error en USD crece con el nivel del precio, el
# **MAPE** (relativo) es la métrica más justa para comparar estrategias; el RMSE sirve para comparar lags y
# horizontes dentro de una misma estrategia.

# %%
fig, axes = plt.subplots(len(STRATS), len(HS), figsize=(15, 9), sharey="row")
for r, st in enumerate(STRATS):
    for c, H in enumerate(HS):
        ax = axes[r, c]
        for lag in LAGS:
            d = mlp.query("estrategia == @st and lag == @lag and H == @H").sort_values("fold")
            ax.plot(d.fold, d.RMSE, marker="o", color=LAG_COLOR[lag], label=f"{lag} lags")
        ax.set_title(f"{STRATEGY_LABEL[st]} · H = {H}", fontsize=10)
        ax.set_xticks(range(1, 6))
        if c == 0:
            ax.set_ylabel("RMSE validación (USD)")
        if r == len(STRATS) - 1:
            ax.set_xlabel("Fold")
axes[0, 0].legend()
savefig(fig, FIG, "04_rmse_por_fold")
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric in zip(axes, ["MAPE", "RMSE"]):
    x = np.arange(len(HS))
    w = 0.26
    for i, st in enumerate(STRATS):
        v = cons[cons.estrategia == st].groupby("H")[metric].mean().reindex(HS)
        ax.bar(x + (i - 1) * w, v.values, width=w, color=STRAT_COLOR[st], label=STRATEGY_LABEL[st])
    ax.set_xticks(x, [f"H = {h}" for h in HS])
    ax.set_ylabel(f"{metric} medio de validación" + (" (%)" if metric == "MAPE" else " (USD)"))
    ax.set_title(f"{metric} por estrategia (promedio sobre lags)")
axes[0].legend()
savefig(fig, FIG, "04_estrategias")
plt.show()

# %% [markdown]
# ## 4.5 Degradación con el horizonte

# %%
fig, axes = plt.subplots(1, len(STRATS), figsize=(15, 4), sharey=True)
for ax, st in zip(axes, STRATS):
    for lag in LAGS:
        v = cons.query("estrategia == @st and lag == @lag").set_index("H").RMSE.reindex(HS)
        ax.plot(HS, v.values, marker="o", color=LAG_COLOR[lag], label=f"{lag} lags")
    ax.set_title(STRATEGY_LABEL[st]); ax.set_xlabel("Horizonte H (días)"); ax.set_xticks(HS)
axes[0].set_ylabel("RMSE medio de validación (USD)")
axes[0].legend()
savefig(fig, FIG, "04_rmse_vs_horizonte")
plt.show()

# %%
fig, axes = plt.subplots(1, len(HS), figsize=(15, 3.8), sharey=True)
ph = per_h[(per_h.modelo == "mlp") & (per_h.estrategia == "forwardChaining")]
for ax, H in zip(axes, HS):
    for lag in LAGS:
        v = ph.query("lag == @lag and H == @H").groupby("h").RMSE.mean()
        ax.plot(v.index, v.values, marker=".", color=LAG_COLOR[lag], label=f"{lag} lags")
    ax.set_title(f"H = {H}"); ax.set_xlabel("Paso h (días adelante)")
axes[0].set_ylabel("RMSE validación (USD)")
axes[0].legend()
fig.suptitle("RMSE medio por paso del horizonte (Forward Chaining)", y=1.02)
savefig(fig, FIG, "04_rmse_por_paso")
plt.show()

# %% [markdown]
# ## 4.6 El MLP frente a las referencias

# %%
vs = (per_fold.groupby(["estrategia", "modelo", "H"])[["RMSE", "MAPE"]].mean().reset_index())
vs_fc = vs[vs.estrategia == "forwardChaining"].pivot(index="H", columns="modelo", values="MAPE")
vs_fc = vs_fc[["mlp", "sklearn", "naive"]].rename(columns={"mlp": "MLP Keras (mejor arq.)", "sklearn": "MLPRegressor (64, 32)",
                                                          "naive": "Ingenuo (último valor)"})
vs.to_csv(RES / "mlp_vs_referencias.csv", index=False)
display(Markdown("**MAPE medio de validación (%) por horizonte — Forward Chaining, promedio sobre lags**"))
display(vs_fc.style.format("{:.2f}").highlight_min(axis=1, props="font-weight:bold"))

fig, ax = plt.subplots(figsize=(8, 3.8))
x = np.arange(len(HS)); w = 0.26
for i, (m, lab) in enumerate([("mlp", "MLP Keras"), ("sklearn", "MLPRegressor"), ("naive", "Ingenuo")]):
    v = vs.query("estrategia == 'forwardChaining' and modelo == @m").set_index("H").MAPE.reindex(HS)
    ax.bar(x + (i - 1) * w, v.values, width=w, color=MODEL_COLOR[m], label=lab)
ax.set_xticks(x, [f"H = {h}" for h in HS]); ax.set_ylabel("MAPE validación (%)")
ax.set_title("MLP frente a las referencias (Forward Chaining)"); ax.legend()
savefig(fig, FIG, "04_mlp_vs_referencias")
plt.show()

# %%
win = (vs_fc["MLP Keras (mejor arq.)"] < vs_fc["Ingenuo (último valor)"])
display(Markdown(
    f"**Lectura automática.** En Forward Chaining, el MLP supera al pronóstico ingenuo en {int(win.sum())} de "
    f"{len(HS)} horizontes (MAPE). Diferencia media MLP − ingenuo: "
    f"{(vs_fc['MLP Keras (mejor arq.)'] - vs_fc['Ingenuo (último valor)']).mean():+.2f} puntos porcentuales. "
    "Un MLP que no le gana al paseo aleatorio no está aprendiendo dinámica útil del precio: solo reproduce el "
    "último nivel con ruido. Esa comparación es la que da sentido a las métricas absolutas."))
