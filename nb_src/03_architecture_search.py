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
# # 3. Búsqueda de la mejor arquitectura MLP
#
# **Objetivo del capítulo.** Encontrar, para cada una de las 16 combinaciones lag × horizonte, la topología del
# MLP (número de capas ocultas, neuronas por capa y Dropout después de cada capa) con menor error de validación
# sin sobreajuste, siguiendo la metodología de `architectures_dropout.ipynb`.
#
# **Espacio de búsqueda (búsqueda exhaustiva).**
#
# * Neuronas por capa ∈ {32, 64, 128, 256}; de 1 a 3 capas ocultas; todas las combinaciones con repetición
#   (`itertools.product`): 4 + 16 + 64 = 84 topologías.
# * Para una red de *n* capas, las 2ⁿ máscaras de Dropout(0,5): 4·2 + 16·4 + 64·8 = **584 candidatos** por
#   configuración, es decir **9 344 redes** en total.
# * Modelo: `Input(lag) → [Dense(u, relu) (+ Dropout)] × capas → Dense(H)`; Adam, pérdida MSE, métrica RMSE;
#   30 épocas, `batch_size = 32`, **sin barajar** (orden temporal).
#
# **Protocolo.** Split de Forward Chaining con origen en el 80 % de la serie de CV (capítulo 2.5); escaladores
# ajustados solo con el entrenamiento. Por candidato: `best_val_rmse`, `last_train_rmse`,
# `gap = |last_train_rmse − best_val_rmse|`, `gap_ratio = gap / best_val_rmse` y `holdout_rmse` (solo como
# evidencia final, nunca para elegir). **Selección:** candidatos con `gap_ratio < 0,15`, ordenados por
# `best_val_rmse`. Los RMSE de este capítulo están en unidades estandarizadas (z) del objetivo.

# %%
import ast
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from fcst.features import holdout_split
from fcst.models import candidates
from fcst.nbutils import LAG_COLOR, MUTED, SERIES, SEQ_CMAP, plt, savefig, setup
from fcst.search import load_curves, run_search, select_best

cfg, RES, FIG = setup()
S = cfg["search"]
series = pd.read_csv(RES / "serie_modelado.csv", parse_dates=["Date"])
y = series.Close.values
sp = holdout_split(len(y), cfg["holdout_frac"], cfg["horizons"])
LAGS, HS = cfg["lags"], cfg["horizons"]
n_cand = len(candidates(S["neurons"], S["max_layers"]))
print(f"{n_cand} candidatos por configuración × {len(LAGS) * len(HS)} configuraciones = {n_cand * len(LAGS) * len(HS):,} redes")

# %% [markdown]
# ## 3.1 Ejecución (en paralelo y reanudable)
#
# Cada configuración guarda su tabla completa en `results/architecture_search_lag{L}_h{H}.csv` y sus curvas de
# aprendizaje en `results/search_curves/`. Si la sesión se interrumpe, al volver a ejecutar se saltan las
# configuraciones ya terminadas.

# %%
t0 = time.time()
searches = {}
for lag in LAGS:
    for H in HS:
        searches[(lag, H)] = run_search(y[:sp.n_cv], y, sp.n_cv, lag, H, S, RES, seed=cfg["seed"])
allres = pd.concat(searches.values(), ignore_index=True)
print(f"Listo en {(time.time() - t0) / 60:.1f} min (incluye configuraciones reanudadas). "
      f"Tiempo total de ajuste registrado: {allres.fit_s.sum() / 3600:.1f} h-núcleo")

# %% [markdown]
# ## 3.2 Mejor arquitectura por (lag, horizonte)

# %%
best_rows = []
for (lag, H), df in searches.items():
    b = select_best(df, S["gap_ratio_max"])
    n_ok = int((df.gap_ratio < S["gap_ratio_max"]).sum())
    best_rows.append({"lag": lag, "H": H, "arquitectura": b.arquitectura, "dropout_mask": b.dropout_mask,
                      "capas": int(b.capas), "params": int(b.params), "best_val_rmse": b.best_val_rmse,
                      "best_epoch": int(b.best_epoch), "gap_ratio": b.gap_ratio, "holdout_rmse": b.holdout_rmse,
                      "candidatos_balanceados": n_ok, "criterio": b.criterio})
best = pd.DataFrame(best_rows)
best.to_csv(RES / "mejor_arquitectura.csv", index=False)
best.style.format({"best_val_rmse": "{:.4f}", "gap_ratio": "{:.3f}", "holdout_rmse": "{:.4f}"})

# %% [markdown]
# ## 3.3 Top-5 de candidatos balanceados por configuración

# %%
top5 = []
for (lag, H), df in searches.items():
    ok = df[df.gap_ratio < S["gap_ratio_max"]].sort_values("best_val_rmse").head(5).copy()
    ok.insert(0, "puesto", range(1, len(ok) + 1))
    top5.append(ok[["lag", "H", "puesto", "arquitectura", "dropout_mask", "best_val_rmse", "gap_ratio", "holdout_rmse"]])
top5 = pd.concat(top5, ignore_index=True)
top5.to_csv(RES / "top5_arquitecturas.csv", index=False)
top5.style.format({"best_val_rmse": "{:.4f}", "gap_ratio": "{:.3f}", "holdout_rmse": "{:.4f}"}).hide(axis="index")

# %% [markdown]
# ## 3.4 Curvas de aprendizaje del Top-3
#
# Una celda por configuración (filas = lag, columnas = horizonte). Línea continua = RMSE de validación; discontinua
# = RMSE de entrenamiento. El color identifica el puesto (1.º, 2.º, 3.º).

# %%
rank_color = [SERIES[0], SERIES[1], SERIES[2]]
fig, axes = plt.subplots(len(LAGS), len(HS), figsize=(14, 11), sharex=True)
for i, lag in enumerate(LAGS):
    for j, H in enumerate(HS):
        ax = axes[i, j]
        curves = load_curves(RES, lag, H)
        t3 = top5.query("lag == @lag and H == @H").head(3)
        for k, r in enumerate(t3.itertuples()):
            tr, va = curves[f"{r.arquitectura}|{r.dropout_mask}"]
            ep = np.arange(1, len(tr) + 1)
            ax.plot(ep, va, color=rank_color[k], lw=1.6, label=f"{k + 1}.º {r.arquitectura}")
            ax.plot(ep, tr, color=rank_color[k], lw=1, ls="--")
        ax.set_title(f"lag {lag} · H {H}", fontsize=10)
        ax.legend(fontsize=6.5, loc="upper right")
        if j == 0:
            ax.set_ylabel("RMSE (z)")
        if i == len(LAGS) - 1:
            ax.set_xlabel("Época")
fig.suptitle("Top-3 por configuración: validación (continua) y entrenamiento (discontinua)", y=1.0, fontsize=12)
savefig(fig, FIG, "03_curvas_top3")
plt.show()

# %% [markdown]
# ## 3.5 Efecto del número de capas, del ancho y del Dropout
#
# Para comparar entre configuraciones (cada una tiene su propia escala de error), cada candidato se expresa como
# su error relativo al mejor de su configuración: `best_val_rmse / mínimo de la configuración`.

# %%
allres["val_rel"] = allres.best_val_rmse / allres.groupby(["lag", "H"]).best_val_rmse.transform("min")
allres["ancho_max"] = allres.arquitectura.map(lambda s: max(ast.literal_eval(s)))
allres["frac_dropout"] = allres.n_dropout / allres.capas
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, lab in [(axes[0], "capas", "Capas ocultas"), (axes[1], "ancho_max", "Neuronas de la capa más ancha"),
                     (axes[2], "frac_dropout", "Fracción de capas con Dropout")]:
    g = allres.groupby(col)
    keys = sorted(g.groups)
    data = [g.get_group(k).val_rel.clip(upper=allres.val_rel.quantile(.98)) for k in keys]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    for b in bp["boxes"]:
        b.set(facecolor="#cde2fb", edgecolor=SERIES[0])
    for m in bp["medians"]:
        m.set(color=SERIES[0], lw=2)
    ax.set_xticks(range(1, len(keys) + 1), [f"{k:.2g}" if isinstance(k, float) else str(k) for k in keys])
    ax.set_xlabel(lab); ax.set_ylabel("RMSE de validación / mejor de su configuración")
fig.suptitle("Efecto de la topología sobre el error de validación (todas las configuraciones)", fontsize=12)
savefig(fig, FIG, "03_efecto_topologia")
plt.show()

# %%
eff = pd.concat([
    allres.groupby("capas")[["val_rel", "gap_ratio"]].median().assign(factor="capas"),
    allres.groupby("ancho_max")[["val_rel", "gap_ratio"]].median().assign(factor="ancho máx."),
    allres.groupby("n_dropout")[["val_rel", "gap_ratio"]].median().assign(factor="capas con Dropout"),
]).rename_axis("nivel").reset_index()
eff["% balanceados"] = pd.concat([
    allres.groupby("capas").gap_ratio.apply(lambda s: 100 * (s < S["gap_ratio_max"]).mean()),
    allres.groupby("ancho_max").gap_ratio.apply(lambda s: 100 * (s < S["gap_ratio_max"]).mean()),
    allres.groupby("n_dropout").gap_ratio.apply(lambda s: 100 * (s < S["gap_ratio_max"]).mean()),
]).values
eff = eff[["factor", "nivel", "val_rel", "gap_ratio", "% balanceados"]].rename(
    columns={"val_rel": "RMSE val relativo (mediana)", "gap_ratio": "gap_ratio (mediana)"})
eff.to_csv(RES / "efecto_topologia.csv", index=False)
eff.style.format({"RMSE val relativo (mediana)": "{:.3f}", "gap_ratio (mediana)": "{:.3f}", "% balanceados": "{:.0f}"}).hide(axis="index")

# %% [markdown]
# ## 3.6 ¿Cambia la arquitectura óptima con el horizonte?

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, col, lab in [(axes[0], "params", "Parámetros de la mejor red"), (axes[1], "best_val_rmse", "best_val_rmse (z)")]:
    mat = best.pivot(index="lag", columns="H", values=col)
    im = ax.imshow(mat.values, cmap=SEQ_CMAP, aspect="auto")
    for (a, b), v in np.ndenumerate(mat.values):
        txt = f"{v:,.0f}" if col == "params" else f"{v:.3f}"
        ax.text(b, a, txt, ha="center", va="center", fontsize=9,
                color="white" if v > np.nanmean(mat.values) else "#0b0b0b")
    ax.set_xticks(range(len(HS)), HS); ax.set_yticks(range(len(LAGS)), LAGS)
    ax.set_xlabel("Horizonte H (días)"); ax.set_ylabel("Lag (días)"); ax.set_title(lab); ax.grid(False)
savefig(fig, FIG, "03_mejor_por_config")
plt.show()

piv = best.pivot(index="lag", columns="H", values="arquitectura")
piv.columns = [f"H = {h}" for h in piv.columns]
piv

# %%
cap_by_h = best.groupby("H").capas.mean()
par_by_h = best.groupby("H").params.median()
gr = allres.groupby("n_dropout").gap_ratio.median()
display(Markdown(
    f"**Lectura automática de esta corrida.** Capas promedio de la red elegida por horizonte: "
    + ", ".join(f"H={h}: {v:.1f}" for h, v in cap_by_h.items())
    + ". Mediana de parámetros: " + ", ".join(f"H={h}: {v:,.0f}" for h, v in par_by_h.items())
    + f". La mediana de gap_ratio pasa de {gr.iloc[0]:.3f} sin Dropout a {gr.iloc[-1]:.3f} con Dropout en todas las capas. "
    f"{int((best.candidatos_balanceados == 0).sum())} de 16 configuraciones no tuvieron ningún candidato con "
    f"gap_ratio < {S['gap_ratio_max']} (en ese caso se aplicó el criterio alterno indicado en la tabla)."))
