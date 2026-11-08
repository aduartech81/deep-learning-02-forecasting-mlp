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
# # 7. Modelación de la volatilidad
#
# **Objetivo del capítulo.** La guía pide un modelo orientado tanto al **nivel** de la serie como a su
# **volatilidad**. Aquí el objetivo pasa a ser la volatilidad histórica realizada a 30 días (anualizada,
# capítulo 1):
#
# $$\sigma_t = \sqrt{\tfrac{1}{n}\textstyle\sum_{i=1}^{n}(r_{t-i}-\bar r)^2}\cdot\sqrt{365},\qquad r_t=\ln(P_t/P_{t-1}),\; n = 30.$$
#
# **Alcance (documentado).** Se reutiliza la mejor arquitectura encontrada para el precio en cada (lag, H) y se
# evalúa con la estrategia final (Forward Chaining) y con el mismo hold-out de fechas. Repetir la búsqueda
# exhaustiva para este segundo objetivo duplicaría el cómputo (otras 9 344 redes); el capítulo mide si la
# arquitectura del precio sirve también para la volatilidad.
#
# **Advertencia.** σₜ es una media móvil de 30 días: dos valores consecutivos comparten 29 retornos, así que la
# serie es muy suave y el pronóstico ingenuo es muy fuerte a horizontes cortos.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from fcst.experiments import fit_final, parse_arch, run_cv
from fcst.features import holdout_split
from fcst.nbutils import INK2, LAG_COLOR, MODEL_COLOR, PART_COLOR, plt, savefig, setup

cfg, RES, FIG = setup()
series = pd.read_csv(RES / "serie_modelado.csv", parse_dates=["Date"])
sp_price = holdout_split(len(series), cfg["holdout_frac"], cfg["horizons"])
vol = series.dropna(subset=["Volatility"]).reset_index(drop=True)
v, vdates = vol.Volatility.values, vol.Date
n_cv = len(v) - sp_price.n_hold        # mismo hold-out (mismas fechas) que el precio
assert vdates.iloc[n_cv] == series.Date.iloc[sp_price.n_cv]
LAGS, HS = cfg["lags"], cfg["horizons"]
best = pd.read_csv(RES / "mejor_arquitectura.csv")
print(f"Volatilidad: {len(v)} días; CV hasta {vdates.iloc[n_cv - 1].date()}, hold-out desde {vdates.iloc[n_cv].date()}")

# %% [markdown]
# ## 7.1 Validación cruzada (Forward Chaining) — métricas de VALIDACIÓN

# %%
rows = []
for lag in LAGS:
    for H in HS:
        b = best.query("lag == @lag and H == @H").iloc[0]
        ep = int(min(max(b.best_epoch, 1), cfg["train"]["epochs_max"]))
        pf, _, _ = run_cv(v[:n_cv], "forwardChaining", lag, H, parse_arch(b.arquitectura), parse_arch(b.dropout_mask),
                          ep, cfg, seed=cfg["seed"], models=("mlp", "naive"))
        rows.append(pf)
vcv = pd.concat(rows, ignore_index=True)
vcv.to_csv(RES / "volatilidad_cv_por_fold.csv", index=False)
vt = vcv.groupby(["modelo", "lag", "H"])[["RMSE", "MAPE", "BDS_pvalue_h1"]].mean().reset_index()
vt.to_csv(RES / "volatilidad_cv_resumen.csv", index=False)
piv = vt.pivot_table(index="lag", columns=["H", "modelo"], values="MAPE")
display(Markdown("**MAPE medio de validación (%) de la volatilidad — MLP frente a ingenuo** (Forward Chaining)"))
piv.style.format("{:.2f}")

# %% [markdown]
# ## 7.2 Hold-out final

# %%
rows, hold_preds = [], {}
for lag in LAGS:
    for H in HS:
        b = best.query("lag == @lag and H == @H").iloc[0]
        ep = int(min(max(b.best_epoch, 1), cfg["train"]["epochs_max"]))
        _, _, res = fit_final(v, n_cv, lag, H, parse_arch(b.arquitectura), parse_arch(b.dropout_mask), ep, cfg, seed=cfg["seed"])
        hold_preds[(lag, H)] = res
        rows.append({"lag": lag, "H": H, "RMSE_hold": res["summary_mlp"]["RMSE"], "MAPE_hold": res["summary_mlp"]["MAPE"],
                     "RMSE_naive": res["summary_naive"]["RMSE"], "MAPE_naive": res["summary_naive"]["MAPE"],
                     "BDS_pvalue_h1": res["summary_mlp"]["BDS_pvalue_h1"]})
vh = pd.DataFrame(rows)
vh["MLP_mejor_que_ingenuo"] = vh.RMSE_hold < vh.RMSE_naive
vh.to_csv(RES / "volatilidad_holdout.csv", index=False)
vh.style.format({"RMSE_hold": "{:.4f}", "MAPE_hold": "{:.2f}", "RMSE_naive": "{:.4f}", "MAPE_naive": "{:.2f}",
                 "BDS_pvalue_h1": "{:.3f}"}).hide(axis="index")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
ax = axes[0]
for lag in LAGS:
    d = vt.query("modelo == 'mlp' and lag == @lag").set_index("H").MAPE.reindex(HS)
    ax.plot(HS, d.values, marker="o", color=LAG_COLOR[lag], label=f"MLP · {lag} lags")
d = vt.query("modelo == 'naive'").groupby("H").MAPE.mean().reindex(HS)
ax.plot(HS, d.values, marker="s", color=MODEL_COLOR["naive"], ls="--", label="Ingenuo")
ax.set_xticks(HS); ax.set_xlabel("Horizonte H (días)"); ax.set_ylabel("MAPE validación (%)")
ax.set_title("Volatilidad: error de validación frente al horizonte"); ax.legend(fontsize=8)
ax = axes[1]
lag_b, H_b = int(vh.loc[vh.RMSE_hold.idxmin(), "lag"]), int(vh.loc[vh.RMSE_hold.idxmin(), "H"])
res = hold_preds[(lag_b, H_b)]
ax.plot(vdates.iloc[n_cv - 200:n_cv], v[n_cv - 200:n_cv], color=PART_COLOR["train"], lw=1, label="previo (real)")
ax.plot(vdates.iloc[res["hold_idx"]], res["y_hold"][:, 0], color=PART_COLOR["hold"], lw=1.2, label="hold-out real")
ax.plot(vdates.iloc[res["hold_idx"]], res["p_hold"][:, 0], color=INK2, ls="--", lw=1, label="MLP (h = 1)")
ax.plot(vdates.iloc[res["hold_idx"] + H_b - 1], res["p_hold"][:, -1], color=MODEL_COLOR["sklearn"], ls=":", lw=1.2,
        label=f"MLP (h = {H_b})")
ax.set_title(f"Hold-out, mejor configuración (lag {lag_b}, H {H_b})"); ax.set_ylabel("Volatilidad anualizada")
ax.legend(fontsize=8); ax.tick_params(axis="x", labelrotation=30)
savefig(fig, FIG, "07_volatilidad")
plt.show()

# %%
display(Markdown(
    f"**Lectura automática.** En el hold-out, el MLP supera al ingenuo en {int(vh.MLP_mejor_que_ingenuo.sum())} de 16 "
    f"configuraciones para la volatilidad (frente a lo observado para el precio en el capítulo 5). MAPE del MLP en el "
    f"hold-out: mínimo {vh.MAPE_hold.min():.1f} %, máximo {vh.MAPE_hold.max():.1f} %. Por la superposición de 29 de "
    "30 retornos entre días consecutivos, cualquier ventaja sobre el ingenuo a h = 1 es pequeña; la ganancia, si la "
    "hay, debe aparecer en los horizontes largos, donde la volatilidad tiende a revertir hacia su media."))
