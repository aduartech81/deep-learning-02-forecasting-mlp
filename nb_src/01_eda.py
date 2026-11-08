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
# # 1. Datos y análisis exploratorio
#
# **Objetivo del capítulo.** Obtener la serie diaria de cierre de Bitcoin (BTCUSDT) desde la API pública de
# Binance, documentar su procedencia y caracterizar su nivel, sus retornos y su variabilidad: estadísticas
# descriptivas, distribución de los retornos, autocorrelación de los retornos y de sus cuadrados
# (heterocedasticidad) y volatilidad histórica realizada.
#
# **Decisiones que salen de aquí:** variable objetivo (precio de cierre; volatilidad como objetivo alterno en el
# capítulo 7), tamaño del hold-out temporal y la advertencia de no estacionariedad que condiciona el escalado.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller

from fcst.data import load_prices, read_manifest
from fcst.features import add_features, holdout_split
from fcst.nbutils import INK2, MUTED, PART_COLOR, SERIES, plt, savefig, setup

cfg, RES, FIG = setup()

# %% [markdown]
# ## 1.1 Procedencia de los datos
#
# Velas diarias (`interval=1d`) del par BTCUSDT desde el endpoint público `/api/v3/klines` de Binance, paginando
# de a 1 000 velas. Se intenta primero `data-api.binance.vision` (espejo oficial de Binance solo para datos de
# mercado) y luego `api.binance.com`, que responde 451 desde algunas regiones, como los servidores de Colab.
# La fecha de corte es fija (`config.yaml`) para que el análisis sea reproducible; el manifiesto guarda la huella
# SHA-256 del archivo descargado.

# %%
raw = load_prices(cfg)
man = read_manifest(cfg)
pd.Series(man, name="valor").to_frame()

# %%
df = add_features(raw, cfg["vol_window"])
print(f"{len(df):,} días · {df.Date.min().date()} → {df.Date.max().date()}")
gaps = df.Date.diff().dt.days.value_counts().rename("frecuencia")
print("Diferencia entre fechas consecutivas (días):", dict(gaps))
assert df.Date.is_monotonic_increasing and df.Date.is_unique

# %% [markdown]
# ## 1.2 Partición: serie de validación cruzada y hold-out temporal final
#
# Antes de mirar cualquier modelo se reserva el último tramo (10 %, nunca menos de 2·H_max = 56 días). El
# hold-out **no participa** en la validación cruzada, en la búsqueda de arquitectura ni en el ajuste de los
# escaladores: solo sirve para la evaluación definitiva del modelo final (capítulo 5).

# %%
series = df.dropna(subset=["LogReturn"]).reset_index(drop=True)   # el primer día no tiene retorno
sp = holdout_split(len(series), cfg["holdout_frac"], cfg["horizons"])
cut_date = series.Date.iloc[sp.n_cv]
part = pd.DataFrame({
    "tramo": ["Validación cruzada (CV)", "Hold-out final"],
    "desde": [series.Date.iloc[0].date(), cut_date.date()],
    "hasta": [series.Date.iloc[sp.n_cv - 1].date(), series.Date.iloc[-1].date()],
    "días": [sp.n_cv, sp.n_hold],
})
part.to_csv(RES / "particion.csv", index=False)
part

# %% [markdown]
# ## 1.3 Estadísticas descriptivas de precios y retornos

# %%
def describe(x: pd.Series) -> dict:
    x = x.dropna()
    return {"n": len(x), "media": x.mean(), "desv. est.": x.std(), "mín": x.min(), "p25": x.quantile(.25),
            "mediana": x.median(), "p75": x.quantile(.75), "máx": x.max(), "asimetría": stats.skew(x),
            "curtosis (exceso)": stats.kurtosis(x)}


desc = pd.DataFrame({"Close (USD)": describe(series.Close), "Retorno log diario": describe(series.LogReturn),
                     f"Volatilidad {cfg['vol_window']} d (anualizada)": describe(series.Volatility)}).T
desc.to_csv(RES / "eda_descriptivos.csv")
desc.style.format("{:,.4f}")

# %% [markdown]
# ## 1.4 Pruebas sobre la serie
#
# * **ADF** (Dickey-Fuller aumentada): H0 = raíz unitaria (no estacionaria).
# * **Jarque-Bera**: H0 = retornos con distribución normal.
# * **Ljung-Box** (10 rezagos): H0 = sin autocorrelación, sobre los retornos y sobre sus cuadrados.
# * **ARCH-LM** (Engle, 10 rezagos): H0 = sin heterocedasticidad condicional.

# %%
r = series.LogReturn.values
tests = []
for name, x in [("Close (nivel)", series.Close.values), ("log(Close)", np.log(series.Close.values)), ("Retorno log", r)]:
    stat, p, *_ = adfuller(x, autolag="AIC")
    tests.append({"prueba": "ADF", "serie": name, "estadístico": stat, "p_valor": p})
jb = stats.jarque_bera(r)
tests.append({"prueba": "Jarque-Bera", "serie": "Retorno log", "estadístico": jb.statistic, "p_valor": jb.pvalue})
for name, x in [("Retorno log", r), ("Retorno log al cuadrado", r ** 2)]:
    lb = acorr_ljungbox(x, lags=[10], return_df=True).iloc[0]
    tests.append({"prueba": "Ljung-Box (10)", "serie": name, "estadístico": lb.lb_stat, "p_valor": lb.lb_pvalue})
lm = het_arch(r, nlags=10)
tests.append({"prueba": "ARCH-LM (10)", "serie": "Retorno log", "estadístico": lm[0], "p_valor": lm[1]})
tests = pd.DataFrame(tests)
tests["decisión (α = 0,05)"] = np.where(tests.p_valor < 0.05, "se rechaza H0", "no se rechaza H0")
tests.to_csv(RES / "eda_pruebas.csv", index=False)
tests.style.format({"estadístico": "{:,.3f}", "p_valor": "{:.2e}"})

# %%
p_adf_close = tests.query("prueba == 'ADF' and serie == 'Close (nivel)'").p_valor.iloc[0]
p_lb2 = tests.query("serie == 'Retorno log al cuadrado'").p_valor.iloc[0]
display(Markdown(
    f"**Lectura.** El precio en nivel {'no es estacionario' if p_adf_close > 0.05 else 'resulta estacionario según ADF'} "
    f"(ADF p = {p_adf_close:.2f}), mientras que los retornos sí lo son. Los retornos tienen colas pesadas "
    f"(curtosis en exceso = {desc.loc['Retorno log diario', 'curtosis (exceso)']:.1f}; Jarque-Bera rechaza normalidad) y sus "
    f"cuadrados están autocorrelacionados (Ljung-Box p = {p_lb2:.1e}): la volatilidad se agrupa en el tiempo. "
    "Dos consecuencias para el modelado: (1) al predecir el **nivel**, el escalador ajustado con el pasado puede quedar "
    "fuera de rango cuando el precio alcanza máximos nuevos (riesgo de extrapolación que se revisará en el hold-out); "
    "(2) es esperable que los residuos no sean iid aunque el error medio sea bajo, lo que medirá el test BDS."))

# %% [markdown]
# ## 1.5 Visualizaciones

# %%
fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
ax = axes[0]
ax.plot(series.Date[:sp.n_cv], series.Close[:sp.n_cv], color=PART_COLOR["train"], label="Serie de CV")
ax.plot(series.Date[sp.n_cv:], series.Close[sp.n_cv:], color=PART_COLOR["hold"], label="Hold-out final")
ax.set_yscale("log"); ax.set_ylabel("Cierre (USD, escala log)"); ax.set_title("Precio de cierre diario BTCUSDT")
ax.legend(loc="upper left")
ax = axes[1]
ax.plot(series.Date, series.LogReturn, color=SERIES[0], lw=0.7)
ax.set_ylabel("Retorno log diario"); ax.set_title("Retornos logarítmicos diarios")
ax = axes[2]
ax.plot(series.Date, series.Volatility, color=SERIES[1])
ax.set_ylabel("Volatilidad anualizada"); ax.set_title(f"Volatilidad histórica realizada (ventana {cfg['vol_window']} días)")
for a in axes:
    a.axvspan(cut_date, series.Date.iloc[-1], color=PART_COLOR["hold"], alpha=0.08, lw=0)
savefig(fig, FIG, "01_serie_retornos_volatilidad")
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]
ax.hist(r, bins=120, density=True, color=SERIES[0], alpha=0.85, label="Retornos observados")
xs = np.linspace(r.min(), r.max(), 400)
ax.plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), color=INK2, lw=1.5, label="Normal con igual media y desv.")
ax.set_title("Histograma de retornos diarios"); ax.set_xlabel("Retorno log"); ax.legend()
ax = axes[1]
stats.probplot(r, dist="norm", plot=ax)
ax.get_lines()[0].set(color=SERIES[0], markersize=3); ax.get_lines()[1].set(color=INK2)
ax.set_title("Gráfico Q-Q frente a la normal")
savefig(fig, FIG, "01_histograma_qq_retornos")
plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
plot_acf(series.Close, lags=60, ax=axes[0], color=SERIES[0], vlines_kwargs={"colors": SERIES[0]})
axes[0].set_title("ACF del precio (nivel)")
plot_acf(r, lags=60, ax=axes[1], color=SERIES[0], vlines_kwargs={"colors": SERIES[0]})
axes[1].set_title("ACF de los retornos")
plot_acf(r ** 2, lags=60, ax=axes[2], color=SERIES[1], vlines_kwargs={"colors": SERIES[1]})
axes[2].set_title("ACF de los retornos al cuadrado")
for a in axes:
    a.set_xlabel("Rezago (días)")
    for c in a.collections:
        c.set_facecolor(MUTED); c.set_alpha(0.2)
savefig(fig, FIG, "01_acf")
plt.show()

# %% [markdown]
# **Cómo leer las ACF.** La ACF del precio decae muy despacio (memoria larga típica de una serie integrada): los
# rezagos del precio contienen mucha información sobre el nivel siguiente, y por eso un pronóstico ingenuo
# ("mañana igual que hoy") será un rival difícil. La ACF de los retornos es casi nula, mientras que la de sus
# cuadrados es positiva y persistente: el **cuándo** del movimiento grande es en parte predecible (volatilidad), no
# su **dirección**. Esto motiva modelar también la volatilidad (capítulo 7).

# %%
series.to_csv(RES / "serie_modelado.csv", index=False)
print("Guardado:", RES / "serie_modelado.csv")
