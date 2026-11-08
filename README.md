# Pronóstico multihorizonte de Bitcoin con MLP y validación cruzada temporal

**Proyecto 02 de la serie de Deep Learning** · Proyecto Integrador 2.21 · Maestría en Ingeniería Biomédica,
Universidad del Norte · Andrés Duarte Chaves

Predicción del cierre diario de Bitcoin (BTCUSDT) a **7, 14, 21 y 28 días** usando solo su propio histórico, con un
MLP multisalida en Keras. El proyecto compara las tres estrategias de validación cruzada de
[`timeseries-cv`](https://pypi.org/project/timeseries-cv/) (Group K-Fold, Forward Chaining y K-Fold), busca de
forma exhaustiva la mejor arquitectura para cada una de las **16 combinaciones lag × horizonte (9 344 redes)**,
diagnostica los residuos con el **test BDS**, modela también la **volatilidad** y despliega el modelo como una
**API FastAPI** con Docker, pruebas unitarias e integración continua.

> **Estado:** código completo y probado de punta a punta con una serie sintética. La ejecución con los datos
> reales de Binance se hace en Google Colab Pro (ver abajo); los resultados se publicarán al terminar.

**Jupyter Book:** https://aduartech81.github.io/deep-learning-02-forecasting-mlp/

## Qué hace diferente a este proyecto

- **Revisión crítica de la librería.** `tsxv` en PyPI usa la firma `f(sequence, numInputs, numOutputs, numJumps)`
  (no los argumentos con nombre del borrador de la guía). Forward Chaining y K-Fold crean un fold por cada salto y
  cada fold valida **una sola ventana** (miles de folds con `numJumps = 1`). Se documenta y se resuelve agrupando
  los folds de tsxv en 5 bloques, sin alterar qué es entrenamiento y qué es validación en cada uno.
- **Sin fuga de información.** Hold-out temporal final (10 %, ≥ 2·H) intocable; escaladores ajustados solo con
  entrenamiento; purga de ventanas cuyo objetivo se solapa con el de validación; pruebas automáticas que
  verifican que Forward Chaining entrena siempre con el pasado.
- **Referencia honesta.** Todo se compara contra el pronóstico ingenuo (último valor), la vara que un modelo de
  precios debe superar para aportar algo, además del MLPRegressor de scikit-learn.
- **Detalle técnico corregido.** El test BDS se calcula con `statsmodels` (`arch.bootstrap` no lo incluye).

## Estructura

```
├── config.yaml                 # perfiles full (datos reales) y smoke (serie sintética)
├── src/fcst/                   # datos (Binance), features, CV con tsxv, modelos, búsqueda, métricas
├── nb_src/                     # fuente de los notebooks (jupytext)
├── notebooks/                  # 9 capítulos ejecutados + figs/
├── results/                    # tablas CSV (búsqueda, métricas por fold, por lag y horizonte, hold-out, BDS)
├── app/                        # API FastAPI + modelos por horizonte (model_h{H}.keras, scalers_h{H}.joblib)
├── tests/                      # pruebas de la API, de los modelos, de la validación temporal y de las métricas
├── Dockerfile · requirements-api.txt · .github/workflows/ci.yml
└── colab/ejecutar_en_colab.ipynb   # ejecución completa en Colab Pro con respaldo en Drive
```

| Capítulo | Contenido |
|---|---|
| 1 · EDA | Procedencia (API pública de Binance), descriptivos, ADF, Jarque-Bera, Ljung-Box, ARCH-LM, ACF y ACF² |
| 2 · Validación | Ventanas multipaso, qué hace realmente cada estrategia de tsxv, esquema e integridad de los folds |
| 3 · Arquitectura | 584 candidatos × 16 configuraciones, Top-5, curvas Top-3, efecto de capas, ancho y Dropout |
| 4 · Entrenamiento | 3 estrategias × 16 configuraciones × 5 folds; tablas por lag y horizonte; MLP vs referencias |
| 5 · Hold-out | Estrategia final, modelos por horizonte, series reales vs predichas (mejor, mediano y peor fold) |
| 6 · Residuos | Test BDS por fold y estrategia, y en el hold-out |
| 7 · Volatilidad | El mismo enfoque con la volatilidad realizada a 30 días como objetivo |
| 8 · Despliegue | API, Docker, pruebas y CI en funcionamiento |
| 9 · Conclusiones | Síntesis, limitaciones y versiones |

## Cómo ejecutarlo

**Google Colab Pro (recomendado):** abra `colab/ejecutar_en_colab.ipynb` desde GitHub en Colab, elija CPU con RAM
alta y ejecute todas las celdas. Respalda en Drive después de cada capítulo y se reanuda si la sesión se corta.

**Local:**

```bash
pip install -r requirements.txt
FC_PROFILE=smoke bash scripts/run_all.sh   # prueba rápida con serie sintética (results_smoke/)
bash scripts/run_all.sh                    # corrida completa con datos reales (horas: búsqueda de 9 344 redes)
```

**API:**

```bash
pip install -r requirements-api.txt && uvicorn app.api:app --port 8000
# o bien: docker build -t btc-forecast-mlp . && docker run -p 8000:8000 btc-forecast-mlp
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"lags": [...], "horizon": 14}'
```

`GET /info` indica cuántos lags espera el modelo de cada horizonte. Documentación interactiva en `/docs`.

## Datos

Velas diarias públicas de Binance (`/api/v3/klines`, intervalo `1d`), descargadas por el propio código con fecha
de corte fija y huella SHA-256 registrada. No se redistribuyen en el repositorio.

---

*English summary:* multi-horizon (7–28 days) forecasting of daily BTCUSDT closes with a Keras multi-output MLP,
three time-series cross-validation schemes from `timeseries-cv` (with a documented fix for its one-window folds),
exhaustive architecture/dropout search (9,344 networks), BDS residual diagnostics, a volatility extension, and a
FastAPI + Docker + CI deployment. Academic exercise; not investment advice.
