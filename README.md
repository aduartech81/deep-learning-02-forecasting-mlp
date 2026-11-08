# Pronóstico multihorizonte de Bitcoin con MLP y validación cruzada temporal

**Proyecto 02 de la serie de Deep Learning** · Proyecto Integrador 2.21 · Maestría en Ingeniería Biomédica,
Universidad del Norte · Andrés Duarte Chaves

Predicción del cierre diario de Bitcoin (BTCUSDT) a **7, 14, 21 y 28 días** usando solo su propio histórico, con un
MLP multisalida en Keras. El proyecto compara las tres estrategias de validación cruzada de
[`timeseries-cv`](https://pypi.org/project/timeseries-cv/) (Group K-Fold, Forward Chaining y K-Fold), busca de
forma exhaustiva la mejor arquitectura para cada una de las **16 combinaciones lag × horizonte (9 344 redes)**,
diagnostica los residuos con el **test BDS**, modela también la **volatilidad** y despliega el modelo como una
**API FastAPI** con Docker, pruebas unitarias e integración continua.

> **Estado:** ejecutado completo con los datos reales de Binance (3 226 días, 2018-01-01 → 2026-10-31) en Google
> Colab Pro (8 núcleos, 51 GB). Los notebooks publicados contienen las salidas reales.

## Resultados principales

| | Resultado |
|---|---|
| Partición | CV: 2 903 días (2018 → 13 dic 2025) · hold-out intocable: 322 días (14 dic 2025 → 31 oct 2026) |
| Búsqueda de arquitectura | 9 344 redes en 2 h 53 min; ganadoras de 2–3 capas, anchas (256) y con Dropout en la última capa |
| **Precio: MLP vs ingenuo (validación, MAPE)** | MLP 7,2 / 10,5 / 13,5 / 12,5 % vs ingenuo 3,7 / 5,1 / 6,3 / 7,2 % (H = 7 / 14 / 21 / 28) |
| **Precio: hold-out** | Los 4 modelos desplegados pierden contra el ingenuo (MAPE 6,0 / 5,1 / 6,2 / 7,4 % vs 3,1 / 4,4 / 5,6 / 6,7 %) |
| Estrategias de validación | Forward Chaining: realista (entrena siempre con el pasado). K-Fold: optimista en los bloques con cambio de régimen. Group K-Fold: la peor estimación (pocas ventanas, fuga en 30/80 folds) |
| Residuos (BDS, h = 1) | No son iid en ningún fold de Forward Chaining ni en el hold-out |
| **Volatilidad** | El MLP sí le gana al ingenuo en RMSE del hold-out en 12 de 16 configuraciones |

**Conclusión.** Usando solo su propio pasado, el cierre de Bitcoin se comporta casi como un paseo aleatorio: el
pronóstico ingenuo ("mañana igual que hoy") le gana al MLP en todos los horizontes. La volatilidad, en cambio, sí tiene
estructura que la red aprovecha. Es un resultado negativo honesto, sostenido por un protocolo sin fuga y por la
comparación contra referencias simples (detalle en el capítulo 9).

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
pip install -r requirements.txt tensorflow-cpu && pip install --no-deps timeseries-cv==0.1.5
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
