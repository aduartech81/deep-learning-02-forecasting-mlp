# Pronóstico multihorizonte de Bitcoin con MLP y validación cruzada temporal

**Proyecto Integrador de Deep Learning (sección 2.21 de la guía del curso)**
Maestría en Ingeniería Biomédica · Universidad del Norte
Asignatura Deep Learning · Profesor: Dr. Lihki Rubio
Estudiante: Andrés Duarte Chaves

## Objetivo

Predecir el precio de cierre diario de Bitcoin (BTCUSDT) a 7, 14, 21 y 28 días usando **solo su propio
histórico**, con una red neuronal MLP multisalida, comparando tres estrategias de validación cruzada temporal de la
librería `timeseries-cv` (Group K-Fold, Forward Chaining y K-Fold), buscando de forma exhaustiva la mejor
arquitectura para cada combinación de ventana de entrada (lags de 7, 14, 21 y 28 días) y horizonte, analizando los
residuos con el test BDS, modelando también la volatilidad y desplegando el modelo como una API.

## Matriz de experimentos

| Dimensión | Valores |
|---|---|
| Ventana de entrada (lags) | 7, 14, 21, 28 días |
| Horizonte de salida H (multisalida) | 7, 14, 21, 28 días |
| Estrategias de validación (tsxv.splitTrainVal) | Group K-Fold, Forward Chaining, K-Fold |
| Búsqueda de arquitectura | 1–3 capas × {32, 64, 128, 256} neuronas × máscaras de Dropout(0,5): 584 candidatos por (lag, H), 9 344 redes |
| Métricas (validación y hold-out) | MAPE, MAE, RMSE, MSE por paso y promedio; p-valor BDS de los residuos de h = 1 |
| Referencias | MLPRegressor (64, 32) de scikit-learn y pronóstico ingenuo (último valor) |

## Protocolo sin fuga de información

1. El último 10 % de la serie (nunca menos de 2·H máx.) se reserva como **hold-out temporal**. No participa en
   la validación cruzada, ni en la búsqueda de arquitectura, ni en el ajuste de los escaladores.
2. En cada fold, los escaladores se ajustan **solo** con el entrenamiento.
3. Las ventanas de entrenamiento cuyo objetivo se solapa con el de una ventana de validación se purgan.
4. La arquitectura y el lag desplegado se eligen con el error de **validación**; el hold-out se mide una sola vez.

## Capítulos

1. **Datos y EDA:** procedencia (API pública de Binance), descriptivos, pruebas y autocorrelación.
2. **Validación temporal:** qué hace realmente cada estrategia de tsxv y cómo se adaptó.
3. **Búsqueda de arquitectura:** 9 344 redes, Top-5, curvas de aprendizaje y efecto de capas, ancho y Dropout.
4. **Entrenamiento fold a fold:** tablas por lag y horizonte, comparación de estrategias y de referencias.
5. **Modelo final y hold-out:** modelos desplegados por horizonte y series reales frente a predichas.
6. **Residuos:** test BDS por estrategia y en el hold-out.
7. **Volatilidad:** el mismo enfoque aplicado a la volatilidad histórica realizada.
8. **Despliegue:** API FastAPI, Docker, pruebas unitarias e integración continua.
9. **Conclusiones y reproducibilidad.**

```{admonition} Aviso
:class: note
Ejercicio académico de pronóstico de series de tiempo. No constituye una recomendación de inversión.
```
