"""API de inferencia: recibe los últimos cierres (lags) y el horizonte, y devuelve H predicciones.

Carga un MLP (la mejor arquitectura encontrada) y un par de escaladores (x, y) por horizonte:
app/model_h{H}.keras y app/scalers_h{H}.joblib. La carpeta se puede cambiar con MODEL_DIR.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402

from app.schemas import HorizonInfo, Prediction, TimeSeriesData  # noqa: E402

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent))
VALID_HORIZONS = (7, 14, 21, 28)

models = {H: tf.keras.models.load_model(MODEL_DIR / f"model_h{H}.keras") for H in VALID_HORIZONS
          if (MODEL_DIR / f"model_h{H}.keras").exists()}
scalers = {H: joblib.load(MODEL_DIR / f"scalers_h{H}.joblib") for H in models}
card_path = MODEL_DIR / "model_card.json"
card = json.loads(card_path.read_text(encoding="utf8")) if card_path.exists() else {}

app = FastAPI(title="Pronóstico multihorizonte de BTCUSDT con MLP",
              description="Proyecto Integrador 2.21 de Deep Learning (Uninorte). Predice el cierre diario de "
                          "Bitcoin 7, 14, 21 o 28 días adelante a partir de sus últimos cierres. "
                          "Uso académico: no es una recomendación de inversión.",
              version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "horizons": sorted(models)}


@app.get("/info", response_model=list[HorizonInfo])
def info():
    out = []
    for H in sorted(models):
        c = card.get("horizontes", {}).get(str(H), {})
        out.append(HorizonInfo(horizon=H, n_lags=int(scalers[H]["x"].n_features_in_),
                               arquitectura=c.get("arquitectura", ""), dropout_mask=c.get("dropout_mask", ""),
                               rmse_validacion_usd=c.get("RMSE_val_USD"), rmse_holdout_usd=c.get("RMSE_holdout_USD")))
    return out


@app.post("/predict", response_model=Prediction)
def predict(data: TimeSeriesData):
    H = data.horizon
    if H not in models:
        raise HTTPException(status_code=503, detail=f"No hay modelo cargado para el horizonte {H}")
    scaler_x, scaler_y = scalers[H]["x"], scalers[H]["y"]
    features = np.asarray(data.lags, dtype=float).reshape(1, -1)
    if features.shape[1] != scaler_x.n_features_in_:
        raise HTTPException(status_code=422, detail=f"Se esperaban {scaler_x.n_features_in_} lags para H={H}, "
                                                    f"se recibieron {features.shape[1]}")
    if not np.all(np.isfinite(features)):
        raise HTTPException(status_code=422, detail="Los lags deben ser números finitos")
    pred_scaled = models[H].predict(scaler_x.transform(features), verbose=0)
    prediction = scaler_y.inverse_transform(pred_scaled)[0]
    return Prediction(horizon=H, n_lags=features.shape[1], prediction=[float(v) for v in prediction])
