"""Modelos Pydantic de entrada y salida de la API."""
from typing import Literal

from pydantic import BaseModel, Field

Horizon = Literal[7, 14, 21, 28]


class TimeSeriesData(BaseModel):
    lags: list[float] = Field(..., description="Últimos cierres observados, del más antiguo al más reciente. "
                                               "Su número debe ser el lag del modelo de ese horizonte (ver /info).")
    horizon: Horizon = Field(7, description="Horizonte de predicción en días: 7, 14, 21 o 28.")

    model_config = {"json_schema_extra": {"examples": [{"lags": [100000.0] * 7, "horizon": 7}]}}


class Prediction(BaseModel):
    horizon: int
    n_lags: int
    prediction: list[float] = Field(..., description="H predicciones (días t+1 ... t+H).")


class HorizonInfo(BaseModel):
    horizon: int
    n_lags: int
    arquitectura: str
    dropout_mask: str
    rmse_validacion_usd: float | None = None
    rmse_holdout_usd: float | None = None
