"""Métricas por paso de horizonte y test BDS de independencia de residuos."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import bds


def per_step(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Métricas para cada paso h = 1..H (columnas de y)."""
    e = y_true - y_pred
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(e) / np.abs(y_true)
    return pd.DataFrame({
        "h": np.arange(1, y_true.shape[1] + 1),
        "MAPE": 100 * np.nanmean(ape, axis=0),
        "MAE": np.mean(np.abs(e), axis=0),
        "MSE": np.mean(e ** 2, axis=0),
        "RMSE": np.sqrt(np.mean(e ** 2, axis=0)),
    })


def bds_pvalue(resid: np.ndarray, max_dim: int = 2) -> float:
    """p-valor del test BDS (dimensión de inmersión 2) sobre los residuos.

    Se usa statsmodels.tsa.stattools.bds: el borrador de la guía importa ``bds`` desde
    ``arch.bootstrap``, pero ese módulo no lo incluye. p > 0,05 -> no se rechaza que los residuos sean iid.
    """
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    if len(resid) < 30 or np.std(resid) == 0:
        return float("nan")
    _, p = bds(resid, max_dim=max_dim)
    return float(np.atleast_1d(p)[0])


def summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Métricas agregadas (promedio sobre h = 1..H) + BDS sobre los residuos de h = 1."""
    ps = per_step(y_true, y_pred)
    return {"MAPE": ps.MAPE.mean(), "MAE": ps.MAE.mean(), "MSE": ps.MSE.mean(), "RMSE": ps.RMSE.mean(),
            "RMSE_h1": ps.RMSE.iloc[0], "RMSE_hH": ps.RMSE.iloc[-1],
            "BDS_pvalue_h1": bds_pvalue(y_true[:, 0] - y_pred[:, 0])}
