"""Retornos, volatilidad histórica y partición con hold-out temporal final."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    """r_t = ln(P_t / P_{t-1});  sigma_t = std(r_{t-n+1..t}) * sqrt(365)  (anualizada; ddof=0 como en la guía)."""
    out = df[["Date", "Close"]].copy()
    out["LogReturn"] = np.log(out.Close / out.Close.shift(1))
    out["Volatility"] = out.LogReturn.rolling(vol_window).std(ddof=0) * np.sqrt(365)
    return out


@dataclass
class HoldoutSplit:
    n_total: int
    n_hold: int

    @property
    def n_cv(self) -> int:
        return self.n_total - self.n_hold


def holdout_split(n: int, frac: float, horizons) -> HoldoutSplit:
    """Reserva el tramo final: max(frac * n, 2 * max(H)) observaciones. No participa en CV, búsqueda ni scalers."""
    n_hold = max(int(n * frac), 2 * max(horizons))
    return HoldoutSplit(n, n_hold)


def holdout_windows(series: np.ndarray, n_cv: int, lag: int, H: int):
    """Ventanas cuyo objetivo (los H valores a predecir) cae completo dentro del hold-out.

    Las entradas pueden incluir los últimos valores de la serie de CV: eso es información pasada,
    disponible en el momento del pronóstico, y no filtra ningún objetivo del hold-out.
    Devuelve (X, y, idx_inicio_objetivo).
    """
    starts = np.arange(n_cv - lag, len(series) - lag - H + 1)
    X = np.stack([series[s:s + lag] for s in starts])
    y = np.stack([series[s + lag:s + lag + H] for s in starts])
    return X, y, starts + lag
