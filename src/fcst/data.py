"""Descarga de velas diarias de Binance (API pública, sin registro) y serie sintética para pruebas."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# La API principal (api.binance.com) responde 451 desde algunas regiones (p. ej. servidores de EE. UU.,
# como los de Colab). data-api.binance.vision es el espejo oficial de Binance solo para datos de mercado.
ENDPOINTS = ["https://data-api.binance.vision", "https://api.binance.com"]
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
              "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def _ms(date: str) -> int:
    return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def download_binance(symbol: str, interval: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    """Descarga velas [start, end) paginando de a 1000. Devuelve (DataFrame, endpoint usado)."""
    import requests

    last_err = None
    for base in ENDPOINTS:
        try:
            rows, start_ms, end_ms = [], _ms(start), _ms(end)
            while start_ms < end_ms:
                r = requests.get(f"{base}/api/v3/klines", timeout=30, params={
                    "symbol": symbol, "interval": interval, "startTime": start_ms,
                    "endTime": end_ms - 1, "limit": 1000})
                r.raise_for_status()
                data = r.json()
                if not data:
                    break
                rows.extend(data)
                start_ms = data[-1][0] + 1
                time.sleep(0.2)  # cortesía con el límite de peticiones
            df = pd.DataFrame(rows, columns=KLINE_COLS)
            df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
            df = df.rename(columns=str.capitalize)[["Date", "Open", "High", "Low", "Close", "Volume", "Trades"]]
            df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            return df.drop_duplicates("Date").sort_values("Date").reset_index(drop=True), base
        except Exception as e:  # noqa: BLE001 - se intenta el siguiente endpoint
            last_err = e
    raise RuntimeError(f"No se pudo descargar de Binance: {last_err}")


def synthetic_series(start: str, end: str, seed: int = 0) -> pd.DataFrame:
    """Serie de precios SINTÉTICA (GBM con cambios de régimen y volatilidad agrupada). Solo para pruebas."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D", inclusive="left")
    n = len(dates)
    sigma = np.empty(n)
    s = 0.03
    for t in range(n):  # volatilidad tipo GARCH(1,1) simplificada
        s = np.sqrt(1e-5 + 0.08 * (rng.standard_normal() * s) ** 2 + 0.9 * s ** 2)
        sigma[t] = s
    drift = np.where(np.arange(n) % 400 < 250, 0.0015, -0.001)
    r = drift + sigma * rng.standard_normal(n)
    close = 10_000 * np.exp(np.cumsum(r))
    return pd.DataFrame({"Date": dates, "Open": close, "High": close, "Low": close, "Close": close,
                         "Volume": rng.uniform(1e3, 1e4, n), "Trades": rng.integers(1e4, 1e5, n)})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prices(cfg) -> pd.DataFrame:
    """Lee el CSV crudo; si no existe, lo descarga (o genera la serie sintética en el perfil smoke)."""
    d = cfg["data"]
    path = cfg.data_csv
    manifest = path.with_suffix(".manifest.json")
    if not path.exists():
        if d["synthetic"]:
            df, src = synthetic_series(d["start"], d["end"], cfg["seed"]), "sintética (GBM con regímenes)"
        else:
            df, src = download_binance(d["symbol"], d["interval"], d["start"], d["end"])
        df.to_csv(path, index=False)
        manifest.write_text(json.dumps({
            "fuente": src, "simbolo": d["symbol"], "intervalo": d["interval"], "desde": d["start"],
            "hasta_excluido": d["end"], "filas": len(df), "primera_fecha": str(df.Date.min().date()),
            "ultima_fecha": str(df.Date.max().date()), "sha256": sha256(path),
            "descargado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2, ensure_ascii=False))
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def read_manifest(cfg) -> dict:
    p = cfg.data_csv.with_suffix(".manifest.json")
    return json.loads(p.read_text()) if p.exists() else {}
