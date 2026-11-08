"""Entrenamiento fold a fold (3 estrategias x lag x horizonte) y modelo final con hold-out."""
from __future__ import annotations

import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .cv import make_folds, windows
from .features import holdout_windows
from .metrics import per_step, summary
from .models import Scalers, fit_keras, fit_sklearn_baseline, naive_forecast, predict_keras


def parse_arch(s):
    return tuple(ast.literal_eval(s)) if isinstance(s, str) else tuple(s)


def run_cv(series_cv: np.ndarray, strategy: str, lag: int, H: int, arch, mask, epochs: int, cfg,
           seed: int = 42, models=("mlp", "sklearn", "naive")):
    """Devuelve (tabla por fold, tabla por paso h, predicciones de validación por fold).

    Métricas en unidades originales (USD), siempre sobre VALIDACIÓN. Para cada fold se guardan las
    predicciones de train y de validación (todas las muestras del fold)."""
    c = cfg["cv"]
    folds = make_folds(strategy, len(series_cv), lag, H, c["n_steps_jump"], c["n_blocks"], c["min_train_frac"])
    rows, steps, preds = [], [], {}
    for f in folds:
        Xtr, ytr = windows(series_cv, f.train, lag, H)
        Xva, yva = windows(series_cv, f.val, lag, H)
        sc = Scalers(Xtr, ytr)
        out = {}
        if "mlp" in models:
            t0 = time.time()
            m, tr_curve, va_curve = fit_keras(arch, mask, sc.tx(Xtr), sc.ty(ytr), sc.tx(Xva), sc.ty(yva),
                                              epochs, cfg["train"]["batch_size"], cfg["search"]["dropout_rate"], seed)
            out["mlp"] = (sc.inv_y(predict_keras(m, sc.tx(Xtr))), sc.inv_y(predict_keras(m, sc.tx(Xva))), time.time() - t0)
        if "sklearn" in models:
            t0 = time.time()
            sk = fit_sklearn_baseline(sc.tx(Xtr), sc.ty(ytr), seed)
            out["sklearn"] = (sc.inv_y(sk.predict(sc.tx(Xtr))), sc.inv_y(sk.predict(sc.tx(Xva))), time.time() - t0)
        if "naive" in models:
            out["naive"] = (naive_forecast(Xtr, H), naive_forecast(Xva, H), 0.0)
        for name, (ptr, pva, secs) in out.items():
            s = summary(yva, pva)
            s_tr = summary(ytr, ptr)
            rows.append({"estrategia": strategy, "lag": lag, "H": H, "modelo": name, "fold": f.k + 1,
                         "n_train": len(f.train), "n_val": len(f.val), "folds_tsxv": f.n_lib_folds,
                         "val_desde": int(f.val.min() + lag), "val_hasta": int(f.val.max() + lag + H - 1),
                         **s, "RMSE_train": s_tr["RMSE"], "fit_s": secs})
            ps = per_step(yva, pva)
            ps.insert(0, "fold", f.k + 1)
            ps.insert(0, "modelo", name)
            steps.append(ps)
            preds[(name, f.k + 1)] = {"train_idx": f.train + lag, "val_idx": f.val + lag,
                                      "y_train": ytr, "p_train": ptr, "y_val": yva, "p_val": pva}
    per_fold = pd.DataFrame(rows)
    per_h = pd.concat(steps, ignore_index=True)
    per_h.insert(0, "H", H)
    per_h.insert(0, "lag", lag)
    per_h.insert(0, "estrategia", strategy)
    return per_fold, per_h, preds


def fold_table(per_fold: pd.DataFrame, model="mlp") -> pd.DataFrame:
    """Formato pedido: filas = folds + 'Media ± Std'; columnas = métricas agregadas + BDS p-valor h1."""
    cols = ["MAPE", "MAE", "RMSE", "MSE", "BDS_pvalue_h1"]
    t = per_fold[per_fold.modelo == model].set_index("fold")[cols].copy()
    t.index = [f"Fold {i}" for i in t.index]
    m, s = t.mean(), t.std()
    t.loc["Media"] = m
    t.loc["Std"] = s
    return t


def fit_final(series_full: np.ndarray, n_cv: int, lag: int, H: int, arch, mask, epochs: int, cfg, seed=42):
    """Modelo final: entrena con TODAS las ventanas cuya salida cae antes del hold-out (scalers con esas
    ventanas) y evalúa una sola vez en el hold-out. Devuelve (modelo, scalers, resultados)."""
    series_cv = series_full[:n_cv]
    starts = np.arange(0, n_cv - lag - H + 1)
    Xtr, ytr = windows(series_cv, starts, lag, H)
    Xho, yho, ho_idx = holdout_windows(series_full, n_cv, lag, H)
    sc = Scalers(Xtr, ytr)
    m, tr_curve, _ = fit_keras(arch, mask, sc.tx(Xtr), sc.ty(ytr), None, None, epochs,
                               cfg["train"]["batch_size"], cfg["search"]["dropout_rate"], seed)
    p_tr = sc.inv_y(predict_keras(m, sc.tx(Xtr)))
    p_ho = sc.inv_y(predict_keras(m, sc.tx(Xho)))
    p_naive = naive_forecast(Xho, H)
    res = {"train_idx": starts + lag, "y_train": ytr, "p_train": p_tr, "hold_idx": ho_idx, "y_hold": yho,
           "p_hold": p_ho, "p_naive": p_naive, "summary_mlp": summary(yho, p_ho),
           "summary_naive": summary(yho, p_naive), "per_h_mlp": per_step(yho, p_ho),
           "per_h_naive": per_step(yho, p_naive), "train_curve": tr_curve}
    return m, sc, res


def save_preds(preds: dict, path: Path):
    """Guarda las predicciones (npz comprimido) para gráficas posteriores."""
    flat = {}
    for (name, k), d in preds.items():
        for key, v in d.items():
            flat[f"{name}__{k}__{key}"] = v
    np.savez_compressed(path, **flat)


def load_preds(path: Path) -> dict:
    z = np.load(path)
    out = {}
    for key in z.files:
        name, k, field = key.split("__")
        out.setdefault((name, int(k)), {})[field] = z[key]
    return out


def write_json(obj, path: Path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float))
