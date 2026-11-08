"""Búsqueda exhaustiva de arquitecturas MLP (capas, neuronas, dropout por capa), en paralelo y reanudable.

Sigue la metodología de architectures_dropout.ipynb: para cada candidato se registran best_val_rmse,
last_train_rmse, gap = |last_train_rmse - best_val_rmse|, gap_ratio = gap / best_val_rmse, holdout_rmse
(solo como evidencia final) y las curvas de aprendizaje. Selección: candidatos con gap_ratio < 0,15,
ordenados por best_val_rmse (nunca por el hold-out).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .cv import search_split, windows
from .features import holdout_windows
from .models import Scalers, candidates

_TF_READY = False


def _init_tf_single_thread():
    """Cada proceso trabajador usa un hilo: con redes tan pequeñas, paralelizar candidatos rinde más."""
    global _TF_READY
    if not _TF_READY:
        import tensorflow as tf
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            pass
        _TF_READY = True


def _run_chunk(chunk, data, epochs, batch_size, dropout_rate, seed):
    _init_tf_single_thread()
    from .models import fit_keras, predict_keras
    Xtr, ytr, Xva, yva, Xho, yho = data
    out = []
    for arch, mask in chunk:
        t0 = time.time()
        model, tr, va = fit_keras(arch, mask, Xtr, ytr, Xva, yva, epochs, batch_size, dropout_rate, seed)
        yhat = predict_keras(model, Xho)
        best_val = float(np.min(va))
        out.append({
            "arquitectura": str(tuple(arch)), "dropout_mask": str(tuple(mask)), "capas": len(arch),
            "neuronas_total": int(sum(arch)), "n_dropout": int(sum(mask)), "params": int(model.count_params()),
            "best_val_rmse": best_val, "best_epoch": int(np.argmin(va)) + 1, "last_train_rmse": float(tr[-1]),
            "last_val_rmse": float(va[-1]), "gap": float(abs(tr[-1] - best_val)),
            "holdout_rmse": float(np.sqrt(np.mean((yho - yhat) ** 2))),
            "fit_s": time.time() - t0, "train_curve": [float(v) for v in tr], "val_curve": [float(v) for v in va],
        })
        del model
    return out


def prepare_search_data(series_cv, series_full, n_cv, lag, H, origin_frac, jump=1):
    """Split de búsqueda (Forward Chaining con origen en origin_frac) + hold-out, escalados con el train."""
    fold = search_split(n_cv, lag, H, origin_frac, jump)
    Xtr, ytr = windows(series_cv, fold.train, lag, H)
    Xva, yva = windows(series_cv, fold.val, lag, H)
    Xho, yho, _ = holdout_windows(series_full, n_cv, lag, H)
    sc = Scalers(Xtr, ytr)
    data = (sc.tx(Xtr), sc.ty(ytr), sc.tx(Xva), sc.ty(yva), sc.tx(Xho), sc.ty(yho))
    return data, fold


def run_search(series_cv, series_full, n_cv, lag, H, scfg: dict, out_dir: Path, seed=42, force=False):
    """Ejecuta (o reanuda) la búsqueda para un (lag, H). Guarda results/architecture_search_lag{L}_h{H}.csv
    y las curvas en results/search_curves/lag{L}_h{H}.json. Devuelve el DataFrame completo."""
    out_dir = Path(out_dir)
    csv = out_dir / f"architecture_search_lag{lag}_h{H}.csv"
    curves_path = out_dir / "search_curves" / f"lag{lag}_h{H}.json"
    if csv.exists() and curves_path.exists() and not force:
        return pd.read_csv(csv)
    from joblib import Parallel, delayed
    data, fold = prepare_search_data(series_cv, series_full, n_cv, lag, H, scfg["origin_frac"])
    cands = candidates(scfg["neurons"], scfg["max_layers"])
    n_jobs = scfg.get("n_jobs", -1)
    import os
    workers = os.cpu_count() if n_jobs in (-1, None) else n_jobs
    chunks = [cands[i::max(workers * 4, 1)] for i in range(max(workers * 4, 1))]
    chunks = [c for c in chunks if c]
    t0 = time.time()
    res = Parallel(n_jobs=workers, backend="loky")(
        delayed(_run_chunk)(c, data, scfg["epochs"], scfg["batch_size"], scfg["dropout_rate"], seed) for c in chunks)
    rows = [r for chunk in res for r in chunk]
    df = pd.DataFrame(rows)
    df["gap_ratio"] = df["gap"] / df["best_val_rmse"]
    df["lag"], df["H"] = lag, H
    df["n_train"], df["n_val"] = len(fold.train), len(fold.val)
    df = df.sort_values("best_val_rmse").reset_index(drop=True)
    curves_path.parent.mkdir(parents=True, exist_ok=True)
    curves_path.write_text(json.dumps({f"{r.arquitectura}|{r.dropout_mask}": [r.train_curve, r.val_curve]
                                       for r in df.itertuples()}))
    df = df.drop(columns=["train_curve", "val_curve"])
    df.to_csv(csv, index=False)
    print(f"lag={lag} H={H}: {len(df)} candidatos en {time.time() - t0:.0f} s")
    return df


def select_best(df: pd.DataFrame, gap_ratio_max: float = 0.15) -> pd.Series:
    """Candidatos balanceados (gap_ratio < umbral) ordenados por best_val_rmse; si ninguno cumple,
    se toma el de menor gap_ratio entre el 10 % con mejor validación (y se deja constancia)."""
    ok = df[df.gap_ratio < gap_ratio_max].sort_values("best_val_rmse")
    if len(ok):
        best = ok.iloc[0].copy()
        best["criterio"] = f"gap_ratio < {gap_ratio_max}"
        return best
    top = df.nsmallest(max(1, len(df) // 10), "best_val_rmse")
    best = top.sort_values("gap_ratio").iloc[0].copy()
    best["criterio"] = "ninguno con gap_ratio bajo el umbral: menor gap_ratio del 10 % con mejor validación"
    return best


def load_curves(out_dir: Path, lag: int, H: int) -> dict:
    return json.loads((Path(out_dir) / "search_curves" / f"lag{lag}_h{H}.json").read_text())
