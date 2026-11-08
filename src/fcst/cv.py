"""Validación cruzada temporal con timeseries-cv (tsxv.splitTrainVal).

Las tres funciones de tsxv comparten firma ``f(sequence, numInputs, numOutputs, numJumps)`` y devuelven
diccionarios ``X, y, Xcv, ycv`` indexados por fold. Para saber exactamente QUÉ ventanas usa cada fold
(y poder dibujar el esquema, purgar solapes y ubicar las predicciones en el tiempo), se llama a tsxv con
la secuencia de índices 0..n-1 en lugar de los valores: cada ventana devuelta empieza en su índice.

Comportamiento de tsxv que condiciona el diseño (verificado en el capítulo 2):

* ``forwardChaining`` y ``kFold`` crean un fold por cada salto y cada fold valida UNA sola ventana
  (con n ≈ 2 800 días y numJumps = 1 son ~2 700 folds). Entrenar una red por fold es inviable y el test
  BDS no se puede calcular con un solo residuo. Por eso se agrupan los folds consecutivos en ``n_blocks``
  bloques: en cada bloque se entrena una vez con las ventanas que son de entrenamiento en TODOS los folds
  del bloque, y se validan todas las ventanas de validación del bloque.
* ``groupKFold`` devuelve siempre 5 folds y se usa tal cual.

En todas las estrategias se purgan las ventanas de entrenamiento cuyo objetivo (los H valores a predecir)
se solapa con el objetivo de alguna ventana de validación del fold: sin esa purga el modelo vería durante el
entrenamiento los mismos valores que luego se le piden en validación.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tsxv.splitTrainVal import (split_train_val_forwardChaining, split_train_val_groupKFold,
                                split_train_val_kFold)

SPLIT_FUNCTIONS = {
    "groupKFold": split_train_val_groupKFold,
    "forwardChaining": split_train_val_forwardChaining,
    "kFold": split_train_val_kFold,
}
STRATEGY_LABEL = {"groupKFold": "Group K-Fold", "forwardChaining": "Forward Chaining", "kFold": "K-Fold"}


CACHE = Path(__file__).resolve().parents[2] / "cache"


def tsxv_window_starts(strategy: str, n: int, lag: int, H: int, jump: int = 1):
    """Llama a tsxv sobre los índices 0..n-1 y devuelve, por fold, (inicios_train, inicios_val).

    El resultado se guarda en cache/ (K-Fold con numJumps = 1 tarda segundos y usa >1 GB de RAM)."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"tsxv_{strategy}_n{n}_l{lag}_h{H}_j{jump}.npz"
    if f.exists():
        z = np.load(f)
        return [(z[f"t{k}"], z[f"v{k}"]) for k in range(int(z["nf"]))]
    out = _tsxv_call(strategy, n, lag, H, jump)
    np.savez_compressed(f, nf=len(out), **{f"t{k}": t for k, (t, _) in enumerate(out)},
                        **{f"v{k}": v for k, (_, v) in enumerate(out)})
    return out


def _tsxv_call(strategy, n, lag, H, jump):
    seq = np.arange(n, dtype=np.int32)
    X, _, Xcv, _ = SPLIT_FUNCTIONS[strategy](seq, lag, H, jump)
    out = []
    for k in sorted(X):
        if len(X[k]) == 0 or len(Xcv[k]) == 0:
            continue
        out.append((X[k][:, 0].astype(int), Xcv[k][:, 0].astype(int)))
    del X, Xcv
    return out


@dataclass
class Fold:
    k: int
    train: np.ndarray   # índices de inicio de ventana (entrenamiento)
    val: np.ndarray     # índices de inicio de ventana (validación)
    n_lib_folds: int    # cuántos folds de tsxv agrupa este fold
    purged: bool = True     # False si la purga dejaba muy pocas ventanas y se usó el fold tal cual
    overlap_frac: float = 0.0  # fracción de ventanas de train con objetivo solapado (si no se purgó)


def _purge(train: np.ndarray, val: np.ndarray, lag: int, H: int) -> np.ndarray:
    """Quita ventanas de train cuyo objetivo [s+lag, s+lag+H) se solapa con el objetivo de alguna de val."""
    if len(val) == 0:
        return train
    t0 = train + lag
    v0 = np.sort(val + lag)
    # para cada objetivo de train, ¿hay algún objetivo de val que empiece en (t0 - H, t0 + H)?
    lo = np.searchsorted(v0, t0 - H + 1, side="left")
    hi = np.searchsorted(v0, t0 + H, side="left")
    return train[hi <= lo]


def make_folds(strategy: str, n: int, lag: int, H: int, jump: int = 1, n_blocks: int = 5,
               min_train_frac: float = 0.5) -> list[Fold]:
    lib = tsxv_window_starts(strategy, n, lag, H, jump)
    if strategy == "groupKFold":
        return [_purged_fold(k, tr, va, 1, lag, H) for k, (tr, va) in enumerate(lib)]
    # forwardChaining / kFold: un fold de tsxv = una ventana de validación -> se agrupan en bloques
    origin = int(min_train_frac * n)
    lib = [(tr, va) for tr, va in lib if va[0] >= origin]
    groups = np.array_split(np.arange(len(lib)), n_blocks)
    folds = []
    for k, g in enumerate(groups):
        if len(g) == 0:
            continue
        val = np.concatenate([lib[i][1] for i in g])
        train = lib[g[0]][0]
        for i in g[1:]:  # ventanas que son de entrenamiento en todos los folds del bloque
            train = np.intersect1d(train, lib[i][0], assume_unique=True)
        train = np.setdiff1d(train, val, assume_unique=True)
        folds.append(_purged_fold(k, train, val, len(g), lag, H))
    return folds


MIN_TRAIN = 50


def _purged_fold(k, train, val, n_lib, lag, H) -> Fold:
    """Aplica la purga; si deja menos de MIN_TRAIN ventanas (pasa en groupKFold de tsxv cuando H es grande
    frente al espaciado entre ventanas de validación), conserva el fold tal cual y deja constancia de la fuga."""
    kept = _purge(train, val, lag, H)
    if len(kept) >= MIN_TRAIN:
        return Fold(k, kept, val, n_lib, True, 0.0)
    return Fold(k, train, val, n_lib, False, 1 - len(kept) / max(len(train), 1))


def search_split(n: int, lag: int, H: int, origin_frac: float, jump: int = 1) -> Fold:
    """Split único para la búsqueda de arquitectura (Forward Chaining con origen en origin_frac):
    entrenamiento = ventanas del fold de tsxv cuya validación empieza en el origen; validación = todas
    las ventanas de validación posteriores (siempre en el futuro respecto al entrenamiento)."""
    lib = tsxv_window_starts("forwardChaining", n, lag, H, jump)
    origin = int(origin_frac * n)
    idx = next(i for i, (_, va) in enumerate(lib) if va[0] >= origin)
    train = lib[idx][0]
    val = np.concatenate([va for _, va in lib[idx:]])
    return Fold(0, _purge(train, val, lag, H), val, len(lib) - idx)


def windows(series: np.ndarray, starts: np.ndarray, lag: int, H: int):
    """Construye (X, y) a partir de los índices de inicio de ventana."""
    X = np.stack([series[s:s + lag] for s in starts])
    y = np.stack([series[s + lag:s + lag + H] for s in starts])
    return X, y


def check_no_leak(fold: Fold, lag: int, H: int, strategy: str) -> dict:
    """Comprobaciones de integridad de un fold (se reportan en el capítulo 2)."""
    tr_end_target = fold.train + lag + H        # fin (excluido) del objetivo de cada ventana de train
    va_start_target = fold.val + lag
    return {
        "solape_objetivos": int(len(fold.train) - len(_purge(fold.train, fold.val, lag, H))),
        "train_todo_en_el_pasado": bool(len(fold.train) and tr_end_target.max() <= va_start_target.min()),
        "ventanas_train": int(len(fold.train)), "ventanas_val": int(len(fold.val)),
        "estrategia": strategy, "purgado": fold.purged, "frac_train_con_fuga": fold.overlap_frac,
    }
