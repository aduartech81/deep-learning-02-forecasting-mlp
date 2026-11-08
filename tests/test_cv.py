"""Pruebas de la validación temporal: sin fuga en Forward Chaining, purga correcta y ventanas bien formadas."""
import numpy as np
import pytest

from fcst.cv import _purge, make_folds, search_split, tsxv_window_starts, windows
from fcst.features import holdout_split, holdout_windows

N = 700


@pytest.mark.parametrize("lag,H", [(7, 7), (14, 28), (28, 14)])
def test_forward_chaining_trains_on_the_past(lag, H):
    for f in make_folds("forwardChaining", N, lag, H, 1, 5, 0.5):
        assert (f.train + lag + H).max() <= (f.val + lag).min()


@pytest.mark.parametrize("strategy", ["groupKFold", "forwardChaining", "kFold"])
def test_tsxv_windows_inside_series(strategy):
    for tr, va in tsxv_window_starts(strategy, N, 14, 14):
        assert tr.min() >= 0 and va.max() + 14 + 14 <= N


def test_purge_removes_overlapping_targets():
    train = np.arange(0, 100)
    kept = _purge(train, np.array([50]), lag=7, H=7)
    assert not np.any((kept > 50 - 7) & (kept < 50 + 7))
    assert 0 in kept and 99 in kept


def test_purged_folds_have_no_target_overlap():
    for st in ["forwardChaining", "kFold", "groupKFold"]:
        for f in make_folds(st, N, 7, 7, 1, 5, 0.5):
            if f.purged:
                assert len(_purge(f.train, f.val, 7, 7)) == len(f.train)


def test_windows_and_holdout():
    s = np.arange(200.0)
    X, y = windows(s, np.array([0, 5]), 7, 3)
    assert X.shape == (2, 7) and y.shape == (2, 3) and y[1, 0] == 12
    sp = holdout_split(200, 0.1, [7, 14, 21, 28])
    assert sp.n_hold == 56  # nunca menor a 2 * max(H)
    Xh, yh, idx = holdout_windows(s, sp.n_cv, 7, 7)
    assert idx.min() == sp.n_cv and (yh[:, -1] < 200).all()


def test_search_split_is_temporal():
    f = search_split(N, 14, 14, 0.8)
    assert (f.train + 14 + 14).max() <= (f.val + 14).min()
