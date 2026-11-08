import numpy as np

from fcst.metrics import bds_pvalue, per_step, summary


def test_per_step_perfect_prediction():
    y = np.arange(1, 41, dtype=float).reshape(10, 4)
    ps = per_step(y, y)
    assert np.allclose(ps[["MAPE", "MAE", "MSE", "RMSE"]].values, 0)


def test_bds_iid_noise_not_rejected_often():
    rng = np.random.default_rng(1)
    ps = [bds_pvalue(rng.standard_normal(400)) for _ in range(20)]
    assert np.mean(np.array(ps) > 0.05) > 0.7


def test_summary_keys():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((100, 7)) + 10
    s = summary(y, y + 0.1)
    assert {"MAPE", "MAE", "MSE", "RMSE", "BDS_pvalue_h1"} <= set(s)
