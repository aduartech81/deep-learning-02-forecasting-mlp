"""Pruebas de los modelos guardados: forma de la salida y escaladores coherentes."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pytest

from conftest import skip_sin_modelos
import tensorflow as tf

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parents[1] / "app"))
HORIZONS = [7, 14, 21, 28]


@skip_sin_modelos
@pytest.mark.parametrize("H", HORIZONS)
def test_model_prediction_shape(H):
    model = tf.keras.models.load_model(MODEL_DIR / f"model_h{H}.keras")
    n_lags = model.input_shape[1]
    y_pred = model.predict(np.random.rand(5, n_lags), verbose=0)
    assert y_pred.shape == (5, H)


@skip_sin_modelos
@pytest.mark.parametrize("H", HORIZONS)
def test_prediction_values(H):
    model = tf.keras.models.load_model(MODEL_DIR / f"model_h{H}.keras")
    y_pred = model.predict(np.random.rand(1, model.input_shape[1]), verbose=0)[0]
    assert len(y_pred) == H
    assert np.all(np.isfinite(y_pred))


@skip_sin_modelos
@pytest.mark.parametrize("H", HORIZONS)
def test_scalers_match_model(H):
    model = tf.keras.models.load_model(MODEL_DIR / f"model_h{H}.keras")
    sc = joblib.load(MODEL_DIR / f"scalers_h{H}.joblib")
    assert sc["x"].n_features_in_ == model.input_shape[1]
    assert sc["y"].n_features_in_ == H


@skip_sin_modelos
def test_model_card():
    card = json.loads((MODEL_DIR / "model_card.json").read_text(encoding="utf8"))
    assert set(card["horizontes"]) == {str(h) for h in HORIZONS}


@skip_sin_modelos
@pytest.mark.parametrize("H", HORIZONS)
def test_numpy_backend_matches_keras(H):
    """La inferencia sin TensorFlow (pesos exportados) debe dar lo mismo que el modelo Keras."""
    from app.api import NumpyMLP
    path = MODEL_DIR / f"weights_h{H}.npz"
    if not path.exists():
        pytest.skip("Sin pesos exportados (scripts/export_numpy.py)")
    model = tf.keras.models.load_model(MODEL_DIR / f"model_h{H}.keras")
    X = np.random.default_rng(H).standard_normal((50, model.input_shape[1])).astype("float32")
    np.testing.assert_allclose(NumpyMLP(path).predict(X), model.predict(X, verbose=0), rtol=1e-4, atol=1e-4)
