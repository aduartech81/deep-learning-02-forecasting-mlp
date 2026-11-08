"""Pruebas de la API (usa los modelos guardados en app/ o en MODEL_DIR)."""
import pytest
from conftest import HAY_MODELOS, skip_sin_modelos
from fastapi.testclient import TestClient

if HAY_MODELOS:
    from app.api import app, models
    client = TestClient(app)

HORIZONS = [7, 14, 21, 28]


def n_lags(H):
    return {d["horizon"]: d["n_lags"] for d in client.get("/info").json()}[H]


@skip_sin_modelos
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["horizons"] == HORIZONS


@skip_sin_modelos
@pytest.mark.parametrize("H", HORIZONS)
def test_predict_endpoint(H):
    lags = [100_000.0 + 50 * i for i in range(n_lags(H))]
    r = client.post("/predict", json={"lags": lags, "horizon": H})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["prediction"], list)
    assert len(data["prediction"]) == H
    assert all(isinstance(v, float) for v in data["prediction"])


@skip_sin_modelos
def test_invalid_horizon():
    r = client.post("/predict", json={"lags": [1.0] * 7, "horizon": 10})
    assert r.status_code == 422


@skip_sin_modelos
@pytest.mark.parametrize("H", HORIZONS)
def test_wrong_number_of_lags(H):
    r = client.post("/predict", json={"lags": [1.0] * (n_lags(H) + 1), "horizon": H})
    assert r.status_code == 422


@skip_sin_modelos
def test_models_loaded():
    assert sorted(models) == HORIZONS
