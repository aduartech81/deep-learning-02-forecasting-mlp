# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 8. Despliegue (MLOps): API, contenedor, pruebas e integración continua
#
# **Objetivo del capítulo.** Dejar el modelo listo para usarse fuera del notebook y comprobar, desde aquí mismo,
# que la API responde como se especifica.
#
# | Pieza | Archivo | Qué hace |
# |---|---|---|
# | API | `app/api.py`, `app/schemas.py` | FastAPI: `POST /predict` recibe `lags` y `horizon ∈ {7, 14, 21, 28}` y devuelve H predicciones; `GET /info` informa cuántos lags espera cada horizonte; `GET /health`. |
# | Modelos | `app/model_h{H}.keras`, `app/scalers_h{H}.joblib`, `app/model_card.json` | Mejor MLP por horizonte (capítulo 5) y sus escaladores ajustados solo con entrenamiento. |
# | Contenedor | `Dockerfile` | Imagen `python:3.12-slim` con TensorFlow CPU; sirve la API con uvicorn en el puerto 8000. |
# | Pruebas | `tests/` | Forma de la salida de cada modelo, coherencia modelo–escaladores, API (códigos 200/422), ausencia de fuga en la validación y métricas. |
# | CI | `.github/workflows/ci.yml` | En cada *push*: flake8 + pytest. |

# %%
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from IPython.display import Markdown, display

from fcst.nbutils import setup

cfg, RES, FIG = setup()
APP = cfg.path("app_dir")
os.environ["MODEL_DIR"] = str(APP)
from fastapi.testclient import TestClient  # noqa: E402

from app.api import app  # noqa: E402

client = TestClient(app)
card = json.loads((APP / "model_card.json").read_text(encoding="utf8"))

# %% [markdown]
# ## 8.1 La API en funcionamiento

# %%
print(client.get("/health").json())
pd.DataFrame(client.get("/info").json())

# %% [markdown]
# Pronóstico con los **últimos cierres conocidos** de la serie (los del final del hold-out), para cada horizonte:

# %%
last = card["ultimos_cierres"]
rows = []
for H in [7, 14, 21, 28]:
    n = {d["horizon"]: d["n_lags"] for d in client.get("/info").json()}[H]
    r = client.post("/predict", json={"lags": last[-n:], "horizon": H})
    p = r.json()["prediction"]
    rows.append({"H": H, "lags usados": n, "estado HTTP": r.status_code, "día +1": p[0], f"día +H": p[-1],
                 "mín": min(p), "máx": max(p)})
display(Markdown(f"Último cierre conocido: **{card['ultimo_cierre_conocido']['close']:,.0f} USD** "
                 f"({card['ultimo_cierre_conocido']['fecha']})."))
pd.DataFrame(rows).style.format({c: "{:,.0f}" for c in ["día +1", "día +H", "mín", "máx"]}).hide(axis="index")

# %%
bad = client.post("/predict", json={"lags": [1.0] * 7, "horizon": 10})
print("Horizonte inválido ->", bad.status_code, bad.json()["detail"][0]["msg"])
bad2 = client.post("/predict", json={"lags": [1.0] * 3, "horizon": 7})
print("Número de lags incorrecto ->", bad2.status_code, bad2.json()["detail"])

# %% [markdown]
# ## 8.2 Pruebas unitarias

# %%
out = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, capture_output=True, text=True,
                     env={**os.environ, "MODEL_DIR": str(APP), "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}"})
print(out.stdout[-1500:])
assert out.returncode == 0, out.stderr[-2000:]

# %% [markdown]
# ## 8.3 Uso local y con Docker
#
# ```bash
# # Local
# pip install -r requirements-api.txt
# uvicorn app.api:app --port 8000
#
# # Docker
# docker build -t btc-forecast-mlp .
# docker run -p 8000:8000 btc-forecast-mlp
#
# # Consulta (horizonte de 14 días; envíe tantos lags como indique /info para ese horizonte)
# curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
#      -d '{"lags": [...], "horizon": 14}'
# ```
#
# La documentación interactiva queda en `http://localhost:8000/docs`. El modelo es un ejercicio académico de
# pronóstico: no constituye una recomendación de inversión.
