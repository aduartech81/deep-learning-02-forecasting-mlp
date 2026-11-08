"""Exporta los pesos de los MLP de app/model_h{H}.keras a app/weights_h{H}.npz para inferencia sin TensorFlow.

En inferencia el MLP es solo una cadena de capas densas (ReLU en las ocultas, lineal en la salida); el Dropout no
actúa al predecir. La API puede usar estos pesos con INFERENCE_BACKEND=numpy (imagen ligera para Render).
"""
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402

APP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "app"
for H in (7, 14, 21, 28):
    m = tf.keras.models.load_model(APP / f"model_h{H}.keras")
    dense = [layer for layer in m.layers if isinstance(layer, tf.keras.layers.Dense)]
    arrays = {}
    for i, layer in enumerate(dense):
        W, b = layer.get_weights()
        arrays[f"W{i}"], arrays[f"b{i}"] = W.astype(np.float32), b.astype(np.float32)
        act = layer.get_config()["activation"]
        assert act == ("linear" if i == len(dense) - 1 else "relu"), (H, i, act)
    np.savez_compressed(APP / f"weights_h{H}.npz", n_layers=len(dense), **arrays)
    print(f"H={H}: {len(dense)} capas densas exportadas")
