"""MLP multisalida en Keras (modelo principal), MLPRegressor (baseline) y pronóstico ingenuo."""
from __future__ import annotations

import itertools
import os

import numpy as np
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def architectures(neurons, max_layers):
    """Todas las combinaciones con repetición: itertools.product(neuronas, repeat=num_capas)."""
    return [c for k in range(1, max_layers + 1) for c in itertools.product(neurons, repeat=k)]


def dropout_masks(n_layers):
    """Las 2^n máscaras booleanas de dropout (activar o no Dropout después de cada capa oculta)."""
    return [tuple(bool((m >> i) & 1) for i in range(n_layers)) for m in range(2 ** n_layers)]


def candidates(neurons, max_layers):
    return [(a, m) for a in architectures(neurons, max_layers) for m in dropout_masks(len(a))]


def build_mlp(arch, mask, input_dim, n_outputs, dropout_rate=0.5, seed=0):
    """Input(lag) -> [Dense(units, relu) (+ Dropout)] x capas -> Dense(H). Adam, MSE, RMSE."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(input_dim,))])
    for units, use_do in zip(arch, mask):
        model.add(tf.keras.layers.Dense(units, activation="relu"))
        if use_do:
            model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(n_outputs))
    # steps_per_execution solo agrupa pasos en una llamada (menos overhead de Python); no cambia el cálculo
    model.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
                  steps_per_execution=16)
    return model


class Scalers:
    """StandardScaler para X y para y, ajustados SOLO con las ventanas de entrenamiento."""

    def __init__(self, X, y):
        self.x = StandardScaler().fit(X)
        self.y = StandardScaler().fit(y)

    def tx(self, X):
        return self.x.transform(X)

    def ty(self, y):
        return self.y.transform(y)

    def inv_y(self, y):
        return self.y.inverse_transform(y)


def fit_keras(arch, mask, Xtr, ytr, Xva=None, yva=None, epochs=30, batch_size=32, dropout_rate=0.5, seed=0):
    """Entrena sin barajar (orden temporal). Devuelve (modelo, curva_train_rmse, curva_val_rmse)."""
    model = build_mlp(arch, mask, Xtr.shape[1], ytr.shape[1], dropout_rate, seed)
    val = (Xva, yva) if Xva is not None else None
    h = model.fit(Xtr, ytr, validation_data=val, epochs=epochs, batch_size=batch_size, shuffle=False, verbose=0)
    return model, h.history["rmse"], h.history.get("val_rmse", [])


def predict_keras(model, X):
    return model.predict(X, verbose=0, batch_size=1024)


def fit_sklearn_baseline(Xtr, ytr, seed=0):
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed).fit(Xtr, ytr)


def naive_forecast(X, H):
    """Pronóstico ingenuo (paseo aleatorio): repetir el último valor observado durante los H pasos."""
    return np.repeat(X[:, -1:], H, axis=1)
