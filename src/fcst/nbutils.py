"""Utilidades de los notebooks: configuración, estilo de gráficos y guardado de figuras."""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt  # noqa: F401  (se reexporta para los notebooks)

from .config import load_config

# Paleta categórica validada (orden fijo; el color sigue a la entidad, no al rango)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SEQ_CMAP = mpl.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ)
STRAT_COLOR = {"groupKFold": SERIES[2], "forwardChaining": SERIES[0], "kFold": SERIES[1]}
LAG_COLOR = {7: SEQ[1], 14: SEQ[2], 21: SEQ[3], 28: SEQ[4]}
MODEL_COLOR = {"mlp": SERIES[0], "sklearn": SERIES[3], "naive": MUTED}
PART_COLOR = {"train": SERIES[0], "val": SERIES[1], "hold": SERIES[2]}


def style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 1.8, "lines.markersize": 6, "figure.dpi": 110, "savefig.dpi": 150,
        "axes.prop_cycle": mpl.cycler(color=SERIES), "font.family": "sans-serif",
    })


def setup(profile=None):
    """Carga configuración, aplica estilo y devuelve (cfg, carpeta de resultados, carpeta de figuras)."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    style()
    cfg = load_config(profile)
    res, fig = cfg.path("results_dir"), cfg.path("figures_dir")
    print(f"Perfil: {cfg.profile}  |  resultados en: {res}")
    if cfg.profile == "smoke":
        print("ATENCIÓN: perfil 'smoke' con una serie SINTÉTICA. Los números no tienen valor empírico.")
    return cfg, res, fig


def savefig(fig, figdir: Path, name: str):
    fig.tight_layout()
    fig.savefig(Path(figdir) / f"{name}.png", bbox_inches="tight")
