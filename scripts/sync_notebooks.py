"""Convierte nb_src/*.py (formato jupytext percent, fuente de verdad) en notebooks/*.ipynb sin salidas,
solo si el .py es más nuevo que el .ipynb o este no existe (no borra salidas de corridas ya hechas)."""
from pathlib import Path

import jupytext

ROOT = Path(__file__).resolve().parents[1]
for src in sorted((ROOT / "nb_src").glob("[0-9][0-9]_*.py")):
    dst = ROOT / "notebooks" / f"{src.stem}.ipynb"
    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
        nb = jupytext.read(src)
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        jupytext.write(nb, dst)
        print("sincronizado:", dst.name)
