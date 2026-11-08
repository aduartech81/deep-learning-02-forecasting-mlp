import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest  # noqa: E402

MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "app"))
HAY_MODELOS = all((MODEL_DIR / f"model_h{h}.keras").exists() for h in (7, 14, 21, 28))
skip_sin_modelos = pytest.mark.skipif(not HAY_MODELOS,
                                      reason="Aún no hay modelos entrenados en app/ (se generan en el capítulo 5)")
