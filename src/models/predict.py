"""Model artifact loading shared by online and offline prediction code."""

from __future__ import annotations

from pathlib import Path

import yaml
from joblib import load
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_model() -> Pipeline:
    """Load the fitted preprocessing-and-classifier pipeline from config."""
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    model_path = PROJECT_ROOT / config["paths"]["model"]
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run `python -m src.models.train` first."
        )
    model = load(model_path)
    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected a Scikit-learn Pipeline at {model_path}")
    return model
