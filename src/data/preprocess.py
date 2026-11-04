"""Clean Telco churn data and create reproducible train/test splits.

The split files retain categorical columns so that the model training step can
fit one complete Scikit-learn preprocessing-and-model pipeline on training
data only. This avoids fitting encoders or scalers before the split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_settings(config_path: Path | None = None) -> dict:
    """Read project paths and split settings from YAML configuration."""
    config_path = config_path or PROJECT_ROOT / "configs" / "config.yaml"
    params_path = PROJECT_ROOT / "params.yaml"
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    with params_path.open(encoding="utf-8") as stream:
        params = yaml.safe_load(stream) or {}
    return {**config, **params}


def load_and_clean_data(path: Path) -> pd.DataFrame:
    """Load the source CSV, validate its schema, and normalize key fields."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the IBM Telco CSV there or "
            "set paths.raw_data in configs/config.yaml."
        )

    data = pd.read_csv(path)
    required = {"customerID", "Churn", "TotalCharges"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data = data.drop_duplicates().copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.drop(columns="customerID")

    # Rows with no target cannot be used for supervised learning. Invalid or
    # blank TotalCharges remain NaN and are imputed inside the training pipeline.
    data = data.dropna(subset=["Churn"])
    data["Churn"] = data["Churn"].astype(str).str.strip().map({"No": 0, "Yes": 1})
    if data["Churn"].isna().any():
        values = sorted(data.loc[data["Churn"].isna(), "Churn"].astype(str).unique())
        raise ValueError("Churn must contain only 'Yes' or 'No'.")
    data["Churn"] = data["Churn"].astype("int8")
    return data


def split_and_save(
    data: pd.DataFrame,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Path, Path]:
    """Stratify by churn and write clean, unencoded splits to disk."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1")

    features = data.drop(columns="Churn")
    target = data["Churn"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    train = x_train.copy()
    train["Churn"] = y_train
    test = x_test.copy()
    test["Churn"] = y_test

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path


def main() -> None:
    """Run the configured data preparation stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional config YAML path")
    args = parser.parse_args()

    settings = read_settings(args.config)
    raw_path = PROJECT_ROOT / settings["paths"]["raw_data"]
    output_dir = PROJECT_ROOT / settings["paths"]["processed_dir"]
    data = load_and_clean_data(raw_path)
    train_path, test_path = split_and_save(
        data,
        output_dir,
        test_size=float(settings.get("test_size", 0.2)),
        random_state=int(settings.get("random_state", 42)),
    )
    print(f"Preprocessing complete: {train_path} and {test_path}")
    print(f"Rows: {len(data)} | Features: {data.shape[1] - 1}")


if __name__ == "__main__":
    main()
