"""Evaluate the selected serialized pipeline on the held-out test split."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET = "Churn"


def main() -> None:
    """Load the saved model, evaluate test data, and save a JSON report."""
    with (PROJECT_ROOT / "configs" / "config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]
    model_path = PROJECT_ROOT / config["paths"]["model"]
    test_path = processed_dir / "test.csv"
    for path in (model_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation input not found: {path}")

    test_data = pd.read_csv(test_path)
    if TARGET not in test_data:
        raise ValueError(f"Expected target column {TARGET!r} in {test_path}")
    X_test = test_data.drop(columns=TARGET)
    y_test = test_data[TARGET].astype(int)
    model = load(model_path)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    reports_dir = PROJECT_ROOT / config["paths"].get("reports_dir", "artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "test_metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Use the same SQLite tracking database as model training.
    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("customer-churn")
    with mlflow.start_run(run_name="best_model_test_evaluation"):
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
        mlflow.log_artifact(str(report_path), artifact_path="evaluation")

    print("Model evaluation completed")
    print("Test metrics: " + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
