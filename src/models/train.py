"""Train candidate churn models, track validation runs, and save the winner."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from joblib import dump
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.features.build_features import build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET = "Churn"


def load_settings() -> tuple[dict, dict]:
    """Load project paths and modeling parameters."""
    with (PROJECT_ROOT / "configs" / "config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as stream:
        params = yaml.safe_load(stream) or {}
    return config, params


def make_candidates(random_state: int) -> dict:
    """Create the three requested classifier candidates."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def calculate_metrics(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict:
    """Calculate the requested binary classification metrics."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
    }


def main() -> None:
    """Fit candidates, choose by validation F1, refit, and save the pipeline."""
    config, params = load_settings()
    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]
    train_path = processed_dir / "train.csv"
    if not train_path.is_file():
        raise FileNotFoundError(f"Training data not found: {train_path}. Run preprocessing first.")

    training_data = pd.read_csv(train_path)
    if TARGET not in training_data:
        raise ValueError(f"Expected target column {TARGET!r} in {train_path}")
    X = training_data.drop(columns=TARGET)
    y = training_data[TARGET].astype(int)
    random_state = int(params.get("random_state", config.get("random_state", 42)))
    validation_size = float(params.get("validation_size", 0.2))
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=random_state,
        stratify=y,
    )

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
    candidates = make_candidates(random_state)

    # SQLite avoids MLflow's deprecated local file-store backend.
    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("customer-churn")
    best_name: str | None = None
    best_f1 = -1.0

    for name, classifier in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_columns, categorical_columns)),
                ("classifier", classifier),
            ]
        )
        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {
                    "model": name,
                    "random_state": random_state,
                    "validation_size": validation_size,
                    "numeric_features": len(numeric_columns),
                    "categorical_features": len(categorical_columns),
                }
            )
            pipeline.fit(X_fit, y_fit)
            metrics = calculate_metrics(pipeline, X_valid, y_valid)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("selection_metric", "f1")
            signature = infer_signature(X_valid, pipeline.predict(X_valid))
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                signature=signature,
                input_example=X_valid.head(3),
            )
            print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if metrics["f1"] > best_f1:
                best_name = name
                best_f1 = metrics["f1"]

    assert best_name is not None
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_columns, categorical_columns)),
            ("classifier", make_candidates(random_state)[best_name]),
        ]
    )
    final_pipeline.fit(X, y)

    model_path = PROJECT_ROOT / config["paths"]["model"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(final_pipeline, model_path)
    validation_summary = {
        "selected_model": best_name,
        "selection_metric": "f1",
        "validation_f1": best_f1,
        "training_rows": int(len(training_data)),
        "features": list(X.columns),
    }
    summary_path = PROJECT_ROOT / config["paths"].get("reports_dir", "artifacts/reports") / Path("training_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")
    print(f"Best model selected: {best_name} (validation F1={best_f1:.4f})")
    print(f"Model saved successfully: {model_path}")


if __name__ == "__main__":
    main()
