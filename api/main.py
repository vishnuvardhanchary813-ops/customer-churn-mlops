"""FastAPI endpoints for customer churn predictions."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

from src.models.predict import load_model


logger = logging.getLogger(__name__)


class CustomerInput(BaseModel):
    """Validated Telco customer fields expected by the trained pipeline."""

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: Annotated[int, Field(ge=0, le=1)]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: Annotated[int, Field(ge=0, le=72)]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: Annotated[float, Field(ge=0)]
    TotalCharges: Annotated[float | None, Field(ge=0)] = None


class PredictionResponse(BaseModel):
    prediction: int
    churn: bool
    label: Literal["Churn", "No churn"]
    churn_probability: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the API starts; report load failures via health."""
    try:
        app.state.model = load_model()
        logger.info("Churn model loaded")
    except Exception:
        app.state.model = None
        logger.exception("Could not load churn model")
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Predicts customer churn using the trained Scikit-learn pipeline.",
    lifespan=lifespan,
)


def get_loaded_model(request: Request) -> Pipeline:
    """Return the application model or an actionable service error."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable. Train the model and restart the API.",
        )
    return model


@app.get("/")
def root() -> dict[str, str]:
    """API welcome and documentation location."""
    return {"service": "Customer Churn Prediction API", "docs": "/docs"}


@app.get("/health")
def health(request: Request) -> dict[str, str | bool]:
    """Report API and model readiness."""
    ready = getattr(request.app.state, "model", None) is not None
    return {"status": "ok" if ready else "degraded", "model_loaded": ready}


@app.get("/model-info")
def model_info(request: Request) -> dict[str, str | int | bool]:
    """Describe the loaded estimator and its input feature count."""
    model = get_loaded_model(request)
    estimator = model.named_steps.get("classifier", model.steps[-1][1])
    return {
        "loaded": True,
        "pipeline_type": type(model).__name__,
        "estimator_type": type(estimator).__name__,
        "input_features": int(model.n_features_in_),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput, request: Request) -> PredictionResponse:
    """Return the predicted churn class and its probability."""
    model = get_loaded_model(request)
    try:
        features = pd.DataFrame([customer.model_dump()])
        features["TotalCharges"] = pd.to_numeric(
            features["TotalCharges"], errors="coerce"
        )
        prediction = int(model.predict(features)[0])
        classes = list(model.classes_)
        churn_index = classes.index(1)
        probability = float(model.predict_proba(features)[0][churn_index])
        return PredictionResponse(
            prediction=prediction,
            churn=bool(prediction),
            label="Churn" if prediction == 1 else "No churn",
            churn_probability=probability,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Check API logs for details.",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
