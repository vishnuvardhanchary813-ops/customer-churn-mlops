"""Shared pytest fixtures for model and API tests."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def customer_payload() -> dict:
    """A valid Telco customer matching the training feature schema."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 55.0,
        "TotalCharges": 660.0,
    }


@pytest.fixture
def api_client():
    """Test client that runs the app lifespan and loads the saved model."""
    with TestClient(app) as client:
        yield client
