"""Tests that the saved model artifact is a usable prediction pipeline."""

import pandas as pd

from src.models.predict import load_model


def test_saved_pipeline_returns_binary_prediction_and_probability(customer_payload):
    model = load_model()
    features = pd.DataFrame([customer_payload])
    features["TotalCharges"] = pd.to_numeric(features["TotalCharges"])

    prediction = model.predict(features)
    probabilities = model.predict_proba(features)

    assert int(prediction[0]) in {0, 1}
    assert probabilities.shape == (1, 2)
    assert 0.0 <= float(probabilities[0, 1]) <= 1.0
