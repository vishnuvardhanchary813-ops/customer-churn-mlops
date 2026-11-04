"""Tests for the FastAPI prediction service."""


def test_root_and_health(api_client):
    root_response = api_client.get("/")
    health_response = api_client.get("/health")

    assert root_response.status_code == 200
    assert root_response.json()["docs"] == "/docs"
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok", "model_loaded": True}


def test_model_info_reports_classifier(api_client):
    response = api_client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["loaded"] is True
    assert response.json()["estimator_type"] == "LogisticRegression"


def test_predict_returns_churn_label_and_probability(api_client, customer_payload):
    response = api_client.post("/predict", json=customer_payload)

    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] in (0, 1)
    assert result["churn"] is (result["prediction"] == 1)
    assert result["label"] in ("Churn", "No churn")
    assert 0.0 <= result["churn_probability"] <= 1.0


def test_predict_rejects_invalid_contract(api_client, customer_payload):
    customer_payload["Contract"] = "Weekly"
    response = api_client.post("/predict", json=customer_payload)

    assert response.status_code == 422
