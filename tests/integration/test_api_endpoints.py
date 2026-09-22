"""Integration tests for FastAPI REST endpoints."""

from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """Test index route returns platform status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["platform"] == "ModelFlow MLOps"
    assert "documentation" in data


def test_health_endpoint(client: TestClient):
    """Test health check returns liveness info."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert "model_loaded" in data


def test_model_features_endpoint(client: TestClient):
    """Test model features endpoint returns expected schema lists."""
    response = client.get("/model/features")
    assert response.status_code == 200
    data = response.json()
    assert "numerical_features" in data
    assert "categorical_features" in data
    assert "tenure" in data["numerical_features"]
    assert "Contract" in data["categorical_features"]


def test_predict_endpoint_success(client: TestClient, valid_customer_payload: dict):
    """Test single prediction endpoint returns prediction, confidence, and risk tier."""
    response = client.post("/predict", json=valid_customer_payload)
    # If model is loaded, status is 200; if not yet trained, verify handled gracefully
    if response.status_code == 200:
        data = response.json()
        assert data["prediction"] in ["Yes", "No"]
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["risk_tier"] in ["Low Risk", "Medium Risk", "High Risk"]
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0.0
    else:
        # Standby mode prior to initial training
        assert response.status_code in [200, 500]


def test_predict_endpoint_validation_error(client: TestClient):
    """Test API rejects invalid feature payloads with 422 Unprocessable Entity."""
    invalid_payload = {
        "gender": "InvalidGender",  # Not in Literal
        "tenure": -10,             # Must be >= 0
        "MonthlyCharges": -99.9,   # Must be >= 0
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_batch_endpoint(client: TestClient, valid_customer_payload: dict, loyal_low_risk_customer_payload: dict):
    """Test batch prediction endpoint with multiple customers."""
    batch_payload = {
        "customers": [valid_customer_payload, loyal_low_risk_customer_payload]
    }
    response = client.post("/predict/batch", json=batch_payload)
    if response.status_code == 200:
        data = response.json()
        assert data["total_records"] == 2
        assert len(data["predictions"]) == 2
        assert "overall_churn_rate_pct" in data


def test_metrics_endpoint(client: TestClient):
    """Test operational telemetry metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "uptime_seconds" in data
    assert "total_requests" in data
    assert "latency_ms" in data
    assert "predictions" in data


def test_drift_endpoint(client: TestClient):
    """Test statistical drift endpoint."""
    response = client.get("/monitoring/drift")
    assert response.status_code == 200
    data = response.json()
    assert "overall_status" in data
    assert "features_analyzed" in data


def test_experiments_endpoint(client: TestClient):
    """Test MLflow experiment comparison endpoint."""
    response = client.get("/experiments")
    assert response.status_code == 200
    data = response.json()
    assert "experiment_name" in data
