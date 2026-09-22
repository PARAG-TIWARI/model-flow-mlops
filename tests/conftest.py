"""Pytest fixtures for ModelFlow MLOps test suite."""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import pandas as pd

# Add project root to sys.path so tests import cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app  # noqa: E402
from src.data.ingest import generate_benchmark_churn_data  # noqa: E402


@pytest.fixture(scope="session")
def client() -> TestClient:
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def sample_benchmark_dataframe() -> pd.DataFrame:
    """Fixture providing small synthetic dataset for fast tests."""
    return generate_benchmark_churn_data(n_samples=200, seed=42)


@pytest.fixture
def valid_customer_payload() -> dict:
    """Fixture providing valid customer feature payload."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.85,
        "TotalCharges": 2156.40,
    }


@pytest.fixture
def loyal_low_risk_customer_payload() -> dict:
    """Fixture providing low-risk loyal customer profile."""
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 68,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 45.20,
        "TotalCharges": 3073.60,
    }
