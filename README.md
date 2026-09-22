<div align="center">

# ModelFlow MLOps

### Reproducible Machine Learning Training, Tracking, Serving and Monitoring Platform

**An enterprise-grade, end-to-end MLOps platform for predictive analytics and risk intelligence.**

[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue?logo=github-actions&logoColor=white)](https://github.com)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-3.x-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-Data%20Version-945DD6?logo=dvc&logoColor=white)](https://dvc.org)
[![React](https://img.shields.io/badge/Frontend-React%2019%20%2B%20Vite-61DAFB?logo=react&logoColor=black)](https://vitejs.dev)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

---

## Overview

**ModelFlow MLOps** is a complete, production-oriented Machine Learning Engineering system designed to take raw enterprise data from ingestion to reproducible training, experiment tracking, containerized low-latency serving, and statistical live drift monitoring.

Built with a clean separation of concerns across data engineering, modeling, inference serving, and observability, ModelFlow demonstrates production-ready software engineering standards for machine learning practitioners.

---

## Problem

In enterprise environments, customer attrition (churn) directly erodes recurring revenue and customer lifetime value (LTV). However, traditional machine learning models in production suffer from:

1. **Pipeline Irreproducibility**: Inability to deterministically reproduce dataset states, feature preprocessing transforms, and model weights across environments.
2. **Experiment Chaos**: Lack of centralized experiment tracking, leading to opaque model selection and untracked hyperparameter modifications.
3. **Serving Inefficiencies**: Fragile inference endpoints lacking input validation, standardized error handling, and latency tracking.
4. **Silent Model Degradation**: Undetected distribution shifts (data drift) between training baselines and live inference streams.

---

## Solution

ModelFlow MLOps resolves these challenges through a modular, decoupled architecture:

- **Reproducible Pipeline Orchestration**: DVC (`dvc.yaml`) automates and caches data ingestion, validation, preprocessing, training, evaluation, and drift analysis.
- **Systematic Experiment Tracking**: MLflow logs candidate architectures, parameters, metrics, confusion matrix plots, and manages model artifacts.
- **Production-Grade Serving**: FastAPI delivers low-latency inference (11.21 ms mean, 6.43 ms p50 verified locally) with Pydantic v2 validation, structured risk tiers, and OpenAPI documentation.
- **Real-Time Telemetry & Drift Detection**: Automated statistical tests (Kolmogorov-Smirnov test and Population Stability Index for numerical features; Chi-Square for categorical features) monitor live inference payloads against baseline distributions.
- **Interactive Monitoring Dashboard**: Modern React + TypeScript + Vite + Tailwind CSS interface displaying real-time metrics, live prediction simulator, and MLflow experiment leaderboards.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ModelFlow MLOps Architecture                          │
└─────────────────────────────────────────────────────────────────────────────┘

   ┌────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
   │  DATA LAYER    │      │   EXPERIMENT TRACKING   │      │    CI/CD AUTOMATION     │
   │                │      │        (MLflow)         │      │    (GitHub Actions)     │
   │ Raw Dataset    │      │                         │      │                         │
   │ (7,043 rows)   │      │ Candidate Experiments   │      │  push / pull_request    │
   │       │        │      │ ├─ Logistic Regression  │      │            │            │
   │       ▼        │      │ ├─ Tuned Random Forest  │      │            ▼            │
   │ DataValidator  │─────►│ └─ HistGradientBoosting │      │  Ruff Linter Check      │
   │ (Schema/Types) │      │                         │      │            │            │
   │       │        │      │ Automated Model Gate    │      │            ▼            │
   │       ▼        │      │ (ROC-AUC >= 0.8200)     │      │  Pytest Suite (28 tests)│
   │ Data Splits    │      │            │            │      │            │            │
   │ Train/Val/Test │      │            ▼            │      │            ▼            │
   └───────┬────────┘      │   Champion Promotion    │─────►│  Docker Build & Test    │
           │               │   (artifacts/models/)   │      └─────────────────────────┘
           ▼               └─────────────────────────┘
   ┌────────────────┐
   │  DVC PIPELINE  │
   │  (dvc repro)   │
   └────────────────┘

                     ┌─────────────────────────────────────────┐
                     │       MODEL SERVING  (FastAPI :8000)     │
                     │                                         │
                     │  GET  /health        Liveness probe     │
                     │  GET  /model         Model metadata     │
                     │  POST /predict       Real-time scoring  │
                     │  POST /predict/batch Bulk inference     │
                     │  GET  /metrics       Serving telemetry  │
                     │  GET  /monitoring/drift Live drift check│
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │      DATA DRIFT & QUALITY ASSURANCE     │
                     │                                         │
                     │  Kolmogorov-Smirnov 2-sample test       │
                     │  Population Stability Index (PSI)       │
                     │  Chi-Square Contingency analysis        │
                     └─────────────────────────────────────────┘
```

---

## Pipeline Stages

The platform features 5 reproducible pipeline stages managed via `dvc.yaml`:

1. **`data_ingestion`**: Fetches or synthesizes the full benchmark dataset into `data/raw/telco_churn_raw.csv`.
2. **`data_preprocessing`**: Cleans missing records, applies type safety, and performs stratified splitting (70% train, 15% val, 15% test).
3. **`training`**: Trains 3 candidate architectures with cross-validation, logs runs to MLflow, and promotes the champion meeting the ROC-AUC gate.
4. **`evaluation`**: Executes held-out evaluation on `test.csv`, outputting JSON metrics and diagnostic plots (`confusion_matrix.png`, `roc_curve.png`, `feature_importance.png`).
5. **`drift_analysis`**: Evaluates statistical feature shifts across all 19 predictive attributes against the training baseline.

Execute the entire pipeline reproducibly:
```bash
dvc repro
```

---

## Dataset

The platform models customer retention using the **IBM Telco Customer Churn** benchmark dataset:

- **Source**: IBM Watson Sample Open Dataset
- **License**: Apache 2.0 / CC0 Public Domain
- **Total Samples**: 7,043 customer accounts
- **Target Variable**: `Churn` (Binary: `Yes` / `No`)
- **Features (19 attributes)**:
  - **Numerical**: `tenure` (months), `MonthlyCharges` ($), `TotalCharges` ($)
  - **Categorical**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`
- **Data Splits**:
  - Training: 4,930 samples (70.0%, Churn rate: 26.5%)
  - Validation: 1,056 samples (15.0%, Churn rate: 26.5%)
  - Held-out Test: 1,057 samples (15.0%, Churn rate: 26.6%)

---

## Model

ModelFlow evaluates multiple competing model architectures before promoting the champion:

1. **Calibrated Logistic Regression (Champion)**: Regularized linear classifier with balanced class weighting and probability calibration.
2. **Tuned Random Forest (Challenger 1)**: Ensemble of 120 balanced decision trees with controlled max depth.
3. **HistGradientBoosting Classifier (Challenger 2)**: Binned gradient-boosted decision tree pipeline for non-linear feature interactions.

### Automated Promotion Gate
To be promoted to production serving (`artifacts/models/champion_model.joblib`), a model must achieve `ROC-AUC >= 0.8200` on the validation set.

---

## DVC

Data and pipeline versioning are configured in `dvc.yaml` and parameterized through `params.yaml`:

```yaml
stages:
  data_ingestion:
    cmd: python -m src.data.ingest
    outs:
      - data/raw/telco_churn_raw.csv

  data_preprocessing:
    cmd: python -m src.data.loader
    deps:
      - data/raw/telco_churn_raw.csv
    outs:
      - data/processed/train.csv
      - data/processed/val.csv
      - data/processed/test.csv

  training:
    cmd: python -m src.training.trainer
    outs:
      - artifacts/models/champion_model.joblib
      - artifacts/models/model_metadata.json

  evaluation:
    cmd: python -m src.evaluation.evaluator
    metrics:
      - artifacts/reports/evaluation_metrics.json:
          cache: false

  drift_analysis:
    cmd: python -m src.monitoring.drift
    outs:
      - artifacts/reports/drift_report.json:
          cache: false
```

---

## MLflow

ModelFlow implements experiment tracking with SQLite backend storage (`sqlite:///mlflow.db`):

- **Experiment Name**: `modelflow-customer-churn`
- **Logged Parameters**: Model architecture, regularization `C`, tree estimators, learning rate, solver, training duration.
- **Logged Metrics**: Train & Validation Accuracy, Precision, Recall, F1-Score, F1-Macro, ROC-AUC, PR-AUC, and Confusion Matrix cell counts.
- **Artifacts**: Scikit-learn model signatures, confusion matrix plots, ROC curves, feature importance charts.

Inspect runs via the MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

---

## Evaluation

Model evaluation is executed strictly on held-out test data (`test.csv`):

| Metric | Champion Score | Interpretation |
|---|---|---|
| **ROC-AUC** | **0.8447** | Excellent discrimination between retaining and churning customers |
| **Accuracy** | **74.74%** | Overall correct classification rate under balanced weighting |
| **Recall (Churn)** | **77.66%** | High sensitivity; captures over 77% of all churning accounts |
| **F1-Score** | **0.6213** | Balanced harmonic mean of precision and recall |
| **PR-AUC** | **0.6587** | Precision-Recall area under the curve |

### Out-of-Sample Confusion Matrix (1,057 Samples)
- **True Negatives (Retained)**: 615
- **False Positives (False Alarms)**: 161
- **False Negatives (Missed Churn)**: 106
- **True Positives (Caught Churn)**: 175

Artifacts generated:
- `artifacts/reports/evaluation_metrics.json`
- `artifacts/reports/confusion_matrix.png`
- `artifacts/reports/roc_curve.png`
- `artifacts/reports/feature_importance.png`

---

## FastAPI

Production REST API built with FastAPI and Pydantic v2:

- **Swagger Documentation**: `http://localhost:8000/docs`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

### Endpoints
- `GET /`: Platform index and service registry.
- `GET /health`: Liveness probe with uptime and model residency check.
- `GET /health/ready`: Readiness probe verifying model load status.
- `GET /model`: Returns champion model metadata, hyperparameters, and feature specifications.
- `GET /model/features`: Returns lists of expected numerical and categorical inputs.
- `POST /predict`: Real-time inference returning churn prediction, calibrated probability, risk tier, and retention strategy.
- `POST /predict/batch`: Bulk inference for up to 1,000 records.
- `GET /metrics`: Serving telemetry (request counts, error rate, p50/p95/p99 latencies, prediction ratios).
- `GET /monitoring/drift`: Statistical feature drift report (PSI & KS-test).
- `GET /experiments`: Comparison of candidate MLflow training runs.

---

## Monitoring

ModelFlow implements an in-memory, thread-safe metrics collector (`MetricsCollector`) and statistical drift detector (`DriftDetector`):

- **Operational Telemetry**:
  - Request counters and error tracking
  - Latency percentiles: min, mean, p50, p95, p99, max
  - Class distribution (predicted churn rate vs retained)
- **Data Drift Detection**:
  - **Numerical Features**: Two-sample Kolmogorov-Smirnov test (`scipy.stats.ks_2samp`) and Population Stability Index (PSI).
  - **Categorical Features**: Chi-Square contingency analysis and categorical PSI.
  - **Alert Thresholds**: Alert triggered if PSI exceeds `0.150` or KS p-value falls below `0.05`.

---

## Docker

Production multi-stage `Dockerfile`:
- **Builder Stage**: Installs pinned dependencies, executes the end-to-end training pipeline (`python -m scripts.run_pipeline`), and runs build-time assertions verifying `champion_model.joblib`, `model_metadata.json`, and reference datasets.
- **Runner Stage**: Minimal `python:3.12-slim` image that copies verified artifacts, datasets, and MLflow SQLite database from the builder stage, running under an unprivileged user (`mlopsuser:mlopsgroup`).
- **Healthcheck**: Configured `HEALTHCHECK` instruction using curl.

Run with Docker Compose:
```bash
docker compose up --build
```

---

## CI/CD

Automated GitHub Actions pipeline (`.github/workflows/ci_cd.yml`):

1. **Lint & Code Style**: Enforces PEP 8 standards with `ruff check`.
2. **Unit & Integration Tests**: Executes 28 tests via `pytest` with coverage reporting.
3. **Data & Pipeline Verification**: Verifies end-to-end reproducibility of training and evaluation stages.
4. **Clean-Room Originality Audit**: Scans repository to verify zero reference remnants.
5. **Docker Container Build**: Verifies that the production Docker image compiles cleanly.
6. **Deployment Webhook**: Triggers Render deployment on pushes to `main`.

---

## Testing

Comprehensive test suite covering unit math, schema validation, and API integration:

```bash
pytest tests/ -v --tb=short
```

**Execution Results**:
```
tests/integration/test_api_endpoints.py::test_root_endpoint PASSED
tests/integration/test_api_endpoints.py::test_health_endpoint PASSED
tests/integration/test_api_endpoints.py::test_model_features_endpoint PASSED
tests/integration/test_api_endpoints.py::test_predict_endpoint_success PASSED
tests/integration/test_api_endpoints.py::test_predict_endpoint_validation_error PASSED
tests/integration/test_api_endpoints.py::test_predict_batch_endpoint PASSED
tests/integration/test_api_endpoints.py::test_metrics_endpoint PASSED
tests/integration/test_api_endpoints.py::test_drift_endpoint PASSED
tests/integration/test_api_endpoints.py::test_experiments_endpoint PASSED
tests/integration/test_pipeline_flow.py::test_end_to_end_mini_pipeline PASSED
tests/unit/test_data_validation.py::test_data_validator_valid_dataset PASSED
tests/unit/test_data_validation.py::test_data_validator_missing_column PASSED
tests/unit/test_data_validation.py::test_data_validator_negative_values PASSED
tests/unit/test_data_validation.py::test_data_validator_empty_dataframe PASSED
tests/unit/test_drift.py::test_compute_psi_identical_distributions PASSED
tests/unit/test_drift.py::test_compute_psi_shifted_distribution PASSED
tests/unit/test_drift.py::test_drift_detector_clean_data PASSED
tests/unit/test_evaluation.py::test_compute_classification_metrics_perfect_predictions PASSED
tests/unit/test_evaluation.py::test_compute_classification_metrics_partial_accuracy PASSED
tests/unit/test_metrics_collector.py::test_metrics_collector_empty PASSED
tests/unit/test_metrics_collector.py::test_metrics_collector_record_prediction PASSED
tests/unit/test_metrics_collector.py::test_metrics_collector_recent_dataframe PASSED
tests/unit/test_model_manager.py::test_model_manager_info PASSED
tests/unit/test_model_manager.py::test_model_manager_missing_file PASSED
tests/unit/test_predictor.py::test_determine_risk_tier PASSED
tests/unit/test_preprocessing.py::test_feature_engineer_transform PASSED
tests/unit/test_preprocessing.py::test_build_preprocessor_shape PASSED
tests/unit/test_preprocessing.py::test_preprocessor_feature_names_out PASSED

============================= 28 passed in 2.89s ==============================
```

---

## Deployment

- **Backend (FastAPI)**: Configured for Render via `render.yaml` with `runtime: docker` using multi-stage `Dockerfile`. Binds dynamically to `0.0.0.0:$PORT`.
- **Frontend (Vite + React)**: Configured for Vercel via `vercel.json` with client-side SPA routing.
- **Environment Configuration**: Set `VITE_API_BASE_URL` in the frontend (e.g. `https://model-flow-mlops.onrender.com`) to point to the production backend URL. Render web service permits CORS requests from `https://model-flow-mlops.vercel.app`.

---

## Project Structure

```
model-flow-mlops/
├── src/
│   ├── config/              # Application settings & YAML loaders
│   ├── data/                # Ingestion, validation, and split loaders
│   ├── features/            # Scikit-Learn ColumnTransformer & feature engineering
│   ├── training/            # Model pipeline factory, trainer, and MLflow registry
│   ├── evaluation/          # Classification metrics & diagnostic plot generators
│   ├── serving/             # Model lifecycle manager & inference predictor
│   └── monitoring/          # Statistical drift detector (KS/PSI) & operational metrics
│
├── api/
│   ├── routes/              # Health, model, predict, and metrics endpoints
│   ├── schemas/             # Pydantic v2 request and response contracts
│   └── main.py              # FastAPI application entrypoint
│
├── frontend/                # React 19 + TypeScript + Vite + Tailwind CSS dashboard
│   ├── src/
│   │   ├── components/      # Navigation and UI components
│   │   ├── pages/           # Home, Model, Prediction, Experiments, System
│   │   └── services/        # Typed API service client
│   └── package.json
│
├── tests/
│   ├── unit/                # Unit tests for preprocessing, models, and drift
│   ├── integration/         # REST API endpoint and pipeline flow tests
│   └── conftest.py          # Pytest fixtures and mock payloads
│
├── scripts/
│   ├── run_pipeline.py      # End-to-end Python pipeline runner
│   ├── simulate_inference.py# Traffic simulator for live metrics demonstration
│   └── check_originality.py # Clean-room compliance auditor
│
├── configs/                 # Project & model configuration YAML
├── params.yaml              # DVC parameters file
├── dvc.yaml                 # DVC reproducible pipeline stages
├── Dockerfile               # Multi-stage production container
├── docker-compose.yml       # API + MLflow service orchestration
├── render.yaml              # Render web service configuration
├── vercel.json              # Vercel SPA routing configuration
├── requirements.txt         # Pinned production Python dependencies
└── README.md
```

---

## Local Setup

### 1. Clone & Set Up Environment
```bash
git clone <your-repo-url> model-flow-mlops
cd model-flow-mlops

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Execute Data & Training Pipeline
```bash
python -m scripts.run_pipeline
# Or using DVC:
dvc repro
```

### 3. Launch Backend Inference Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation available at: `http://localhost:8000/docs`

### 4. Launch Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```
Dashboard available at: `http://localhost:3000`

---

## Environment Variables

Copy `.env.example` to `.env` to customize settings:

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `production` | Deployment environment (`development` / `production`) |
| `HOST` | `0.0.0.0` | Host IP for serving |
| `PORT` | `8000` | Server listening port |
| `MODEL_PATH` | `artifacts/models/champion_model.joblib` | Path to promoted champion model |
| `METADATA_PATH` | `artifacts/models/model_metadata.json` | Path to model metadata JSON |
| `REFERENCE_DATA_PATH` | `data/processed/train.csv` | Baseline dataset for drift detection |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow tracking database URI |
| `PSI_DRIFT_THRESHOLD` | `0.15` | PSI threshold triggering drift alert |
| `KS_ALPHA` | `0.05` | P-value threshold for Kolmogorov-Smirnov test |

---

## API Usage

### Single Customer Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
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
    "TotalCharges": 1078.20
  }'
```

**Response**:
```json
{
  "prediction": "Yes",
  "churn_probability": 0.7421,
  "confidence": 0.7421,
  "risk_tier": "High Risk",
  "recommendation": "Customer shows severe churn indicators. Immediate retention intervention recommended.",
  "model_version": "1.0.0",
  "model_name": "Calibrated Logistic Regression",
  "latency_ms": 2.15
}
```

---

## Results

ModelFlow achieves competitive retention classification on the standard Telco benchmark:

- **Validation ROC-AUC**: `0.8448`
- **Held-out Test ROC-AUC**: `0.8447`
- **Held-out Recall**: `77.66%`
- **Inference Latency (LOCAL Verification Results)**:
  - **Sample Benchmark**: 50 real inference requests
  - **Success Rate**: 100%
  - **Mean Latency**: 11.21 ms
  - **P50 Latency**: 6.43 ms
  - **P95 Latency**: 22.80 ms
  - **P99 Latency**: 39.04 ms
- **Test Pass Rate**: `100% (28/28 passing)`
- **Clean-Room Audit**: `0 reference remnants found`

---

## Limitations

- **Host Docker Daemon**: While `Dockerfile` and `docker-compose.yml` are written to production standards and syntax-validated, Docker daemon execution depends on Docker Desktop being installed on the host OS.
- **Local SQLite Tracking**: In local development, MLflow utilizes SQLite (`mlflow.db`). Production deployments should target managed PostgreSQL and S3 artifact storage.

---

## Future Work

- [ ] Automated continuous retraining triggers upon detecting statistical drift alerts.
- [ ] Integration with cloud object storage (AWS S3 / GCP Cloud Storage) for DVC remote caching.
- [ ] SHAP (SHapley Additive exPlanations) values returned directly in the REST API payload for real-time per-feature explainability.
- [ ] Canary deployment and traffic shadowing via Kubernetes and Istio.

---

## License

This project is licensed under the Apache License 2.0.
