"""Operational telemetry and statistical drift monitoring endpoints."""

import json
import logging
from fastapi import APIRouter
from api.schemas.response import OperationalMetricsResponse, DriftReportResponse
from src.config.settings import BASE_DIR, load_yaml_params
from src.monitoring.drift import DriftDetector
from src.monitoring.metrics_collector import operational_metrics

logger = logging.getLogger("modelflow.api.metrics")
router = APIRouter(tags=["Monitoring & Telemetry"])


@router.get(
    "/metrics",
    response_model=OperationalMetricsResponse,
    summary="Operational Serving Telemetry",
    description="Returns request volume, error counts, latency percentiles (p50, p95, p99), and predicted churn distribution.",
)
async def get_operational_metrics():
    """Operational metrics endpoint for Prometheus and dashboards."""
    return operational_metrics.get_summary()


@router.get(
    "/monitoring/drift",
    response_model=DriftReportResponse,
    summary="Data Drift Analysis",
    description="Computes Kolmogorov-Smirnov and Population Stability Index (PSI) to detect statistical distribution shifts against reference baseline.",
)
async def get_data_drift():
    """Calculate or retrieve current feature drift report."""
    recent_df = operational_metrics.get_recent_dataframe()

    detector = DriftDetector()
    if not recent_df.empty and len(recent_df) >= 10:
        logger.info(f"Computing real-time drift on {len(recent_df)} recent live requests...")
        report = detector.check_drift(recent_df)
    else:
        # Fallback to test evaluation split if live stream buffer is still warming up
        params = load_yaml_params()
        report_path = BASE_DIR / params.get("monitoring", {}).get("drift_report_path", "artifacts/reports/drift_report.json")
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
        else:
            test_path = BASE_DIR / params.get("data", {}).get("processed_test_path", "data/processed/test.csv")
            if test_path.exists():
                import pandas as pd
                test_df = pd.read_csv(test_path)
                report = detector.check_drift(test_df)
            else:
                report = {
                    "has_drift": False,
                    "overall_status": "Baseline ready — awaiting inference volume",
                    "features_analyzed": 0,
                    "drifted_features_count": 0,
                    "drift_percentage": 0.0,
                    "reference_samples": len(detector.reference_df) if not detector.reference_df.empty else 0,
                    "current_samples": 0,
                    "feature_reports": {},
                }

    return report


@router.get(
    "/experiments",
    summary="MLflow Experiment Comparison",
    description="Returns candidate models comparison, parameters, and validation metrics for dashboard visualization.",
)
async def get_experiments():
    """Return experiment runs comparison from MLflow or evaluation reports."""
    params = load_yaml_params()
    reports_path = BASE_DIR / "artifacts" / "reports" / "evaluation_metrics.json"
    meta_path = BASE_DIR / "artifacts" / "models" / "model_metadata.json"

    eval_data = {}
    if reports_path.exists():
        with open(reports_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

    meta_data = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

    # Fetch from MLflow if tracking database exists
    runs_list = []
    try:
        import mlflow
        tracking_uri = params.get("mlflow", {}).get("tracking_uri", f"sqlite:///{BASE_DIR / 'mlflow.db'}")
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(params.get("mlflow", {}).get("experiment_name", "modelflow-customer-churn"))
        if exp:
            runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=10)
            for r in runs:
                runs_list.append({
                    "run_id": r.info.run_id,
                    "run_name": r.data.tags.get("model_name", r.info.run_name),
                    "status": r.info.status,
                    "start_time": r.info.start_time,
                    "duration_sec": r.data.params.get("training_duration_seconds"),
                    "params": r.data.params,
                    "metrics": {k: round(v, 4) for k, v in r.data.metrics.items()},
                })
    except Exception as e:
        logger.warning(f"Could not read MLflow client runs directly: {e}")

    return {
        "experiment_name": params.get("mlflow", {}).get("experiment_name", "modelflow-customer-churn"),
        "champion_model": meta_data.get("model_name", "None"),
        "test_metrics": eval_data,
        "runs": runs_list,
    }
