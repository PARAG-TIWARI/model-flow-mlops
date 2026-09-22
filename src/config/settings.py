"""Configuration and settings management for ModelFlow MLOps."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base project root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class ProjectSettings(BaseSettings):
    """Application and pipeline configuration settings."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Core
    environment: str = Field(default="development", alias="ENVIRONMENT")
    project_name: str = Field(default="ModelFlow MLOps", alias="PROJECT_NAME")
    project_version: str = "1.0.0"
    api_v1_str: str = "/api/v1"

    # Server settings
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    cors_origins: List[str] = ["*"]

    # Model & storage paths
    base_dir: Path = BASE_DIR
    model_path: Path = Field(
        default=BASE_DIR / "artifacts" / "models" / "champion_model.joblib",
        alias="MODEL_PATH"
    )
    metadata_path: Path = Field(
        default=BASE_DIR / "artifacts" / "models" / "model_metadata.json",
        alias="METADATA_PATH"
    )
    reference_data_path: Path = Field(
        default=BASE_DIR / "data" / "processed" / "train.csv",
        alias="REFERENCE_DATA_PATH"
    )
    metrics_report_path: Path = Field(
        default=BASE_DIR / "artifacts" / "reports" / "evaluation_metrics.json"
    )

    # MLflow settings
    mlflow_tracking_uri: str = Field(
        default=f"sqlite:///{BASE_DIR / 'mlflow.db'}",
        alias="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field(
        default="modelflow-customer-churn",
        alias="MLFLOW_EXPERIMENT_NAME"
    )

    # Monitoring & Drift
    ks_alpha: float = Field(default=0.05, alias="KS_ALPHA")
    psi_drift_threshold: float = Field(default=0.15, alias="PSI_DRIFT_THRESHOLD")


def load_yaml_params(params_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration dictionary from params.yaml."""
    target = params_path or (BASE_DIR / "params.yaml")
    if not target.exists():
        raise FileNotFoundError(f"Parameters file not found at: {target}")
    with open(target, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Singleton instance
settings = ProjectSettings()
