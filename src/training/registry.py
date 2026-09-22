"""MLflow tracking integration and model registry management."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import mlflow
import mlflow.sklearn

from src.config.settings import BASE_DIR, load_yaml_params

logger = logging.getLogger("modelflow.training.registry")


class ModelRegistryManager:
    """Manages experiment tracking runs, metric logging, and artifact persistence."""

    def __init__(self, tracking_uri: Optional[str] = None, experiment_name: Optional[str] = None):
        params = load_yaml_params()
        mlflow_cfg = params.get("mlflow", {})
        self.tracking_uri = tracking_uri or mlflow_cfg.get("tracking_uri", f"sqlite:///{BASE_DIR / 'mlflow.db'}")
        self.experiment_name = experiment_name or mlflow_cfg.get("experiment_name", "modelflow-customer-churn")
        self.registered_model_name = mlflow_cfg.get("registered_model_name", "CustomerChurnClassifier")

        # Configure MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        try:
            self.experiment = mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(f"Could not set MLflow experiment: {e}. Falling back to default.")

    def log_training_run(
        self,
        model_name: str,
        pipeline: Any,
        params: Dict[str, Any],
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        training_duration_sec: float,
        artifacts_dict: Optional[Dict[str, Path]] = None,
    ) -> str:
        """Log a complete experiment run to MLflow."""
        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id

            # Log tags
            mlflow.set_tags({
                "project": "ModelFlow MLOps",
                "model_name": model_name,
                "framework": "scikit-learn",
                "task": "binary_classification",
                "target": "CustomerChurn",
            })

            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("training_duration_seconds", round(training_duration_sec, 3))

            # Log metrics (unpacking nested dicts like confusion matrix)
            for k, v in train_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"train_{k}", float(v))
                elif isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (int, float)):
                            mlflow.log_metric(f"train_{k}_{sub_k}", float(sub_v))

            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"val_{k}", float(v))
                elif isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (int, float)):
                            mlflow.log_metric(f"val_{k}_{sub_k}", float(sub_v))

            # Log model artifact with signature
            try:
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path="model",
                )
            except Exception as e:
                logger.warning(f"MLflow model logging warning: {e}")

            # Log custom plot artifacts if provided
            if artifacts_dict:
                for name, path in artifacts_dict.items():
                    if path.exists():
                        mlflow.log_artifact(str(path), artifact_path="plots")

            logger.info(f"Logged MLflow run '{model_name}' (ID: {run_id}) with val_roc_auc={val_metrics.get('roc_auc', 0.0):.4f}")
            return run_id

    def promote_champion_model(
        self,
        champion_pipeline: Any,
        champion_name: str,
        champion_metrics: Dict[str, float],
        champion_params: Dict[str, Any],
        run_id: str,
        model_save_path: Optional[Path] = None,
        metadata_save_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Save champion model artifact and structured JSON metadata for serving."""
        params = load_yaml_params()
        sel_cfg = params.get("selection", {})

        model_path = model_save_path or (BASE_DIR / sel_cfg.get("model_save_path", "artifacts/models/champion_model.joblib"))
        meta_path = metadata_save_path or (BASE_DIR / sel_cfg.get("metadata_save_path", "artifacts/models/model_metadata.json"))

        model_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Save joblib binary
        joblib.dump(champion_pipeline, model_path)
        logger.info(f"Champion model '{champion_name}' persisted to {model_path}")

        # 2. Build metadata schema
        feat_cfg = params.get("features", {})
        metadata = {
            "model_name": champion_name,
            "version": "1.0.0",
            "framework": "scikit-learn",
            "pipeline_type": type(champion_pipeline.named_steps["classifier"]).__name__,
            "mlflow_run_id": run_id,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "target": params.get("data", {}).get("target_column", "Churn"),
            "classes": ["No", "Yes"],
            "features": {
                "numerical": feat_cfg.get("numerical", []),
                "categorical": feat_cfg.get("categorical", []),
            },
            "hyperparameters": champion_params,
            "validation_metrics": champion_metrics,
            "promotion_criteria": {
                "metric": sel_cfg.get("primary_metric", "roc_auc"),
                "threshold": sel_cfg.get("min_roc_auc_threshold", 0.82),
                "achieved": champion_metrics.get(sel_cfg.get("primary_metric", "roc_auc"), 0.0),
            }
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model metadata persisted to {meta_path}")
        return metadata
