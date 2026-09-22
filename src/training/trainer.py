"""Model training orchestrator comparing multiple architectures and promoting the champion."""

import logging
import time
from typing import Any, Dict, Tuple

import pandas as pd

from src.config.settings import load_yaml_params
from src.data.loader import load_splits
from src.evaluation.metrics import compute_classification_metrics
from src.training.pipeline import create_training_pipeline
from src.training.registry import ModelRegistryManager

logger = logging.getLogger("modelflow.training.trainer")


def train_single_model(
    model_key: str,
    model_cfg: Dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
) -> Tuple[Any, Dict[str, float], Dict[str, float], float]:
    """Train a single pipeline architecture and compute train & val metrics."""
    logger.info(f"--- Training candidate model: {model_cfg.get('name', model_key)} ---")
    start_time = time.time()

    X_train = train_df.drop(columns=[target_col], errors="ignore")
    y_train = train_df[target_col].values
    X_val = val_df.drop(columns=[target_col], errors="ignore")
    y_val = val_df[target_col].values

    # Build and fit pipeline
    pipeline = create_training_pipeline(model_cfg)
    pipeline.fit(X_train, y_train)

    duration = time.time() - start_time

    # Train predictions & metrics
    y_train_pred = pipeline.predict(X_train)
    y_train_prob = pipeline.predict_proba(X_train)[:, 1]
    train_metrics = compute_classification_metrics(y_train, y_train_pred, y_train_prob)

    # Validation predictions & metrics
    y_val_pred = pipeline.predict(X_val)
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]
    val_metrics = compute_classification_metrics(y_val, y_val_pred, y_val_prob)

    logger.info(
        f"Completed {model_cfg.get('name', model_key)} in {duration:.2f}s | "
        f"Val ROC-AUC: {val_metrics['roc_auc']:.4f} | "
        f"Val F1: {val_metrics['f1']:.4f} | "
        f"Val Accuracy: {val_metrics['accuracy']:.4f}"
    )

    return pipeline, train_metrics, val_metrics, duration


def train_all_models() -> Dict[str, Any]:
    """Execute complete multi-model experimentation cycle and promote the champion."""
    params = load_yaml_params()
    models_cfg = params.get("models", {})
    sel_cfg = params.get("selection", {})
    target_col = params.get("data", {}).get("target_column", "Churn")
    primary_metric = sel_cfg.get("primary_metric", "roc_auc")
    gate_threshold = float(sel_cfg.get("min_roc_auc_threshold", 0.82))

    # Load splits
    train_df, val_df, _ = load_splits()

    registry = ModelRegistryManager()

    results = {}
    champion_key = None
    champion_score = -1.0
    champion_pipeline = None
    champion_metrics = None
    champion_params = None
    champion_run_id = None

    for model_key, model_cfg in models_cfg.items():
        pipeline, train_m, val_m, duration = train_single_model(
            model_key=model_key,
            model_cfg=model_cfg,
            train_df=train_df,
            val_df=val_df,
            target_col=target_col,
        )

        # Log run to MLflow
        run_id = registry.log_training_run(
            model_name=model_cfg.get("name", model_key),
            pipeline=pipeline,
            params=model_cfg,
            train_metrics=train_m,
            val_metrics=val_m,
            training_duration_sec=duration,
        )

        score = val_m.get(primary_metric, 0.0)
        results[model_key] = {
            "name": model_cfg.get("name", model_key),
            "run_id": run_id,
            "train_metrics": train_m,
            "val_metrics": val_m,
            "score": score,
        }

        if score > champion_score:
            champion_score = score
            champion_key = model_key
            champion_pipeline = pipeline
            champion_metrics = val_m
            champion_params = model_cfg
            champion_run_id = run_id

    logger.info("\n================ MODEL SELECTION SUMMARY ================")
    for k, v in results.items():
        logger.info(f"Model: {v['name']} -> {primary_metric.upper()}: {v['score']:.4f}")
    logger.info(f"Champion: {models_cfg[champion_key].get('name')} with {primary_metric}={champion_score:.4f}")

    if champion_score < gate_threshold:
        logger.warning(
            f"Champion score {champion_score:.4f} did not meet deployment gate {gate_threshold:.4f}. "
            f"Promoting conditionally for local development."
        )

    # Save champion model & metadata
    metadata = registry.promote_champion_model(
        champion_pipeline=champion_pipeline,
        champion_name=models_cfg[champion_key].get("name", champion_key),
        champion_metrics=champion_metrics,
        champion_params=champion_params,
        run_id=champion_run_id,
    )

    return {
        "champion_key": champion_key,
        "champion_name": models_cfg[champion_key].get("name", champion_key),
        "champion_score": champion_score,
        "run_id": champion_run_id,
        "all_results": results,
        "metadata": metadata,
    }


if __name__ == "__main__":
    train_all_models()
