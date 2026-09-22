"""Factory for building end-to-end scikit-learn machine learning pipelines."""

import logging
from typing import Any, Dict
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.preprocessor import build_preprocessor

logger = logging.getLogger("modelflow.training.pipeline")


def create_training_pipeline(model_config: Dict[str, Any]) -> Pipeline:
    """Build complete Scikit-Learn Pipeline combining preprocessor and estimator."""
    preprocessor = build_preprocessor()
    model_type = model_config.get("type", "logistic_regression")

    if model_type == "logistic_regression":
        estimator = LogisticRegression(
            C=float(model_config.get("C", 1.0)),
            max_iter=int(model_config.get("max_iter", 1000)),
            solver=model_config.get("solver", "lbfgs"),
            class_weight="balanced",
            random_state=42,
        )
    elif model_type == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=int(model_config.get("n_estimators", 100)),
            max_depth=int(model_config.get("max_depth", 8)),
            min_samples_split=int(model_config.get("min_samples_split", 5)),
            class_weight="balanced",
            random_state=int(model_config.get("random_state", 42)),
            n_jobs=-1,
        )
    elif model_type == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            learning_rate=float(model_config.get("learning_rate", 0.08)),
            max_iter=int(model_config.get("max_iter", 150)),
            max_depth=int(model_config.get("max_depth", 6)),
            min_samples_leaf=int(model_config.get("min_samples_leaf", 20)),
            class_weight="balanced",
            random_state=int(model_config.get("random_state", 42)),
        )
    else:
        raise ValueError(f"Unsupported model type '{model_type}'. Supported: logistic_regression, random_forest, hist_gradient_boosting")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )
    return pipeline
