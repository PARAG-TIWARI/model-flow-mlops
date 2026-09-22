"""Scikit-Learn preprocessor pipeline definition."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.settings import BASE_DIR, load_yaml_params
from src.data.loader import load_splits

logger = logging.getLogger("modelflow.features.preprocessor")


def build_preprocessor(
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Build Scikit-Learn ColumnTransformer for mixed numerical and categorical data."""
    params = load_yaml_params()
    feat_cfg = params.get("features", {})

    num_features = numerical_cols or feat_cfg.get("numerical", ["tenure", "MonthlyCharges", "TotalCharges"])
    cat_features = categorical_cols or feat_cfg.get(
        "categorical",
        [
            "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
            "Contract", "PaperlessBilling", "PaymentMethod"
        ],
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def fit_and_save_preprocessor(
    train_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Tuple[ColumnTransformer, Path]:
    """Fit preprocessor on training data and persist to disk."""
    params = load_yaml_params()
    target_col = params.get("data", {}).get("target_column", "Churn")

    features_df = train_df.drop(columns=[target_col], errors="ignore")
    preprocessor = build_preprocessor()
    preprocessor.fit(features_df)

    out_file = output_path or (BASE_DIR / "artifacts" / "models" / "preprocessor.joblib")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, out_file)
    logger.info(f"Preprocessor fitted on {len(features_df)} records and saved to {out_file}")

    return preprocessor, out_file


def load_preprocessor(path: Optional[Path] = None) -> ColumnTransformer:
    """Load persisted preprocessor from disk."""
    target = path or (BASE_DIR / "artifacts" / "models" / "preprocessor.joblib")
    if not target.exists():
        raise FileNotFoundError(f"Preprocessor not found at: {target}")
    return joblib.load(target)


if __name__ == "__main__":
    train_df, _, _ = load_splits()
    fit_and_save_preprocessor(train_df)
