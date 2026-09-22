"""Data splitting, transformation, and split loader module."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import BASE_DIR, load_yaml_params
from src.data.ingest import ingest_raw_data
from src.data.validator import DataValidator

logger = logging.getLogger("modelflow.data.loader")


def clean_churn_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize data types and clean missing or whitespace values."""
    df_clean = df.copy()

    # Drop customerID if present for model training dataframe
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])

    # TotalCharges can contain whitespace or blank entries in raw data
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(
            df_clean["TotalCharges"].astype(str).str.strip(),
            errors="coerce"
        )
        # Impute missing TotalCharges with MonthlyCharges * tenure
        missing_mask = df_clean["TotalCharges"].isnull()
        if missing_mask.any():
            imputed = df_clean.loc[missing_mask, "MonthlyCharges"] * df_clean.loc[missing_mask, "tenure"].clip(lower=1)
            df_clean.loc[missing_mask, "TotalCharges"] = imputed

    # Convert SeniorCitizen to string category for uniform categorical processing
    if "SeniorCitizen" in df_clean.columns:
        df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].astype(str)

    # Standardize target Churn column to 0/1 binary integer
    if "Churn" in df_clean.columns:
        if df_clean["Churn"].dtype == object:
            df_clean["Churn"] = (df_clean["Churn"].str.strip().str.lower() == "yes").astype(int)

    return df_clean


def prepare_data_splits(raw_data_path: Optional[Path] = None) -> Tuple[Path, Path, Path]:
    """Load raw dataset, clean, validate, split into train/val/test, and save to disk."""
    params = load_yaml_params()
    data_cfg = params.get("data", {})

    raw_path = raw_data_path or (BASE_DIR / data_cfg.get("raw_path", "data/raw/telco_churn_raw.csv"))
    if not raw_path.exists():
        logger.info(f"Raw data file not found at {raw_path}. Ingesting raw data first...")
        raw_path = ingest_raw_data(raw_path)

    df_raw = pd.read_csv(raw_path)
    logger.info(f"Loaded raw dataset with {len(df_raw)} records.")

    # Validate raw structure
    validator = DataValidator(params)
    val_result = validator.validate(df_raw, is_training=True)
    if not val_result.is_valid:
        raise ValueError(f"Raw data validation failed: {val_result.errors}")

    # Clean dataset
    df_clean = clean_churn_dataframe(df_raw)

    # Stratified Split: Train (70%), Val (15%), Test (15%)
    random_state = data_cfg.get("random_state", 42)
    train_ratio = data_cfg.get("train_split", 0.70)
    val_ratio = data_cfg.get("val_split", 0.15)
    test_ratio = data_cfg.get("test_split", 0.15)

    remaining_ratio = val_ratio + test_ratio
    val_share_of_remaining = val_ratio / remaining_ratio

    train_df, temp_df = train_test_split(
        df_clean,
        train_size=train_ratio,
        random_state=random_state,
        stratify=df_clean["Churn"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_share_of_remaining,
        random_state=random_state,
        stratify=temp_df["Churn"],
    )

    # Paths
    train_path = BASE_DIR / data_cfg.get("processed_train_path", "data/processed/train.csv")
    val_path = BASE_DIR / data_cfg.get("processed_val_path", "data/processed/val.csv")
    test_path = BASE_DIR / data_cfg.get("processed_test_path", "data/processed/test.csv")
    ref_path = BASE_DIR / data_cfg.get("reference_path", "data/reference/reference_baseline.csv")

    for p in [train_path, val_path, test_path, ref_path]:
        p.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    # Save reference baseline for production drift monitoring
    train_df.to_csv(ref_path, index=False)

    logger.info(
        f"Data splits successfully saved:\n"
        f"  - Train: {train_path} ({len(train_df)} rows, Churn rate: {train_df['Churn'].mean():.1%})\n"
        f"  - Val:   {val_path} ({len(val_df)} rows, Churn rate: {val_df['Churn'].mean():.1%})\n"
        f"  - Test:  {test_path} ({len(test_df)} rows, Churn rate: {test_df['Churn'].mean():.1%})"
    )

    return train_path, val_path, test_path


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, val, and test splits from processed data directory."""
    params = load_yaml_params()
    data_cfg = params.get("data", {})

    train_path = BASE_DIR / data_cfg.get("processed_train_path", "data/processed/train.csv")
    val_path = BASE_DIR / data_cfg.get("processed_val_path", "data/processed/val.csv")
    test_path = BASE_DIR / data_cfg.get("processed_test_path", "data/processed/test.csv")

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        logger.info("Splits not found on disk. Generating splits now...")
        prepare_data_splits()

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df


if __name__ == "__main__":
    prepare_data_splits()
