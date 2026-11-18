""""Data splitting, transformation, and split loader module."""

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
            errors="coerce",
        )
        # Impute missing TotalCharges with MonthlyCharges * tenure
        missing_mask = df_clean["TotalCharges"].isnull()
        if missing_mask.any():
            imputed = (
                df_clean.loc[missing_mask, "MonthlyCharges"]
                * df_clean.loc[missing_mask, "tenure"].clip(lower=1)
            )
            df_clean.loc[missing_mask, "TotalCharges"] = imputed

    # Convert SeniorCitizen to string category for uniform categorical processing
    if "SeniorCitizen" in df_clean.columns:
        df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].astype(str)

    # Standardize target Churn to 0/1 for object, pandas StringDtype,
    # nullable numeric, and regular numeric inputs. pandas.read_csv can infer
    # extension string dtypes in some environments, so a dtype == object
    # check is not sufficient here.
    if "Churn" in df_clean.columns:
        raw_target = df_clean["Churn"]
        normalized_target = raw_target.astype("string").str.strip().str.lower()
        label_values = normalized_target.map({"yes": 1, "no": 0})
        numeric_values = pd.to_numeric(raw_target, errors="coerce")
        numeric_values = numeric_values.where(numeric_values.isin([0, 1]))
        cleaned_target = label_values.fillna(numeric_values)

        invalid_mask = cleaned_target.isna()
        if invalid_mask.any():
            invalid_values = raw_target[invalid_mask].drop_duplicates().tolist()
            raise ValueError(
                "Churn must contain only Yes/No or 0/1 values; "
                f"found invalid values: {invalid_values}"
            )

        df_clean["Churn"] = cleaned_target.astype("int8")

    return df_clean


def prepare_data_splits(raw_data_path: Optional[Path] = None) -> Tuple[Path, Path, Path]:
    """Load raw data, clean, validate, split, and save the three datasets."""
    params = load_yaml_params()
    data_cfg = params.get("data", {})

    raw_path = raw_data_path or (
        BASE_DIR / data_cfg.get("raw_path", "data/raw/telco_churn_raw.csv")
    )
    if not raw_path.exists():
        logger.info("Raw data file not found at %s. Ingesting raw data first...", raw_path)
        raw_path = ingest_raw_data(raw_path)

    df_raw = pd.read_csv(raw_path)
    logger.info("Loaded raw dataset with %s records.", len(df_raw))

    # Validate raw structure
    validator = DataValidator(params)
    val_result = validator.validate(df_raw, is_training=True)
    if not val_result.is_valid:
        raise ValueError(f"Raw data validation failed: {val_result.errors}")

    # Clean dataset
    df_clean = clean_churn_dataframe(df_raw)

    # Stratified split: Train (70%), Val (15%), Test (15%)
    random_state = data_cfg.get("random_state", 42)
    train_ratio = data_cfg.get("train_split", 0.70)
    val_ratio = data_cfg.get("val_split", 0.15)
    test_ratio = data_cfg.get("test_split", 0.15)

    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("Train, validation, and test split ratios must each be between 0 and 1.")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("Train, validation, and test split ratios must sum to 1.0.")

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

    for path in (train_path, val_path, test_path, ref_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    # Save the training split as the reference baseline for drift monitoring.
    train_df.to_csv(ref_path, index=False)

    logger.info(
        "Data splits saved: Train=%s rows (churn %.1f%%), "
        "Val=%s rows (churn %.1f%%), Test=%s rows (churn %.1f%%)",
        len(train_df),
        train_df["Churn"].mean() * 100,
        len(val_df),
        val_df["Churn"].mean() * 100,
        len(test_df),
        test_df["Churn"].mean() * 100,
    )

    return train_path, val_path, test_path


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test splits from the processed-data directory."""
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
