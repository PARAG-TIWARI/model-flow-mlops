"""Data validation module for ModelFlow MLOps.

Validates schemas, ranges, null percentages, and categorical constraints.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.config.settings import load_yaml_params

logger = logging.getLogger("modelflow.data.validator")


@dataclass
class ValidationResult:
    """Encapsulates validation outcome and diagnostic details."""

    is_valid: bool
    row_count: int
    column_count: int
    missing_values: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class DataValidator:
    """Validates tabular churn datasets against project specifications."""

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or load_yaml_params()
        self.feature_config = self.params.get("features", {})
        self.numerical_cols = self.feature_config.get("numerical", [])
        self.categorical_cols = self.feature_config.get("categorical", [])
        self.target_col = self.params.get("data", {}).get("target_column", "Churn")

    def validate(self, df: pd.DataFrame, is_training: bool = True) -> ValidationResult:
        """Run structural and statistical integrity checks."""
        errors: List[str] = []
        warnings: List[str] = []

        if df.empty:
            return ValidationResult(
                is_valid=False,
                row_count=0,
                column_count=0,
                errors=["Dataset is empty"],
            )

        # 1. Check required feature columns.
        expected_features = self.numerical_cols + self.categorical_cols
        missing_features = [col for col in expected_features if col not in df.columns]
        if missing_features:
            errors.append(f"Missing required feature columns: {missing_features}")

        if is_training and self.target_col not in df.columns:
            errors.append(f"Training dataset missing target column '{self.target_col}'")

        # 2. Check null values.
        null_counts = df.isnull().sum().to_dict()
        for col, count in null_counts.items():
            if count > 0:
                pct = (count / len(df)) * 100
                if col == self.target_col and is_training:
                    errors.append(f"Target column '{col}' contains {count} missing values")
                elif pct > 10.0:
                    errors.append(
                        f"Column '{col}' has excessive missing values: {count} ({pct:.1f}%)"
                    )
                else:
                    warnings.append(
                        f"Column '{col}' contains {count} ({pct:.1f}%) missing values"
                    )

        # 3. Numeric range checks.
        for num_col in self.numerical_cols:
            if num_col in df.columns:
                series = pd.to_numeric(df[num_col], errors="coerce")
                if series.isnull().any():
                    warnings.append(f"Non-numeric values found and coerced in column '{num_col}'")
                if (series < 0).any():
                    errors.append(f"Negative values found in column '{num_col}' (must be >= 0)")

        # 4. Validate accepted target representations consistently with the cleaner.
        # Accept Yes/No labels (case/whitespace-insensitive) or numeric 0/1 values.
        if is_training and self.target_col in df.columns:
            target = df[self.target_col]
            normalized = target.astype("string").str.strip().str.lower()
            label_valid = normalized.isin(["yes", "no"])
            numeric = pd.to_numeric(target, errors="coerce")
            numeric_valid = numeric.isin([0, 1])
            valid_target = (label_valid | numeric_valid).fillna(False)
            invalid_mask = ~valid_target

            if invalid_mask.any():
                invalid_values = target[invalid_mask].drop_duplicates().tolist()
                errors.append(f"Invalid target values found: {invalid_values}")

        is_valid = len(errors) == 0
        if is_valid:
            logger.info(
                "Data validation passed successfully: %s rows, %s columns.",
                len(df),
                len(df.columns),
            )
        else:
            logger.error("Data validation failed with %s errors: %s", len(errors), errors)

        return ValidationResult(
            is_valid=is_valid,
            row_count=len(df),
            column_count=len(df.columns),
            missing_values={key: value for key, value in null_counts.items() if value > 0},
            errors=errors,
            warnings=warnings,
        )
