"""Unit tests for data validation and schema integrity."""

import pandas as pd
import pytest

from src.data.loader import clean_churn_dataframe
from src.data.validator import DataValidator


def test_data_validator_valid_dataset(sample_benchmark_dataframe):
    """Test validator accepts conforming benchmark dataset."""
    validator = DataValidator()
    result = validator.validate(sample_benchmark_dataframe, is_training=True)
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.row_count == len(sample_benchmark_dataframe)


def test_data_validator_missing_column(sample_benchmark_dataframe):
    """Test validator rejects dataframe missing a required feature."""
    broken_df = sample_benchmark_dataframe.drop(columns=["tenure"])
    validator = DataValidator()
    result = validator.validate(broken_df, is_training=True)
    assert result.is_valid is False
    assert any("tenure" in err for err in result.errors)


def test_data_validator_negative_values(sample_benchmark_dataframe):
    """Test validator detects illegal negative numerical values."""
    corrupted_df = sample_benchmark_dataframe.copy()
    corrupted_df.loc[0, "MonthlyCharges"] = -50.0
    validator = DataValidator()
    result = validator.validate(corrupted_df, is_training=True)
    assert result.is_valid is False
    assert any("MonthlyCharges" in err for err in result.errors)


def test_data_validator_empty_dataframe():
    """Test validator handles empty dataframe safely."""
    empty_df = pd.DataFrame()
    validator = DataValidator()
    result = validator.validate(empty_df, is_training=True)
    assert result.is_valid is False
    assert any("empty" in err.lower() for err in result.errors)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (pd.Series(["Yes", "No", " YES ", " no "], dtype="string"), [1, 0, 1, 0]),
        (pd.Series(["Yes", "No"], dtype="object"), [1, 0]),
        (pd.Series([1, 0, 1, 0], dtype="Int64"), [1, 0, 1, 0]),
        (pd.Series([1, 0, 1, 0], dtype="int64"), [1, 0, 1, 0]),
        (pd.Series(["1", "0"], dtype="string"), [1, 0]),
    ],
)
def test_clean_churn_dataframe_normalizes_supported_target_dtypes(target, expected):
    """Normalize common CSV and pandas extension dtypes to an integer target."""
    frame = pd.DataFrame({"tenure": range(len(target)), "Churn": target})

    cleaned = clean_churn_dataframe(frame)

    assert cleaned["Churn"].tolist() == expected
    assert pd.api.types.is_integer_dtype(cleaned["Churn"].dtype)


def test_clean_churn_dataframe_rejects_unknown_target_values():
    """Reject ambiguous labels rather than silently mapping them to no churn."""
    frame = pd.DataFrame({"Churn": pd.Series(["Yes", "Maybe"], dtype="string")})

    with pytest.raises(ValueError, match="Churn must contain only Yes/No or 0/1"):
        clean_churn_dataframe(frame)
