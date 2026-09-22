"""Unit tests for feature engineering and preprocessor pipelines."""

import numpy as np
from src.features.preprocessor import build_preprocessor
from src.features.transformer import FeatureEngineer


def test_feature_engineer_transform(sample_benchmark_dataframe):
    """Test custom feature engineering transformer generates expected financial ratios."""
    fe = FeatureEngineer(add_ratio_features=True)
    transformed = fe.transform(sample_benchmark_dataframe)

    assert "ChargesRatio" in transformed.columns
    assert "ChargeDiff" in transformed.columns
    assert "HasSecuritySupportBundle" in transformed.columns
    assert "HasFullStreamingBundle" in transformed.columns
    assert len(transformed) == len(sample_benchmark_dataframe)


def test_build_preprocessor_shape(sample_benchmark_dataframe):
    """Test column transformer transforms mixed inputs into numerical arrays."""
    preprocessor = build_preprocessor()
    X = sample_benchmark_dataframe.drop(columns=["customerID", "Churn"], errors="ignore")

    X_trans = preprocessor.fit_transform(X)
    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == len(sample_benchmark_dataframe)
    # Output columns should be >= input features due to one-hot encoding
    assert X_trans.shape[1] > 19
    # No NaNs in transformed output
    assert not np.isnan(X_trans).any()


def test_preprocessor_feature_names_out(sample_benchmark_dataframe):
    """Test column transformer generates human-interpretable feature names."""
    preprocessor = build_preprocessor()
    X = sample_benchmark_dataframe.drop(columns=["customerID", "Churn"], errors="ignore")
    preprocessor.fit(X)

    feature_names = preprocessor.get_feature_names_out()
    assert len(feature_names) > 0
    assert any("tenure" in fn for fn in feature_names)
    assert any("Contract" in fn for fn in feature_names)
