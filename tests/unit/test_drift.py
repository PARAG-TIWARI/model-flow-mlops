"""Unit tests for statistical drift detection and PSI computation."""

import numpy as np
from src.monitoring.drift import DriftDetector, compute_psi


def test_compute_psi_identical_distributions():
    """Test PSI between identical distributions is approximately zero."""
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=50.0, scale=10.0, size=1000)
    cur = rng.normal(loc=50.0, scale=10.0, size=1000)

    psi = compute_psi(ref, cur)
    assert psi < 0.05


def test_compute_psi_shifted_distribution():
    """Test PSI detects significant distribution shift."""
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=20.0, scale=5.0, size=1000)
    cur = rng.normal(loc=80.0, scale=5.0, size=1000)

    psi = compute_psi(ref, cur)
    assert psi > 0.25


def test_drift_detector_clean_data(sample_benchmark_dataframe):
    """Test drift detector reports stable status when comparing splits from same distribution."""
    half = len(sample_benchmark_dataframe) // 2
    ref_df = sample_benchmark_dataframe.iloc[:half]
    cur_df = sample_benchmark_dataframe.iloc[half:]

    detector = DriftDetector(reference_df=ref_df)
    report = detector.check_drift(cur_df)

    assert "has_drift" in report
    assert "features_analyzed" in report
    assert report["features_analyzed"] > 0
    assert report["overall_status"] == "Stable"
