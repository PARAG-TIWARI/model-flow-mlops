"""Unit tests for the inference predictor and risk tier categorizer."""

from src.serving.predictor import ChurnPredictor


def test_determine_risk_tier():
    """Test risk tier boundaries and actionable advice."""
    predictor = ChurnPredictor()

    tier_high, rec_high = predictor._determine_risk_tier(0.85)
    assert tier_high == "High Risk"
    assert "retention" in rec_high.lower()

    tier_med, rec_med = predictor._determine_risk_tier(0.50)
    assert tier_med == "Medium Risk"
    assert "engagement" in rec_med.lower()

    tier_low, rec_low = predictor._determine_risk_tier(0.15)
    assert tier_low == "Low Risk"
    assert "stable" in rec_low.lower()
