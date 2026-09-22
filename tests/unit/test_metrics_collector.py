"""Unit tests for thread-safe operational metrics collector."""

from src.monitoring.metrics_collector import MetricsCollector


def test_metrics_collector_empty():
    """Test metrics summary on freshly initialized collector."""
    collector = MetricsCollector()
    summary = collector.get_summary()

    assert summary["total_requests"] == 0
    assert summary["successful_requests"] == 0
    assert summary["failed_requests"] == 0
    assert summary["latency_ms"]["mean"] == 0.0
    assert summary["buffered_inferences_count"] == 0


def test_metrics_collector_record_prediction():
    """Test recording successful inference updates throughput and latency distributions."""
    collector = MetricsCollector()
    sample_feat = {"tenure": 24, "MonthlyCharges": 70.0}

    collector.record_prediction(sample_feat, "Yes", 0.75, 4.2)
    collector.record_prediction(sample_feat, "No", 0.20, 2.8)
    collector.record_prediction(sample_feat, "No", 0.15, 3.1)
    collector.record_error()

    summary = collector.get_summary()
    assert summary["total_requests"] == 4
    assert summary["successful_requests"] == 3
    assert summary["failed_requests"] == 1
    assert summary["error_rate_pct"] == 25.0
    assert summary["predictions"]["distribution"]["Yes"] == 1
    assert summary["predictions"]["distribution"]["No"] == 2
    assert summary["latency_ms"]["min"] == 2.8
    assert summary["latency_ms"]["max"] == 4.2
    assert summary["buffered_inferences_count"] == 3


def test_metrics_collector_recent_dataframe():
    """Test exporting buffered inferences to DataFrame."""
    collector = MetricsCollector()
    collector.record_prediction({"tenure": 12}, "Yes", 0.82, 3.0)
    df = collector.get_recent_dataframe()

    assert not df.empty
    assert len(df) == 1
    assert "_prediction" in df.columns
    assert "_probability" in df.columns
    assert df["_prediction"].iloc[0] == "Yes"
