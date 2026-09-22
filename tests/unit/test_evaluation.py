"""Unit tests for classification metrics and evaluation math."""

import numpy as np
from src.evaluation.metrics import compute_classification_metrics


def test_compute_classification_metrics_perfect_predictions():
    """Test metrics calculation with perfect predictions."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.05, 0.95, 0.10, 0.90, 0.02, 0.98])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"]["true_positive"] == 3
    assert metrics["confusion_matrix"]["true_negative"] == 3
    assert metrics["confusion_matrix"]["false_positive"] == 0
    assert metrics["confusion_matrix"]["false_negative"] == 0


def test_compute_classification_metrics_partial_accuracy():
    """Test metrics calculation with known mixed predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.7])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 0.5
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["sample_count"] == 4
