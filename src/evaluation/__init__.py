"""Model evaluation and metrics calculation module."""
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.evaluator import evaluate_model_on_test_split

__all__ = ["compute_classification_metrics", "evaluate_model_on_test_split"]
