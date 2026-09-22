"""Evaluation runner generating machine-readable metrics and diagnostic plots."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.config.settings import BASE_DIR, load_yaml_params
from src.evaluation.metrics import compute_classification_metrics

logger = logging.getLogger("modelflow.evaluation.evaluator")


def generate_evaluation_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cm_dict: Dict[str, int],
    pipeline: Any,
    feature_names: list,
    output_dir: Path,
) -> Dict[str, Path]:
    """Generate confusion matrix, ROC curve, and feature importance plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = {}

    # 1. Confusion Matrix Plot
    cm_path = output_dir / "confusion_matrix.png"
    plt.figure(figsize=(6, 5))
    cm_matrix = np.array([
        [cm_dict["true_negative"], cm_dict["false_positive"]],
        [cm_dict["false_negative"], cm_dict["true_positive"]],
    ])
    plt.imshow(cm_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("ModelFlow — Out-of-Sample Confusion Matrix", fontsize=12, fontweight="bold", pad=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Retained (0)", "Churned (1)"])
    plt.yticks(tick_marks, ["Retained (0)", "Churned (1)"])

    thresh = cm_matrix.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = cm_matrix[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, f"{val:,}", horizontalalignment="center", verticalalignment="center", color=color, fontweight="bold", fontsize=13)

    plt.ylabel("Actual Label", fontweight="bold")
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()
    generated_plots["confusion_matrix"] = cm_path

    # 2. ROC Curve Plot
    roc_path = output_dir / "roc_curve.png"
    plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    from sklearn.metrics import roc_auc_score
    auc_val = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"Champion Model (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()
    generated_plots["roc_curve"] = roc_path

    # 3. Feature Importance Plot
    fi_path = output_dir / "feature_importance.png"
    try:
        classifier = pipeline.named_steps.get("classifier")
        preprocessor = pipeline.named_steps.get("preprocessor")

        importances = None
        names = []

        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
            if hasattr(preprocessor, "get_feature_names_out"):
                names = list(preprocessor.get_feature_names_out())
        elif hasattr(classifier, "coef_"):
            importances = np.abs(classifier.coef_[0])
            if hasattr(preprocessor, "get_feature_names_out"):
                names = list(preprocessor.get_feature_names_out())

        if importances is not None and len(names) == len(importances):
            top_indices = np.argsort(importances)[-12:]
            top_names = [names[i] for i in top_indices]
            top_vals = importances[top_indices]

            plt.figure(figsize=(8, 6))
            plt.barh(range(len(top_indices)), top_vals, color="#3B82F6", align="center")
            plt.yticks(range(len(top_indices)), top_names, fontsize=9)
            plt.xlabel("Relative Feature Contribution / Importance", fontweight="bold")
            plt.title("Top 12 Predictive Features for Customer Churn", fontsize=12, fontweight="bold", pad=12)
            plt.tight_layout()
            plt.savefig(fi_path, dpi=200)
            plt.close()
            generated_plots["feature_importance"] = fi_path
    except Exception as e:
        logger.warning(f"Could not generate feature importance plot: {e}")

    return generated_plots


def evaluate_model_on_test_split(
    model_path: Optional[Path] = None,
    test_data_path: Optional[Path] = None,
    output_metrics_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run full evaluation on held-out test split, saving JSON metrics and plots."""
    params = load_yaml_params()
    eval_cfg = params.get("evaluation", {})
    data_cfg = params.get("data", {})
    sel_cfg = params.get("selection", {})

    target_model_path = model_path or (BASE_DIR / sel_cfg.get("model_save_path", "artifacts/models/champion_model.joblib"))
    target_test_path = test_data_path or (BASE_DIR / data_cfg.get("processed_test_path", "data/processed/test.csv"))
    target_metrics_path = output_metrics_path or (BASE_DIR / eval_cfg.get("metrics_output", "artifacts/reports/evaluation_metrics.json"))
    target_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if not target_model_path.exists():
        raise FileNotFoundError(f"Champion model not found at {target_model_path}. Train model first.")
    if not target_test_path.exists():
        raise FileNotFoundError(f"Test split not found at {target_test_path}.")

    pipeline = joblib.load(target_model_path)
    test_df = pd.read_csv(target_test_path)

    target_col = data_cfg.get("target_column", "Churn")
    X_test = test_df.drop(columns=[target_col], errors="ignore")
    y_test = test_df[target_col].values

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Compute metrics
    metrics = compute_classification_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)

    # Generate diagnostic plots
    plots_dir = target_metrics_path.parent
    plots = generate_evaluation_plots(
        y_true=y_test,
        y_prob=y_prob,
        cm_dict=metrics["confusion_matrix"],
        pipeline=pipeline,
        feature_names=list(X_test.columns),
        output_dir=plots_dir,
    )
    metrics["artifacts"] = {k: str(v) for k, v in plots.items()}

    # Save to JSON
    with open(target_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        f"Evaluation complete on {len(test_df)} test samples:\n"
        f"  - Accuracy: {metrics['accuracy']:.4f}\n"
        f"  - ROC-AUC:  {metrics['roc_auc']:.4f}\n"
        f"  - F1-Score: {metrics['f1']:.4f}\n"
        f"  - PR-AUC:   {metrics['pr_auc']:.4f}\n"
        f"Saved metrics report to: {target_metrics_path}"
    )

    return metrics


if __name__ == "__main__":
    evaluate_model_on_test_split()
