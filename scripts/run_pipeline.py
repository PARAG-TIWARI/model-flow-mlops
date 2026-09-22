"""End-to-end pipeline execution script for ModelFlow MLOps."""

import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path for reliable module resolution across all environments
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("modelflow.pipeline")


def run_full_pipeline():
    """Execute all pipeline stages in sequence."""
    total_start = time.time()
    logger.info("==================================================")
    logger.info("    ModelFlow MLOps — Starting Pipeline Run       ")
    logger.info("==================================================")

    # Stage 1: Ingestion
    logger.info("\n>>> [STAGE 1/5] Ingesting Raw Dataset...")
    from src.data.ingest import ingest_raw_data
    raw_path = ingest_raw_data()
    logger.info(f"Raw data ready: {raw_path}")

    # Stage 2: Validation and Splitting
    logger.info("\n>>> [STAGE 2/5] Validating and Splitting Dataset...")
    from src.data.loader import prepare_data_splits
    train_p, val_p, test_p = prepare_data_splits()
    logger.info("Splits generated successfully.")

    # Stage 3: Multi-model Training & MLflow Logging
    logger.info("\n>>> [STAGE 3/5] Training Models & Tracking Experiments...")
    from src.training.trainer import train_all_models
    train_results = train_all_models()
    logger.info(f"Champion promoted: {train_results['champion_name']} (ROC-AUC: {train_results['champion_score']:.4f})")

    # Stage 4: Out-of-sample Evaluation & Diagnostic Plots
    logger.info("\n>>> [STAGE 4/5] Evaluating Champion Model on Test Split...")
    from src.evaluation.evaluator import evaluate_model_on_test_split
    eval_metrics = evaluate_model_on_test_split()
    logger.info(f"Evaluation complete: Test ROC-AUC={eval_metrics['roc_auc']}, F1={eval_metrics['f1']}")

    # Stage 5: Baseline Drift Analysis
    logger.info("\n>>> [STAGE 5/5] Performing Baseline Drift Analysis...")
    from src.monitoring.drift import calculate_dataset_drift
    drift_report = calculate_dataset_drift()
    logger.info(f"Drift check complete: {drift_report['overall_status']}")

    elapsed = round(time.time() - total_start, 2)
    logger.info("\n==================================================")
    logger.info(f"  ModelFlow MLOps Pipeline Finished in {elapsed}s ")
    logger.info("==================================================")


if __name__ == "__main__":
    import traceback
    try:
        run_full_pipeline()
    except Exception:
        err_msg = traceback.format_exc()
        print(err_msg, file=sys.stderr)
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path:
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(f"### Pipeline Execution Error\n```\n{err_msg}\n```\n")
        sys.exit(1)

