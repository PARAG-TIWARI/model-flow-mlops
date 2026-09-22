"""Inference engine performing real-time and batch customer churn predictions."""

import logging
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.monitoring.metrics_collector import operational_metrics
from src.serving.manager import ModelManager, model_manager

logger = logging.getLogger("modelflow.serving.predictor")


class ChurnPredictor:
    """Performs inference, calculates risk tiers, and collects operational telemetry."""

    def __init__(self, manager: ModelManager = model_manager):
        self.manager = manager

    def _determine_risk_tier(self, probability: float) -> Tuple[str, str]:
        """Classify churn probability into business actionable risk tiers."""
        if probability >= 0.65:
            return "High Risk", "Customer shows severe churn indicators. Immediate retention intervention recommended."
        elif probability >= 0.35:
            return "Medium Risk", "Customer has moderate churn risk. Proactive engagement suggested."
        else:
            return "Low Risk", "Customer indicates stable retention patterns."

    def predict_single(self, input_features: Dict[str, Any]) -> Dict[str, Any]:
        """Execute single-customer real-time inference."""
        start_time = time.time()
        pipeline = self.manager.pipeline
        meta = self.manager.metadata

        # Convert to single-row DataFrame
        df_input = pd.DataFrame([input_features])

        # Prediction and probabilities
        try:
            proba_array = pipeline.predict_proba(df_input)[0]
            # Probability of positive class (Churn = Yes, index 1)
            churn_prob = float(proba_array[1])
            pred_class = "Yes" if churn_prob >= 0.50 else "No"
        except Exception as e:
            operational_metrics.record_error()
            logger.error(f"Inference execution failed: {e}")
            raise

        latency_ms = round((time.time() - start_time) * 1000, 2)
        risk_tier, recommendation = self._determine_risk_tier(churn_prob)
        confidence = round(float(np.max(proba_array)), 4)

        # Record metrics
        operational_metrics.record_prediction(
            features=input_features,
            prediction=pred_class,
            probability=churn_prob,
            latency_ms=latency_ms,
        )

        return {
            "prediction": pred_class,
            "churn_probability": round(churn_prob, 4),
            "confidence": confidence,
            "risk_tier": risk_tier,
            "recommendation": recommendation,
            "model_version": meta.get("version", "1.0.0"),
            "model_name": meta.get("model_name", "Champion Model"),
            "latency_ms": latency_ms,
        }

    def predict_batch(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute bulk inference across a list of customer records."""
        start_time = time.time()
        pipeline = self.manager.pipeline
        meta = self.manager.metadata

        df_input = pd.DataFrame(items)
        try:
            proba_array = pipeline.predict_proba(df_input)
            churn_probs = proba_array[:, 1]
            pred_classes = np.where(churn_probs >= 0.50, "Yes", "No")
        except Exception as e:
            operational_metrics.record_error()
            logger.error(f"Batch inference failed: {e}")
            raise

        results = []
        high_risk_count = 0

        for i, (pred, prob) in enumerate(zip(pred_classes, churn_probs)):
            prob_float = float(round(prob, 4))
            tier, rec = self._determine_risk_tier(prob_float)
            if tier == "High Risk":
                high_risk_count += 1

            results.append({
                "record_index": i,
                "prediction": str(pred),
                "churn_probability": prob_float,
                "confidence": round(float(np.max(proba_array[i])), 4),
                "risk_tier": tier,
            })

        total_latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "total_records": len(items),
            "high_risk_count": high_risk_count,
            "overall_churn_rate_pct": round((sum(1 for r in results if r['prediction'] == 'Yes') / max(1, len(items))) * 100, 2),
            "total_latency_ms": total_latency_ms,
            "average_latency_per_record_ms": round(total_latency_ms / max(1, len(items)), 3),
            "model_version": meta.get("version", "1.0.0"),
            "predictions": results,
        }


# Global singleton instance
predictor = ChurnPredictor()
