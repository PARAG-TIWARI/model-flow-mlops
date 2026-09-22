"""In-memory operational metrics collector for real-time serving telemetry."""

from collections import deque
from datetime import datetime, timezone
import logging
import threading
import time
from typing import Any, Deque, Dict
import numpy as np

logger = logging.getLogger("modelflow.monitoring.metrics")


class MetricsCollector:
    """Thread-safe collector for request throughput, latency, and prediction statistics."""

    def __init__(self, max_recent: int = 1000):
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.latencies_ms: Deque[float] = deque(maxlen=max_recent)
        self.prediction_counts: Dict[str, int] = {"No": 0, "Yes": 0}
        self.recent_inferences: Deque[Dict[str, Any]] = deque(maxlen=max_recent)

    def record_prediction(self, features: Dict[str, Any], prediction: str, probability: float, latency_ms: float):
        """Record an inference event."""
        with self._lock:
            self.total_requests += 1
            self.successful_requests += 1
            self.latencies_ms.append(latency_ms)
            if prediction in self.prediction_counts:
                self.prediction_counts[prediction] += 1
            else:
                self.prediction_counts[prediction] = 1

            record = dict(features)
            record["_prediction"] = prediction
            record["_probability"] = probability
            record["_timestamp"] = datetime.now(timezone.utc).isoformat()
            self.recent_inferences.append(record)

    def record_error(self):
        """Record a failed request."""
        with self._lock:
            self.total_requests += 1
            self.failed_requests += 1

    def get_summary(self) -> Dict[str, Any]:
        """Compute summary statistics for API health and operational monitoring."""
        with self._lock:
            uptime_seconds = round(time.time() - self.start_time, 1)
            latencies = list(self.latencies_ms)

            if latencies:
                arr = np.array(latencies)
                p50 = float(np.percentile(arr, 50))
                p95 = float(np.percentile(arr, 95))
                p99 = float(np.percentile(arr, 99))
                avg_lat = float(np.mean(arr))
                min_lat = float(np.min(arr))
                max_lat = float(np.max(arr))
            else:
                p50 = p95 = p99 = avg_lat = min_lat = max_lat = 0.0

            total_preds = sum(self.prediction_counts.values())
            churn_rate = (
                round((self.prediction_counts.get("Yes", 0) / total_preds) * 100, 2)
                if total_preds > 0
                else 0.0
            )

            return {
                "uptime_seconds": uptime_seconds,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "error_rate_pct": round((self.failed_requests / max(1, self.total_requests)) * 100, 2),
                "latency_ms": {
                    "p50": round(p50, 2),
                    "p95": round(p95, 2),
                    "p99": round(p99, 2),
                    "mean": round(avg_lat, 2),
                    "min": round(min_lat, 2),
                    "max": round(max_lat, 2),
                },
                "predictions": {
                    "distribution": dict(self.prediction_counts),
                    "predicted_churn_rate_pct": churn_rate,
                },
                "buffered_inferences_count": len(self.recent_inferences),
            }

    def get_recent_dataframe(self):
        """Return buffered recent requests as pandas DataFrame for live drift evaluation."""
        import pandas as pd
        with self._lock:
            if not self.recent_inferences:
                return pd.DataFrame()
            return pd.DataFrame(list(self.recent_inferences))


# Global singleton instance
operational_metrics = MetricsCollector()
