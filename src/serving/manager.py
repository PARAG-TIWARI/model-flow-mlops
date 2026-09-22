"""Model lifecycle manager handling artifact loading, validation, and caching."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import joblib

from src.config.settings import settings

logger = logging.getLogger("modelflow.serving.manager")


class ModelManager:
    """Encapsulates loading and serving of champion models and metadata."""

    def __init__(self, model_path: Optional[Path] = None, metadata_path: Optional[Path] = None):
        self.model_path = model_path or settings.model_path
        self.metadata_path = metadata_path or settings.metadata_path
        self._pipeline = None
        self._metadata = None
        self._loaded_at = None

    def load(self, force_reload: bool = False):
        """Load or reload model and metadata into memory."""
        if self._pipeline is not None and not force_reload:
            return self._pipeline

        logger.info(f"Loading champion model from: {self.model_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{self.model_path}'. "
                "Please run training first (e.g., `python -m src.training.trainer`)."
            )

        self._pipeline = joblib.load(self.model_path)

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {
                "model_name": "Champion Classifier",
                "version": "1.0.0",
                "framework": "scikit-learn",
                "promoted_at": datetime.now(timezone.utc).isoformat(),
            }

        self._loaded_at = datetime.now(timezone.utc).isoformat()
        logger.info(f"Successfully loaded model '{self._metadata.get('model_name')}' v{self._metadata.get('version')}")
        return self._pipeline

    @property
    def pipeline(self):
        """Access loaded Scikit-Learn pipeline."""
        if self._pipeline is None:
            self.load()
        return self._pipeline

    @property
    def metadata(self) -> Dict[str, Any]:
        """Access loaded model metadata."""
        if self._metadata is None:
            self.load()
        return self._metadata or {}

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently resident in memory."""
        return self._pipeline is not None

    def get_info(self) -> Dict[str, Any]:
        """Return comprehensive metadata for /model API endpoint."""
        meta = self.metadata
        return {
            "status": "ready" if self.is_loaded else "uninitialized",
            "model_name": meta.get("model_name", "Unknown"),
            "version": meta.get("version", "1.0.0"),
            "framework": meta.get("framework", "scikit-learn"),
            "pipeline_type": meta.get("pipeline_type", "Pipeline"),
            "promoted_at": meta.get("promoted_at"),
            "loaded_at": self._loaded_at,
            "target": meta.get("target", "Churn"),
            "features": meta.get("features", {}),
            "hyperparameters": meta.get("hyperparameters", {}),
            "validation_metrics": meta.get("validation_metrics", {}),
            "promotion_criteria": meta.get("promotion_criteria", {}),
        }


# Global singleton instance
model_manager = ModelManager()
