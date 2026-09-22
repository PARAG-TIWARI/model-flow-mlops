"""Pydantic response schemas for structured API output."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check status response."""
    status: str = Field(..., examples=["healthy"])
    uptime_seconds: float = Field(..., examples=[128.4])
    model_loaded: bool = Field(..., examples=[True])
    model_version: str = Field(..., examples=["1.0.0"])
    environment: str = Field(..., examples=["production"])


class PredictionResponse(BaseModel):
    """Real-time customer churn prediction response."""
    prediction: str = Field(..., description="'Yes' for Churn, 'No' for Retained", examples=["Yes"])
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated churn likelihood", examples=[0.7421])
    confidence: float = Field(..., ge=0.0, le=1.0, description="Highest class confidence", examples=[0.7421])
    risk_tier: str = Field(..., description="Risk category: Low, Medium, or High Risk", examples=["High Risk"])
    recommendation: str = Field(..., description="Actionable retention recommendation")
    model_version: str = Field(..., examples=["1.0.0"])
    model_name: str = Field(..., examples=["HistGradientBoosting Classifier"])
    latency_ms: float = Field(..., description="Inference latency in milliseconds", examples=[2.15])


class BatchRecordResult(BaseModel):
    """Prediction outcome for an individual record inside batch payload."""
    record_index: int
    prediction: str
    churn_probability: float
    confidence: float
    risk_tier: str


class BatchPredictionResponse(BaseModel):
    """Batch inference aggregate and per-record response."""
    total_records: int
    high_risk_count: int
    overall_churn_rate_pct: float
    total_latency_ms: float
    average_latency_per_record_ms: float
    model_version: str
    predictions: List[BatchRecordResult]


class ModelMetadataResponse(BaseModel):
    """Serving model metadata, features, and evaluation performance."""
    status: str
    model_name: str
    version: str
    framework: str
    pipeline_type: str
    promoted_at: Optional[str] = None
    loaded_at: Optional[str] = None
    target: str
    features: Dict[str, List[str]]
    hyperparameters: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    promotion_criteria: Dict[str, Any]


class OperationalMetricsResponse(BaseModel):
    """Real-time API performance and prediction volume statistics."""
    uptime_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate_pct: float
    latency_ms: Dict[str, float]
    predictions: Dict[str, Any]
    buffered_inferences_count: int


class DriftReportResponse(BaseModel):
    """Statistical data drift report."""
    has_drift: bool
    overall_status: str
    features_analyzed: int
    drifted_features_count: int
    drift_percentage: float
    reference_samples: int
    current_samples: int
    feature_reports: Dict[str, Any]
