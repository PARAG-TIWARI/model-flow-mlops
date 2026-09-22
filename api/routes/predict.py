"""Inference endpoints for real-time and batch customer churn predictions."""

import logging

from fastapi import APIRouter, HTTPException, status

from api.schemas.request import BatchInferenceRequest, CustomerFeatures
from api.schemas.response import BatchPredictionResponse, PredictionResponse
from src.serving.predictor import predictor

logger = logging.getLogger("modelflow.api.predict")
router = APIRouter(tags=["Inference"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Real-time Customer Churn Prediction",
    description="Predicts churn likelihood, confidence score, risk tier, and retention recommendation for a single customer profile.",
)
async def predict_churn(customer: CustomerFeatures):
    """Single-customer real-time inference endpoint."""
    try:
        payload = customer.to_dict()
        result = predictor.predict_single(payload)
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Customer Churn Inference",
    description="Processes up to 1000 customer records in bulk, computing individual predictions and aggregate risk distribution.",
)
async def predict_batch_churn(batch_request: BatchInferenceRequest):
    """Bulk customer inference endpoint."""
    try:
        items = [c.to_dict() for c in batch_request.customers]
        result = predictor.predict_batch(items)
        return result
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}",
        )
