"""Model metadata and schema endpoints."""

from fastapi import APIRouter, HTTPException

from api.schemas.response import ModelMetadataResponse
from src.config.settings import load_yaml_params
from src.serving.manager import model_manager

router = APIRouter(prefix="/model", tags=["Model"])


@router.get(
    "",
    response_model=ModelMetadataResponse,
    summary="Get Model Information",
    description="Returns metadata of the deployed champion model, hyperparameters, and validation metrics.",
)
async def get_model_info():
    """Fetch current model metadata and training performance."""
    try:
        info = model_manager.get_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model metadata: {e}")


@router.get(
    "/features",
    summary="Get Expected Features",
    description="Returns list of required numerical and categorical features expected by the pipeline.",
)
async def get_model_features():
    """Return feature schema specifications."""
    params = load_yaml_params()
    feat_cfg = params.get("features", {})
    return {
        "numerical_features": feat_cfg.get("numerical", []),
        "categorical_features": feat_cfg.get("categorical", []),
        "target": params.get("data", {}).get("target_column", "Churn"),
    }
