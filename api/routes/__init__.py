"""API route endpoints package."""
from api.routes.health import router as health_router
from api.routes.metrics import router as metrics_router
from api.routes.model import router as model_router
from api.routes.predict import router as predict_router

__all__ = ["health_router", "model_router", "predict_router", "metrics_router"]
