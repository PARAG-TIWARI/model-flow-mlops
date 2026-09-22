"""Health and readiness check endpoints."""

import time

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from api.schemas.response import HealthResponse
from src.config.settings import settings
from src.serving.manager import model_manager

router = APIRouter(tags=["Health"])

START_TIME = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness Probe",
    description="Returns service uptime, health status, and whether champion model is resident in memory.",
)
async def health_check():
    """Liveness probe verifying API availability."""
    uptime = round(time.time() - START_TIME, 2)
    is_loaded = model_manager.is_loaded
    version = model_manager.metadata.get("version", "1.0.0") if is_loaded else "unknown"

    return HealthResponse(
        status="healthy",
        uptime_seconds=uptime,
        model_loaded=is_loaded,
        model_version=version,
        environment=settings.environment,
    )


@router.get(
    "/health/ready",
    summary="Readiness Probe",
    description="Kubernetes/container readiness probe returning 200 OK only when model is fully loaded.",
)
async def readiness_check():
    """Readiness probe checking model readiness."""
    if not model_manager.is_loaded:
        try:
            model_manager.load()
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "error": str(e)},
            )
    return {"status": "ready", "model_version": model_manager.metadata.get("version", "1.0.0")}
