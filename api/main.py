"""FastAPI Main Application for ModelFlow MLOps Platform."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.health import router as health_router
from api.routes.metrics import router as metrics_router
from api.routes.model import router as model_router
from api.routes.predict import router as predict_router
from src.config.settings import settings
from src.serving.manager import model_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("modelflow.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle hook: Preloads champion model at startup."""
    logger.info("Initializing ModelFlow MLOps Serving Engine...")
    try:
        model_manager.load()
        logger.info("Champion model successfully loaded at startup.")
    except Exception as e:
        logger.warning(
            f"Champion model could not be loaded at startup: {e}. "
            "Server running in standby mode. Train a model to enable live predictions."
        )
    yield
    logger.info("Shutting down ModelFlow MLOps Serving Engine.")


app = FastAPI(
    title="ModelFlow MLOps",
    description="Reproducible Machine Learning Training, Tracking, Serving and Monitoring Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware for modern frontend consumption
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include route handlers
app.include_router(health_router)
app.include_router(model_router)
app.include_router(predict_router)
app.include_router(metrics_router)


@app.get("/", tags=["Root"])
async def root():
    """Root platform index returning service status and documentation endpoints."""
    return {
        "platform": "ModelFlow MLOps",
        "description": "Reproducible Machine Learning Training, Tracking, Serving and Monitoring Platform",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "model": "/model",
        "metrics": "/metrics",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.host, port=settings.port, reload=False)
