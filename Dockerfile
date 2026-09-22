# ==============================================================================
# ModelFlow MLOps — Multi-Stage Production Dockerfile
# ==============================================================================

# Stage 1: Build Dependencies & Execute Training Pipeline
FROM python:3.12-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt

# Copy source code, scripts, configurations, data, artifacts, and parameters
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY params.yaml ./params.yaml
COPY data/ ./data/
COPY artifacts/ ./artifacts/

# Set environment variables for pipeline execution
ENV PYTHONPATH="/build" \
    PYTHONUNBUFFERED=1

# Execute the complete 5-stage training, evaluation, and drift pipeline
RUN /opt/venv/bin/python -m scripts.run_pipeline

# Verify that champion model and pipeline artifacts exist before proceeding
RUN test -f artifacts/models/champion_model.joblib && \
    test -f artifacts/models/model_metadata.json && \
    test -f artifacts/reports/evaluation_metrics.json && \
    test -f artifacts/reports/drift_report.json && \
    test -f data/processed/train.csv

# Stage 2: Production Serving Image
FROM python:3.12-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PORT=8000 \
    ENVIRONMENT=production

# Install curl for container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-privileged system user for security
RUN groupadd -r mlopsgroup && useradd -r -g mlopsgroup -u 10001 mlopsuser

# Copy application source code and configurations
COPY src/ /app/src/
COPY api/ /app/api/
COPY configs/ /app/configs/
COPY params.yaml ./params.yaml

# Copy trained model artifacts, reports, datasets, and tracking database from builder
COPY --from=builder /build/artifacts /app/artifacts
COPY --from=builder /build/data /app/data
COPY --from=builder /build/mlflow.db* /app/

# Build-time verification inside runtime image
RUN test -f /app/artifacts/models/champion_model.joblib && \
    test -f /app/artifacts/models/model_metadata.json && \
    test -f /app/data/processed/train.csv

# Adjust file ownership to non-root user
RUN chown -R mlopsuser:mlopsgroup /app

USER mlopsuser

EXPOSE 8000

# Docker Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production entrypoint
CMD ["sh", "-c", "/opt/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
