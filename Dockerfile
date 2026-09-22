# ==============================================================================
# ModelFlow MLOps — Multi-Stage Production Dockerfile
# ==============================================================================

# Stage 1: Build Dependencies
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

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 2: Production Serving Image
FROM python:3.12-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
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

# Copy application source code, configurations, and trained artifacts
COPY src/ /app/src/
COPY api/ /app/api/
COPY configs/ /app/configs/
COPY params.yaml /app/params.yaml
COPY artifacts/ /app/artifacts/
COPY data/ /app/data/

# Adjust file ownership to non-root user
RUN chown -R mlopsuser:mlopsgroup /app

USER mlopsuser

EXPOSE 8000

# Docker Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production entrypoint
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
