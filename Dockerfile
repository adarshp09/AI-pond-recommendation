# syntax=docker/dockerfile:1
# AI Pond Recommendation — multi-stage image (Phase 06).
#
# Builder stage installs the geospatial toolchain and Python deps into a venv;
# the runtime stage copies only the venv + runtime libraries, runs as a non-root
# user, and serves the API on :8000.

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# System-level spatial/build dependencies for pyogrio, shapely, GDAL, rasterio.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    LOG_FORMAT=json \
    LAND_PROVIDER_MODE=local_only \
    OVERPASS_FALLBACK_ENABLED=false \
    CACHE_ENABLED=true \
    CACHE_DIR=/app/.cache \
    LOCAL_OSM_GPKG_PATH=/app/data/osm/prepared/land_context.gpkg \
    LOCAL_OSM_MANIFEST_PATH=/app/data/osm/manifest.json

# Runtime spatial libraries + curl (used by the HEALTHCHECK).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY . /app

# Non-root user + writable working/cache/data directories.
RUN groupadd --system appuser \
    && useradd --system --gid appuser --home-dir /app --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/.cache /app/data \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health/datasets || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
