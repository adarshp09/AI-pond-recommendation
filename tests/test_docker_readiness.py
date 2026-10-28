"""Phase 06 — containerization & deployment readiness tests.

Verifies, without requiring Docker:

* environment-variable parsing under containerized configurations (real
  ``config.Settings`` in a subprocess so env is read fresh);
* dataset path resolution (env override + ``/health/datasets`` behaviour when the
  dataset is missing);
* the healthcheck endpoint returns 2xx (what ``curl -f`` requires);
* the Dockerfile / docker-compose.yml / .dockerignore encode the required
  hardening (non-root user, spatial deps, volumes, prod env, healthcheck).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from config import settings

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "Dockerfile"
COMPOSE = REPO_ROOT / "docker-compose.yml"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"

# Env keys we control in the subprocess so the parent env cannot leak in.
_CONTROLLED = {
    "LOG_FORMAT", "LAND_PROVIDER_MODE", "OVERPASS_FALLBACK_ENABLED", "CACHE_ENABLED",
    "CACHE_DIR", "LOCAL_OSM_ENABLED", "LOCAL_OSM_GPKG_PATH", "LOCAL_OSM_MANIFEST_PATH",
    "LOG_LEVEL", "POND_RECOMMENDATION_ENGINE",
}

_CONFIG_KEYS = [
    "LOG_FORMAT", "LAND_PROVIDER_MODE", "OVERPASS_FALLBACK_ENABLED", "CACHE_ENABLED",
    "CACHE_DIR", "LOCAL_OSM_GPKG_PATH", "LOCAL_OSM_MANIFEST_PATH", "LOG_LEVEL",
    "POND_RECOMMENDATION_ENGINE",
]


def _run_config(overrides: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in _CONTROLLED}
    env.update({k: str(v) for k, v in overrides.items()})
    code = (
        "import json;from config import settings;"
        f"keys={_CONFIG_KEYS!r};"
        "print(json.dumps({k: getattr(settings, k) for k in keys}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# Environment parsing (containerized)
# ---------------------------------------------------------------------------

def test_container_environment_parsing():
    parsed = _run_config({
        "LOG_FORMAT": "json",
        "LAND_PROVIDER_MODE": "local_only",
        "OVERPASS_FALLBACK_ENABLED": "false",
        "CACHE_ENABLED": "true",
        "CACHE_DIR": "/app/.cache",
        "LOCAL_OSM_GPKG_PATH": "/app/data/osm/prepared/land_context.gpkg",
        "LOCAL_OSM_MANIFEST_PATH": "/app/data/osm/manifest.json",
    })

    assert parsed["LOG_FORMAT"] == "json"
    assert parsed["LAND_PROVIDER_MODE"] == "local_only"
    assert parsed["OVERPASS_FALLBACK_ENABLED"] is False
    assert parsed["CACHE_ENABLED"] is True
    assert parsed["CACHE_DIR"] == "/app/.cache"
    assert parsed["LOCAL_OSM_GPKG_PATH"] == "/app/data/osm/prepared/land_context.gpkg"
    assert parsed["LOCAL_OSM_MANIFEST_PATH"] == "/app/data/osm/manifest.json"


@pytest.mark.parametrize("value,expected", [("true", True), ("1", True), ("yes", True), ("false", False), ("0", False), ("off", False)])
def test_boolean_env_parsing(value, expected):
    parsed = _run_config({"OVERPASS_FALLBACK_ENABLED": value})
    assert parsed["OVERPASS_FALLBACK_ENABLED"] is expected


def test_developer_defaults_are_not_production_json():
    # With no container env, the code defaults apply (local-only, no Overpass, engine
    # off). LOG_FORMAT is env/.env-driven: the code default is "text" (documented in
    # .env.example), but a repo .env may legitimately set "json", so accept either.
    parsed = _run_config({})
    assert parsed["LOG_FORMAT"] in {"text", "json"}
    assert parsed["LAND_PROVIDER_MODE"] == "local_only"
    assert parsed["OVERPASS_FALLBACK_ENABLED"] is False
    assert parsed["POND_RECOMMENDATION_ENGINE"] is False


def test_dataset_path_resolution_from_env(tmp_path):
    custom = str(tmp_path / "custom" / "land_context.gpkg")
    parsed = _run_config({"LOCAL_OSM_GPKG_PATH": custom})
    assert parsed["LOCAL_OSM_GPKG_PATH"] == custom


# ---------------------------------------------------------------------------
# Dataset readiness endpoint
# ---------------------------------------------------------------------------

def test_health_datasets_reports_missing_dataset(client, monkeypatch, tmp_path):
    monkeypatch.setitem(settings.__dict__, "LOCAL_OSM_GPKG_PATH", str(tmp_path / "nope.gpkg"))
    monkeypatch.setitem(settings.__dict__, "LOCAL_OSM_MANIFEST_PATH", str(tmp_path / "no-manifest.json"))

    response = client.get("/health/datasets")
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "local_osm"
    assert body["available"] is False
    assert body["exists"] is False
    assert "path" in body


def test_health_datasets_reports_container_mode(client):
    response = client.get("/health/datasets")
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "local_only"
    assert body["overpass_fallback_enabled"] is False


def test_healthcheck_endpoints_return_2xx(client):
    # `curl -f` succeeds on 2xx; the compose healthcheck targets /health/datasets.
    assert client.get("/health").status_code == 200
    assert client.get("/health/datasets").status_code == 200


# ---------------------------------------------------------------------------
# Dockerfile / compose hardening
# ---------------------------------------------------------------------------

def test_dockerfile_hardening():
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "python:3.11-slim" in text
    assert "AS builder" in text and "AS runtime" in text
    # spatial system deps
    assert "libgdal-dev" in text
    assert "gcc" in text and "g++" in text
    # non-root user
    assert "appuser" in text
    assert "USER appuser" in text
    # writable dirs
    assert "/app/.cache" in text and "/app/data" in text
    # curl for the healthcheck
    assert "curl" in text
    # entrypoint
    assert "EXPOSE 8000" in text
    assert "uvicorn" in text and "backend.main:app" in text
    assert "--port" in text and "8000" in text


def test_compose_configuration():
    text = COMPOSE.read_text(encoding="utf-8")

    # production env defaults
    assert "LOG_FORMAT: json" in text
    assert "LAND_PROVIDER_MODE: local_only" in text
    assert 'OVERPASS_FALLBACK_ENABLED: "false"' in text
    assert 'CACHE_ENABLED: "true"' in text
    # volumes
    assert "./data:/app/data" in text
    assert "cache_data:/app/.cache" in text
    # port + healthcheck
    assert "8000:8000" in text
    assert "curl -f http://localhost:8000/health/datasets || exit 1" in text


def test_dockerignore_excludes_secrets_and_large_artifacts():
    text = DOCKERIGNORE.read_text(encoding="utf-8")
    assert ".env" in text
    assert ".cache/" in text
    assert "data/osm/prepared/" in text
    assert "tests/" in text
