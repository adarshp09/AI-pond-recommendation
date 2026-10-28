from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str | int | float | bool | None = None, *, cast=str):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        if cast is bool:
            return value.strip().lower() in {"1", "true", "yes", "on"}
        if cast is int:
            return int(value)
        if cast is float:
            return float(value)
        if cast is str:
            return value
    except (TypeError, ValueError):
        return default
    return value


# ---------------------------------------------------------------------------
# Container -> native fallbacks
# ---------------------------------------------------------------------------
# The same environment may be reused inside a container (docker compose) and on
# a bare host (e.g. sys1, where Docker is unavailable). docker-compose service
# names (jaeger, db, redis, ...) do not resolve on a bare host, so they are
# rewritten to "localhost" when we are NOT running inside a container. Inside a
# container nothing is rewritten, so compose networking keeps working.
#
# Note: this project has no database and no Redis/queue, so the only URL that
# actually needs this today is the Jaeger OTLP endpoint.
_CONTAINER_SERVICE_HOSTS = {
    "jaeger",
    "otel-collector",
    "otel_collector",
    "db",
    "postgres",
    "postgresql",
    "redis",
    "mongodb",
    "mongo",
    "mysql",
    "rabbitmq",
}


def running_in_container() -> bool:
    """True when the process runs inside Docker/Podman/Kubernetes."""

    return (
        os.path.exists("/.dockerenv")
        or os.path.exists("/run/.containerenv")
        or bool(os.getenv("KUBERNETES_SERVICE_HOST"))
    )


def native_service_url(value: str | None, default: str | None = None) -> str | None:
    """Point a container-service URL at ``localhost`` when running natively.

    ``http://jaeger:4318`` -> ``http://localhost:4318``. The rewrite is skipped
    inside a container, and any host that is not a known compose service name is
    returned unchanged.
    """

    if value is None:
        return default
    if running_in_container():
        return value
    try:
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(value)
    except ValueError:
        return value
    host = parts.hostname
    if not host or host not in _CONTAINER_SERVICE_HOSTS:
        return value
    netloc = parts.netloc.replace(host, "localhost", 1)
    return urlunsplit(parts._replace(netloc=netloc))


@dataclass(frozen=True)
class Settings:
    OPENTOPOGRAPHY_API_KEY: str | None = _env("OPENTOPOGRAPHY_API_KEY", None, cast=str)
    OPENTOPOGRAPHY_DATASET: str = _env("OPENTOPOGRAPHY_DATASET", "SRTMGL1", cast=str)
    OPENTOPOGRAPHY_URL: str = _env("OPENTOPOGRAPHY_URL", "https://portal.opentopography.org/API/globaldem", cast=str)
    OPENTOPOGRAPHY_CONNECT_TIMEOUT: float = _env("OPENTOPOGRAPHY_CONNECT_TIMEOUT", 10.0, cast=float)
    OPENTOPOGRAPHY_READ_TIMEOUT: float = _env("OPENTOPOGRAPHY_READ_TIMEOUT", 120.0, cast=float)
    OPENTOPOGRAPHY_MAX_RETRIES: int = _env("OPENTOPOGRAPHY_MAX_RETRIES", 2, cast=int)
    OPENMETEO_ARCHIVE_URL: str = _env("OPENMETEO_ARCHIVE_URL", "https://archive-api.open-meteo.com/v1/archive", cast=str)
    OVERPASS_URL: str = _env("OVERPASS_URL", "https://overpass-api.de/api/interpreter", cast=str)
    OVERPASS_FALLBACK_URL: str = _env("OVERPASS_FALLBACK_URL", "https://overpass.kumi.systems/api/interpreter", cast=str)
    OVERPASS_SECONDARY_FALLBACK_URL: str = _env("OVERPASS_SECONDARY_FALLBACK_URL", "https://overpass.private.coffee/api/interpreter", cast=str)
    OVERPASS_CONNECT_TIMEOUT: float = _env("OVERPASS_CONNECT_TIMEOUT", 5.0, cast=float)
    OVERPASS_READ_TIMEOUT: float = _env("OVERPASS_READ_TIMEOUT", 25.0, cast=float)
    OVERPASS_RETRIES: int = _env("OVERPASS_RETRIES", 2, cast=int)
    OVERPASS_USER_AGENT: str = _env("OVERPASS_USER_AGENT", "AI-Pond-App/1.0", cast=str)

    # Phase 03 — land provider selection & Overpass policy.
    # Runtime geographic enrichment is served exclusively from the prepared local
    # GeoPackage. Overpass is NOT queried at runtime; the Overpass chain
    # (land_service) is retained ONLY for explicit offline tooling
    # (mode="overpass_only" in scripts/). See docs/architecture.md §7.
    LOCAL_OSM_ENABLED: bool = _env("LOCAL_OSM_ENABLED", True, cast=bool)
    LOCAL_OSM_GPKG_PATH: str = _env(
        "LOCAL_OSM_GPKG_PATH",
        str(Path(__file__).resolve().parent / "data" / "osm" / "prepared" / "land_context.gpkg"),
        cast=str,
    )
    LOCAL_OSM_MANIFEST_PATH: str = _env(
        "LOCAL_OSM_MANIFEST_PATH",
        str(Path(__file__).resolve().parent / "data" / "osm" / "manifest.json"),
        cast=str,
    )
    LOCAL_OSM_MAX_AGE_DAYS: float = _env("LOCAL_OSM_MAX_AGE_DAYS", 180.0, cast=float)
    LOCAL_OSM_QUERY_BUFFER_M: float = _env("LOCAL_OSM_QUERY_BUFFER_M", 0.0, cast=float)
    # "local_first" | "local_only" | "overpass_only" — default local_only.
    LAND_PROVIDER_MODE: str = _env("LAND_PROVIDER_MODE", "local_only", cast=str)
    OVERPASS_FALLBACK_ENABLED: bool = _env("OVERPASS_FALLBACK_ENABLED", False, cast=bool)

    # CORS — origins allowed to call the API from the browser.
    # CORS_ALLOW_ORIGINS: extra comma-separated exact origins.
    # CORS_ALLOW_ORIGIN_REGEX: default allows the frontend port on localhost and
    # private (RFC1918) addresses, so the UI works when served from another host.
    CORS_ALLOW_ORIGINS: str = _env("CORS_ALLOW_ORIGINS", "", cast=str)
    CORS_ALLOW_ORIGIN_REGEX: str = _env(
        "CORS_ALLOW_ORIGIN_REGEX",
        r"^https?://(localhost|127\.0\.0\.1|\[::1?\]"
        r"|10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        r"|192\.168\.\d{1,3}\.\d{1,3}"
        r"|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(:\d{1,5})?$",
        cast=str,
    )

    NOMINATIM_URL: str = _env("NOMINATIM_URL", "https://nominatim.openstreetmap.org/search", cast=str)
    PHOTON_URL: str = _env("PHOTON_URL", "https://photon.komoot.io/api/", cast=str)
    HTTP_TIMEOUT_SECONDS: float = _env("HTTP_TIMEOUT_SECONDS", 30.0, cast=float)
    HTTP_RETRY_COUNT: int = _env("HTTP_RETRY_COUNT", 3, cast=int)
    CACHE_DIR: str = _env("CACHE_DIR", str(Path(__file__).resolve().parent / ".cache"), cast=str)
    CACHE_TTL_SECONDS: int = _env("CACHE_TTL_SECONDS", 3600, cast=int)
    DEM_RESOLUTION_M: float = _env("DEM_RESOLUTION_M", 5.0, cast=float)

    # Phase 02 — opt-in recommendation engine.
    #   false (default) = legacy heuristic pipeline (suitability + recommendation.py)
    #   true            = tested engine: candidates -> geographic_context ->
    #                     rainfall_runoff -> recommendation_engine (real distances,
    #                     feasibility, confidence, deterministic ranking).
    # Kept default-false so production behaviour is unchanged unless opted in.
    POND_RECOMMENDATION_ENGINE: bool = _env("POND_RECOMMENDATION_ENGINE", False, cast=bool)
    MAX_UPLOAD_SIZE_MB: int = _env("MAX_UPLOAD_SIZE_MB", 100, cast=int)
    LOG_LEVEL: str = _env("LOG_LEVEL", "INFO", cast=str).upper()
    # Phase 05 — log rendering: "text" for local dev, "json" for production.
    LOG_FORMAT: str = _env("LOG_FORMAT", "text", cast=str).lower()
    CACHE_ENABLED: bool = _env("CACHE_ENABLED", True, cast=bool)
    ENABLE_RETRIES: bool = _env("ENABLE_RETRIES", True, cast=bool)

    # Phase 04 — layered caching TTLs (seconds) and provider retries.
    OPENTOPOGRAPHY_CACHE_TTL_S: int = _env("OPENTOPOGRAPHY_CACHE_TTL_S", 60 * 60 * 24 * 30, cast=int)
    RAINFALL_CACHE_TTL_S: int = _env("RAINFALL_CACHE_TTL_S", 60 * 60 * 24 * 30, cast=int)
    RAINFALL_RETRIES: int = _env("RAINFALL_RETRIES", 2, cast=int)
    GEOCODING_CACHE_TTL_S: int = _env("GEOCODING_CACHE_TTL_S", 60 * 60 * 24 * 7, cast=int)
    GEOCODING_RETRIES: int = _env("GEOCODING_RETRIES", 2, cast=int)
    GEOCODING_MIN_INTERVAL_S: float = _env("GEOCODING_MIN_INTERVAL_S", 1.0, cast=float)
    PLACE_SEARCH_CACHE_TTL_S: int = _env("PLACE_SEARCH_CACHE_TTL_S", 60 * 60 * 24 * 7, cast=int)
    PLACE_SEARCH_RETRIES: int = _env("PLACE_SEARCH_RETRIES", 2, cast=int)
    LOCAL_OSM_CACHE_TTL_S: int = _env("LOCAL_OSM_CACHE_TTL_S", 3600, cast=int)
    # Optional admin token required by POST /cache/clear (empty = open in dev).
    CACHE_ADMIN_TOKEN: str = _env("CACHE_ADMIN_TOKEN", "", cast=str)

    # Phase 08 — OpenTelemetry tracing + Prometheus metrics.
    # OTEL_ENABLED=false by default: no exporters/instrumentation are registered.
    OTEL_ENABLED: bool = _env("OTEL_ENABLED", False, cast=bool)
    OTEL_SERVICE_NAME: str = _env("OTEL_SERVICE_NAME", "ai-pond-recommendation", cast=str)
    # Container compose points this at the "jaeger" service; natively the same
    # value is auto-rewritten to localhost (see native_service_url above).
    OTEL_EXPORTER_OTLP_ENDPOINT: str = (
        native_service_url(_env("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318", cast=str))
        or "http://localhost:4318"
    )
    PROMETHEUS_METRICS_ENABLED: bool = _env("PROMETHEUS_METRICS_ENABLED", True, cast=bool)

    # Phase 09 — river/waterway exclusion.
    # Reject pond candidates that fall within ``WATERBODY_BUFFER_METERS`` of any
    # waterway or water body in the local GeoPackage (safety constraint).
    RIVER_EXCLUSION_ENABLED: bool = _env("RIVER_EXCLUSION_ENABLED", True, cast=bool)
    WATERBODY_BUFFER_METERS: float = _env("WATERBODY_BUFFER_METERS", 30.0, cast=float)

    @property
    def MAX_UPLOAD_SIZE_BYTES(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


settings = Settings()


def redact_secret(value: str | None) -> str:
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "***"
    return f"{value[:2]}***{value[-2:]}"


def redact_url(url: str) -> str:
    if not url:
        return url
    try:
        from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
        parsed = urlsplit(url)
        query = parse_qsl(parsed.query, keep_blank_values=True)
        redacted = []
        for key, val in query:
            if key.lower() in {"api_key", "key", "apikey", "token", "access_token"}:
                redacted.append((key, "[REDACTED]"))
            else:
                redacted.append((key, val))
        new_query = urlencode(redacted, doseq=True)
        return urlunsplit(parsed._replace(query=new_query))
    except Exception:
        return url
