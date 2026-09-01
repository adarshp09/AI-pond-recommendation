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


@dataclass(frozen=True)
class Settings:
    OPENTOPOGRAPHY_API_KEY: str | None = _env("OPENTOPOGRAPHY_API_KEY", None, cast=str)
    OPENTOPOGRAPHY_DATASET: str = _env("OPENTOPOGRAPHY_DATASET", "SRTMGL1", cast=str)
    OPENTOPOGRAPHY_URL: str = _env("OPENTOPOGRAPHY_URL", "https://portal.opentopography.org/API/globaldem", cast=str)
    OPENMETEO_ARCHIVE_URL: str = _env("OPENMETEO_ARCHIVE_URL", "https://archive-api.open-meteo.com/v1/archive", cast=str)
    OVERPASS_URL: str = _env("OVERPASS_URL", "https://overpass-api.de/api/interpreter", cast=str)
    NOMINATIM_URL: str = _env("NOMINATIM_URL", "https://nominatim.openstreetmap.org/search", cast=str)
    PHOTON_URL: str = _env("PHOTON_URL", "https://photon.komoot.io/api/", cast=str)
    HTTP_TIMEOUT_SECONDS: float = _env("HTTP_TIMEOUT_SECONDS", 30.0, cast=float)
    HTTP_RETRY_COUNT: int = _env("HTTP_RETRY_COUNT", 3, cast=int)
    CACHE_DIR: str = _env("CACHE_DIR", str(Path(__file__).resolve().parent / ".cache"), cast=str)
    CACHE_TTL_SECONDS: int = _env("CACHE_TTL_SECONDS", 3600, cast=int)
    DEM_RESOLUTION_M: float = _env("DEM_RESOLUTION_M", 5.0, cast=float)
    MAX_UPLOAD_SIZE_MB: int = _env("MAX_UPLOAD_SIZE_MB", 100, cast=int)
    LOG_LEVEL: str = _env("LOG_LEVEL", "INFO", cast=str).upper()
    CACHE_ENABLED: bool = _env("CACHE_ENABLED", True, cast=bool)
    ENABLE_RETRIES: bool = _env("ENABLE_RETRIES", True, cast=bool)

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
