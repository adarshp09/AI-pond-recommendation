from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import httpx

from config import settings
from services.cache import get_or_set

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_USER_AGENT = "pond-recommendation-app/1.0"


class GeocodingError(Exception):
    pass


# Nominatim usage policy asks for at most ~1 request/second. Serialise calls and
# space them out (configurable; 0 disables the throttle).
_throttle_lock = threading.Lock()
_last_call_monotonic = 0.0


def _throttle() -> None:
    global _last_call_monotonic

    interval = float(getattr(settings, "GEOCODING_MIN_INTERVAL_S", 1.0))
    if interval <= 0:
        return
    with _throttle_lock:
        now = time.monotonic()
        wait = interval - (now - _last_call_monotonic)
        if wait > 0:
            time.sleep(wait)
        _last_call_monotonic = time.monotonic()


def _fetch_once(query: str, timeout: float) -> Dict[str, Any]:
    params = {"q": query.strip(), "format": "jsonv2", "limit": 5}

    try:
        response = httpx.get(
            NOMINATIM_URL,
            params=params,
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
    except TimeoutError as exc:
        raise GeocodingError(f"Unable to connect to Nominatim: timeout ({timeout}s)") from exc
    except httpx.HTTPError as exc:
        raise GeocodingError(f"Unable to connect to Nominatim: {exc}") from exc

    payload = response.json()
    results = []
    for item in payload:
        results.append({
            "display_name": item.get("display_name"),
            "latitude": float(item.get("lat", 0.0)),
            "longitude": float(item.get("lon", 0.0)),
            "address": item.get("address", {}),
            "source": "Nominatim API",
        })

    return {"query": query.strip(), "results": results, "source": "Nominatim API"}


def geocode_place(query: str, timeout: float = 30.0) -> Dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise GeocodingError("query must be a non-empty string.")

    cache_key = {"provider": "nominatim", "query": query.strip()}

    def _compute() -> Dict[str, Any]:
        retries = int(getattr(settings, "GEOCODING_RETRIES", 2))
        last_error: Optional[GeocodingError] = None
        for attempt in range(retries + 1):
            try:
                _throttle()
                return _fetch_once(query, timeout)
            except GeocodingError as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(min(0.1 * (2 ** attempt), 0.5))
        assert last_error is not None
        raise last_error

    return get_or_set(
        "geocoding",
        cache_key,
        _compute,
        ttl_seconds=int(getattr(settings, "GEOCODING_CACHE_TTL_S", 60 * 60 * 24 * 7)),
    )
