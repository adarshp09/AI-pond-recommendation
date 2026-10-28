from __future__ import annotations

import time
from typing import Any, Dict, Optional

import httpx

from config import settings
from services.cache import get_or_set

PHOTON_URL = "https://photon.komoot.io/api/"


class PlaceSearchError(Exception):
    pass


def _fetch_once(query: str, timeout: float) -> Dict[str, Any]:
    params = {"q": query.strip(), "limit": 5, "lang": "en"}

    try:
        response = httpx.get(PHOTON_URL, params=params, timeout=timeout)
        response.raise_for_status()
    except TimeoutError as exc:
        raise PlaceSearchError(f"Unable to connect to Photon: timeout ({timeout}s)") from exc
    except httpx.HTTPError as exc:
        raise PlaceSearchError(f"Unable to connect to Photon: {exc}") from exc

    payload = response.json()
    results = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates", [0.0, 0.0])
        results.append({
            "name": props.get("name"),
            "country": props.get("country"),
            "latitude": float(coords[1]),
            "longitude": float(coords[0]),
            "source": "Photon API",
        })

    return {"query": query.strip(), "results": results, "source": "Photon API"}


def search_places(query: str, timeout: float = 30.0) -> Dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise PlaceSearchError("query must be a non-empty string.")

    cache_key = {"provider": "photon", "query": query.strip()}

    def _compute() -> Dict[str, Any]:
        retries = int(getattr(settings, "PLACE_SEARCH_RETRIES", 2))
        last_error: Optional[PlaceSearchError] = None
        for attempt in range(retries + 1):
            try:
                return _fetch_once(query, timeout)
            except PlaceSearchError as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(min(0.1 * (2 ** attempt), 0.5))
        assert last_error is not None
        raise last_error

    return get_or_set(
        "place_search",
        cache_key,
        _compute,
        ttl_seconds=int(getattr(settings, "PLACE_SEARCH_CACHE_TTL_S", 60 * 60 * 24 * 7)),
    )
