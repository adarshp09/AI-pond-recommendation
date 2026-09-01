from __future__ import annotations

from typing import Any, Dict

import httpx


class GeocodingError(Exception):
    pass


def geocode_place(query: str, timeout: float = 30.0) -> Dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise GeocodingError("query must be a non-empty string.")

    params = {"q": query.strip(), "format": "jsonv2", "limit": 5}

    try:
        response = httpx.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers={"User-Agent": "pond-recommendation-app/1.0"},
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
