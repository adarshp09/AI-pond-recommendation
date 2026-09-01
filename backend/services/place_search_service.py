from __future__ import annotations

from typing import Any, Dict

import httpx


class PlaceSearchError(Exception):
    pass


def search_places(query: str, timeout: float = 30.0) -> Dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise PlaceSearchError("query must be a non-empty string.")

    params = {"q": query.strip(), "limit": 5, "lang": "en"}

    try:
        response = httpx.get(
            "https://photon.komoot.io/api/",
            params=params,
            timeout=timeout,
        )
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
