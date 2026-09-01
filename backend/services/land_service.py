from __future__ import annotations

from typing import Any, Dict, List

import httpx


class LandError(Exception):
    pass


def _validate_bbox(south: float, north: float, west: float, east: float) -> None:
    if not all(isinstance(v, (int, float)) for v in [south, north, west, east]):
        raise LandError("All bounding-box values must be numeric.")
    if south >= north:
        raise LandError("south must be smaller than north.")
    if west >= east:
        raise LandError("west must be smaller than east.")


def fetch_land_context(
    south: float,
    west: float,
    north: float,
    east: float,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    _validate_bbox(south, north, west, east)

    overpass_query = f"""
    [out:json][timeout:25];
    (
      way[water][bbox:{south},{west},{north},{east}];
      way[waterway][bbox:{south},{west},{north},{east}];
      way[highway][bbox:{south},{west},{north},{east}];
      node[building][bbox:{south},{west},{north},{east}];
      way[building][bbox:{south},{west},{north},{east}];
    );
    out center tags geom;
    """

    try:
        response = httpx.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": overpass_query},
            timeout=timeout,
            headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
        )
        response.raise_for_status()
    except TimeoutError as exc:
        raise LandError(f"Unable to connect to OSM Overpass: timeout ({timeout}s)") from exc
    except httpx.HTTPError as exc:
        raise LandError(f"Unable to connect to OSM Overpass: {exc}") from exc

    payload = response.json()
    elements = payload.get("elements", [])

    water = []
    roads = []
    buildings = []

    for element in elements:
        tags = element.get("tags", {})
        if tags.get("water") or tags.get("waterway"):
            water.append(element)
        elif tags.get("highway"):
            roads.append(element)
        elif tags.get("building"):
            buildings.append(element)

    return {
        "bbox": {
            "south": float(south),
            "west": float(west),
            "north": float(north),
            "east": float(east),
        },
        "water_bodies": water,
        "roads": roads,
        "buildings": buildings,
        "source": "OpenStreetMap Overpass API",
    }
