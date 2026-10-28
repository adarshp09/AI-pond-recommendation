"""Phase 09 — river / waterway exclusion for pond candidates.

Pond candidates produced from DEM flow accumulation can land directly inside
active river beds or stream channels. This module rejects candidate points that
fall within a configurable safety buffer (default 30 m) of any water body or
waterway in the prepared local GeoPackage.

Design notes
------------
* Reads the ``water_areas`` (polygons) and ``waterways`` (lines) layers from the
  same GeoPackage the runtime land provider uses; both are buffered in metres and
  dissolved into one exclusion zone.
* Distances are computed in a metric CRS (UTM zone derived from the candidates'
  centroid), so ``buffer_meters`` is truly metres regardless of latitude.
* Fully guarded: a missing dataset or absent geospatial stack returns the input
  candidates unchanged (the filter never fabricates exclusions and never crashes
  the pipeline).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("pond_recommendation")

DEFAULT_LAND_GPKG_PATH = "data/osm/prepared/land_context.gpkg"
DEFAULT_BUFFER_METERS = 30.0
_WATER_LAYERS = ("water_areas", "waterways")


def _utm_epsg(longitude: float, latitude: float) -> str:
    zone = int((float(longitude) + 180.0) / 6.0) + 1
    zone = min(max(zone, 1), 60)
    base = 32600 if float(latitude) >= 0 else 32700
    return f"EPSG:{base + zone}"


def _coordinate(candidate: Dict[str, Any], key: str) -> Optional[float]:
    value = candidate.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def exclude_river_candidates(
    candidates: List[Dict[str, Any]],
    land_gpkg_path: str = DEFAULT_LAND_GPKG_PATH,
    buffer_meters: float = DEFAULT_BUFFER_METERS,
) -> List[Dict[str, Any]]:
    """Drop candidate points within ``buffer_meters`` of any water feature.

    Reads ``water_areas`` and ``waterways`` from ``land_gpkg_path``, buffers them
    by ``buffer_meters`` (in a local metric CRS), and returns the candidates that
    do **not** intersect that zone.

    Degrades safely: if the dataset is missing/unreadable or geopandas/pyogrio
    are unavailable, the candidates are returned unchanged.
    """

    candidates = list(candidates or [])
    if not candidates:
        return candidates

    try:
        buffer = float(buffer_meters)
    except (TypeError, ValueError):
        buffer = DEFAULT_BUFFER_METERS
    if buffer < 0:
        buffer = 0.0

    path = Path(land_gpkg_path) if land_gpkg_path else None
    if path is None or not path.exists():
        logger.debug("river_exclusion_dataset_missing", extra={"path": str(land_gpkg_path)})
        return candidates

    try:
        import pyogrio
        from pyproj import Transformer
        from shapely.geometry import Point
        from shapely.ops import unary_union
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("river_exclusion_unavailable", extra={"error": str(exc)})
        return candidates

    longitudes = [v for v in (_coordinate(c, "longitude") for c in candidates) if v is not None]
    latitudes = [v for v in (_coordinate(c, "latitude") for c in candidates) if v is not None]
    if not longitudes or not latitudes:
        return candidates

    mean_lon = sum(longitudes) / len(longitudes)
    mean_lat = sum(latitudes) / len(latitudes)
    metric_crs = _utm_epsg(mean_lon, mean_lat)

    # bbox-limited read (WGS84) padded by the buffer so edge features are included.
    pad_lat = (buffer + 1.0) / 111_320.0
    cos_lat = max(abs(math.cos(math.radians(mean_lat))), 1e-6)
    pad_lon = pad_lat / cos_lat
    bbox = (
        min(longitudes) - pad_lon,
        min(latitudes) - pad_lat,
        max(longitudes) + pad_lon,
        max(latitudes) + pad_lat,
    )

    geometries: List[Any] = []
    for layer in _WATER_LAYERS:
        try:
            frame = pyogrio.read_dataframe(str(path), layer=layer, bbox=bbox, on_invalid="ignore")
        except Exception as exc:
            logger.debug("river_exclusion_layer_failed", extra={"layer": layer, "error": str(exc)})
            continue
        if frame is None or len(frame) == 0:
            continue
        try:
            if frame.crs is None:
                frame = frame.set_crs("EPSG:4326")
            frame = frame.to_crs(metric_crs)
        except Exception as exc:  # pragma: no cover - CRS edge cases
            logger.debug("river_exclusion_reproject_failed", extra={"layer": layer, "error": str(exc)})
            continue
        for geometry in frame.geometry:
            if geometry is not None and not geometry.is_empty:
                geometries.append(geometry)

    if not geometries:
        logger.debug("river_exclusion_no_water_features", extra={"bbox": bbox})
        return candidates

    try:
        water_zone = unary_union(geometries).buffer(buffer)
    except Exception as exc:  # pragma: no cover - geometry edge cases
        logger.warning("river_exclusion_buffer_failed", extra={"error": str(exc)})
        return candidates

    transformer = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    kept: List[Dict[str, Any]] = []
    excluded = 0
    for candidate in candidates:
        longitude = _coordinate(candidate, "longitude")
        latitude = _coordinate(candidate, "latitude")
        if longitude is None or latitude is None:
            kept.append(candidate)
            continue
        try:
            x, y = transformer.transform(longitude, latitude)
        except Exception:  # pragma: no cover - projection edge cases
            kept.append(candidate)
            continue
        if water_zone.intersects(Point(x, y)):
            excluded += 1
            continue
        kept.append(candidate)

    logger.info(
        "river_exclusion_applied",
        extra={
            "input": len(candidates),
            "kept": len(kept),
            "excluded": excluded,
            "buffer_m": buffer,
        },
    )
    return kept


__all__ = ["DEFAULT_BUFFER_METERS", "DEFAULT_LAND_GPKG_PATH", "exclude_river_candidates"]
