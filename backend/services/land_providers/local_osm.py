"""Local OSM GeoPackage provider.

Reads the spatially indexed GeoPackage produced by
``scripts/prepare_osm_data.py`` and returns the same normalised land-context payload
as the Overpass provider — without any network access.

Heavy geospatial dependencies (``geopandas``/``pyogrio``) are imported *lazily* so this
module imports cleanly on machines without the geospatial stack; ``available()`` then
returns ``False`` and the selector falls back per policy instead of crashing.

Resource management: the dataset descriptor (path, manifest, coverage, layers) is
validated once and cached, guarded by a lock, and reused across queries. GDAL opens the
GeoPackage per query and uses the R-tree spatial index for the bounding-box read.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import settings

try:
    from services.cache import cache_is_enabled, get_cache
except ImportError:  # pragma: no cover - alternative import path
    from ..cache import cache_is_enabled, get_cache

from .base import (
    COVERAGE_FULL,
    COVERAGE_NONE,
    COVERAGE_PARTIAL,
    STATUS_AVAILABLE,
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_MISSING,
    STATUS_OUT_OF_COVERAGE,
    STATUS_STALE,
    STATUS_UNAVAILABLE,
    STATUS_UNREADABLE,
    LandContextResult,
    LandProvider,
    ProviderUnavailable,
    bbox_dict,
    bbox_relation,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME = "local_osm"
SOURCE_TEMPLATE = "Local OSM GeoPackage ({path})"

# Layer -> the payload list it feeds.
_LAYER_TO_PAYLOAD: Dict[str, str] = {
    "water_areas": "water_bodies",
    "waterways": "water_bodies",
    "roads": "roads",
    "buildings": "buildings",
}
_REQUIRED_LAYERS = tuple(_LAYER_TO_PAYLOAD)

_DESCRIPTOR_CACHE: Dict[Tuple[str, float], "_DatasetDescriptor"] = {}
_DESCRIPTOR_LOCK = threading.Lock()


def _geostack():
    """Import the geospatial stack lazily; return ``(gpd, pyogrio)`` or ``None``."""

    try:
        import geopandas as gpd  # noqa: WPS433
        import pyogrio  # noqa: WPS433
    except ImportError:  # pragma: no cover - environment dependent
        return None
    return gpd, pyogrio


def _parse_iso(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass
class _DatasetDescriptor:
    path: str
    exists: bool
    readable: bool
    manifest: Optional[Dict[str, Any]]
    coverage: Optional[Tuple[float, float, float, float]]
    generated_at: Optional[str]
    age_days: Optional[float]
    layers: Tuple[str, ...]
    error: Optional[str] = None

    @property
    def has_coverage(self) -> bool:
        return self.coverage is not None


def _coverage_from_manifest(manifest: Dict[str, Any] | None) -> Optional[Tuple[float, float, float, float]]:
    if not manifest:
        return None
    area = manifest.get("study_area") or {}
    try:
        return (
            float(area["south"]),
            float(area["west"]),
            float(area["north"]),
            float(area["east"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


class LocalOSMProvider(LandProvider):
    name = PROVIDER_NAME

    def __init__(
        self,
        gpkg_path: str | Path | None,
        *,
        manifest_path: str | Path | None = None,
        max_age_days: float | None = None,
        layers: Tuple[str, ...] = _REQUIRED_LAYERS,
        query_buffer_m: float = 0.0,
    ) -> None:
        self.gpkg_path = str(gpkg_path) if gpkg_path else ""
        self.manifest_path = str(manifest_path) if manifest_path else None
        self.max_age_days = max_age_days
        self.layers = tuple(layers)
        self.query_buffer_m = float(query_buffer_m or 0.0)
        self._log = logger

    # -- lifecycle ---------------------------------------------------------

    def _descriptor(self) -> _DatasetDescriptor:
        path = Path(self.gpkg_path) if self.gpkg_path else Path("")
        if not self.gpkg_path:
            return _DatasetDescriptor("", False, False, None, None, None, None, (), "no dataset path configured")

        try:
            mtime = path.stat().st_mtime
        except OSError as exc:
            return _DatasetDescriptor(str(path), False, False, None, None, None, None, (), str(exc))

        key = (str(path), mtime)
        with _DESCRIPTOR_LOCK:
            cached = _DESCRIPTOR_CACHE.get(key)
        if cached is not None:
            return cached

        descriptor = self._build_descriptor(path)
        with _DESCRIPTOR_LOCK:
            _DESCRIPTOR_CACHE.clear()
            _DESCRIPTOR_CACHE[key] = descriptor
        return descriptor

    def _build_descriptor(self, path: Path) -> _DatasetDescriptor:
        manifest: Optional[Dict[str, Any]] = None
        manifest_candidates = [Path(self.manifest_path)] if self.manifest_path else []
        manifest_candidates.append(path.parent.parent / "manifest.json")
        for candidate in manifest_candidates:
            if candidate and candidate.exists():
                try:
                    manifest = json.loads(candidate.read_text(encoding="utf-8"))
                    break
                except (OSError, json.JSONDecodeError) as exc:
                    self._log.debug("local_osm_manifest_unreadable", extra={"error": str(exc)})

        coverage = _coverage_from_manifest(manifest)
        generated_at = (manifest or {}).get("generated_at")
        age_days: Optional[float] = None
        parsed = _parse_iso(generated_at)
        if parsed is not None:
            age_days = (datetime.now(timezone.utc) - parsed).total_seconds() / 86_400.0

        stack = _geostack()
        if stack is None:
            return _DatasetDescriptor(
                str(path), True, False, manifest, coverage, generated_at, age_days, (),
                "geopandas/pyogrio not installed",
            )
        _, pyogrio = stack
        try:
            found = {name for name, _ in pyogrio.list_layers(path)}
        except Exception as exc:  # corrupt / not a GPKG
            return _DatasetDescriptor(
                str(path), True, False, manifest, coverage, generated_at, age_days, (), str(exc)
            )

        missing = [layer for layer in self.layers if layer not in found]
        if missing:
            return _DatasetDescriptor(
                str(path), True, False, manifest, coverage, generated_at, age_days, tuple(found),
                f"missing layers: {missing}",
            )

        if coverage is None:
            return _DatasetDescriptor(
                str(path), True, False, manifest, None, generated_at, age_days, tuple(found),
                "manifest has no study_area coverage",
            )

        return _DatasetDescriptor(
            str(path), True, True, manifest, coverage, generated_at, age_days, tuple(found)
        )

    # -- inspection --------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        descriptor = self._descriptor()
        return {
            "provider": self.name,
            "path": descriptor.path,
            "exists": descriptor.exists,
            "readable": descriptor.readable,
            "coverage": list(descriptor.coverage) if descriptor.coverage else None,
            "generated_at": descriptor.generated_at,
            "age_days": round(descriptor.age_days, 2) if descriptor.age_days is not None else None,
            "max_age_days": self.max_age_days,
            "layers": list(descriptor.layers),
            "error": descriptor.error,
        }

    def available(self) -> bool:
        descriptor = self._descriptor()
        if not descriptor.readable:
            return False
        if self.max_age_days is not None:
            if descriptor.age_days is None or descriptor.age_days > float(self.max_age_days):
                return False
        return True

    def is_stale(self) -> bool:
        if self.max_age_days is None:
            return False
        descriptor = self._descriptor()
        if descriptor.age_days is None:
            return True
        return descriptor.age_days > float(self.max_age_days)

    def coverage_status(self, south: float, west: float, north: float, east: float) -> str:
        descriptor = self._descriptor()
        if descriptor.coverage is None:
            return COVERAGE_NONE
        return bbox_relation((south, west, north, east), descriptor.coverage)

    def covers(self, south: float, west: float, north: float, east: float) -> bool:
        return self.available() and self.coverage_status(south, west, north, east) == COVERAGE_FULL

    # -- query -------------------------------------------------------------

    def _buffer(self, south: float, west: float, north: float, east: float):
        if self.query_buffer_m <= 0:
            return south, west, north, east
        lat_pad = self.query_buffer_m / 111_320.0
        centre_lat = (south + north) / 2.0
        cos_lat = max(abs(math.cos(math.radians(centre_lat))), 1e-6)
        lon_pad = self.query_buffer_m / (111_320.0 * cos_lat)
        return south - lat_pad, west - lon_pad, north + lat_pad, east + lon_pad

    def fetch(self, south: float, west: float, north: float, east: float) -> LandContextResult:
        descriptor = self._descriptor()
        bbox = bbox_dict(south, west, north, east)

        if not self.gpkg_path:
            raise ProviderUnavailable(STATUS_MISSING, "no local OSM dataset path configured")
        if not descriptor.exists:
            raise ProviderUnavailable(STATUS_MISSING, f"local OSM dataset not found: {descriptor.path}")
        if not descriptor.readable:
            raise ProviderUnavailable(STATUS_UNREADABLE, f"local OSM dataset unusable: {descriptor.error}")

        relation = self.coverage_status(south, west, north, east)
        if relation != COVERAGE_FULL:
            raise ProviderUnavailable(
                STATUS_OUT_OF_COVERAGE,
                f"query bbox {relation} coverage for local OSM dataset",
            )
        if self.is_stale():
            age = (
                f"{descriptor.age_days:.1f} days old"
                if descriptor.age_days is not None
                else "of unknown age"
            )
            raise ProviderUnavailable(
                STATUS_STALE,
                f"local OSM dataset is {age} (max {self.max_age_days})",
            )

        # Layer-2 cache (off by default in tests): reuse the normalised result for
        # an identical query against the same dataset revision. Keyed by the
        # dataset path + mtime so a rebuilt dataset never serves stale features.
        cache = None
        cache_key = None
        if cache_is_enabled():
            try:
                mtime = Path(descriptor.path).stat().st_mtime
            except OSError:
                mtime = None
            if mtime is not None:
                cache = get_cache()
                cache_key = {
                    "provider": "local_osm",
                    "path": descriptor.path,
                    "mtime": mtime,
                    "bbox": [float(south), float(west), float(north), float(east)],
                    "buffer": self.query_buffer_m,
                    "layers": list(self.layers),
                }
                cached = cache.get(
                    "local_osm",
                    cache_key,
                    ttl_seconds=int(getattr(settings, "LOCAL_OSM_CACHE_TTL_S", 3600)),
                )
                if isinstance(cached, dict):
                    return LandContextResult(
                        bbox=bbox,
                        water_bodies=cached.get("water_bodies", []),
                        roads=cached.get("roads", []),
                        buildings=cached.get("buildings", []),
                        source=SOURCE_TEMPLATE.format(path=Path(descriptor.path).name),
                        provider=self.name,
                        status=cached.get("status", STATUS_AVAILABLE),
                        coverage=relation,
                        feature_counts=cached.get("feature_counts"),
                    )

        stack = _geostack()
        if stack is None:  # pragma: no cover - guarded by descriptor.readable
            raise ProviderUnavailable(STATUS_UNAVAILABLE, "geopandas/pyogrio not installed")
        gpd, pyogrio = stack  # noqa: F841

        qs, qw, qn, qe = self._buffer(south, west, north, east)
        began = time.perf_counter()
        elements: Dict[str, List[Dict[str, Any]]] = {"water_bodies": [], "roads": [], "buildings": []}
        try:
            for layer, payload_key in _LAYER_TO_PAYLOAD.items():
                frame = pyogrio.read_dataframe(
                    descriptor.path, layer=layer, bbox=(qw, qs, qe, qn), on_invalid="ignore"
                )
                elements[payload_key].extend(_frame_to_elements(frame, payload_key))
        except Exception as exc:  # pragma: no cover - IO/corruption
            raise ProviderUnavailable(STATUS_ERROR, f"local OSM query failed: {exc}") from exc

        elapsed_ms = (time.perf_counter() - began) * 1000.0
        counts = {key: len(value) for key, value in elements.items()}
        status = STATUS_AVAILABLE if any(counts.values()) else STATUS_EMPTY
        result = LandContextResult(
            bbox=bbox,
            water_bodies=elements["water_bodies"],
            roads=elements["roads"],
            buildings=elements["buildings"],
            source=SOURCE_TEMPLATE.format(path=Path(descriptor.path).name),
            provider=self.name,
            status=status,
            coverage=relation,
            feature_counts=counts,
        )
        self._log.debug(
            "local_osm_query",
            extra={"counts": counts, "elapsed_ms": round(elapsed_ms, 2), "status": status},
        )
        if cache is not None and cache_key is not None:
            try:
                cache.set(
                    "local_osm",
                    cache_key,
                    {
                        "status": status,
                        "feature_counts": counts,
                        "water_bodies": result.water_bodies,
                        "roads": result.roads,
                        "buildings": result.buildings,
                    },
                )
            except Exception:  # pragma: no cover - cache write is best-effort
                pass
        return result


# ---------------------------------------------------------------------------
# Geometry -> Overpass-style element conversion
# ---------------------------------------------------------------------------

def _coords_lonlat(ring) -> List[Dict[str, float]]:
    return [{"lat": float(lat), "lon": float(lon)} for lon, lat in ring]


def _geometry_parts(geometry) -> List[Any]:
    """Flatten any (possibly nested) collection into atomic geometries."""

    geom_type = getattr(geometry, "geom_type", "")
    if geom_type in {"MultiPolygon", "MultiLineString", "GeometryCollection"}:
        parts: List[Any] = []
        for part in geometry.geoms:
            parts.extend(_geometry_parts(part))
        return [part for part in parts if not getattr(part, "is_empty", False)]
    return [geometry]


def _element_for_part(part, osm_id: str, osm_type: str, tags: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    geom_type = getattr(part, "geom_type", "")
    element_id = osm_id if index == 0 else f"{osm_id}#{index}"

    if geom_type == "Point":
        return {
            "type": osm_type if osm_type in {"node", "way"} else "node",
            "id": element_id,
            "tags": dict(tags),
            "lat": float(part.y),
            "lon": float(part.x),
        }
    if geom_type == "LineString":
        coordinates = _coords_lonlat(part.coords)
        if len(coordinates) < 2:
            return None
        return {"type": osm_type, "id": element_id, "tags": dict(tags), "geometry": coordinates}
    if geom_type == "Polygon":
        coordinates = _coords_lonlat(part.exterior.coords)
        if len(coordinates) < 4:
            return None
        element = {"type": osm_type, "id": element_id, "tags": dict(tags), "geometry": coordinates}
        holes = [
            _coords_lonlat(ring.coords)
            for ring in part.interiors
            if len(ring.coords) >= 4
        ]
        if holes:
            # Interior rings are preserved so downstream containment/distance math
            # treats a hole as *not* part of the feature.
            element["holes"] = holes
        return element
    return None


def _frame_to_elements(frame, payload_key: str) -> List[Dict[str, Any]]:
    elements: List[Dict[str, Any]] = []
    if frame is None or len(frame) == 0:
        return elements

    has_tags = "tags" in frame.columns
    has_osm_id = "osm_id" in frame.columns
    has_osm_type = "osm_type" in frame.columns

    for record in frame.itertuples(index=False):
        geometry = getattr(record, "geometry", None)
        if geometry is None or getattr(geometry, "is_empty", True):
            continue
        raw_tags = getattr(record, "tags", None) if has_tags else None
        try:
            tags = json.loads(raw_tags) if raw_tags else {}
        except (TypeError, json.JSONDecodeError):
            tags = {}
        osm_id = str(getattr(record, "osm_id", "") or "?") if has_osm_id else "?"
        osm_type = str(getattr(record, "osm_type", "") or "way") if has_osm_type else "way"
        for index, part in enumerate(_geometry_parts(geometry)):
            element = _element_for_part(part, osm_id, osm_type, tags, index)
            if element is not None:
                elements.append(element)
    return elements


__all__ = ["LocalOSMProvider", "PROVIDER_NAME"]
