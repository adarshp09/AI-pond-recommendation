"""Phase 4 — real-world GIS context and geographic feasibility.

Enriches Phase-3 candidates with **real provider data** (OSM Overpass via
``services.land_service``) and determines an explicit
``feasible`` / ``rejected`` / ``unknown`` status.

Design:

* candidates are **clustered by metric proximity** and each cluster is queried
  once (never one request per candidate);
* identical queries are de-duplicated within a run and cached across runs
  (``services.cache.FileCache``);
* requests run with **bounded concurrency** (thread pool), a **timeout**, and
  **capped retries**;
* feature geometries are transformed to a **projected CRS** so all distances
  are computed in metres with a Shapely STRtree;
* missing data is never replaced by ``0`` or a fabricated distance — unknown
  stays ``None`` and yields status ``unknown``.

Scope: geographic context only. No rainfall/runoff, no ML ranking, no
distributed workers, no API restructuring.
"""

from __future__ import annotations

import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import transform as shapely_transform
from shapely.strtree import STRtree

from backend.recommendation import DEFAULT_REJECTION_CONSTRAINTS
from services.land_service import fetch_land_context

# Kind -> the Overpass payload key that holds its elements.
_KIND_PAYLOAD_KEY: Dict[str, str] = {
    "water": "water_bodies",
    "road": "roads",
    "building": "buildings",
}

DEFAULT_CLUSTER_RADIUS_M = 1000.0
DEFAULT_BBOX_BUFFER_M = 1000.0
DEFAULT_MAX_CONCURRENT_REQUESTS = 4
DEFAULT_PROVIDER_TIMEOUT_S = 20.0
DEFAULT_PROVIDER_RETRIES = 2
DEFAULT_CACHE_TTL_S = 86_400
DEFAULT_NEARBY_RADIUS_M = 500.0
DEFAULT_MAX_ENRICHED_CANDIDATES = 200

PROVIDER_NAME = "OpenStreetMap Overpass API (land_service.fetch_land_context)"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return int(default)


def _utm_epsg(longitude: float, latitude: float) -> str:
    zone = int((longitude + 180.0) / 6.0) + 1
    zone = min(max(zone, 1), 60)
    base = 32600 if latitude >= 0 else 32700
    return f"EPSG:{base + zone}"


@lru_cache(maxsize=64)
def _cached_transformer(source_crs: str, target_crs: str) -> Transformer:
    """Memoise Transformer construction (expensive: PROJ database lookups)."""

    return Transformer.from_crs(source_crs, target_crs, always_xy=True)


def project_lonlat(longitude: float, latitude: float) -> Tuple[float, float, str]:
    """Project a WGS84 point to a metric UTM CRS derived from its own location."""

    crs = _utm_epsg(float(longitude), float(latitude))
    transformer = _cached_transformer("EPSG:4326", crs)
    x, y = transformer.transform(float(longitude), float(latitude))
    return float(x), float(y), crs


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GeographicContextConfig:
    """Deterministic geographic-context configuration."""

    max_enriched_candidates: int = DEFAULT_MAX_ENRICHED_CANDIDATES
    cluster_radius_m: float = DEFAULT_CLUSTER_RADIUS_M
    bbox_buffer_m: float = DEFAULT_BBOX_BUFFER_M
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS
    provider_timeout_s: float = DEFAULT_PROVIDER_TIMEOUT_S
    provider_retries: int = DEFAULT_PROVIDER_RETRIES
    cache_enabled: bool = True
    cache_ttl_s: int = DEFAULT_CACHE_TTL_S
    nearby_radius_m: float = DEFAULT_NEARBY_RADIUS_M

    # Feasibility thresholds reuse the existing rejection-constraint contract
    # (which itself reads POND_MIN_* environment variables).
    min_road_distance_m: float = field(
        default_factory=lambda: float(DEFAULT_REJECTION_CONSTRAINTS["min_road_distance_m"])
    )
    min_building_distance_m: float = field(
        default_factory=lambda: float(DEFAULT_REJECTION_CONSTRAINTS["min_building_distance_m"])
    )
    reject_inside_water: bool = True

    @classmethod
    def from_env(cls) -> "GeographicContextConfig":
        return cls(
            max_enriched_candidates=_env_int("POND_MAX_ENRICHED_CANDIDATES", DEFAULT_MAX_ENRICHED_CANDIDATES),
            cluster_radius_m=_env_float("POND_GEO_CLUSTER_RADIUS_M", DEFAULT_CLUSTER_RADIUS_M),
            bbox_buffer_m=_env_float("POND_GEO_BBOX_BUFFER_M", DEFAULT_BBOX_BUFFER_M),
            max_concurrent_requests=_env_int("POND_GEO_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENT_REQUESTS),
            provider_timeout_s=_env_float("POND_GEO_PROVIDER_TIMEOUT_S", DEFAULT_PROVIDER_TIMEOUT_S),
            provider_retries=_env_int("POND_GEO_PROVIDER_RETRIES", DEFAULT_PROVIDER_RETRIES),
            cache_enabled=_env_int("POND_GEO_CACHE_ENABLED", 1) == 1,
            cache_ttl_s=_env_int("POND_GEO_CACHE_TTL_S", DEFAULT_CACHE_TTL_S),
            nearby_radius_m=_env_float("POND_GEO_NEARBY_RADIUS_M", DEFAULT_NEARBY_RADIUS_M),
        )


# ---------------------------------------------------------------------------
# Run statistics (provider latency/counts, cache hit rate)
# ---------------------------------------------------------------------------

@dataclass
class GeographicStats:
    provider_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    provider_errors: int = 0
    provider_timeouts: int = 0
    provider_ms_total: float = 0.0
    clusters: int = 0

    def to_dict(self) -> Dict[str, Any]:
        lookups = self.cache_hits + self.cache_misses
        return {
            "provider_calls": int(self.provider_calls),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "cache_hit_rate": (float(self.cache_hits) / lookups) if lookups else None,
            "provider_errors": int(self.provider_errors),
            "provider_timeouts": int(self.provider_timeouts),
            "provider_ms_total": round(float(self.provider_ms_total), 3),
            "clusters": int(self.clusters),
        }


_STATS = GeographicStats()
_STATS_LOCK = threading.Lock()


def _record(**deltas: float) -> None:
    with _STATS_LOCK:
        for key, value in deltas.items():
            setattr(_STATS, key, getattr(_STATS, key) + value)


def reset_stats() -> None:
    with _STATS_LOCK:
        for key in GeographicStats.__dataclass_fields__:
            setattr(_STATS, key, 0.0 if key == "provider_ms_total" else 0)


def get_stats() -> Dict[str, Any]:
    with _STATS_LOCK:
        return _STATS.to_dict()


class _NullStage:
    def set_output(self, **_: Any) -> "_NullStage":
        return self

    def add_metric(self, *_: Any, **__: Any) -> "_NullStage":
        return self


class _NullRecorder:
    def stage(self, *_: Any, **__: Any):
        import contextlib

        return contextlib.nullcontext(_NullStage())


# ---------------------------------------------------------------------------
# Clustering + bounding boxes
# ---------------------------------------------------------------------------

def cluster_candidates(candidates: Sequence[Dict[str, Any]], radius_m: float) -> List[List[int]]:
    """Group candidate indices whose metric coordinates are within ``radius_m``."""

    if radius_m <= 0:
        return [[index] for index in range(len(candidates))]

    bins: Dict[Tuple[int, int], List[int]] = {}
    clusters: List[List[int]] = []
    radius_sq = radius_m * radius_m

    for index, candidate in enumerate(candidates):
        bx = int(math.floor(float(candidate["x"]) / radius_m))
        by = int(math.floor(float(candidate["y"]) / radius_m))

        placed = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for cluster_index in bins.get((bx + dx, by + dy), ()):
                    representative = candidates[clusters[cluster_index][0]]
                    ddx = float(candidate["x"]) - float(representative["x"])
                    ddy = float(candidate["y"]) - float(representative["y"])
                    if ddx * ddx + ddy * ddy <= radius_sq:
                        clusters[cluster_index].append(index)
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break

        if not placed:
            clusters.append([index])
            bins.setdefault((bx, by), []).append(len(clusters) - 1)

    return clusters


def cluster_bbox(cluster: Sequence[Dict[str, Any]], buffer_m: float) -> Tuple[float, float, float, float]:
    """Return a rounded ``(south, west, north, east)`` bbox covering a cluster."""

    latitudes = [float(c["latitude"]) for c in cluster]
    longitudes = [float(c["longitude"]) for c in cluster]
    centre_lat = float(np.mean(latitudes))

    lat_pad = buffer_m / 111_320.0
    cos_lat = max(abs(math.cos(math.radians(centre_lat))), 1e-6)
    lon_pad = buffer_m / (111_320.0 * cos_lat)

    return (
        round(min(latitudes) - lat_pad, 4),
        round(min(longitudes) - lon_pad, 4),
        round(max(latitudes) + lat_pad, 4),
        round(max(longitudes) + lon_pad, 4),
    )


def _bboxes_equal(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(abs(float(a) - float(b)) < 1e-9 for a, b in zip(left, right))


# ---------------------------------------------------------------------------
# Geometry construction
# ---------------------------------------------------------------------------

def _element_geometry(element: Dict[str, Any], kind: str) -> Optional[Any]:
    """Build a WGS84 geometry from an Overpass element (never fabricates one)."""

    coordinates = element.get("geometry")
    if coordinates and isinstance(coordinates, list):
        points = [
            (float(item["lon"]), float(item["lat"]))
            for item in coordinates
            if isinstance(item, dict) and "lon" in item and "lat" in item
        ]
        if len(points) >= 2:
            closed = points[0] == points[-1]
            # Roads stay lines even when closed; areas become polygons.
            if kind != "road" and closed and len(points) >= 4:
                try:
                    return Polygon(points)
                except Exception:
                    return None
            return LineString(points)

    latitude = element.get("lat")
    longitude = element.get("lon")
    if latitude is None or longitude is None:
        centre = element.get("center") or {}
        latitude = centre.get("lat")
        longitude = centre.get("lon")
    if latitude is None or longitude is None:
        return None
    return Point(float(longitude), float(latitude))


@dataclass
class _FeatureIndex:
    geometries: List[Any]
    summaries: List[Dict[str, Any]]
    tree: Optional[STRtree]


def _build_feature_index(
    elements: Sequence[Dict[str, Any]],
    transformer: Transformer,
    kind: str,
) -> _FeatureIndex:
    """Build a projected feature index for one feature kind.

    ``kind`` comes from the provider payload key, so features the provider has
    already partitioned are used as-is (never re-classified/dropped here).
    """

    geometries: List[Any] = []
    summaries: List[Dict[str, Any]] = []

    for element in elements or []:
        geometry = _element_geometry(element, kind)
        if geometry is None:
            continue
        try:
            projected = shapely_transform(lambda x, y: transformer.transform(x, y), geometry)
        except Exception:
            continue
        if projected.is_empty:
            continue
        geometries.append(projected)
        summaries.append(
            {
                "id": f"{element.get('type', 'way')}/{element.get('id', '?')}",
                "kind": kind,
                "tags": dict(element.get("tags") or {}),
            }
        )

    return _FeatureIndex(
        geometries=geometries,
        summaries=summaries,
        tree=STRtree(geometries) if geometries else None,
    )


def _nearest_distance(index: _FeatureIndex, point: Point) -> Optional[float]:
    if index.tree is None:
        return None
    nearest = index.tree.nearest(point)
    if nearest is None:
        return None
    return float(point.distance(index.geometries[int(nearest)]))


def _count_within(index: _FeatureIndex, point: Point, radius_m: float) -> int:
    if index.tree is None or radius_m <= 0:
        return 0
    # Buffer the point and use the tree's bbox prefilter, then exact distance.
    candidates = index.tree.query(point.buffer(radius_m))
    return int(sum(1 for i in candidates if point.distance(index.geometries[int(i)]) <= radius_m))


# ---------------------------------------------------------------------------
# Provider access (bounded concurrency, timeout, retries, cache)
# ---------------------------------------------------------------------------

def _cache_key(bbox: Sequence[float]) -> Dict[str, Any]:
    return {"provider": "osm_overpass_land_context", "bbox": [float(v) for v in bbox]}


def _fetch_with_retries(
    provider: Callable[..., Dict[str, Any]],
    bbox: Tuple[float, float, float, float],
    config: GeographicContextConfig,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    south, west, north, east = bbox
    last_error: Optional[str] = None

    for attempt in range(int(config.provider_retries) + 1):
        started = time.perf_counter()
        try:
            payload = provider(south, west, north, east)
            _record(provider_calls=1, provider_ms_total=(time.perf_counter() - started) * 1000.0)
            return payload, None
        except Exception as exc:
            _record(provider_errors=1, provider_ms_total=(time.perf_counter() - started) * 1000.0)
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < int(config.provider_retries):
                time.sleep(min(0.01 * (2 ** attempt), 0.05))

    return None, last_error


def _resolve_payloads(
    bboxes: Sequence[Tuple[float, float, float, float]],
    config: GeographicContextConfig,
    provider: Callable[..., Dict[str, Any]],
    cache: Any,
    recorder: Any,
) -> Dict[int, Dict[str, Any]]:
    """Resolve one land-context payload per cluster bbox, with dedupe + cache."""

    results: Dict[int, Dict[str, Any]] = {}
    scheduled: List[Tuple[int, Tuple[float, float, float, float]]] = []

    for index, bbox in enumerate(bboxes):
        duplicate = next((i for i, existing in scheduled if _bboxes_equal(existing, bbox)), None)
        if duplicate is not None:
            results[index] = {"_alias_of": duplicate}
            _record(cache_hits=1)
            continue

        if cache is not None and config.cache_enabled:
            cached = cache.get("land_context", _cache_key(bbox))
            if cached is not None:
                results[index] = {"_payload": cached}
                _record(cache_hits=1)
                continue

        scheduled.append((index, bbox))

    with recorder.stage("geographic_provider") as handle:
        if scheduled:
            workers = max(1, min(int(config.max_concurrent_requests), len(scheduled)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [(pool.submit(_fetch_with_retries, provider, bbox, config), index) for index, bbox in scheduled]
                for future, index in futures:
                    try:
                        payload, error = future.result(timeout=float(config.provider_timeout_s))
                    except FutureTimeoutError:
                        payload, error = None, "provider timeout"
                        _record(provider_timeouts=1)

                    if payload is None:
                        results[index] = {"_error": error or "provider returned no payload"}
                        continue

                    results[index] = {"_payload": payload}
                    if cache is not None and config.cache_enabled:
                        cache.set("land_context", _cache_key(bboxes[index]), payload)
                        _record(cache_misses=1)

        handle.set_output(scheduled_queries=len(scheduled))

    for index, entry in list(results.items()):
        if "_alias_of" in entry:
            results[index] = results[entry["_alias_of"]]

    return results


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------

def decide_feasibility(
    distances: Dict[str, Optional[float]],
    config: GeographicContextConfig,
    provider_ok: bool,
) -> Tuple[str, Optional[str]]:
    """Explicit ``feasible`` / ``rejected`` / ``unknown`` decision.

    Road/building clearance and water overlap are hard rejections. Absent
    provider data yields ``unknown`` — never a fabricated pass.
    """

    if not provider_ok:
        return "unknown", "provider_unavailable"

    water = distances.get("water_body")
    building = distances.get("building")
    road = distances.get("road")

    if config.reject_inside_water and water is not None and water <= 0.0:
        return "rejected", "inside_or_on_water_body"
    if building is not None and building < config.min_building_distance_m:
        return "rejected", "insufficient_building_clearance"
    if road is not None and road < config.min_road_distance_m:
        return "rejected", "insufficient_road_clearance"

    return "feasible", None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def enrich_candidates(
    candidates: Sequence[Dict[str, Any]],
    *,
    config: Optional[GeographicContextConfig] = None,
    provider: Optional[Callable[..., Dict[str, Any]]] = None,
    cache: Any = None,
    recorder: Any = None,
) -> Dict[str, Any]:
    """Enrich candidates with real geographic context and feasibility.

    Returns ``{"candidates": [...], "diagnostics": {...}}``. Distances are
    ``None`` wherever provider data is unavailable — never ``0``.
    """

    active_config = config or GeographicContextConfig.from_env()
    active_provider = provider if provider is not None else fetch_land_context
    active_recorder = recorder if recorder is not None else _NullRecorder()

    active_cache = cache
    if active_cache is None and active_config.cache_enabled:
        try:
            from services.cache import get_cache

            active_cache = get_cache(ttl_seconds=int(active_config.cache_ttl_s), enabled=True)
        except Exception:
            active_cache = None

    candidates = list(candidates)
    enrich_limit = int(active_config.max_enriched_candidates)
    enrichable = candidates[:enrich_limit]
    remainder = candidates[enrich_limit:]

    with active_recorder.stage("geographic_clustering") as handle:
        clusters = cluster_candidates(enrichable, active_config.cluster_radius_m)
        _record(clusters=len(clusters))
        handle.set_output(candidate_count=len(candidates), enriched_count=len(enrichable), cluster_count=len(clusters))

    bboxes = [cluster_bbox(cluster_as_candidates, active_config.bbox_buffer_m)
              for cluster_as_candidates in ([enrichable[i] for i in cluster] for cluster in clusters)]
    payloads = _resolve_payloads(bboxes, active_config, active_provider, active_cache, active_recorder)

    status_counts = {"feasible": 0, "rejected": 0, "unknown": 0}
    reasons: Dict[str, int] = {}
    enriched: List[Dict[str, Any]] = []

    with active_recorder.stage("geographic_features") as handle:
        for cluster_index, member_indices in enumerate(clusters):
            entry = payloads.get(cluster_index, {"_error": "no provider result"})
            payload = entry.get("_payload")
            provider_ok = isinstance(payload, dict)

            feature_index: Dict[str, _FeatureIndex] = {}
            if provider_ok:
                metric_crs = next((enrichable[i].get("crs") for i in member_indices if enrichable[i].get("crs")), None)
                if metric_crs is None:
                    metric_crs = _utm_epsg(
                        float(np.mean([enrichable[i]["longitude"] for i in member_indices])),
                        float(np.mean([enrichable[i]["latitude"] for i in member_indices])),
                    )
                try:
                    transformer = _cached_transformer("EPSG:4326", str(metric_crs))
                    for kind, payload_key in _KIND_PAYLOAD_KEY.items():
                        feature_index[kind] = _build_feature_index(payload.get(payload_key) or [], transformer, kind)
                except Exception as exc:
                    provider_ok = False
                    entry = {"_error": f"transform failed: {exc}"}

            for member in member_indices:
                candidate = dict(enrichable[member])

                if not provider_ok:
                    reason = entry.get("_error") or "provider_unavailable"
                    candidate.update(
                        {
                            "road_distance_m": None,
                            "building_distance_m": None,
                            "water_distance_m": None,
                            "nearby_feature_counts": {"roads": 0, "buildings": 0, "water_bodies": 0},
                            "land_context": {"water_bodies": [], "roads": [], "buildings": []},
                            "geographic_source": PROVIDER_NAME,
                            "geographic_status": "unknown",
                            "feasibility": "unknown",
                            "feasibility_reason": reason,
                            "provider_ok": False,
                        }
                    )
                    status_counts["unknown"] += 1
                    reasons[reason] = reasons.get(reason, 0) + 1
                    enriched.append(candidate)
                    continue

                point = Point(float(candidate["x"]), float(candidate["y"]))
                distances = {
                    "road": _nearest_distance(feature_index["road"], point),
                    "building": _nearest_distance(feature_index["building"], point),
                    "water_body": _nearest_distance(feature_index["water"], point),
                }
                counts = {
                    "roads": _count_within(feature_index["road"], point, active_config.nearby_radius_m),
                    "buildings": _count_within(feature_index["building"], point, active_config.nearby_radius_m),
                    "water_bodies": _count_within(feature_index["water"], point, active_config.nearby_radius_m),
                }

                status, reason = decide_feasibility(distances, active_config, True)
                status_counts[status] += 1
                if reason:
                    reasons[reason] = reasons.get(reason, 0) + 1

                candidate.update(
                    {
                        "road_distance_m": distances["road"],
                        "building_distance_m": distances["building"],
                        "water_distance_m": distances["water_body"],
                        "nearby_feature_counts": counts,
                        "land_context": {
                            "water_bodies": feature_index["water"].summaries[:3],
                            "roads": feature_index["road"].summaries[:3],
                            "buildings": feature_index["building"].summaries[:3],
                        },
                        "geographic_source": payload.get("source", PROVIDER_NAME),
                        "geographic_status": "available",
                        "feasibility": status,
                        "feasibility_reason": reason,
                        "provider_ok": True,
                    }
                )
                enriched.append(candidate)

        handle.set_output(**status_counts)

    for candidate in remainder:
        candidate = dict(candidate)
        candidate.update(
            {
                "road_distance_m": None,
                "building_distance_m": None,
                "water_distance_m": None,
                "nearby_feature_counts": {"roads": 0, "buildings": 0, "water_bodies": 0},
                "land_context": {"water_bodies": [], "roads": [], "buildings": []},
                "geographic_source": PROVIDER_NAME,
                "geographic_status": "unknown",
                "feasibility": "unknown",
                "feasibility_reason": "not_enriched_candidate_limit",
                "provider_ok": False,
            }
        )
        status_counts["unknown"] += 1
        reasons["not_enriched_candidate_limit"] = reasons.get("not_enriched_candidate_limit", 0) + 1
        enriched.append(candidate)

    diagnostics = {
        "provider": PROVIDER_NAME,
        "stats": get_stats(),
        "feasible_count": status_counts["feasible"],
        "rejected_count": status_counts["rejected"],
        "unknown_count": status_counts["unknown"],
        "feasibility_reasons": dict(sorted(reasons.items())),
        "cluster_count": len(clusters),
        "config": {
            "cluster_radius_m": active_config.cluster_radius_m,
            "bbox_buffer_m": active_config.bbox_buffer_m,
            "max_concurrent_requests": active_config.max_concurrent_requests,
            "provider_timeout_s": active_config.provider_timeout_s,
            "provider_retries": active_config.provider_retries,
            "cache_enabled": bool(active_config.cache_enabled),
            "min_road_distance_m": active_config.min_road_distance_m,
            "min_building_distance_m": active_config.min_building_distance_m,
            "reject_inside_water": active_config.reject_inside_water,
        },
    }

    return {"candidates": enriched, "diagnostics": diagnostics}


__all__ = [
    "GeographicContextConfig",
    "GeographicStats",
    "cluster_bbox",
    "cluster_candidates",
    "decide_feasibility",
    "enrich_candidates",
    "get_stats",
    "project_lonlat",
    "reset_stats",
]
