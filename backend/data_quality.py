"""Baseline data-quality metric collection.

Every function here is a pure reader of values the pipeline *already* produces
(contours, DEM, hydrology, catchment, candidates). Nothing in this module
changes how those values are computed; it only summarises them so that future
algorithm work (Phase 2+) can be compared against a recorded baseline.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


def _finite(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _stats(values: Any) -> Dict[str, Optional[float]]:
    finite = _finite(values)
    if finite.size == 0:
        return {"min": None, "max": None, "mean": None, "median": None, "range": None}
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    return {
        "min": minimum,
        "max": maximum,
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "range": float(maximum - minimum),
    }


def contour_metrics(contour_data: Any) -> Dict[str, Any]:
    """Contour count, coordinate count, and elevation summary."""

    contours = getattr(contour_data, "contours", None) or []
    coordinate_count = 0
    elevations: List[float] = []
    for contour in contours:
        coordinate_count += len(getattr(contour, "coordinates", []) or [])
        elevations.append(float(getattr(contour, "elevation_m", 0.0)))

    elevation_stats = _stats(elevations)
    return {
        "contour_count": int(len(contours)),
        "coordinate_count": int(coordinate_count),
        "elevation_min_m": elevation_stats["min"],
        "elevation_max_m": elevation_stats["max"],
        "elevation_range_m": elevation_stats["range"],
        "elevation_mean_m": elevation_stats["mean"],
    }


def dem_metrics(dem: Any, validation: Any = None) -> Dict[str, Any]:
    """DEM shape, resolution, NaN percentage, elevation and slope summaries."""

    elevation = np.asarray(getattr(dem, "elevation", np.empty((0, 0))), dtype=float)
    if elevation.ndim != 2 or elevation.size == 0:
        return {
            "dem_rows": 0,
            "dem_columns": 0,
            "dem_cells": 0,
            "dem_resolution_m": None,
            "dem_nan_percentage": None,
            "elevation_min_m": None,
            "elevation_max_m": None,
            "elevation_range_m": None,
            "slope_min_degrees": None,
            "slope_max_degrees": None,
            "slope_mean_degrees": None,
            "valid_cell_fraction": None,
            "dem_status": getattr(validation, "status", None),
            "dem_quality_score": None,
        }

    finite_mask = np.isfinite(elevation)
    nan_percentage = float((1.0 - float(np.mean(finite_mask))) * 100.0)

    slope = getattr(dem, "slope_degrees", None)
    slope_stats = _stats(slope) if slope is not None else {
        "min": None, "max": None, "mean": None, "median": None, "range": None
    }
    elevation_stats = _stats(elevation)

    return {
        "dem_rows": int(elevation.shape[0]),
        "dem_columns": int(elevation.shape[1]),
        "dem_cells": int(elevation.size),
        "dem_resolution_m": (
            None if getattr(dem, "resolution_m", None) is None else float(dem.resolution_m)
        ),
        "dem_nan_percentage": nan_percentage,
        "elevation_min_m": elevation_stats["min"],
        "elevation_max_m": elevation_stats["max"],
        "elevation_range_m": elevation_stats["range"],
        "slope_min_degrees": slope_stats["min"],
        "slope_max_degrees": slope_stats["max"],
        "slope_mean_degrees": slope_stats["mean"],
        "valid_cell_fraction": float(np.mean(finite_mask)),
        "dem_status": getattr(validation, "status", None),
        "dem_quality_score": (
            None if validation is None else float(getattr(validation, "score", 0.0))
        ),
    }


def hydrology_metrics(hydrology: Any) -> Dict[str, Any]:
    """Flow-direction resolution, sinks, and flow-accumulation statistics.

    ``valid_flow_direction_percentage`` is the share of valid cells that have a
    resolved downstream direction (``flow_direction >= 0``). Cells without a
    downstream neighbour are sinks or edge-outflow cells and are reported
    separately. ``unresolved_cycle_cells`` counts valid cells left as NaN by the
    topological accumulation pass (i.e. flow cycles).
    """

    flow_direction = np.asarray(getattr(hydrology, "flow_direction", np.empty((0, 0))))
    valid_mask = np.asarray(getattr(hydrology, "valid_mask", np.empty((0, 0))), dtype=bool)
    accumulation = np.asarray(getattr(hydrology, "flow_accumulation", np.empty((0, 0))), dtype=float)

    if flow_direction.size == 0 or valid_mask.size == 0:
        return {
            "hydrology_valid_cells": 0,
            "valid_flow_direction_percentage": None,
            "sink_cells": int(getattr(hydrology, "sink_cells", 0) or 0),
            "edge_outflow_cells": int(getattr(hydrology, "edge_outflow_cells", 0) or 0),
            "unresolved_cycle_cells": 0,
            "flow_accumulation_min_cells": None,
            "flow_accumulation_max_cells": None,
            "flow_accumulation_mean_cells": None,
            "flow_accumulation_median_cells": None,
        }

    valid_count = int(np.sum(valid_mask))
    resolved = int(np.sum(valid_mask & (flow_direction >= 0)))
    unresolved_cycle = int(np.sum(valid_mask & ~np.isfinite(accumulation)))
    accumulation_stats = _stats(accumulation[valid_mask])

    return {
        "hydrology_valid_cells": valid_count,
        "valid_flow_direction_percentage": (
            float(resolved / valid_count * 100.0) if valid_count else None
        ),
        "sink_cells": int(getattr(hydrology, "sink_cells", 0) or 0),
        "edge_outflow_cells": int(getattr(hydrology, "edge_outflow_cells", 0) or 0),
        "unresolved_cycle_cells": unresolved_cycle,
        "flow_accumulation_min_cells": accumulation_stats["min"],
        "flow_accumulation_max_cells": accumulation_stats["max"],
        "flow_accumulation_mean_cells": accumulation_stats["mean"],
        "flow_accumulation_median_cells": accumulation_stats["median"],
    }


def catchment_metrics(catchment_stats: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    stats = catchment_stats or {}
    return {
        "catchment_area_m2": float(stats.get("area_m2", 0.0) or 0.0),
        "catchment_area_hectares": float(stats.get("area_hectares", 0.0) or 0.0),
        "catchment_area_km2": float(stats.get("area_km2", 0.0) or 0.0),
        "catchment_cell_count": int(stats.get("cell_count", 0) or 0),
    }


def candidate_metrics(
    candidates: Optional[Iterable[Mapping[str, Any]]] = None,
    *,
    rejected_count: Optional[int] = None,
    recommendation_count: Optional[int] = None,
) -> Dict[str, Any]:
    candidate_list = list(candidates or [])
    return {
        "candidate_count": int(len(candidate_list)),
        "rejected_candidate_count": (
            None if rejected_count is None else int(rejected_count)
        ),
        "recommendation_count": (
            int(recommendation_count)
            if recommendation_count is not None
            else int(len(candidate_list))
        ),
    }


def external_availability(
    *,
    dem: Any = None,
    rainfall: Any = None,
    land_context: Any = None,
    place_search: Any = None,
    geocoding: Any = None,
) -> Dict[str, bool]:
    def available(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, Mapping) and value.get("status") == "unavailable":
            return False
        return True

    return {
        "dem": available(dem),
        "rainfall": available(rainfall),
        "land_context": available(land_context),
        "place_search": available(place_search),
        "geocoding": available(geocoding),
    }


def build_data_quality(
    *,
    contour_data: Any = None,
    dem: Any = None,
    validation: Any = None,
    hydrology: Any = None,
    catchment_stats: Optional[Mapping[str, Any]] = None,
    candidates: Optional[Sequence[Mapping[str, Any]]] = None,
    rejected_count: Optional[int] = None,
    recommendation_count: Optional[int] = None,
    external: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate the baseline data-quality metrics into a single dictionary."""

    metrics: Dict[str, Any] = {}
    if contour_data is not None:
        metrics.update(contour_metrics(contour_data))
    if dem is not None:
        metrics.update(dem_metrics(dem, validation))
    if hydrology is not None:
        metrics.update(hydrology_metrics(hydrology))
    if catchment_stats is not None:
        metrics.update(catchment_metrics(catchment_stats))
    if candidates is not None or rejected_count is not None or recommendation_count is not None:
        metrics.update(
            candidate_metrics(
                candidates,
                rejected_count=rejected_count,
                recommendation_count=recommendation_count,
            )
        )
    if external is not None:
        metrics.update({"external_data": dict(external)})
    return metrics


__all__ = [
    "build_data_quality",
    "candidate_metrics",
    "catchment_metrics",
    "contour_metrics",
    "dem_metrics",
    "external_availability",
    "hydrology_metrics",
]
