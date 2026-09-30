# catchment.py
"""Catchment delineation and raster -> GeoJSON conversion.

Catchments are traced on the D8 flow network produced by :mod:`hydrology`.
``mask_to_polygon`` decomposes the raster mask into horizontal runs (rectangles)
before unioning with Shapely, which is O(rows x runs) instead of the previous
O(cells) per-cell polygon union — and it preserves holes and disconnected
regions exactly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.hydrology import downstream_flat_index

try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
except ImportError:  # pragma: no cover - shapely is a hard dependency
    Polygon = None
    MultiPolygon = None
    unary_union = None


D8_OFFSETS = np.array(
    [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ],
    dtype=np.int8,
)


# ---------------------------------------------------------------------------
# Upstream tracing
# ---------------------------------------------------------------------------

def _upstream_csr(flow_direction: np.ndarray, valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build a CSR-style upstream adjacency from the flat downstream index.

    Returns ``(sources, offsets)`` such that the upstream cells of flat index
    ``i`` are ``sources[offsets[i]:offsets[i + 1]]``.
    """

    rows, columns = flow_direction.shape
    total = rows * columns

    downstream = downstream_flat_index(flow_direction, valid_mask)
    valid_flat = np.asarray(valid_mask, dtype=bool).ravel()

    sources = np.flatnonzero(valid_flat & (downstream >= 0))
    targets = downstream[sources]

    order = np.argsort(targets, kind="stable")
    sources = sources[order]
    targets = targets[order]

    counts = np.bincount(targets, minlength=total).astype(np.int64)
    offsets = np.zeros(total + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    return sources, offsets


def delineate_catchment(
    flow_direction: np.ndarray,
    valid_mask: np.ndarray,
    outlet: Tuple[int, int],
) -> np.ndarray:
    """Return the boolean mask of every cell that drains to ``outlet``."""

    flow_direction = np.asarray(flow_direction)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    rows, columns = flow_direction.shape
    r0, c0 = outlet

    if not (0 <= r0 < rows and 0 <= c0 < columns):
        raise ValueError("Outlet is outside the DEM.")

    if not valid_mask[r0, c0]:
        raise ValueError("Outlet is not located on a valid DEM cell.")

    sources, offsets = _upstream_csr(flow_direction, valid_mask)

    catchment = np.zeros(rows * columns, dtype=bool)
    outlet_flat = r0 * columns + c0
    catchment[outlet_flat] = True

    stack = [outlet_flat]
    while stack:
        cell = stack.pop()
        for upstream in sources[offsets[cell]:offsets[cell + 1]]:
            upstream = int(upstream)
            if not catchment[upstream]:
                catchment[upstream] = True
                stack.append(upstream)

    return catchment.reshape(rows, columns)


# ---------------------------------------------------------------------------
# Area
# ---------------------------------------------------------------------------

def catchment_area(
    catchment_mask: np.ndarray,
    resolution_m: float,
) -> Dict[str, float]:
    """Area is the count of catchment cells x cell area (never a full-grid area)."""

    cell_count = int(np.sum(catchment_mask))
    cell_area = resolution_m * resolution_m
    area_m2 = cell_count * cell_area

    return {
        "area_m2": float(area_m2),
        "area_hectares": float(area_m2 / 10000.0),
        "area_km2": float(area_m2 / 1_000_000.0),
        "cell_count": cell_count,
    }


# ---------------------------------------------------------------------------
# Raster -> polygon
# ---------------------------------------------------------------------------

def _row_runs(mask: np.ndarray) -> List[Tuple[int, int, int]]:
    """Decompose a boolean mask into horizontal runs ``(row, col_start, col_end)``."""

    runs: List[Tuple[int, int, int]] = []
    for row_index in range(mask.shape[0]):
        columns = np.flatnonzero(mask[row_index])
        if columns.size == 0:
            continue
        start = previous = int(columns[0])
        for value in columns[1:]:
            current = int(value)
            if current == previous + 1:
                previous = current
                continue
            runs.append((row_index, start, previous))
            start = previous = current
        runs.append((row_index, start, previous))
    return runs


def mask_to_polygon(
    mask: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> Optional[Any]:
    """Convert a raster mask to a Shapely geometry via horizontal-run rectangles."""

    if Polygon is None:
        return None

    mask = np.asarray(mask, dtype=bool)
    rows, columns = mask.shape

    if len(x_coords) != columns:
        raise ValueError("x_coords length does not match mask columns.")

    if len(y_coords) != rows:
        raise ValueError("y_coords length does not match mask rows.")

    if columns > 1:
        dx = abs(float(np.median(np.diff(x_coords))))
    else:
        dx = 1.0

    if rows > 1:
        dy = abs(float(np.median(np.diff(y_coords))))
    else:
        dy = 1.0

    half_x = dx / 2.0
    half_y = dy / 2.0

    rectangles = []
    for row_index, col_start, col_end in _row_runs(mask):
        y = float(y_coords[row_index])
        x_left = float(x_coords[col_start]) - half_x
        x_right = float(x_coords[col_end]) + half_x
        rectangles.append(
            Polygon(
                [
                    (x_left, y - half_y),
                    (x_right, y - half_y),
                    (x_right, y + half_y),
                    (x_left, y + half_y),
                ]
            )
        )

    if not rectangles:
        return None

    return unary_union(rectangles)


def polygon_to_geojson(geometry: Any) -> Optional[Dict[str, Any]]:
    """Serialise a Shapely geometry to a GeoJSON geometry dict."""

    if geometry is None:
        return None

    if getattr(geometry, "is_empty", False):
        return None

    return {
        "type": geometry.geom_type,
        "coordinates": list(geometry.__geo_interface__["coordinates"]),
    }


def validate_catchment_geometry(geometry: Any) -> Dict[str, Any]:
    """Validate a catchment geometry before it is serialised."""

    if geometry is None:
        return {"valid": False, "geom_type": None, "reason": "missing_geometry"}

    if getattr(geometry, "is_empty", False):
        return {"valid": False, "geom_type": geometry.geom_type, "reason": "empty_geometry"}

    valid = bool(getattr(geometry, "is_valid", False))
    geom_type = geometry.geom_type

    reason = None
    if not valid:
        reason = "invalid_geometry"
    elif geom_type not in {"Polygon", "MultiPolygon"}:
        reason = f"unexpected_geometry_type:{geom_type}"

    return {
        "valid": valid and reason is None,
        "geom_type": geom_type,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Area consistency checks
# ---------------------------------------------------------------------------

def validate_catchment_area(
    catchment_mask: np.ndarray,
    flow_accumulation: np.ndarray,
    outlet: Tuple[int, int],
    resolution_m: float,
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    """Compare cell-derived area against accumulation-derived area."""

    r, c = outlet
    cell_count = int(np.sum(catchment_mask))
    catchment_area_m2 = cell_count * resolution_m * resolution_m

    accumulation_cells = float(flow_accumulation[r, c])
    accumulation_area_m2 = accumulation_cells * resolution_m * resolution_m

    if accumulation_area_m2 > 0:
        relative_difference = abs(catchment_area_m2 - accumulation_area_m2) / accumulation_area_m2
    else:
        relative_difference = 1.0

    return {
        "catchment_area_m2": float(catchment_area_m2),
        "accumulation_area_m2": float(accumulation_area_m2),
        "relative_difference": float(relative_difference),
        "consistent": relative_difference <= tolerance,
    }


def find_candidate_outlets(
    flow_direction: np.ndarray,
    valid_mask: np.ndarray,
    top_n: int = 5,
) -> List[Tuple[int, int]]:
    """Return terminal drainage cells (deterministic row-major order)."""

    flow_direction = np.asarray(flow_direction)
    valid = np.asarray(valid_mask, dtype=bool)
    terminal = valid & (flow_direction == -1)
    cells = [(int(r), int(c)) for r, c in np.argwhere(terminal)]
    return cells[: max(1, int(top_n))]


def analyze_catchment(
    flow_direction: np.ndarray,
    flow_accumulation: np.ndarray,
    valid_mask: np.ndarray,
    outlet: Tuple[int, int],
    resolution_m: float,
    x_coords: Optional[np.ndarray] = None,
    y_coords: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Convenience wrapper: delineate, measure, validate, and (optionally) polygonise."""

    mask = delineate_catchment(flow_direction, valid_mask, outlet)
    area = catchment_area(mask, resolution_m)
    consistency = validate_catchment_area(mask, flow_accumulation, outlet, resolution_m)

    result: Dict[str, Any] = {
        **area,
        "area_validation": consistency,
        "mask": mask,
        "boundary": None,
    }

    if x_coords is not None and y_coords is not None:
        geometry = mask_to_polygon(mask, x_coords, y_coords)
        result["boundary"] = polygon_to_geojson(geometry)

    return result


__all__ = [
    "analyze_catchment",
    "catchment_area",
    "delineate_catchment",
    "find_candidate_outlets",
    "mask_to_polygon",
    "polygon_to_geojson",
    "validate_catchment_area",
    "validate_catchment_geometry",
]
