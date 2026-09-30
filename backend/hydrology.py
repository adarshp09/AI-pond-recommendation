"""Correct, deterministic D8 hydrology engine.

Pipeline
--------
1. DEM pre-processing diagnostics (:mod:`depression`).
2. Priority-Flood + epsilon depression handling (internal artificial sinks are
   removed; genuinely enclosed regions are left untouched and reported).
3. D8 flow direction on the filled surface.
4. Topological flow accumulation.
5. Outlet detection.

D8 direction encoding (clockwise from North; ``row`` increases southward)::

    index  direction  (dr, dc)
      0       N        (-1,  0)
      1       NE       (-1, +1)
      2       E        ( 0, +1)
      3       SE       (+1, +1)
      4       S        (+1,  0)
      5       SW       (+1, -1)
      6       W        ( 0, -1)
      7       NW       (-1, -1)

``flow_direction == -1`` means "no downstream cell" (a terminal cell: either a
boundary outlet or an unresolved internal sink).

Tie-breaking: on equal steepest descent the **lowest direction index** wins
(directions are evaluated in index order with a strict ``>`` comparison), so
results are fully deterministic.

Invalid / NoData cells are never assigned a direction and are never routed
through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.depression import (
    DEFAULT_EPSILON_M,
    boundary_ring,
    fill_depressions,
    preprocess_dem,
)

# D8 offsets, clockwise from North.
D8_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-1, 0),   # 0 N
    (-1, 1),   # 1 NE
    (0, 1),    # 2 E
    (1, 1),    # 3 SE
    (1, 0),    # 4 S
    (1, -1),   # 5 SW
    (0, -1),   # 6 W
    (-1, -1),  # 7 NW
)

D8_DIRECTION_NAMES: Tuple[str, ...] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

D8_DISTANCES: np.ndarray = np.array(
    [1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0)],
    dtype=np.float64,
)


@dataclass
class HydrologyResult:
    flow_direction: np.ndarray
    flow_accumulation: np.ndarray
    slope_degrees: np.ndarray
    valid_mask: np.ndarray

    sink_mask: np.ndarray
    edge_outflow_mask: np.ndarray

    total_cells: int
    valid_cells: int
    sink_cells: int
    edge_outflow_cells: int

    max_accumulation_cells: float
    mean_accumulation_cells: float

    cell_area_m2: float

    # --- Phase-2 additive diagnostics (defaults keep the dataclass compatible) ---
    filled_elevation: Optional[np.ndarray] = None
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    depressions: Dict[str, Any] = field(default_factory=dict)
    outlet_selection: Dict[str, Any] = field(default_factory=dict)
    flow_direction_fraction: float = 0.0
    unresolved_cells: int = 0
    cycle_count: int = 0


# ---------------------------------------------------------------------------
# Array shifting helpers
# ---------------------------------------------------------------------------

def _shifted_neighbours(values: np.ndarray, dr: int, dc: int, fill_value: float) -> np.ndarray:
    """Return ``out[r, c] = values[r + dr, c + dc]`` with out-of-grid = fill_value."""

    rows, columns = values.shape
    out = np.full(values.shape, fill_value, dtype=values.dtype)

    r_start = max(0, -dr)
    r_end = rows - max(0, dr)
    c_start = max(0, -dc)
    c_end = columns - max(0, dc)
    if r_start >= r_end or c_start >= c_end:
        return out

    out[r_start:r_end, c_start:c_end] = values[
        r_start + dr:r_end + dr, c_start + dc:c_end + dc
    ]
    return out


def _shifted_index_neighbours(values: np.ndarray, dr: int, dc: int, fill_value: int = -1) -> np.ndarray:
    """Index version of :func:`_shifted_neighbours` (int64, out-of-grid = -1)."""

    rows, columns = values.shape
    out = np.full(values.shape, fill_value, dtype=np.int64)

    r_start = max(0, -dr)
    r_end = rows - max(0, dr)
    c_start = max(0, -dc)
    c_end = columns - max(0, dc)
    if r_start >= r_end or c_start >= c_end:
        return out

    out[r_start:r_end, c_start:c_end] = values[
        r_start + dr:r_end + dr, c_start + dc:c_end + dc
    ]
    return out


# ---------------------------------------------------------------------------
# Slope
# ---------------------------------------------------------------------------

def calculate_slope(dem: np.ndarray, resolution_m: float) -> np.ndarray:
    """Finite-difference slope in degrees."""

    dem = np.asarray(dem, dtype=np.float64)

    if dem.ndim != 2:
        raise ValueError("DEM must be a 2D array.")
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive.")

    dz_dy, dz_dx = np.gradient(dem, resolution_m, resolution_m)
    gradient = np.sqrt(np.square(dz_dx) + np.square(dz_dy))
    slope = np.degrees(np.arctan(gradient))
    slope[~np.isfinite(slope)] = np.nan
    return slope


# ---------------------------------------------------------------------------
# Sink classification
# ---------------------------------------------------------------------------

def _neighbour_values(elevation: np.ndarray, valid: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Neighbour elevation at ``(dr, dc)``; invalid/out-of-grid neighbours are +inf."""

    neighbour = _shifted_neighbours(elevation, dr, dc, np.inf)
    neighbour_valid = _shifted_neighbours(valid.astype(float), dr, dc, 0.0) > 0.5
    return np.where(neighbour_valid, neighbour, np.inf)


def _has_lower_neighbour(elevation: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """True where a strictly-lower *valid* D8 neighbour exists."""

    elevation = np.asarray(elevation, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)

    has_lower = np.zeros(elevation.shape, dtype=bool)
    for dr, dc in D8_OFFSETS:
        neighbour = _neighbour_values(elevation, valid, dr, dc)
        has_lower |= valid & (elevation > neighbour)
    return has_lower


def count_internal_sinks(elevation: np.ndarray, valid_mask: np.ndarray) -> int:
    """Count internal sinks (no lower neighbour, not on the boundary ring).

    Boundary cells with no lower neighbour are genuine edge outlets, not sinks.
    """

    elevation = np.asarray(elevation, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(elevation)
    edge = valid & boundary_ring(elevation.shape)
    has_lower = _has_lower_neighbour(elevation, valid)
    return int(np.sum(valid & ~has_lower & ~edge))


# ---------------------------------------------------------------------------
# D8 flow direction
# ---------------------------------------------------------------------------

def calculate_d8_flow_direction(
    dem: np.ndarray,
    resolution_m: float,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steepest-descent D8 flow direction.

    Returns ``(flow_direction, sink_mask, edge_outflow_mask)`` where

    * ``flow_direction`` int8, -1 = no downstream cell,
    * ``sink_mask`` = valid, no lower neighbour, *not* on the boundary ring
      (genuine internal sinks / terminal cells),
    * ``edge_outflow_mask`` = valid cells on the boundary ring (their flow
      exits the grid).

    Equal-slope ties resolve to the lowest direction index. Invalid cells are
    never routed through and always get ``-1``.
    """

    dem = np.asarray(dem, dtype=np.float64)

    if dem.ndim != 2:
        raise ValueError("DEM must be a 2D array.")
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive.")

    if valid_mask is None:
        valid_mask = np.isfinite(dem)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(dem)

    if valid.shape != dem.shape:
        raise ValueError("valid_mask must have the same shape as DEM.")

    flow_direction = np.full(dem.shape, -1, dtype=np.int8)
    best_slope = np.zeros(dem.shape, dtype=float)

    for direction, (dr, dc) in enumerate(D8_OFFSETS):
        neighbour = _neighbour_values(dem, valid, dr, dc)
        drop = dem - neighbour
        slope = drop / (resolution_m * D8_DISTANCES[direction])
        better = (drop > 0) & (slope > best_slope)
        flow_direction[better] = direction
        best_slope[better] = slope[better]

    flow_direction[~valid] = -1

    has_lower = best_slope > 0
    edge_outflow_mask = valid & boundary_ring(dem.shape)
    sink_mask = valid & ~has_lower & ~edge_outflow_mask

    return flow_direction, sink_mask, edge_outflow_mask


def calculate_d8_flow(
    dem: np.ndarray,
    resolution_m: float,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible helper returning ``(flow_direction, sink_mask)``.

    .. note::
       Phase 1's version returned ``(flow_direction, flow_direction)`` — the
       second value was the flow direction again, which was misleading. It now
       correctly returns the **sink mask** as the second value.
    """

    flow_direction, sink_mask, _ = calculate_d8_flow_direction(dem, resolution_m, valid_mask)
    return flow_direction, sink_mask


# ---------------------------------------------------------------------------
# Flow accumulation
# ---------------------------------------------------------------------------

def downstream_flat_index(flow_direction: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Flat downstream index per cell (-1 when there is no valid downstream cell)."""

    rows, columns = flow_direction.shape
    flat = np.arange(rows * columns, dtype=np.int64).reshape(rows, columns)

    downstream = np.full((rows, columns), -1, dtype=np.int64)
    for direction, (dr, dc) in enumerate(D8_OFFSETS):
        neighbour = _shifted_index_neighbours(flat, dr, dc, -1)
        selected = (flow_direction == direction) & valid_mask & (neighbour >= 0)
        downstream[selected] = neighbour[selected]

    return downstream.ravel()


def calculate_flow_accumulation(
    flow_direction: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Topological D8 flow accumulation (each valid cell contributes 1).

    Cells that cannot be resolved because they belong to (or drain through) a
    flow cycle are left as ``NaN`` — accumulation is never fabricated.
    """

    flow = np.asarray(flow_direction, dtype=np.int8)
    if flow.ndim != 2:
        raise ValueError("flow_direction must be a 2D array.")

    rows, columns = flow.shape

    if valid_mask is None:
        valid = flow >= -1
    else:
        valid = np.asarray(valid_mask, dtype=bool)
    valid = valid & (flow >= -1)

    total = rows * columns
    valid_flat = valid.ravel()

    downstream = downstream_flat_index(flow, valid)

    sources = np.flatnonzero(valid_flat & (downstream >= 0))
    targets = downstream[sources]
    keep = valid_flat[targets]
    sources = sources[keep]
    targets = targets[keep]

    downstream_where = np.full(total, -1, dtype=np.int64)
    downstream_where[sources] = targets

    indegree = np.zeros(total, dtype=np.int64)
    np.add.at(indegree, targets, 1)

    accumulation = np.where(valid_flat, 1.0, 0.0)

    queue = list(np.flatnonzero(valid_flat & (indegree == 0)))
    head = 0
    while head < len(queue):
        cell = queue[head]
        head += 1
        target = downstream_where[cell]
        if target >= 0:
            accumulation[target] += accumulation[cell]
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    unresolved = valid_flat & (indegree > 0)
    accumulation[unresolved] = np.nan

    return accumulation.reshape(rows, columns)


# ---------------------------------------------------------------------------
# Outlet detection
# ---------------------------------------------------------------------------

def find_outlets(
    flow_direction: np.ndarray,
    valid_mask: np.ndarray,
    flow_accumulation: Optional[np.ndarray] = None,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Identify drainage outlets, deterministically.

    A terminal cell (``flow_direction == -1``) is a *boundary outlet* when it
    lies on the grid boundary (its flow exits the grid) or an *internal
    endpoint* otherwise (an unresolved sink / enclosed drainage endpoint).

    Outlets are ordered by upstream accumulation (descending), then by
    ``(row, column)`` — so the primary outlet is the drainage point that
    collects the most upstream area, with ties broken deterministically.
    """

    flow_direction = np.asarray(flow_direction)
    valid = np.asarray(valid_mask, dtype=bool)

    if flow_direction.ndim != 2:
        raise ValueError("flow_direction must be a 2D array.")
    if valid.shape != flow_direction.shape:
        raise ValueError("valid_mask must have the same shape as flow_direction.")

    terminal = valid & (flow_direction == -1)
    edge = valid & boundary_ring(flow_direction.shape)
    boundary_outlets = terminal & edge
    internal_endpoints = terminal & ~edge

    if flow_accumulation is None:
        accumulation = np.zeros(flow_direction.shape, dtype=float)
    else:
        accumulation = np.asarray(flow_accumulation, dtype=float)
        if accumulation.shape != flow_direction.shape:
            raise ValueError("flow_accumulation must have the same shape as flow_direction.")

    def _rank(mask: np.ndarray) -> List[Tuple[float, int, int]]:
        ranked: List[Tuple[float, int, int]] = []
        for r, c in np.argwhere(mask):
            value = accumulation[int(r), int(c)]
            score = float(value) if np.isfinite(value) else 0.0
            ranked.append((-score, int(r), int(c)))
        ranked.sort()
        return ranked

    ordered: List[Dict[str, Any]] = []
    for kind, mask in (("boundary", boundary_outlets), ("internal", internal_endpoints)):
        for neg_score, r, c in _rank(mask):
            ordered.append(
                {
                    "row": r,
                    "col": c,
                    "kind": kind,
                    "accumulation_cells": float(-neg_score),
                }
            )

    if not ordered:
        return {
            "primary": None,
            "kind": None,
            "outlets": [],
            "reason": "no_valid_outlet",
            "no_valid_outlet": True,
        }

    outlets = ordered[: max(1, int(top_n))]
    for index, outlet in enumerate(outlets):
        outlet["selected"] = index == 0

    primary = outlets[0]
    reason = (
        "max_upstream_accumulation_boundary_outlet"
        if primary["kind"] == "boundary"
        else "internal_endpoint_no_boundary_outlet"
    )

    return {
        "primary": (primary["row"], primary["col"]),
        "kind": primary["kind"],
        "outlets": outlets,
        "reason": reason,
        "no_valid_outlet": False,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def analyze_hydrology(
    dem: np.ndarray,
    resolution_m: float,
    valid_mask: Optional[np.ndarray] = None,
    *,
    fill: bool = True,
    epsilon_m: float = DEFAULT_EPSILON_M,
    outlet_top_n: int = 5,
) -> HydrologyResult:
    """Complete, corrected hydrology calculation."""

    dem = np.asarray(dem, dtype=np.float64)
    if dem.ndim != 2:
        raise ValueError("DEM elevation must be a 2D array.")

    preprocessing = preprocess_dem(dem, valid_mask, resolution_m)

    if valid_mask is None:
        valid = np.isfinite(dem)
    else:
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(dem)

    original_sink_count = count_internal_sinks(dem, valid)

    depression_result = None
    if fill:
        depression_result = fill_depressions(dem, valid, resolution_m, epsilon_m)
        routing_elevation = depression_result.filled_elevation
    else:
        routing_elevation = np.where(valid, dem, np.nan)

    flow_direction, sink_mask, edge_outflow_mask = calculate_d8_flow_direction(
        routing_elevation, resolution_m, valid
    )

    accumulation = calculate_flow_accumulation(flow_direction, valid)

    slope = calculate_slope(dem, resolution_m)

    valid_cells = int(np.sum(valid))
    remaining_sink_count = int(np.sum(sink_mask))
    resolved_sink_count = max(0, original_sink_count - remaining_sink_count)
    unresolved_cells = int(np.sum(valid & ~np.isfinite(accumulation)))

    valid_accumulation = accumulation[np.isfinite(accumulation) & valid]
    if valid_accumulation.size:
        max_accumulation = float(np.max(valid_accumulation))
        mean_accumulation = float(np.mean(valid_accumulation))
    else:
        max_accumulation = 0.0
        mean_accumulation = 0.0

    resolved_directions = int(np.sum(valid & (flow_direction >= 0)))
    flow_direction_fraction = (
        float(resolved_directions / valid_cells) if valid_cells else 0.0
    )

    outlet_selection = find_outlets(flow_direction, valid, accumulation, outlet_top_n)

    depressions: Dict[str, Any] = {
        "original_sink_count": int(original_sink_count),
        "resolved_sink_count": int(resolved_sink_count),
        "remaining_sink_count": int(remaining_sink_count),
        "max_fill_depth_m": float(depression_result.max_fill_depth_m) if depression_result else 0.0,
        "modified_cell_count": int(depression_result.modified_cell_count) if depression_result else 0,
        "unreachable_cell_count": int(depression_result.unreachable_cell_count) if depression_result else 0,
        "epsilon_m": float(epsilon_m),
        "filling_applied": bool(fill),
        "warnings": list(depression_result.warnings) if depression_result else [],
    }

    return HydrologyResult(
        flow_direction=flow_direction,
        flow_accumulation=accumulation,
        slope_degrees=slope,
        valid_mask=valid,
        sink_mask=sink_mask & valid,
        edge_outflow_mask=edge_outflow_mask & valid,
        total_cells=int(dem.size),
        valid_cells=valid_cells,
        sink_cells=remaining_sink_count,
        edge_outflow_cells=int(np.sum(edge_outflow_mask & valid)),
        max_accumulation_cells=max_accumulation,
        mean_accumulation_cells=mean_accumulation,
        cell_area_m2=float(resolution_m * resolution_m),
        filled_elevation=(depression_result.filled_elevation if depression_result else None),
        preprocessing=preprocessing.to_dict(),
        depressions=depressions,
        outlet_selection=outlet_selection,
        flow_direction_fraction=flow_direction_fraction,
        unresolved_cells=unresolved_cells,
        cycle_count=unresolved_cells,
    )


__all__ = [
    "D8_DIRECTION_NAMES",
    "D8_DISTANCES",
    "D8_OFFSETS",
    "HydrologyResult",
    "analyze_hydrology",
    "calculate_d8_flow",
    "calculate_d8_flow_direction",
    "calculate_flow_accumulation",
    "calculate_slope",
    "count_internal_sinks",
    "downstream_flat_index",
    "find_outlets",
]
