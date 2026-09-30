"""Deterministic DEM pre-processing and depression (sink) handling.

This module implements the standard **Priority-Flood + epsilon** algorithm
(Barnes, Lehman & Mulla, 2014) to produce a hydrologically-correct,
strictly-draining surface from an arbitrary DEM:

1. Every valid cell on the grid border is a seed (genuine boundary drainage).
2. Cells are visited in increasing elevation order (a min-heap).
3. A neighbour that is not higher than the cell currently being processed is
   raised to ``processed_elevation + epsilon``.

The result is a filled surface in which every valid cell that is *connected to
the boundary* has a strictly lower downstream neighbour. This removes internal
artificial sinks without inventing drainage: valid cells that are fully
enclosed by NoData (no path to the boundary) are left untouched and remain
genuine unresolved sinks, which the validation layer reports.

NoData cells are never modified and never participate in routing.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

# D8 neighbour offsets, clockwise from North (+row = south, +col = east).
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

DEFAULT_EPSILON_M = 1e-4


def boundary_ring(shape: Tuple[int, int]) -> np.ndarray:
    """Return a boolean mask that is True on the outer ring of the grid."""

    rows, columns = shape
    mask = np.zeros((rows, columns), dtype=bool)
    if rows == 0 or columns == 0:
        return mask
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


@dataclass
class DemPreprocessing:
    """Diagnostics describing the DEM handed to the hydrology engine."""

    rows: int
    columns: int
    resolution_m: float
    total_cells: int
    valid_cells: int
    invalid_cells: int
    valid_fraction: float
    nan_count: int
    elevation_min_m: Optional[float]
    elevation_max_m: Optional[float]
    elevation_range_m: Optional[float]
    warnings: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": int(self.rows),
            "columns": int(self.columns),
            "resolution_m": float(self.resolution_m),
            "total_cells": int(self.total_cells),
            "valid_cells": int(self.valid_cells),
            "invalid_cells": int(self.invalid_cells),
            "valid_fraction": float(self.valid_fraction),
            "nan_count": int(self.nan_count),
            "elevation_min_m": self.elevation_min_m,
            "elevation_max_m": self.elevation_max_m,
            "elevation_range_m": self.elevation_range_m,
            "warnings": list(self.warnings),
        }


@dataclass
class DepressionResult:
    """Outcome of priority-flood depression handling."""

    filled_elevation: np.ndarray
    modified_mask: np.ndarray
    reachable_mask: np.ndarray
    max_fill_depth_m: float
    modified_cell_count: int
    unreachable_cell_count: int
    epsilon_m: float
    warnings: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epsilon_m": float(self.epsilon_m),
            "max_fill_depth_m": float(self.max_fill_depth_m),
            "modified_cell_count": int(self.modified_cell_count),
            "unreachable_cell_count": int(self.unreachable_cell_count),
            "warnings": list(self.warnings),
        }


def preprocess_dem(
    elevation: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    resolution_m: float = 5.0,
) -> DemPreprocessing:
    """Validate DEM shape/resolution/finite values and collect diagnostics."""

    elevation = np.asarray(elevation, dtype=float)

    if elevation.ndim != 2:
        raise ValueError("DEM elevation must be a two-dimensional array.")

    if elevation.size == 0:
        raise ValueError("DEM contains zero cells.")

    if resolution_m <= 0:
        raise ValueError("DEM resolution must be greater than zero.")

    if valid_mask is None:
        valid_mask = np.isfinite(elevation)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape != elevation.shape:
        raise ValueError("valid_mask must have the same shape as the DEM.")

    finite = np.isfinite(elevation)
    valid = valid_mask & finite

    total = int(elevation.size)
    valid_cells = int(np.sum(valid))
    invalid_cells = total - valid_cells
    nan_count = int(elevation.size - int(np.sum(finite)))

    warnings: list = []
    if valid_cells == 0:
        raise ValueError("DEM contains no valid (finite) elevation cells.")
    if nan_count:
        warnings.append(f"DEM contains {nan_count} NaN/infinite cell(s).")
    if valid_cells < total:
        warnings.append(f"{invalid_cells} cell(s) are NoData and will not be routed.")

    finite_values = elevation[valid]
    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))

    return DemPreprocessing(
        rows=int(elevation.shape[0]),
        columns=int(elevation.shape[1]),
        resolution_m=float(resolution_m),
        total_cells=total,
        valid_cells=valid_cells,
        invalid_cells=invalid_cells,
        valid_fraction=float(valid_cells / total),
        nan_count=nan_count,
        elevation_min_m=minimum,
        elevation_max_m=maximum,
        elevation_range_m=maximum - minimum,
        warnings=warnings,
    )


def fill_depressions(
    elevation: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    resolution_m: float = 5.0,
    epsilon_m: float = DEFAULT_EPSILON_M,
) -> DepressionResult:
    """Priority-Flood + epsilon depression fill.

    Returns a filled elevation array in which every cell connected to the grid
    boundary drains strictly downhill toward it. NoData cells are preserved as
    NaN and never modified. Cells enclosed by NoData (unreachable from any
    boundary cell) keep their original elevation and are reported separately.
    """

    elevation = np.asarray(elevation, dtype=float)

    if elevation.ndim != 2 or elevation.size == 0:
        raise ValueError("DEM elevation must be a non-empty two-dimensional array.")

    if epsilon_m <= 0:
        raise ValueError("epsilon_m must be greater than zero.")

    if valid_mask is None:
        valid_mask = np.isfinite(elevation)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(elevation)

    rows, columns = elevation.shape
    filled = elevation.astype(float, copy=True)
    filled[~valid] = np.nan

    visited = np.zeros((rows, columns), dtype=bool)
    heap: list = []

    seeds = boundary_ring((rows, columns)) & valid
    for r, c in np.argwhere(seeds):
        visited[r, c] = True
        heap.append((float(filled[r, c]), int(r) * columns + int(c)))
    heapq.heapify(heap)

    while heap:
        processed_elevation, flat = heapq.heappop(heap)
        r, c = divmod(flat, columns)

        for dr, dc in D8_OFFSETS:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= columns:
                continue
            if not valid[nr, nc] or visited[nr, nc]:
                continue

            neighbour_elevation = filled[nr, nc]
            if neighbour_elevation <= processed_elevation:
                neighbour_elevation = processed_elevation + epsilon_m
                filled[nr, nc] = neighbour_elevation

            visited[nr, nc] = True
            heapq.heappush(heap, (float(neighbour_elevation), nr * columns + nc))

    modified = valid & (filled > elevation + epsilon_m * 0.5)
    reachable = visited & valid
    unreachable = valid & ~reachable

    depth = filled - elevation
    max_depth = float(np.nanmax(depth[modified])) if np.any(modified) else 0.0

    warnings: list = []
    if np.any(unreachable):
        warnings.append(
            f"{int(np.sum(unreachable))} valid cell(s) are enclosed by NoData and cannot drain to the boundary; "
            "they were left unmodified."
        )

    return DepressionResult(
        filled_elevation=filled,
        modified_mask=modified,
        reachable_mask=reachable,
        max_fill_depth_m=max_depth,
        modified_cell_count=int(np.sum(modified)),
        unreachable_cell_count=int(np.sum(unreachable)),
        epsilon_m=float(epsilon_m),
        warnings=warnings,
    )


__all__ = [
    "D8_OFFSETS",
    "DEFAULT_EPSILON_M",
    "DemPreprocessing",
    "DepressionResult",
    "boundary_ring",
    "fill_depressions",
    "preprocess_dem",
]
