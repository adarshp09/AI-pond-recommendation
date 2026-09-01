# hydrology.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# D8 directions.
#
# Index:
#   0 = N
#   1 = NE
#   2 = E
#   3 = SE
#   4 = S
#   5 = SW
#   6 = W
#   7 = NW
#
# The value stored in flow_direction is the direction index.
D8_OFFSETS = np.array(
    [
        (-1, 0),   # N
        (-1, 1),   # NE
        (0, 1),    # E
        (1, 1),    # SE
        (1, 0),    # S
        (1, -1),   # SW
        (0, -1),   # W
        (-1, -1),  # NW
    ],
    dtype=np.int8,
)

D8_DISTANCES = np.array(
    [
        1.0,
        np.sqrt(2.0),
        1.0,
        np.sqrt(2.0),
        1.0,
        np.sqrt(2.0),
        1.0,
        np.sqrt(2.0),
    ],
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


def calculate_slope(
    dem: np.ndarray,
    resolution_m: float,
) -> np.ndarray:
    """
    Calculate terrain slope from DEM using finite differences.

    Returns slope in degrees.
    """

    dem = np.asarray(dem, dtype=np.float64)

    if dem.ndim != 2:
        raise ValueError("DEM must be a 2D array.")

    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive.")

    dz_dy, dz_dx = np.gradient(
        dem,
        resolution_m,
        resolution_m,
    )

    gradient = np.sqrt(
        np.square(dz_dx) +
        np.square(dz_dy)
    )

    slope = np.degrees(np.arctan(gradient))

    slope[~np.isfinite(slope)] = np.nan

    return slope


def calculate_d8_flow_direction(
    dem: np.ndarray,
    resolution_m: float,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate D8 steepest-descent flow direction.

    Returns:

        flow_direction:
            int8 array.
            -1 = no downstream cell.

        sink_mask:
            True where no valid lower neighbouring cell exists.

        edge_outflow_mask:
            True where flow exits the DEM boundary.
    """

    dem = np.asarray(dem, dtype=np.float64)

    rows, cols = dem.shape

    if valid_mask is None:
        valid_mask = np.isfinite(dem)

    valid_mask = np.asarray(valid_mask, dtype=bool)

    if valid_mask.shape != dem.shape:
        raise ValueError(
            "valid_mask must have the same shape as DEM."
        )

    flow_direction = np.full(
        (rows, cols),
        -1,
        dtype=np.int8,
    )

    sink_mask = np.zeros(
        (rows, cols),
        dtype=bool,
    )

    edge_outflow_mask = np.zeros(
        (rows, cols),
        dtype=bool,
    )

    for r in range(rows):
        for c in range(cols):

            if not valid_mask[r, c]:
                continue

            z = dem[r, c]

            best_direction = -1
            best_slope = 0.0

            for direction, (dr, dc) in enumerate(D8_OFFSETS):

                nr = r + int(dr)
                nc = c + int(dc)

                if nr < 0 or nr >= rows:
                    edge_outflow_mask[r, c] = True
                    continue

                if nc < 0 or nc >= cols:
                    edge_outflow_mask[r, c] = True
                    continue

                if not valid_mask[nr, nc]:
                    continue

                dz = z - dem[nr, nc]

                if dz <= 0:
                    continue

                distance = resolution_m * D8_DISTANCES[direction]

                local_slope = dz / distance

                if local_slope > best_slope:
                    best_slope = local_slope
                    best_direction = direction

            flow_direction[r, c] = best_direction

            if best_direction == -1:
                sink_mask[r, c] = True

    return (
        flow_direction,
        sink_mask,
        edge_outflow_mask,
    )


def _downstream_cell(
    r: int,
    c: int,
    direction: int,
    rows: int,
    cols: int,
) -> Optional[Tuple[int, int]]:
    """
    Return downstream cell for a D8 direction.
    """

    if direction < 0 or direction >= 8:
        return None

    dr, dc = D8_OFFSETS[direction]

    nr = r + int(dr)
    nc = c + int(dc)

    if nr < 0 or nr >= rows:
        return None

    if nc < 0 or nc >= cols:
        return None

    return nr, nc


def calculate_flow_accumulation(
    flow_direction: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calculate topological D8 flow accumulation.

    Each valid cell contributes one unit of upstream area.

    A cell containing only itself therefore has accumulation = 1.
    """

    flow_direction = np.asarray(
        flow_direction,
        dtype=np.int8,
    )

    rows, cols = flow_direction.shape

    if valid_mask is None:
        valid_mask = flow_direction >= -1

    valid_mask = np.asarray(
        valid_mask,
        dtype=bool,
    )

    accumulation = np.zeros(
        (rows, cols),
        dtype=np.float64,
    )

    accumulation[valid_mask] = 1.0

    # Number of upstream dependencies for every cell.
    indegree = np.zeros(
        (rows, cols),
        dtype=np.int32,
    )

    downstream = {}

    for r in range(rows):
        for c in range(cols):

            if not valid_mask[r, c]:
                continue

            direction = int(flow_direction[r, c])

            if direction < 0:
                continue

            target = _downstream_cell(
                r,
                c,
                direction,
                rows,
                cols,
            )

            if target is None:
                continue

            nr, nc = target

            if not valid_mask[nr, nc]:
                continue

            indegree[nr, nc] += 1
            downstream[(r, c)] = (nr, nc)

    # Start with cells that have no upstream dependencies.
    queue = []

    for r in range(rows):
        for c in range(cols):

            if not valid_mask[r, c]:
                continue

            if indegree[r, c] == 0:
                queue.append((r, c))

    processed = 0
    head = 0

    while head < len(queue):

        r, c = queue[head]
        head += 1

        processed += 1

        target = downstream.get((r, c))

        if target is None:
            continue

        nr, nc = target

        accumulation[nr, nc] += accumulation[r, c]

        indegree[nr, nc] -= 1

        if indegree[nr, nc] == 0:
            queue.append((nr, nc))

    # If a DEM contains a flow cycle due to numerical issues,
    # unresolved cells remain. Do not fabricate accumulation.
    unresolved = valid_mask & (indegree > 0)

    if np.any(unresolved):
        accumulation[unresolved] = np.nan

    return accumulation


def analyze_hydrology(
    dem: np.ndarray,
    resolution_m: float,
    valid_mask: Optional[np.ndarray] = None,
) -> HydrologyResult:
    """
    Complete hydrology calculation.
    """

    dem = np.asarray(
        dem,
        dtype=np.float64,
    )

    if valid_mask is None:
        valid_mask = np.isfinite(dem)

    valid_mask = np.asarray(
        valid_mask,
        dtype=bool,
    )

    slope = calculate_slope(
        dem,
        resolution_m,
    )

    (
        flow_direction,
        sink_mask,
        edge_outflow_mask,
    ) = calculate_d8_flow_direction(
        dem,
        resolution_m,
        valid_mask,
    )

    accumulation = calculate_flow_accumulation(
        flow_direction,
        valid_mask,
    )

    valid_accumulation = accumulation[
        np.isfinite(accumulation)
        & valid_mask
    ]

    if valid_accumulation.size:
        max_accumulation = float(
            np.max(valid_accumulation)
        )

        mean_accumulation = float(
            np.mean(valid_accumulation)
        )
    else:
        max_accumulation = 0.0
        mean_accumulation = 0.0

    valid_sink_mask = sink_mask & valid_mask

    return HydrologyResult(
        flow_direction=flow_direction,
        flow_accumulation=accumulation,
        slope_degrees=slope,
        valid_mask=valid_mask,

        sink_mask=valid_sink_mask,
        edge_outflow_mask=edge_outflow_mask & valid_mask,

        total_cells=int(dem.size),
        valid_cells=int(np.sum(valid_mask)),
        sink_cells=int(np.sum(valid_sink_mask)),
        edge_outflow_cells=int(
            np.sum(edge_outflow_mask & valid_mask)
        ),

        max_accumulation_cells=max_accumulation,
        mean_accumulation_cells=mean_accumulation,

        cell_area_m2=float(
            resolution_m * resolution_m
        ),
    )