# catchment.py

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
except ImportError:
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


def get_downstream(
    r: int,
    c: int,
    direction: int,
    shape: Tuple[int, int],
) -> Optional[Tuple[int, int]]:

    if direction < 0 or direction > 7:
        return None

    dr, dc = D8_OFFSETS[direction]

    nr = r + int(dr)
    nc = c + int(dc)

    rows, cols = shape

    if nr < 0 or nr >= rows:
        return None

    if nc < 0 or nc >= cols:
        return None

    return nr, nc


def build_upstream_graph(
    flow_direction: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:

    rows, cols = flow_direction.shape

    upstream = {}

    for r in range(rows):
        for c in range(cols):

            if not valid_mask[r, c]:
                continue

            direction = int(
                flow_direction[r, c]
            )

            downstream = get_downstream(
                r,
                c,
                direction,
                (rows, cols),
            )

            if downstream is None:
                continue

            upstream.setdefault(
                downstream,
                [],
            ).append(
                (r, c)
            )

    return upstream


def delineate_catchment(
    flow_direction: np.ndarray,
    valid_mask: np.ndarray,
    outlet: Tuple[int, int],
) -> np.ndarray:
    """
    Reverse trace D8 flow paths from outlet.

    Every upstream cell connected to the outlet is included.
    """

    rows, cols = flow_direction.shape

    r0, c0 = outlet

    if not (
        0 <= r0 < rows
        and 0 <= c0 < cols
    ):
        raise ValueError(
            "Outlet is outside the DEM."
        )

    if not valid_mask[r0, c0]:
        raise ValueError(
            "Outlet is not located on a valid DEM cell."
        )

    upstream = build_upstream_graph(
        flow_direction,
        valid_mask,
    )

    catchment = np.zeros(
        (rows, cols),
        dtype=bool,
    )

    queue = deque(
        [(r0, c0)]
    )

    catchment[r0, c0] = True

    while queue:

        cell = queue.popleft()

        for upstream_cell in upstream.get(
            cell,
            [],
        ):

            r, c = upstream_cell

            if catchment[r, c]:
                continue

            catchment[r, c] = True

            queue.append(
                upstream_cell
            )

    return catchment


def catchment_area(
    catchment_mask: np.ndarray,
    resolution_m: float,
) -> Dict[str, float]:

    cell_count = int(
        np.sum(catchment_mask)
    )

    cell_area = (
        resolution_m * resolution_m
    )

    area_m2 = (
        cell_count * cell_area
    )

    return {
        "area_m2": float(area_m2),
        "area_hectares": float(
            area_m2 / 10000.0
        ),
        "area_km2": float(
            area_m2 / 1_000_000.0
        ),
        "cell_count": cell_count,
    }


def mask_to_polygon(
    mask: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> Optional[Any]:
    """
    Convert raster catchment cells to a polygon.

    Uses union of individual cell polygons.

    This is intentionally independent of any sample-map coordinates.
    """

    if Polygon is None:
        return None

    rows, cols = mask.shape

    if len(x_coords) != cols:
        raise ValueError(
            "x_coords length does not match mask columns."
        )

    if len(y_coords) != rows:
        raise ValueError(
            "y_coords length does not match mask rows."
        )

    if cols > 1:
        dx = float(
            np.median(
                np.diff(x_coords)
            )
        )
    else:
        dx = 1.0

    if rows > 1:
        dy = float(
            np.median(
                np.diff(y_coords)
            )
        )
    else:
        dy = 1.0

    half_x = abs(dx) / 2.0
    half_y = abs(dy) / 2.0

    polygons = []

    active_cells = np.argwhere(mask)

    for r, c in active_cells:

        x = float(
            x_coords[c]
        )

        y = float(
            y_coords[r]
        )

        polygon = Polygon(
            [
                (x - half_x, y - half_y),
                (x + half_x, y - half_y),
                (x + half_x, y + half_y),
                (x - half_x, y + half_y),
            ]
        )

        polygons.append(polygon)

    if not polygons:
        return None

    return unary_union(polygons)


def polygon_to_geojson(
    geometry: Any,
) -> Optional[Dict[str, Any]]:

    if geometry is None:
        return None

    return {
        "type": geometry.geom_type,
        "coordinates": list(
            geometry.__geo_interface__["coordinates"]
        ),
    }


def validate_catchment_area(
    catchment_mask: np.ndarray,
    flow_accumulation: np.ndarray,
    outlet: Tuple[int, int],
    resolution_m: float,
    tolerance: float = 0.05,
) -> Dict[str, Any]:

    r, c = outlet

    cell_count = int(
        np.sum(catchment_mask)
    )

    catchment_area_m2 = (
        cell_count
        * resolution_m
        * resolution_m
    )

    accumulation_cells = float(
        flow_accumulation[r, c]
    )

    accumulation_area_m2 = (
        accumulation_cells
        * resolution_m
        * resolution_m
    )

    if accumulation_area_m2 > 0:

        relative_difference = abs(
            catchment_area_m2
            - accumulation_area_m2
        ) / accumulation_area_m2

    else:
        relative_difference = 1.0

    return {
        "catchment_area_m2": float(
            catchment_area_m2
        ),
        "accumulation_area_m2": float(
            accumulation_area_m2
        ),
        "relative_difference": float(
            relative_difference
        ),
        "consistent": (
            relative_difference <= tolerance
        ),
    }


def find_candidate_outlets(
    flow_direction: np.ndarray,
    valid_mask: np.ndarray,
    top_n: int = 5,
) -> List[Tuple[int, int]]:
    """Return candidate outlet cells, preferring sinks and edge exits."""
    rows, cols = flow_direction.shape
    candidates: List[Tuple[int, int]] = []

    for r in range(rows):
        for c in range(cols):
            if not valid_mask[r, c]:
                continue

            direction = int(flow_direction[r, c])
            if direction < 0:
                candidates.append((r, c))
                continue

            nr = r + int(D8_OFFSETS[direction][0])
            nc = c + int(D8_OFFSETS[direction][1])

            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                candidates.append((r, c))

    unique = []
    seen = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            unique.append(item)

    return unique[:max(1, top_n)]


def analyze_catchment(
    flow_direction: np.ndarray,
    flow_accumulation: np.ndarray,
    valid_mask: np.ndarray,
    outlet: Tuple[int, int],
    resolution_m: float,
    x_coords: Optional[np.ndarray] = None,
    y_coords: Optional[np.ndarray] = None,
) -> Dict[str, Any]:

    mask = delineate_catchment(
        flow_direction,
        valid_mask,
        outlet,
    )

    area = catchment_area(
        mask,
        resolution_m,
    )

    consistency = validate_catchment_area(
        mask,
        flow_accumulation,
        outlet,
        resolution_m,
    )

    result = {
        **area,
        "area_validation": consistency,
        "mask": mask,
        "boundary": None,
    }

    if (
        x_coords is not None
        and y_coords is not None
    ):

        geometry = mask_to_polygon(
            mask,
            x_coords,
            y_coords,
        )

        result["boundary"] = polygon_to_geojson(
            geometry
        )

    return result