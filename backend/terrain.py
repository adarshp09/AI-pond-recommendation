from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from models import ContourData, DEM, TerrainResult


DEFAULT_GRID_RESOLUTION_M = 5.0


# ---------------------------------------------------------------------------
# CONTOUR SAMPLE EXTRACTION
# ---------------------------------------------------------------------------

def _build_interpolation_samples(
    contour_data: ContourData,
):
    """
    Convert contour vertices into interpolation samples.

    Each contour vertex gets the elevation of its contour.
    """

    points = []
    values = []

    for contour in contour_data.contours:

        for x, y in contour.coordinates:

            points.append(
                (float(x), float(y))
            )

            values.append(
                float(contour.elevation_m)
            )

    if not points:
        raise ValueError(
            "No contour interpolation samples available"
        )

    points = np.asarray(
        points,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    # Remove invalid rows.
    mask = (
        np.isfinite(points[:, 0])
        & np.isfinite(points[:, 1])
        & np.isfinite(values)
    )

    points = points[mask]
    values = values[mask]

    if len(points) == 0:
        raise ValueError(
            "All contour interpolation samples are invalid"
        )

    return points, values


# ---------------------------------------------------------------------------
# GRID
# ---------------------------------------------------------------------------

def _create_grid(
    contour_data: ContourData,
    resolution_m: float,
):
    """
    Create a regular metric grid covering the contour extent.
    """

    if resolution_m <= 0:
        raise ValueError(
            "Grid resolution must be greater than zero"
        )

    x = np.arange(
        contour_data.min_x,
        contour_data.max_x + resolution_m,
        resolution_m,
        dtype=float,
    )

    y = np.arange(
        contour_data.min_y,
        contour_data.max_y + resolution_m,
        resolution_m,
        dtype=float,
    )

    if len(x) < 2 or len(y) < 2:
        raise ValueError(
            "Contour extent is too small for DEM generation"
        )

    return x, y


# ---------------------------------------------------------------------------
# INTERPOLATION
# ---------------------------------------------------------------------------

def _interpolate_dem(
    points: np.ndarray,
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
):
    """
    Linear interpolation with nearest-neighbour fallback.

    The returned array ALWAYS has shape:

        (len(y), len(x))
    """

    xx, yy = np.meshgrid(
        x,
        y,
    )

    query_points = np.column_stack(
        [
            xx.ravel(),
            yy.ravel(),
        ]
    )

    # Remove duplicate XY samples because duplicate contour vertices can
    # otherwise cause interpolation instability.
    unique_points, unique_indices = np.unique(
        points,
        axis=0,
        return_index=True,
    )

    unique_values = values[unique_indices]

    linear = LinearNDInterpolator(
        unique_points,
        unique_values,
        fill_value=np.nan,
    )

    z_flat = np.asarray(
        linear(query_points),
        dtype=float,
    )

    z = z_flat.reshape(
        xx.shape
    )

    # Nearest-neighbour fallback for cells outside the linear interpolation
    # convex hull.
    missing = ~np.isfinite(z)

    if np.any(missing):

        tree = cKDTree(
            unique_points
        )

        missing_points = query_points[missing.ravel()]

        _, indices = tree.query(
            missing_points,
            k=1,
        )

        z_flat[missing.ravel()] = (
            unique_values[indices]
        )

        z = z_flat.reshape(
            xx.shape
        )

    return z


# ---------------------------------------------------------------------------
# SLOPE
# ---------------------------------------------------------------------------

def calculate_slope(
    elevation: np.ndarray,
    resolution_m: float,
):
    """
    Calculate terrain slope.

    slope = atan(
        sqrt(
            dz/dx² + dz/dy²
        )
    )
    """

    if elevation.ndim != 2:
        raise ValueError(
            "DEM elevation must be a 2-D array"
        )

    if elevation.shape[0] < 2:
        raise ValueError(
            "DEM requires at least two rows"
        )

    if elevation.shape[1] < 2:
        raise ValueError(
            "DEM requires at least two columns"
        )

    dz_dy, dz_dx = np.gradient(
        elevation,
        resolution_m,
        resolution_m,
    )

    gradient = np.sqrt(
        dz_dx ** 2
        + dz_dy ** 2
    )

    slope_degrees = np.degrees(
        np.arctan(
            gradient
        )
    )

    return (
        slope_degrees,
        gradient,
    )


def calculate_d8_flow(
    dem: np.ndarray,
    resolution_m: float,
    valid_mask: Optional[np.ndarray] = None,
):
    """Backward-compatible alias for the D8 flow-direction utilities."""
    from hydrology import calculate_d8_flow_direction

    flow_direction, sink_mask, edge_outflow_mask = calculate_d8_flow_direction(
        dem,
        resolution_m,
        valid_mask,
    )
    return flow_direction, flow_direction


def calculate_flow_accumulation(
    flow_direction: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Backward-compatible alias for flow accumulation."""
    from hydrology import calculate_flow_accumulation as _calculate_flow_accumulation

    return _calculate_flow_accumulation(
        flow_direction,
        valid_mask,
    )


# ---------------------------------------------------------------------------
# DEM GENERATION
# ---------------------------------------------------------------------------

def build_dem(
    contour_data: ContourData,
    resolution_m: float = DEFAULT_GRID_RESOLUTION_M,
) -> DEM:
    """
    Build DEM from contour data.
    """

    points, values = _build_interpolation_samples(
        contour_data
    )

    x, y = _create_grid(
        contour_data,
        resolution_m,
    )

    elevation = _interpolate_dem(
        points,
        values,
        x,
        y,
    )

    if elevation.shape != (
        len(y),
        len(x),
    ):
        raise RuntimeError(
            "DEM shape mismatch: "
            f"expected {(len(y), len(x))}, "
            f"got {elevation.shape}"
        )

    slope_degrees, gradient = calculate_slope(
        elevation,
        resolution_m,
    )

    valid_mask = (
        np.isfinite(elevation)
        & np.isfinite(slope_degrees)
        & np.isfinite(gradient)
    )

    return DEM(
        elevation=elevation,
        x=x,
        y=y,
        resolution_m=float(resolution_m),
        crs=contour_data.target_crs,
        slope_degrees=slope_degrees,
        gradient_m_per_m=gradient,
        valid_mask=valid_mask,
    )


# ---------------------------------------------------------------------------
# TERRAIN ANALYSIS
# ---------------------------------------------------------------------------

def analyze_terrain(
    contour_data: ContourData,
    resolution_m: float = DEFAULT_GRID_RESOLUTION_M,
) -> TerrainResult:
    """
    Build DEM and calculate basic terrain statistics.
    """

    dem = build_dem(
        contour_data,
        resolution_m,
    )

    valid = dem.valid_mask

    if valid is None:
        valid = np.isfinite(
            dem.elevation
        )

    valid_fraction = float(
        np.mean(valid)
    )

    valid_slopes = dem.slope_degrees[
        valid
    ]

    if len(valid_slopes) == 0:
        raise ValueError(
            "DEM contains no valid slope cells"
        )

    return TerrainResult(
        dem=dem,
        mean_slope_degrees=float(
            np.mean(valid_slopes)
        ),
        max_slope_degrees=float(
            np.max(valid_slopes)
        ),
        valid_cell_fraction=valid_fraction,
    )