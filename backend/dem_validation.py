from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.spatial import cKDTree

from models import ContourData, DEM, DEMValidationResult


# ---------------------------------------------------------------------------
# BASIC STATISTICS
# ---------------------------------------------------------------------------

def _percentile(
    values: np.ndarray,
    percentile: float,
) -> float:
    if len(values) == 0:
        return float("nan")

    return float(
        np.percentile(
            values,
            percentile,
        )
    )


# ---------------------------------------------------------------------------
# LOCAL ELEVATION CHANGE
# ---------------------------------------------------------------------------

def _local_elevation_change(
    elevation: np.ndarray,
):
    """
    Calculate local absolute elevation changes.

    Uses horizontal and vertical neighbours.
    """

    changes = []

    if elevation.shape[1] > 1:

        horizontal = np.abs(
            elevation[:, 1:]
            - elevation[:, :-1]
        )

        changes.append(
            horizontal.ravel()
        )

    if elevation.shape[0] > 1:

        vertical = np.abs(
            elevation[1:, :]
            - elevation[:-1, :]
        )

        changes.append(
            vertical.ravel()
        )

    if not changes:
        return np.array([], dtype=float)

    result = np.concatenate(
        changes
    )

    return result[
        np.isfinite(result)
    ]


# ---------------------------------------------------------------------------
# CONTOUR CONSISTENCY
# ---------------------------------------------------------------------------

def _sample_contour_points(
    contour_data: ContourData,
    max_points: int = 200_000,
):
    """
    Extract contour vertices for DEM-vs-contour validation.
    """

    points = []
    values = []

    for contour in contour_data.contours:

        for x, y in contour.coordinates:

            points.append(
                (x, y)
            )

            values.append(
                contour.elevation_m
            )

    if not points:
        return (
            np.empty((0, 2)),
            np.empty((0,)),
        )

    points = np.asarray(
        points,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(points) > max_points:

        rng = np.random.default_rng(
            42
        )

        indices = rng.choice(
            len(points),
            size=max_points,
            replace=False,
        )

        points = points[indices]
        values = values[indices]

    return points, values


def _contour_consistency(
    dem: DEM,
    contour_data: ContourData,
) -> Dict[str, float]:
    """
    Compare DEM elevation against elevations of sampled contour points.
    """

    points, contour_values = (
        _sample_contour_points(
            contour_data
        )
    )

    if len(points) == 0:
        return {
            "sampled_points": 0,
            "mean_absolute_error_m": float("nan"),
            "rmse_m": float("nan"),
            "max_absolute_error_m": float("nan"),
            "within_1m_fraction": 0.0,
            "within_5m_fraction": 0.0,
            "quality_score": 0.0,
        }

    grid_x, grid_y = np.meshgrid(
        dem.x,
        dem.y,
    )

    grid_points = np.column_stack(
        [
            grid_x.ravel(),
            grid_y.ravel(),
        ]
    )

    tree = cKDTree(
        grid_points
    )

    _, nearest_indices = tree.query(
        points,
        k=1,
    )

    dem_values = dem.elevation.ravel()[
        nearest_indices
    ]

    errors = (
        dem_values
        - contour_values
    )

    errors = errors[
        np.isfinite(errors)
    ]

    if len(errors) == 0:
        return {
            "sampled_points": 0,
            "mean_absolute_error_m": float("nan"),
            "rmse_m": float("nan"),
            "max_absolute_error_m": float("nan"),
            "within_1m_fraction": 0.0,
            "within_5m_fraction": 0.0,
            "quality_score": 0.0,
        }

    absolute = np.abs(
        errors
    )

    mae = float(
        np.mean(absolute)
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    max_error = float(
        np.max(absolute)
    )

    within_1m = float(
        np.mean(
            absolute <= 1.0
        )
    )

    within_5m = float(
        np.mean(
            absolute <= 5.0
        )
    )

    # A simple quality score.
    #
    # 100% within 1m => excellent.
    # Remaining points are partially credited if they are within 5m.
    quality_score = (
        within_1m
        + 0.5
        * max(
            within_5m
            - within_1m,
            0.0,
        )
    )

    quality_score = float(
        np.clip(
            quality_score,
            0.0,
            1.0,
        )
    )

    return {
        "sampled_points": int(len(errors)),
        "mean_absolute_error_m": mae,
        "rmse_m": rmse,
        "max_absolute_error_m": max_error,
        "within_1m_fraction": within_1m,
        "within_5m_fraction": within_5m,
        "quality_score": quality_score,
    }


# ---------------------------------------------------------------------------
# QUALITY SCORE
# ---------------------------------------------------------------------------

def _calculate_quality_score(
    valid_fraction: float,
    nan_fraction: float,
    contour_score: float,
    extreme_gradient_fraction: float,
    flat_fraction: float,
) -> float:
    """
    Calculate overall DEM quality.

    The score intentionally focuses more strongly on:
      - valid cells
      - contour consistency
      - absence of NaNs
      - absence of extreme gradients

    Flat cells are not automatically treated as errors because flat
    terrain can legitimately occur in pond-suitable regions.
    """

    valid_score = np.clip(
        valid_fraction,
        0.0,
        1.0,
    )

    nan_score = np.clip(
        1.0 - nan_fraction,
        0.0,
        1.0,
    )

    gradient_score = np.clip(
        1.0
        - extreme_gradient_fraction,
        0.0,
        1.0,
    )

    flat_score = 1.0

    score = (
        0.30 * valid_score
        + 0.20 * nan_score
        + 0.40 * contour_score
        + 0.10 * gradient_score
    )

    # Keep flat_score available for future calibration without penalizing
    # naturally flat terrain in the current version.
    _ = flat_score

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


# ---------------------------------------------------------------------------
# MAIN VALIDATION
# ---------------------------------------------------------------------------

def validate_dem(
    dem: DEM,
    contour_data: ContourData,
) -> DEMValidationResult:
    """
    Validate generated DEM.
    """

    warnings: List[str] = []

    elevation = np.asarray(
        dem.elevation,
        dtype=float,
    )

    if elevation.ndim != 2:
        raise ValueError(
            "DEM elevation must be two-dimensional"
        )

    total_cells = elevation.size

    if total_cells == 0:
        raise ValueError(
            "DEM contains zero cells"
        )

    finite_mask = np.isfinite(
        elevation
    )

    valid_fraction = float(
        np.mean(finite_mask)
    )

    nan_fraction = float(
        1.0 - valid_fraction
    )

    if valid_fraction < 0.95:
        warnings.append(
            "Less than 95% of DEM cells are valid"
        )

    if nan_fraction > 0:
        warnings.append(
            "DEM contains NaN or infinite elevation cells"
        )

    valid_elevation = elevation[
        finite_mask
    ]

    # ---------------------------------------------------------------
    # Elevation statistics
    # ---------------------------------------------------------------

    elevation_stats = {
        "min_m": float(
            np.min(valid_elevation)
        ),
        "max_m": float(
            np.max(valid_elevation)
        ),
        "mean_m": float(
            np.mean(valid_elevation)
        ),
        "median_m": float(
            np.median(valid_elevation)
        ),
        "std_m": float(
            np.std(valid_elevation)
        ),
        "p01_m": _percentile(
            valid_elevation,
            1,
        ),
        "p05_m": _percentile(
            valid_elevation,
            5,
        ),
        "p25_m": _percentile(
            valid_elevation,
            25,
        ),
        "p50_m": _percentile(
            valid_elevation,
            50,
        ),
        "p75_m": _percentile(
            valid_elevation,
            75,
        ),
        "p95_m": _percentile(
            valid_elevation,
            95,
        ),
        "p99_m": _percentile(
            valid_elevation,
            99,
        ),
    }

    # ---------------------------------------------------------------
    # Local elevation changes
    # ---------------------------------------------------------------

    local_change = _local_elevation_change(
        elevation
    )

    if len(local_change):

        local_stats = {
            "mean_absolute_change_m": float(
                np.mean(local_change)
            ),
            "max_absolute_change_m": float(
                np.max(local_change)
            ),
            "p95_absolute_change_m": _percentile(
                local_change,
                95,
            ),
        }

    else:

        local_stats = {
            "mean_absolute_change_m": 0.0,
            "max_absolute_change_m": 0.0,
            "p95_absolute_change_m": 0.0,
        }

    # ---------------------------------------------------------------
    # Slope
    # ---------------------------------------------------------------

    if dem.slope_degrees is not None:

        slope = np.asarray(
            dem.slope_degrees,
            dtype=float,
        )

        slope_valid = slope[
            np.isfinite(slope)
        ]

    else:

        slope_valid = np.array(
            [],
            dtype=float,
        )

    if len(slope_valid):

        slope_stats = {
            "min_degrees": float(
                np.min(slope_valid)
            ),
            "mean_degrees": float(
                np.mean(slope_valid)
            ),
            "median_degrees": float(
                np.median(slope_valid)
            ),
            "p05_degrees": _percentile(
                slope_valid,
                5,
            ),
            "p95_degrees": _percentile(
                slope_valid,
                95,
            ),
            "max_degrees": float(
                np.max(slope_valid)
            ),
        }

    else:

        slope_stats = {
            "min_degrees": 0.0,
            "mean_degrees": 0.0,
            "median_degrees": 0.0,
            "p05_degrees": 0.0,
            "p95_degrees": 0.0,
            "max_degrees": 0.0,
        }

    # ---------------------------------------------------------------
    # Gradient
    # ---------------------------------------------------------------

    if dem.gradient_m_per_m is not None:

        gradient = np.asarray(
            dem.gradient_m_per_m,
            dtype=float,
        )

        gradient_valid = gradient[
            np.isfinite(gradient)
        ]

    else:

        gradient_valid = np.array(
            [],
            dtype=float,
        )

    if len(gradient_valid):

        extreme_mask = (
            gradient_valid > 1.0
        )

        extreme_fraction = float(
            np.mean(extreme_mask)
        )

        gradient_stats = {
            "mean_m_per_m": float(
                np.mean(gradient_valid)
            ),
            "max_m_per_m": float(
                np.max(gradient_valid)
            ),
            "extreme_gradient_fraction":
                extreme_fraction,
        }

    else:

        gradient_stats = {
            "mean_m_per_m": 0.0,
            "max_m_per_m": 0.0,
            "extreme_gradient_fraction": 0.0,
        }

        extreme_fraction = 0.0

    if extreme_fraction > 0.01:
        warnings.append(
            "More than 1% of DEM cells have extreme gradients"
        )

    # ---------------------------------------------------------------
    # Flat cells
    # ---------------------------------------------------------------

    if len(gradient_valid):

        flat_fraction = float(
            np.mean(
                gradient_valid < 1e-3
            )
        )

    else:

        flat_fraction = 0.0

    # ---------------------------------------------------------------
    # Contour consistency
    # ---------------------------------------------------------------

    contour_stats = _contour_consistency(
        dem,
        contour_data,
    )

    contour_score = contour_stats[
        "quality_score"
    ]

    if contour_score < 0.90:
        warnings.append(
            "DEM-to-contour consistency is below 0.90"
        )

    # ---------------------------------------------------------------
    # Overall score
    # ---------------------------------------------------------------

    score = _calculate_quality_score(
        valid_fraction=valid_fraction,
        nan_fraction=nan_fraction,
        contour_score=contour_score,
        extreme_gradient_fraction=extreme_fraction,
        flat_fraction=flat_fraction,
    )

    if score >= 0.90:
        status = "good"

    elif score >= 0.75:
        status = "acceptable"

    elif score >= 0.50:
        status = "poor"

    else:
        status = "invalid"

    if status == "invalid":
        warnings.append(
            "DEM failed minimum quality requirements"
        )

    return DEMValidationResult(
        status=status,
        score=score,
        shape={
            "rows": int(elevation.shape[0]),
            "columns": int(elevation.shape[1]),
        },
        resolution_m=float(
            dem.resolution_m
        ),
        valid_cell_fraction=valid_fraction,
        nan_fraction=nan_fraction,
        elevation=elevation_stats,
        local_elevation_change=local_stats,
        slope=slope_stats,
        gradient=gradient_stats,
        flat_fraction=flat_fraction,
        contour_consistency=contour_stats,
        warnings=warnings,
    )