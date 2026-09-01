# hydrology_validation.py

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def _safe_fraction(
    numerator: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0

    return float(numerator / denominator)


def validate_flow_direction(
    flow_direction: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Validate D8 flow direction values.
    """

    valid_values = flow_direction[valid_mask]

    if valid_values.size == 0:
        return {
            "valid": False,
            "valid_direction_fraction": 0.0,
            "invalid_direction_cells": 0,
        }

    valid_direction_values = (
        (valid_values >= -1)
        & (valid_values <= 7)
    )

    invalid_count = int(
        np.sum(~valid_direction_values)
    )

    valid_fraction = float(
        np.mean(valid_direction_values)
    )

    return {
        "valid": invalid_count == 0,
        "valid_direction_fraction": valid_fraction,
        "invalid_direction_cells": invalid_count,
    }


def validate_flow_accumulation(
    flow_accumulation: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Validate accumulation values.

    Basic expectations:

    accumulation >= 1 for valid cells
    accumulation is finite
    accumulation does not exceed number of valid cells
    """

    values = flow_accumulation[valid_mask]

    if values.size == 0:
        return {
            "valid": False,
            "finite_fraction": 0.0,
            "minimum_cells": None,
            "maximum_cells": None,
            "invalid_cells": 0,
        }

    finite = np.isfinite(values)

    positive = values >= 1.0

    max_allowed = float(
        np.sum(valid_mask)
    )

    bounded = values <= max_allowed + 1e-9

    valid = finite & positive & bounded

    return {
        "valid": bool(np.all(valid)),
        "finite_fraction": float(np.mean(finite)),
        "minimum_cells": float(
            np.nanmin(values)
        ),
        "maximum_cells": float(
            np.nanmax(values)
        ),
        "invalid_cells": int(
            np.sum(~valid)
        ),
    }


def validate_sinks(
    sink_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Quantify sink cells.

    Natural sinks can exist in real terrain, so this is a warning
    metric rather than an automatic failure.
    """

    valid_cells = int(
        np.sum(valid_mask)
    )

    sink_cells = int(
        np.sum(
            sink_mask
            & valid_mask
        )
    )

    sink_fraction = _safe_fraction(
        sink_cells,
        valid_cells,
    )

    return {
        "sink_cells": sink_cells,
        "sink_fraction": sink_fraction,
        "status": (
            "high"
            if sink_fraction > 0.10
            else "moderate"
            if sink_fraction > 0.03
            else "low"
        ),
    }


def validate_edge_outflow(
    edge_outflow_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Quantify cells whose flow reaches the DEM boundary.
    """

    valid_cells = int(
        np.sum(valid_mask)
    )

    edge_cells = int(
        np.sum(
            edge_outflow_mask
            & valid_mask
        )
    )

    fraction = _safe_fraction(
        edge_cells,
        valid_cells,
    )

    return {
        "edge_outflow_cells": edge_cells,
        "edge_outflow_fraction": fraction,
    }


def validate_catchment(
    catchment_mask: np.ndarray,
    outlet: Tuple[int, int],
    flow_accumulation: np.ndarray,
    cell_area_m2: float,
) -> Dict[str, Any]:
    """
    Validate a delineated catchment.

    Checks:

    1. outlet exists inside catchment
    2. catchment is non-empty
    3. cell-derived area agrees with accumulation-derived area
    """

    catchment_mask = np.asarray(
        catchment_mask,
        dtype=bool,
    )

    rows, cols = catchment_mask.shape

    r, c = outlet

    outlet_inside = (
        0 <= r < rows
        and 0 <= c < cols
        and bool(catchment_mask[r, c])
    )

    cell_count = int(
        np.sum(catchment_mask)
    )

    area_from_cells = (
        cell_count * cell_area_m2
    )

    outlet_accumulation = float(
        flow_accumulation[r, c]
    ) if (
        0 <= r < rows
        and 0 <= c < cols
        and np.isfinite(flow_accumulation[r, c])
    ) else 0.0

    area_from_accumulation = (
        outlet_accumulation * cell_area_m2
    )

    if area_from_accumulation > 0:
        relative_difference = abs(
            area_from_cells
            - area_from_accumulation
        ) / area_from_accumulation
    else:
        relative_difference = 1.0

    area_consistent = (
        relative_difference <= 0.05
    )

    return {
        "valid": (
            outlet_inside
            and cell_count > 0
            and area_consistent
        ),
        "outlet_inside": outlet_inside,
        "cell_count": cell_count,
        "area_from_cells_m2": float(
            area_from_cells
        ),
        "area_from_accumulation_m2": float(
            area_from_accumulation
        ),
        "relative_area_difference": float(
            relative_difference
        ),
        "area_consistent": area_consistent,
    }


def validate_hydrology(
    flow_direction: np.ndarray,
    flow_accumulation: np.ndarray,
    valid_mask: np.ndarray,
    sink_mask: np.ndarray,
    edge_outflow_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Complete hydrological QA before catchment selection.
    """

    direction = validate_flow_direction(
        flow_direction,
        valid_mask,
    )

    accumulation = validate_flow_accumulation(
        flow_accumulation,
        valid_mask,
    )

    sinks = validate_sinks(
        sink_mask,
        valid_mask,
    )

    edge = validate_edge_outflow(
        edge_outflow_mask,
        valid_mask,
    )

    warnings = []

    if not direction["valid"]:
        warnings.append(
            "Invalid D8 flow direction values detected."
        )

    if not accumulation["valid"]:
        warnings.append(
            "Flow accumulation contains invalid values."
        )

    if sinks["sink_fraction"] > 0.10:
        warnings.append(
            "More than 10% of valid cells are sinks. "
            "Consider depression filling before final hydrological analysis."
        )

    if edge["edge_outflow_fraction"] > 0.50:
        warnings.append(
            "More than 50% of valid cells can flow toward "
            "the DEM boundary."
        )

    # Component score.
    direction_score = (
        1.0
        if direction["valid"]
        else direction["valid_direction_fraction"]
    )

    accumulation_score = (
        1.0
        if accumulation["valid"]
        else accumulation["finite_fraction"]
    )

    sink_penalty = min(
        sinks["sink_fraction"] * 2.0,
        1.0,
    )

    edge_penalty = min(
        edge["edge_outflow_fraction"] * 0.5,
        0.5,
    )

    score = (
        0.35 * direction_score
        + 0.40 * accumulation_score
        + 0.25 * (1.0 - sink_penalty)
        - edge_penalty
    )

    score = float(
        np.clip(score, 0.0, 1.0)
    )

    if score >= 0.85 and not warnings:
        status = "good"
    elif score >= 0.65:
        status = "acceptable"
    else:
        status = "poor"

    return {
        "status": status,
        "score": score,

        "flow_direction": direction,
        "flow_accumulation": accumulation,
        "sinks": sinks,
        "edge_outflow": edge,

        "warnings": warnings,
    }