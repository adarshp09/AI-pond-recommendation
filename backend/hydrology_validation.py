# hydrology_validation.py
"""Hydrology QA layer.

Detects invalid flow directions, cycles, unresolved sinks, impossible
accumulation, missing outlets, and malformed catchment geometry, and reports a
status consistent with DEM validation (``good`` / ``acceptable`` / ``poor`` /
``invalid``). Wired into the live pipeline by ``backend/main.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:  # geometry validation lives with the catchment code (single source of truth)
    from catchment import validate_catchment_geometry
except ImportError:  # pragma: no cover
    validate_catchment_geometry = None  # type: ignore[assignment]


def _safe_fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def validate_flow_direction(
    flow_direction: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, Any]:
    """Validate D8 flow direction values (must be within -1..7)."""

    flow_direction = np.asarray(flow_direction)
    valid = np.asarray(valid_mask, dtype=bool)
    values = flow_direction[valid]

    if values.size == 0:
        return {"valid": False, "valid_direction_fraction": 0.0, "invalid_direction_cells": 0}

    in_range = (values >= -1) & (values <= 7)
    invalid_count = int(np.sum(~in_range))

    return {
        "valid": invalid_count == 0,
        "valid_direction_fraction": float(np.mean(in_range)),
        "invalid_direction_cells": invalid_count,
    }


def validate_flow_accumulation(
    flow_accumulation: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, Any]:
    """Validate accumulation: finite, >= 1 and never above the cell count."""

    accumulation = np.asarray(flow_accumulation, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    values = accumulation[valid]

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
    bounded = values <= float(np.sum(valid)) + 1e-9
    ok = finite & positive & bounded

    return {
        "valid": bool(np.all(ok)),
        "finite_fraction": float(np.mean(finite)),
        "minimum_cells": float(np.nanmin(values)),
        "maximum_cells": float(np.nanmax(values)),
        "invalid_cells": int(np.sum(~ok)),
    }


def validate_cycles(
    flow_accumulation: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, Any]:
    """Count valid cells left unresolved by accumulation (flow cycles)."""

    accumulation = np.asarray(flow_accumulation, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    valid_cells = int(np.sum(valid))

    unresolved = int(np.sum(valid & ~np.isfinite(accumulation)))

    return {
        "valid": unresolved == 0,
        "unresolved_cells": unresolved,
        "unresolved_fraction": _safe_fraction(unresolved, valid_cells),
    }


def validate_sinks(
    sink_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, Any]:
    """Quantify remaining internal sinks (informational / warning metric)."""

    valid = np.asarray(valid_mask, dtype=bool)
    sink_mask = np.asarray(sink_mask, dtype=bool)

    valid_cells = int(np.sum(valid))
    sink_cells = int(np.sum(sink_mask & valid))
    sink_fraction = _safe_fraction(sink_cells, valid_cells)

    return {
        "sink_cells": sink_cells,
        "sink_fraction": sink_fraction,
        "status": "high" if sink_fraction > 0.10 else "moderate" if sink_fraction > 0.03 else "low",
    }


def validate_edge_outflow(
    edge_outflow_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, Any]:
    """Quantify cells on the DEM boundary (potential edge outlets)."""

    valid = np.asarray(valid_mask, dtype=bool)
    edge_outflow_mask = np.asarray(edge_outflow_mask, dtype=bool)

    valid_cells = int(np.sum(valid))
    edge_cells = int(np.sum(edge_outflow_mask & valid))

    return {
        "edge_outflow_cells": edge_cells,
        "edge_outflow_fraction": _safe_fraction(edge_cells, valid_cells),
    }


def validate_outlet(
    outlet: Optional[Tuple[int, int]],
    valid_mask: np.ndarray,
    flow_direction: np.ndarray,
) -> Dict[str, Any]:
    """Validate that a drainage outlet was selected and is usable."""

    valid = np.asarray(valid_mask, dtype=bool)
    flow_direction = np.asarray(flow_direction)

    if outlet is None:
        return {"valid": False, "present": False, "reason": "no_outlet_selected"}

    rows, columns = valid.shape
    r, c = outlet
    inside = 0 <= r < rows and 0 <= c < columns
    on_valid_cell = bool(valid[r, c]) if inside else False
    terminal = bool(flow_direction[r, c] == -1) if inside else False

    ok = inside and on_valid_cell
    return {
        "valid": ok,
        "present": True,
        "inside_grid": inside,
        "on_valid_cell": on_valid_cell,
        "terminal_cell": terminal,
        "reason": None if ok else "outlet_off_grid_or_on_invalid_cell",
    }


def validate_catchment(
    catchment_mask: np.ndarray,
    outlet: Tuple[int, int],
    flow_accumulation: np.ndarray,
    cell_area_m2: float,
) -> Dict[str, Any]:
    """Validate a delineated catchment (outlet inside, non-empty, area consistent)."""

    catchment_mask = np.asarray(catchment_mask, dtype=bool)
    rows, columns = catchment_mask.shape
    r, c = outlet

    outlet_inside = (
        0 <= r < rows and 0 <= c < columns and bool(catchment_mask[r, c])
    )

    cell_count = int(np.sum(catchment_mask))
    area_from_cells = cell_count * cell_area_m2

    accumulation_value = (
        float(flow_accumulation[r, c])
        if 0 <= r < rows and 0 <= c < columns and np.isfinite(flow_accumulation[r, c])
        else 0.0
    )
    area_from_accumulation = accumulation_value * cell_area_m2

    if area_from_accumulation > 0:
        relative_difference = abs(area_from_cells - area_from_accumulation) / area_from_accumulation
    else:
        relative_difference = 1.0

    area_consistent = relative_difference <= 0.05

    return {
        "valid": outlet_inside and cell_count > 0 and area_consistent,
        "outlet_inside": outlet_inside,
        "cell_count": cell_count,
        "area_from_cells_m2": float(area_from_cells),
        "area_from_accumulation_m2": float(area_from_accumulation),
        "relative_area_difference": float(relative_difference),
        "area_consistent": area_consistent,
    }


def validate_hydrology(
    flow_direction: np.ndarray,
    flow_accumulation: np.ndarray,
    valid_mask: np.ndarray,
    sink_mask: np.ndarray,
    edge_outflow_mask: np.ndarray,
    *,
    primary_outlet: Optional[Tuple[int, int]] = None,
    catchment_validation: Optional[Dict[str, Any]] = None,
    geometry_validation: Optional[Dict[str, Any]] = None,
    cycle_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Complete hydrological QA with a ``good``/``acceptable``/``poor``/``invalid`` status."""

    direction = validate_flow_direction(flow_direction, valid_mask)
    accumulation = validate_flow_accumulation(flow_accumulation, valid_mask)
    sinks = validate_sinks(sink_mask, valid_mask)
    edge = validate_edge_outflow(edge_outflow_mask, valid_mask)
    cycles = validate_cycles(flow_accumulation, valid_mask)

    if cycle_count is not None:
        cycles["unresolved_cells"] = int(cycle_count)
        cycles["valid"] = int(cycle_count) == 0
        cycles["unresolved_fraction"] = _safe_fraction(
            int(cycle_count), int(np.sum(np.asarray(valid_mask, dtype=bool)))
        )

    outlet = validate_outlet(primary_outlet, valid_mask, flow_direction)

    warnings: list = []
    invalid_reasons: list = []

    if not direction["valid"]:
        warnings.append("Invalid D8 flow direction values detected.")
        invalid_reasons.append("invalid_flow_direction")

    non_cycle_invalid = int(accumulation["invalid_cells"]) - int(cycles["unresolved_cells"])
    if non_cycle_invalid > 0:
        warnings.append("Flow accumulation contains invalid values.")
        invalid_reasons.append("invalid_accumulation")

    if cycles["unresolved_cells"] > 0:
        warnings.append(
            f"{cycles['unresolved_cells']} valid cell(s) are unresolved because they lie on a flow cycle."
        )

    if sinks["sink_fraction"] > 0.10:
        warnings.append("More than 10% of valid cells remain internal sinks after depression handling.")
    elif sinks["sink_cells"] > 0:
        warnings.append(f"{sinks['sink_cells']} internal sink cell(s) remain after depression handling.")

    if edge["edge_outflow_fraction"] > 0.50:
        warnings.append("More than 50% of valid cells can flow toward the DEM boundary.")

    if not outlet["valid"]:
        warnings.append("No valid drainage outlet could be selected.")
        invalid_reasons.append("missing_outlet")

    if catchment_validation is not None and not catchment_validation.get("valid"):
        warnings.append("Catchment area is inconsistent with the flow network.")
        invalid_reasons.append("invalid_catchment_area")

    if geometry_validation is not None and not geometry_validation.get("valid"):
        warnings.append("Catchment boundary geometry is malformed.")
        invalid_reasons.append("malformed_catchment_geometry")

    direction_score = 1.0 if direction["valid"] else direction["valid_direction_fraction"]
    accumulation_score = 1.0 if accumulation["valid"] else accumulation["finite_fraction"]
    sink_penalty = min(sinks["sink_fraction"] * 2.0, 1.0)
    edge_penalty = min(edge["edge_outflow_fraction"] * 0.5, 0.5)
    cycle_penalty = min(cycles["unresolved_fraction"] * 2.0, 1.0)

    score = (
        0.35 * direction_score
        + 0.35 * accumulation_score
        + 0.20 * (1.0 - sink_penalty)
        + 0.10 * (1.0 - cycle_penalty)
        - edge_penalty
    )
    score = float(np.clip(score, 0.0, 1.0))

    if invalid_reasons:
        status = "invalid"
    elif score >= 0.85 and not warnings:
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
        "cycles": cycles,
        "sinks": sinks,
        "edge_outflow": edge,
        "outlet": outlet,
        "catchment": catchment_validation,
        "geometry": geometry_validation,
        "invalid_reasons": invalid_reasons,
        "warnings": warnings,
    }


__all__ = [
    "validate_catchment",
    "validate_catchment_geometry",
    "validate_cycles",
    "validate_edge_outflow",
    "validate_flow_accumulation",
    "validate_flow_direction",
    "validate_hydrology",
    "validate_outlet",
    "validate_sinks",
]
