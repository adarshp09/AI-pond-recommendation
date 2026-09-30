"""Phase 3 — hydrologically valid pond candidate generation.

This module consumes the **corrected Phase-2 hydrology raster** directly (it
never recomputes DEM/hydrology) and produces deterministic pond candidates on
the flow network:

    extract_candidate_seeds   -> flow-accumulation local maxima + detected outlets
    build_candidates          -> attach features (elevation, slope, drainage area, outlet/catchment)
    apply_hard_constraints    -> reject only fundamental geometric/hydrological violations
    deduplicate_candidates    -> spatial separation (grid-binned, deterministic)
    score_and_rank_candidates -> deterministic, explainable ranking

Why accumulation peaks? A pond occupies a point where water naturally
concentrates. Local maxima of D8 flow accumulation mark exactly those
convergence points on the corrected raster, and they are cheap to find
(vectorised neighbourhood maximum + connected-component reduction). They are
also bounded in number, so we never emit one candidate per cell.

Hard constraints vs soft factors are strictly separated: hard constraints reject
only invalid/impossible sites; everything else (slope, accumulation magnitude,
elevation, relief) is exposed as a *soft* feature for Phase 4–6 scoring.

External geographic context (roads/buildings/water bodies) and rainfall/runoff
are intentionally **not** part of candidate validity.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pyproj import CRS, Transformer
from scipy import ndimage
from functools import lru_cache

from backend.hydrology import downstream_flat_index
from backend.suitability import catchment_area_suitability, slope_suitability


# ---------------------------------------------------------------------------
# Projected (metric) coordinates
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _cached_transformer(source_crs: str, target_crs: str) -> Transformer:
    """Memoise Transformer construction (expensive: PROJ database lookups)."""

    return Transformer.from_crs(source_crs, target_crs, always_xy=True)


def _utm_epsg(longitude: float, latitude: float) -> str:
    zone = int((longitude + 180.0) / 6.0) + 1
    zone = min(max(zone, 1), 60)
    base = 32600 if latitude >= 0 else 32700
    return f"EPSG:{base + zone}"


def _is_projected_crs(crs_value: Any) -> Optional[bool]:
    if crs_value is None:
        return None
    try:
        return bool(CRS.from_user_input(crs_value).is_projected)
    except Exception:
        return None


def metric_crs(dem: Any) -> Optional[str]:
    """Return the projected CRS used for metric separation/distance work.

    A projected source CRS is kept as-is; a geographic source (e.g. EPSG:4326)
    yields a UTM zone derived from the raster centroid. ``None`` means the DEM
    coordinates are already metric (or unknown) and are used unchanged.
    """

    crs_value = getattr(dem, "crs", None)
    projected = _is_projected_crs(crs_value)
    if projected is None:
        return None
    if projected:
        return str(crs_value)
    x = np.asarray(getattr(dem, "x"), dtype=float)
    y = np.asarray(getattr(dem, "y"), dtype=float)
    return _utm_epsg(float(np.mean(x)), float(np.mean(y)))


def metric_xy(dem: Any, x: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Project DEM coordinates to metres (identity when already metric)."""

    target = metric_crs(dem)
    if target is None:
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    source = getattr(dem, "crs", None)
    if source is not None and str(source) == target:
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    try:
        transformer = _cached_transformer(str(source), str(target))
        mx, my = transformer.transform(x, y)
        return np.asarray(mx, dtype=float), np.asarray(my, dtype=float)
    except Exception:
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Phase-3 ranking is a *heuristic ordering*, not the final pond suitability
# score (that arrives in Phase 4–6 with rainfall / land context / runoff).
DEFAULT_CANDIDATE_WEIGHTS: Dict[str, float] = {
    "catchment_area": 0.35,
    "flow_accumulation": 0.25,
    "slope": 0.20,
    "hydrological_confidence": 0.10,
    "elevation": 0.10,
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return int(default)


@dataclass
class CandidateConfig:
    """Deterministic candidate-generation configuration."""

    min_separation_m: float = 50.0
    min_drainage_area_cells: float = 2.0
    max_candidates: int = 25
    peak_neighbourhood: int = 3
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CANDIDATE_WEIGHTS))

    @classmethod
    def from_env(cls) -> "CandidateConfig":
        return cls(
            min_separation_m=_env_float("POND_MIN_CANDIDATE_SEPARATION_M", 50.0),
            min_drainage_area_cells=_env_float("POND_MIN_DRAINAGE_AREA_CELLS", 2.0),
            max_candidates=_env_int("POND_MAX_CANDIDATES", 25),
        )


@dataclass
class CandidateSet:
    """Final, ranked candidates plus a full rejection ledger."""

    candidates: List[Dict[str, Any]]
    rejected: List[Dict[str, Any]]
    seed_count: int
    accepted_count: int
    deduplicated_count: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class _NullStage:
    def set_output(self, **_: Any) -> "_NullStage":
        return self

    def add_metric(self, *_: Any, **__: Any) -> "_NullStage":
        return self

    def warn(self, *_: Any, **__: Any) -> "_NullStage":
        return self


class _NullRecorder:
    """No-op stage recorder so the pipeline can run without instrumentation."""

    def stage(self, *_: Any, **__: Any):
        import contextlib

        return contextlib.nullcontext(_NullStage())


# ---------------------------------------------------------------------------
# Stage 1 — seed extraction
# ---------------------------------------------------------------------------

def _reduce_plateau_peaks(peak_mask: np.ndarray, accumulation: np.ndarray) -> List[Tuple[int, int]]:
    """Return one representative cell per connected plateau of the peak mask.

    The representative is the cell with the highest accumulation, breaking ties
    by the lowest flat index — fully deterministic.
    """

    labels, component_count = ndimage.label(peak_mask)
    if component_count == 0:
        return []

    flat_indices = np.flatnonzero(peak_mask.ravel())
    component_of = labels.ravel()[flat_indices]
    accumulation_of = accumulation.ravel()[flat_indices]

    # lexsort: primary = component, secondary = -accumulation, tertiary = index
    order = np.lexsort((flat_indices, -accumulation_of, component_of))
    first_per_component = np.unique(component_of[order], return_index=True)[1]
    selected = flat_indices[order[first_per_component]]

    rows, columns = peak_mask.shape
    return [(int(index // columns), int(index % columns)) for index in np.sort(selected)]


def extract_candidate_seeds(hydrology: Any, dem: Any, config: CandidateConfig) -> List[Tuple[int, int]]:
    """Deterministically extract candidate seed cells from the hydrology raster."""

    accumulation = np.asarray(hydrology.flow_accumulation, dtype=float)
    valid = np.asarray(hydrology.valid_mask, dtype=bool) & np.isfinite(accumulation)

    finite_accumulation = np.where(valid, accumulation, -np.inf)
    neighbourhood_max = ndimage.maximum_filter(
        finite_accumulation,
        size=max(3, int(config.peak_neighbourhood)),
        mode="constant",
        cval=-np.inf,
    )
    peak_mask = valid & (finite_accumulation >= neighbourhood_max)

    seeds = set(_reduce_plateau_peaks(peak_mask, accumulation))

    # Always include the detected outlets so genuine drainage points are never missed.
    outlets = getattr(hydrology, "outlet_selection", {}) or {}
    for outlet in outlets.get("outlets", []) or []:
        row, column = int(outlet["row"]), int(outlet["col"])
        if valid[row, column]:
            seeds.add((row, column))

    return sorted(seeds)


# ---------------------------------------------------------------------------
# Stage 2 — feature enrichment
# ---------------------------------------------------------------------------

def _resolve_outlets(flow_direction: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Return the flat index of the terminal (outlet) cell each valid cell drains to.

    Uses pointer doubling on the D8 downstream index: O(n log n), vectorised,
    and deterministic. Invalid cells map to -1.
    """

    rows, columns = flow_direction.shape
    total = rows * columns
    downstream = downstream_flat_index(flow_direction, valid_mask)
    valid_flat = np.asarray(valid_mask, dtype=bool).ravel()

    self_index = np.arange(total, dtype=np.int64)
    pointer = np.where(valid_flat & (downstream >= 0), downstream, self_index)

    steps = int(math.ceil(math.log2(max(total, 2)))) + 1
    for _ in range(steps):
        jumped = pointer[pointer]
        if np.array_equal(jumped, pointer):
            break
        pointer = jumped

    return np.where(valid_flat, pointer, -1)


def build_candidates(
    seeds: Sequence[Tuple[int, int]],
    hydrology: Any,
    dem: Any,
    config: CandidateConfig,
) -> List[Dict[str, Any]]:
    """Attach every feature needed for filtering, ranking and explainability."""

    elevation = np.asarray(dem.elevation, dtype=float)
    if getattr(dem, "slope_degrees", None) is not None:
        slope = np.asarray(dem.slope_degrees, dtype=float)
    elif getattr(hydrology, "slope_degrees", None) is not None:
        # Reuse the slope already computed by the Phase-2 hydrology stage;
        # never recompute terrain here.
        slope = np.asarray(hydrology.slope_degrees, dtype=float)
    else:
        slope = np.full(elevation.shape, np.nan)
    accumulation = np.asarray(hydrology.flow_accumulation, dtype=float)
    valid = np.asarray(hydrology.valid_mask, dtype=bool)
    flow_direction = np.asarray(hydrology.flow_direction)
    rows, columns = elevation.shape
    cell_area = float(hydrology.cell_area_m2)

    outlet_of = _resolve_outlets(flow_direction, valid).reshape(elevation.shape)

    outlets = getattr(hydrology, "outlet_selection", {}) or {}
    outlet_kind = {}
    for outlet in outlets.get("outlets", []) or []:
        outlet_kind[int(outlet["row"]) * columns + int(outlet["col"])] = outlet.get("kind")

    transformer = _cached_transformer(str(dem.crs), "EPSG:4326")
    x_metric, y_metric = metric_xy(dem, dem.x, dem.y)
    metric_reference = metric_crs(dem)

    candidates: List[Dict[str, Any]] = []
    for row, column in seeds:
        x = float(dem.x[column])
        y = float(dem.y[row])
        try:
            longitude, latitude = transformer.transform(x, y)
            longitude, latitude = float(longitude), float(latitude)
        except Exception:
            longitude = latitude = float("nan")

        outlet_flat = int(outlet_of[row, column]) if valid[row, column] else -1
        if outlet_flat >= 0:
            outlet_row, outlet_column = divmod(outlet_flat, columns)
        else:
            outlet_row = outlet_column = -1

        candidates.append(
            {
                "candidate_id": f"cand-{row:05d}-{column:05d}",
                "row": int(row),
                "column": int(column),
                "x": float(x_metric[column]),
                "y": float(y_metric[row]),
                "crs": metric_reference,
                "latitude": latitude,
                "longitude": longitude,
                "elevation_m": _finite(elevation[row, column]),
                "slope_degrees": _finite(slope[row, column]),
                "flow_accumulation_cells": _finite(accumulation[row, column]),
                "drainage_area_m2": (float(accumulation[row, column]) * cell_area) if np.isfinite(accumulation[row, column]) else None,
                "outlet_id": f"outlet-{outlet_row}-{outlet_column}" if outlet_flat >= 0 else None,
                "catchment_id": f"catchment-{outlet_flat}" if outlet_flat >= 0 else None,
                "outlet_kind": outlet_kind.get(outlet_flat, "internal" if outlet_flat >= 0 else None),
                "valid": bool(valid[row, column]),
                "nodata": not bool(valid[row, column]),
                "generation_reason": "local_flow_accumulation_maximum",
                "rejection_reason": None,
            }
        )

    return candidates


# ---------------------------------------------------------------------------
# Stage 3 — hard constraints
# ---------------------------------------------------------------------------

def apply_hard_constraints(
    candidates: Sequence[Dict[str, Any]],
    hydrology: Any,
    dem: Any,
    config: CandidateConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Reject candidates that violate fundamental geometric/hydrological rules.

    Every rejection carries a machine-readable ``rejection_reason``. Soft factors
    (slope magnitude, accumulation magnitude, elevation, relief) never reject.
    """

    min_area = float(config.min_drainage_area_cells) * float(hydrology.cell_area_m2)
    x_metric, y_metric = metric_xy(dem, dem.x, dem.y)
    x_lo, x_hi = float(np.min(x_metric)), float(np.max(x_metric))
    y_lo, y_hi = float(np.min(y_metric)), float(np.max(y_metric))

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for candidate in candidates:
        reason = None

        if candidate["nodata"] or not candidate["valid"]:
            reason = "nodata_cell"
        elif candidate["elevation_m"] is None:
            reason = "invalid_elevation"
        elif candidate["slope_degrees"] is None:
            reason = "invalid_slope"
        elif candidate["flow_accumulation_cells"] is None or candidate["flow_accumulation_cells"] < 1.0:
            reason = "invalid_accumulation"
        elif candidate["outlet_id"] is None:
            reason = "unresolved_hydrology"
        elif not (math.isfinite(candidate["latitude"]) and math.isfinite(candidate["longitude"])):
            reason = "invalid_coordinate"
        elif not (x_lo <= candidate["x"] <= x_hi and y_lo <= candidate["y"] <= y_hi):
            reason = "outside_analysis_region"
        elif float(candidate["drainage_area_m2"] or 0.0) < min_area:
            reason = "insufficient_drainage_area"

        if reason is not None:
            rejected.append({**candidate, "rejection_reason": reason})
        else:
            accepted.append(candidate)

    return accepted, rejected


def validate_candidate_catchments(
    candidates: Sequence[Dict[str, Any]],
    hydrology: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Reject candidates whose drainage catchment is impossible/invalid.

    This is an **analytic** check (no per-outlet delineation, which would be
    redundant and expensive): a candidate's catchment is valid iff the cell is
    valid, its flow path resolves to a real terminal outlet, and its
    accumulation is finite, >= 1 and bounded by the number of valid cells.
    Full geometric delineation is still performed and validated for the final
    primary candidate when the response catchment block is built.
    """

    valid = np.asarray(hydrology.valid_mask, dtype=bool)
    accumulation = np.asarray(hydrology.flow_accumulation, dtype=float)
    rows, columns = valid.shape
    valid_cell_count = float(np.sum(valid))

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for candidate in candidates:
        row, column = int(candidate["row"]), int(candidate["column"])
        on_valid_cell = 0 <= row < rows and 0 <= column < columns and bool(valid[row, column])

        accumulating = candidate.get("flow_accumulation_cells")
        accumulating = float(accumulating) if accumulating is not None else None

        catchment_ok = (
            on_valid_cell
            and candidate.get("outlet_id") is not None
            and accumulating is not None
            and math.isfinite(accumulating)
            and 1.0 <= accumulating <= valid_cell_count + 1e-9
        )

        if catchment_ok:
            accepted.append(candidate)
        else:
            rejected.append({**candidate, "rejection_reason": "invalid_catchment"})

    return accepted, rejected


# ---------------------------------------------------------------------------
# Stage 4 — deterministic spatial deduplication
# ---------------------------------------------------------------------------

def _strength(candidate: Dict[str, Any]) -> Tuple[float, int, int]:
    """Hydrological strength order: accumulation desc, then row/col asc."""

    accumulation = candidate.get("flow_accumulation_cells")
    accumulation = float(accumulation) if accumulation is not None else 0.0
    return (-accumulation, int(candidate["row"]), int(candidate["column"]))


def deduplicate_candidates(
    candidates: Sequence[Dict[str, Any]],
    config: CandidateConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Greedily keep the strongest candidate within ``min_separation_m``.

    Uses a spatial hash grid (bins of ``min_separation_m``) so the neighbour
    search is O(N x k) rather than O(N^2).
    """

    separation = float(config.min_separation_m)
    ordered = sorted(candidates, key=_strength)

    if separation <= 0:
        return list(ordered), []

    bin_size = separation
    grid: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    separation_sq = separation * separation

    for candidate in ordered:
        bx = int(math.floor(candidate["x"] / bin_size))
        by = int(math.floor(candidate["y"] / bin_size))

        conflict = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in grid.get((bx + dx, by + dy), ()):
                    ddx = candidate["x"] - other["x"]
                    ddy = candidate["y"] - other["y"]
                    if ddx * ddx + ddy * ddy < separation_sq:
                        conflict = other
                        break
                if conflict:
                    break
            if conflict:
                break

        if conflict is not None:
            rejected.append(
                {
                    **candidate,
                    "rejection_reason": "candidate_too_close",
                    "duplicate_of": conflict["candidate_id"],
                }
            )
            continue

        kept.append(candidate)
        grid.setdefault((bx, by), []).append(candidate)

    return kept, rejected


# ---------------------------------------------------------------------------
# Stage 5 — deterministic ranking
# ---------------------------------------------------------------------------

def _explanation(candidate: Dict[str, Any], components: Dict[str, float]) -> List[str]:
    """Generate explanation phrases from computed features only."""

    area = float(candidate.get("drainage_area_m2") or 0.0)
    accumulation = float(candidate.get("flow_accumulation_cells") or 0.0)
    slope = float(candidate.get("slope_degrees") or 0.0)
    elevation = float(candidate.get("elevation_m") or 0.0)
    kind = candidate.get("outlet_kind") or "unknown"

    slope_score = components["slope"]
    if slope_score >= 0.6:
        slope_phrase = "gentle slope within the preferred embankment range"
    elif slope_score >= 0.3:
        slope_phrase = "moderate slope"
    else:
        slope_phrase = "steep slope reduces embankment suitability"

    return [
        f"Contributing drainage area {area:,.0f} m2 across {accumulation:,.0f} upstream cells "
        f"(area score {components['catchment_area']:.2f})",
        f"Flow accumulation relative peak score {components['flow_accumulation']:.2f}",
        f"Slope {slope:.1f} degrees - {slope_phrase} (slope score {slope_score:.2f})",
        f"Site elevation {elevation:.1f} m (elevation score {components['elevation']:.2f})",
        f"Drains to a {kind} outlet (hydrological confidence {components['hydrological_confidence']:.2f})",
    ]


def score_and_rank_candidates(
    candidates: Sequence[Dict[str, Any]],
    hydrology: Any,
    dem: Any,
    config: CandidateConfig,
) -> List[Dict[str, Any]]:
    """Deterministically score and rank candidates using Phase-3 features only."""

    if not candidates:
        return []

    weights = dict(config.weights)
    total_weight = sum(weights.values()) or 1.0

    accumulation = np.asarray(hydrology.flow_accumulation, dtype=float)
    valid = np.asarray(hydrology.valid_mask, dtype=bool)
    finite = accumulation[np.isfinite(accumulation) & valid]
    max_accumulation = float(np.max(finite)) if finite.size else 1.0
    max_accumulation = max(max_accumulation, 1.0)

    elevation = np.asarray(dem.elevation, dtype=float)
    finite_elevation = elevation[np.isfinite(elevation) & valid]
    if finite_elevation.size:
        min_elevation = float(np.min(finite_elevation))
        max_elevation = float(np.max(finite_elevation))
    else:
        min_elevation = max_elevation = 0.0
    elevation_range = max_elevation - min_elevation

    scored: List[Dict[str, Any]] = []
    for candidate in candidates:
        accumulation_cells = float(candidate.get("flow_accumulation_cells") or 0.0)
        area = float(candidate.get("drainage_area_m2") or 0.0)
        slope = float(candidate.get("slope_degrees") or 0.0)
        elevation_value = float(candidate.get("elevation_m") or 0.0)

        components = {
            "catchment_area": _clamp(catchment_area_suitability(area)),
            "flow_accumulation": _clamp(accumulation_cells / max_accumulation),
            "slope": _clamp(slope_suitability(slope)),
            "elevation": _clamp(1.0 - ((elevation_value - min_elevation) / elevation_range)) if elevation_range > 0 else 1.0,
            "hydrological_confidence": 1.0 if candidate.get("outlet_kind") == "boundary" else 0.6,
        }

        score = sum(weights.get(key, 0.0) * value for key, value in components.items()) / total_weight
        score = _clamp(score)

        scored.append(
            {
                **candidate,
                "candidate_score": round(float(score), 6),
                "score_components": {key: round(float(value), 6) for key, value in components.items()},
                "explanation": _explanation(candidate, components),
            }
        )

    scored.sort(
        key=lambda item: (
            -float(item["candidate_score"]),
            -float(item.get("flow_accumulation_cells") or 0.0),
            int(item["row"]),
            int(item["column"]),
        )
    )

    for rank, item in enumerate(scored, start=1):
        item["candidate_rank"] = rank

    return scored


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def _rejection_counts(rejected: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in rejected:
        reason = item.get("rejection_reason") or "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def generate_candidates(
    hydrology: Any,
    dem: Any,
    config: Optional[CandidateConfig] = None,
    recorder: Any = None,
) -> CandidateSet:
    """Full Phase-3 pipeline: seeds -> features -> constraints -> dedup -> rank.

    ``recorder`` may be the :mod:`instrumentation` module (or any object exposing
    ``stage(name)`` returning a context manager that yields a handle with
    ``set_output``); the default is a no-op recorder.
    """

    active_config = config or CandidateConfig.from_env()
    active_recorder = recorder if recorder is not None else _NullRecorder()

    with active_recorder.stage("candidates_generation") as handle:
        seeds = extract_candidate_seeds(hydrology, dem, active_config)
        built = build_candidates(seeds, hydrology, dem, active_config)
        handle.set_output(seed_count=len(seeds), candidate_count=len(built))

    with active_recorder.stage("candidates_filtering") as handle:
        constrained, constraint_rejected = apply_hard_constraints(built, hydrology, dem, active_config)
        catchment_ok, catchment_rejected = validate_candidate_catchments(constrained, hydrology)
        rejected = constraint_rejected + catchment_rejected
        handle.set_output(
            accepted_count=len(catchment_ok),
            rejected_count=len(rejected),
            rejection_reasons=_rejection_counts(rejected),
        )

    with active_recorder.stage("candidates_deduplication") as handle:
        deduplicated, duplicate_rejected = deduplicate_candidates(catchment_ok, active_config)
        rejected = rejected + duplicate_rejected
        handle.set_output(
            kept_count=len(deduplicated),
            duplicate_count=len(duplicate_rejected),
        )

    with active_recorder.stage("candidates_ranking") as handle:
        ranked = score_and_rank_candidates(deduplicated, hydrology, dem, active_config)
        if active_config.max_candidates and len(ranked) > active_config.max_candidates:
            overflow = ranked[active_config.max_candidates:]
            ranked = ranked[: active_config.max_candidates]
            rejected = rejected + [
                {**item, "rejection_reason": "candidate_limit_exceeded"} for item in overflow
            ]
        handle.set_output(
            candidate_count=len(ranked),
            rank1_score=round(float(ranked[0]["candidate_score"]), 6) if ranked else None,
        )

    diagnostics = {
        "seed_count": len(seeds),
        "accepted_count": len(catchment_ok),
        "deduplicated_count": len(deduplicated),
        "final_count": len(ranked),
        "rejected_count": len(rejected),
        "rejection_reasons": _rejection_counts(rejected),
        "min_separation_m": float(active_config.min_separation_m),
        "min_drainage_area_cells": float(active_config.min_drainage_area_cells),
        "max_candidates": int(active_config.max_candidates),
    }

    return CandidateSet(
        candidates=ranked,
        rejected=rejected,
        seed_count=len(seeds),
        accepted_count=len(catchment_ok),
        deduplicated_count=len(deduplicated),
        diagnostics=diagnostics,
    )


__all__ = [
    "CandidateConfig",
    "CandidateSet",
    "DEFAULT_CANDIDATE_WEIGHTS",
    "apply_hard_constraints",
    "build_candidates",
    "deduplicate_candidates",
    "extract_candidate_seeds",
    "generate_candidates",
    "metric_crs",
    "metric_xy",
    "score_and_rank_candidates",
    "validate_candidate_catchments",
]
