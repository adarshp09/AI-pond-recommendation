"""Phase 02 — adapter between the API and the tested recommendation engine.

The API's legacy pipeline (``backend/main.py`` + ``backend/suitability.py`` +
``backend/recommendation.py``) remains the production default. This module is an
**opt-in** bridge to the Phase 3–6 engine modules that already exist and are
tested:

    candidates.generate_candidates
        -> geographic_context.enrich_candidates   (real metric distances + feasibility)
        -> rainfall_runoff.compute_rainfall_runoff_for_candidates
        -> recommendation_engine.run_recommendation_engine
               (hard eligibility, six-component score, confidence, explanation,
                deterministic ranking)

It is activated only when ``POND_RECOMMENDATION_ENGINE`` is truthy. When the
flag is unset/false the API never calls anything here, so legacy behaviour is
byte-for-byte unchanged.

The adapter returns a dict of *response overrides* (``pond_candidate``,
``alternative_candidates``, ``suitability``, ``recommendation``) shaped exactly
like the existing response blocks, plus diagnostic counters. It never mutates
production configuration and does not enable caching yet (Phase 5).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

try:  # imported as ``backend.pipeline`` (normal) …
    from backend.candidates import CandidateConfig, generate_candidates
    from backend.geographic_context import GeographicContextConfig, enrich_candidates
    from backend.recommendation_engine import (
        RecommendationEngineConfig,
        run_recommendation_engine,
    )
except ImportError:  # pragma: no cover - imported with backend/ on sys.path
    from candidates import CandidateConfig, generate_candidates  # type: ignore
    from geographic_context import GeographicContextConfig, enrich_candidates  # type: ignore
    from recommendation_engine import (  # type: ignore
        RecommendationEngineConfig,
        run_recommendation_engine,
    )

try:  # Phase 08 — tracing/metrics for engine stages
    from backend.utils.telemetry import trace_stage
except ImportError:  # pragma: no cover
    from utils.telemetry import trace_stage  # type: ignore

try:  # Phase 09 — river/waterway exclusion
    from backend.utils.spatial_filter import exclude_river_candidates
except ImportError:  # pragma: no cover
    from utils.spatial_filter import exclude_river_candidates  # type: ignore

try:
    from config import settings
except Exception:  # pragma: no cover - defensive
    settings = None  # type: ignore[assignment]

logger = logging.getLogger("pond_recommendation")

ENGINE_ENV = "POND_RECOMMENDATION_ENGINE"

# Engine component keys (mirrors recommendation_engine.RecommendationEngineConfig.weights).
ENGINE_COMPONENT_KEYS = (
    "hydrology",
    "terrain",
    "geography",
    "rainfall",
    "runoff",
    "storage",
)

_TRUTHY = {"1", "true", "yes", "on"}

_ALTERNATIVE_ID = "alternative-{}"


# ---------------------------------------------------------------------------
# Flag
# ---------------------------------------------------------------------------

def recommendation_engine_enabled() -> bool:
    """Return whether the engine is enabled for this process.

    Reads ``POND_RECOMMENDATION_ENGINE`` from the environment first (so tests and
    operators can toggle it without reloading config), then falls back to the
    central :class:`config.Settings` value (default ``False``).
    """

    value = os.getenv(ENGINE_ENV)
    if value is not None:
        return value.strip().lower() in _TRUTHY
    try:
        from config import settings

        return bool(getattr(settings, ENGINE_ENV, False))
    except Exception:  # pragma: no cover - defensive
        return False


# ---------------------------------------------------------------------------
# Provider / rainfall resolution (dynamically resolved so tests can patch them)
# ---------------------------------------------------------------------------

def _resolve_land_provider():
    """Resolve ``fetch_land_context`` at call time.

    Resolving dynamically (instead of binding at import) means the offline test
    suite and operators can patch ``services.land_service.fetch_land_context``
    and the adapter will use the patched callable.
    """

    try:
        from services import land_service

        return land_service.fetch_land_context
    except Exception:  # pragma: no cover - defensive
        from backend.services import land_service as _ls

        return _ls.fetch_land_context


def _compute_runoff(candidates: Sequence[Dict[str, Any]], recorder: Any) -> List[Dict[str, Any]]:
    """Run rainfall/runoff per candidate with caching disabled (Phase 5 owns caching)."""

    try:
        from backend.rainfall_runoff import (
            RainfallRunoffConfig,
            compute_rainfall_runoff_for_candidates,
        )
    except ImportError:  # pragma: no cover
        from rainfall_runoff import (  # type: ignore
            RainfallRunoffConfig,
            compute_rainfall_runoff_for_candidates,
        )

    config = RainfallRunoffConfig()
    # Do not read/write the shared disk cache in Phase 2.
    config.cache_enabled = False
    return compute_rainfall_runoff_for_candidates(
        list(candidates), config=config, cache=None, recorder=recorder
    )


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def _runoff_fields(entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    entry = entry or {}
    return {
        "rainfall_mm": entry.get("rainfall_mm"),
        "estimated_runoff_m3": entry.get("estimated_runoff_m3"),
        "estimated_storage_m3": entry.get("estimated_storage_m3"),
    }


def _catchment_area_m2(candidate: Dict[str, Any], cell_area_m2: Optional[float]) -> float:
    area = candidate.get("drainage_area_m2")
    if area is None:
        accumulation = candidate.get("flow_accumulation_cells")
        if accumulation is not None and cell_area_m2:
            area = float(accumulation) * float(cell_area_m2)
    try:
        return float(area) if area is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _candidate_dict(ranked, index: int, elevation_by_id: Dict[str, Any]) -> Dict[str, Any]:
    """Map a ``RankedCandidate`` to the existing ``pond_candidate`` shape + engine fields."""

    factors = ranked.factors or {}
    catchment = dict(ranked.catchment_area or {})
    overall = float(ranked.suitability.get("overall_score", 0.0)) if ranked.suitability else 0.0

    return {
        "id": "primary" if index == 1 else _ALTERNATIVE_ID.format(index - 1),
        "label": "Pond Candidate" if index == 1 else "Alternative Candidate",
        "latitude": ranked.latitude,
        "longitude": ranked.longitude,
        "elevation_m": elevation_by_id.get(ranked.id),
        "slope_degrees": factors.get("slope_degrees"),
        "catchment_area_m2": catchment.get("area_m2", 0.0),
        "catchment_area": catchment,
        "suitability_score": overall,
        # --- engine-only fields (absent from the legacy candidate) ---
        "confidence": float(ranked.confidence),
        "feasibility": (ranked.eligibility.details or {}).get("feasibility", "feasible"),
        "feasibility_reason": (ranked.eligibility.details or {}).get("feasibility_reason"),
        "road_distance_m": factors.get("road_distance_m"),
        "building_distance_m": factors.get("building_distance_m"),
        "water_distance_m": factors.get("water_distance_m"),
        "explanation": ranked.explanation,
    }


# ---------------------------------------------------------------------------
# River/waterway exclusion (shared by the engine and legacy paths)
# ---------------------------------------------------------------------------

def apply_river_exclusion(
    candidates: Sequence[Dict[str, Any]],
    warnings: Optional[List[str]] = None,
    *,
    land_gpkg_path: Optional[str] = None,
    buffer_meters: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Drop candidates inside the river/water safety buffer (Phase 09).

    Shared by the engine pipeline (``engine_analysis``) and the legacy path in
    ``backend/main.py`` so both emit the same user-facing warning:

        "<N> candidate(s) excluded within <B> m of rivers/water bodies."

    Returns the candidates unchanged when ``RIVER_EXCLUSION_ENABLED`` is false.
    """

    candidates = list(candidates or [])
    if not candidates:
        return candidates

    if settings is None or not getattr(settings, "RIVER_EXCLUSION_ENABLED", True):
        return candidates

    path = (
        land_gpkg_path
        if land_gpkg_path is not None
        else getattr(settings, "LOCAL_OSM_GPKG_PATH", "data/osm/prepared/land_context.gpkg")
    )
    try:
        buffer = float(
            buffer_meters
            if buffer_meters is not None
            else getattr(settings, "WATERBODY_BUFFER_METERS", 30.0)
        )
    except (TypeError, ValueError):
        buffer = 30.0

    with trace_stage("river_exclusion"):
        filtered = exclude_river_candidates(
            candidates,
            land_gpkg_path=path,
            buffer_meters=buffer,
        )

    excluded = len(candidates) - len(filtered)
    if excluded > 0 and warnings is not None:
        warnings.append(
            f"{excluded} candidate(s) excluded within {buffer:g} m of rivers/water bodies."
        )
    return filtered


# ---------------------------------------------------------------------------
# Public adapter
# ---------------------------------------------------------------------------

def engine_analysis(
    dem: Any,
    hydrology: Any,
    *,
    rainfall: Optional[Dict[str, Any]] = None,
    land_context: Optional[Dict[str, Any]] = None,  # noqa: ARG001 - reserved
    land_provider: Any = None,
    recorder: Any = None,
) -> Optional[Dict[str, Any]]:
    """Run the engine and return response overrides, or ``None`` if it cannot.

    ``None`` means "nothing better than legacy" (no candidates, or every
    candidate was ineligible). Callers keep the legacy response in that case.
    The function is defensive: any engine failure is logged and returns ``None``
    so an opt-in engine can never break the API.
    """

    cell_area_m2 = getattr(hydrology, "cell_area_m2", None)
    warnings: List[str] = []

    try:
        with trace_stage("candidate_generation"):
            candidate_set = generate_candidates(
                hydrology, dem, config=CandidateConfig.from_env(), recorder=recorder
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("recommendation_engine_candidates_failed", extra={"error": str(exc)})
        return None

    if not candidate_set.candidates:
        logger.info("recommendation_engine_no_candidates")
        return None

    engine_candidates = apply_river_exclusion(candidate_set.candidates, warnings)
    if not engine_candidates:
        logger.info("recommendation_engine_all_candidates_river_excluded")
        return None

    geo_config = GeographicContextConfig.from_env()
    geo_config.cache_enabled = False
    provider = land_provider if land_provider is not None else _resolve_land_provider()

    try:
        with trace_stage("geographic_enrichment"):
            geo = enrich_candidates(
                engine_candidates,
                config=geo_config,
                provider=provider,
                cache=None,
                recorder=recorder,
            )
        enriched = [dict(candidate) for candidate in geo.get("candidates", [])]
    except Exception as exc:
        logger.warning("recommendation_engine_geographic_failed", extra={"error": str(exc)})
        warnings.append(
            f"Geographic context unavailable ({type(exc).__name__}); "
            "feasibility and distances are reported as unknown."
        )
        enriched = [dict(candidate) for candidate in engine_candidates]
        geo = {"diagnostics": {}}

    if not enriched:
        return None

    # Surface partially-degraded geographic data (mapped by geographic_context to
    # ``feasibility=unknown`` / distances ``None``) as an actionable warning.
    _geo_diagnostics = (geo or {}).get("diagnostics") or {}
    _unknown = int(_geo_diagnostics.get("unknown_count", 0) or 0)
    if _unknown > 0:
        warnings.append(
            f"Geographic context unavailable for {_unknown} candidate(s); "
            "their feasibility is marked unknown."
        )

    try:
        with trace_stage("rainfall_analysis"):
            runoff = _compute_runoff(enriched, recorder)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("recommendation_engine_rainfall_failed", extra={"error": str(exc)})
        warnings.append(
            f"Rainfall/runoff data unavailable ({type(exc).__name__}); "
            "rainfall, runoff and storage factors use neutral values."
        )
        runoff = [{} for _ in enriched]

    elevation_by_id: Dict[str, Any] = {}
    merged: List[Dict[str, Any]] = []

    for position, candidate in enumerate(enriched):
        runoff_entry = runoff[position] if position < len(runoff) else {}
        marked = dict(candidate)
        candidate_id = (
            marked.get("candidate_id") or marked.get("id") or f"candidate-{position}"
        )
        marked["id"] = str(candidate_id)
        elevation_by_id[marked["id"]] = marked.get("elevation_m")

        runoff_fields = _runoff_fields(runoff_entry)
        area_m2 = _catchment_area_m2(marked, cell_area_m2)
        marked["catchment_area_m2"] = area_m2
        marked["flow_accumulation_cells"] = marked.get("flow_accumulation_cells")
        marked.update(runoff_fields)

        # recommendation_engine.extract_features reads these from ``factors``;
        # check_hard_eligibility reads them from the top level. Populate both.
        marked["factors"] = {
            "slope_degrees": marked.get("slope_degrees"),
            "catchment_area_m2": area_m2,
            "rainfall_mm": runoff_fields["rainfall_mm"],
            "water_distance_m": marked.get("water_distance_m"),
            "road_distance_m": marked.get("road_distance_m"),
            "building_distance_m": marked.get("building_distance_m"),
        }
        merged.append(marked)

    try:
        with trace_stage("scoring"):
            ranked = run_recommendation_engine(
                merged, config=RecommendationEngineConfig.from_env()
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("recommendation_engine_ranking_failed", extra={"error": str(exc)})
        return None

    if not ranked:
        logger.info("recommendation_engine_no_eligible_candidates")
        return None

    pond_candidate = _candidate_dict(ranked[0], 1, elevation_by_id)
    alternatives = [
        _candidate_dict(candidate, index, elevation_by_id)
        for index, candidate in enumerate(ranked[1:], start=2)
    ]

    suitability = dict(ranked[0].suitability or {})
    suitability.setdefault("component_scores", {})
    suitability.setdefault("weights", {})
    suitability.setdefault("overall_score", 0.0)

    recommendation = {
        "best_location": {
            "latitude": pond_candidate["latitude"],
            "longitude": pond_candidate["longitude"],
            "source": "recommendation_engine",
        },
        "alternatives": alternatives,
        "suitability_score": float(suitability.get("overall_score", 0.0)),
        "explanation": pond_candidate["explanation"],
    }

    diagnostics = {
        "candidate_count": len(merged),
        "ranked_count": len(ranked),
        "geographic": (geo or {}).get("diagnostics", {}),
    }

    return {
        "pond_candidate": pond_candidate,
        "alternative_candidates": alternatives,
        "suitability": suitability,
        "recommendation": recommendation,
        "diagnostics": diagnostics,
        "warnings": warnings,
    }


__all__ = [
    "ENGINE_COMPONENT_KEYS",
    "ENGINE_ENV",
    "apply_river_exclusion",
    "engine_analysis",
    "recommendation_engine_enabled",
]
