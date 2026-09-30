from __future__ import annotations

"""Compatibility shim for legacy `explainability.build_explainability`.

Delegates to `backend.recommendation_engine.generate_explanation`.
"""

from typing import Any, Mapping

from backend.recommendation_engine import generate_explanation


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_component_scores(payload: Mapping[str, Any]) -> dict[str, float]:
    suitability = payload.get("suitability") or {}
    component_scores = suitability.get("component_scores") or {}
    result: dict[str, float] = {}
    for key in [
        "slope",
        "catchment_area",
        "rainfall",
        "water_distance",
        "road_distance",
        "building_distance",
        "road_building_distance",
        "land_use",
        "hydrology",
        "terrain",
        "geography",
        "runoff",
        "storage",
    ]:
        result[key] = _safe_float(component_scores.get(key, 0.0), 0.0)
    return result


def _derive_positive_negative_factors(payload: Mapping[str, Any]) -> dict[str, list[str]]:
    positive: list[str] = []
    negative: list[str] = []

    suitability = payload.get("suitability") or {}
    overall_score = _safe_float(suitability.get("overall_score", 0.0), 0.0)
    components = _coerce_component_scores(payload)

    if components.get("slope", 0.0) >= 0.6:
        positive.append("Slope remains within the preferred pond embankment range.")
    elif components.get("slope", 0.0) >= 0.3:
        negative.append("Slope is moderately steep for embankment construction.")
    else:
        negative.append("Slope is too steep for safe pond construction.")

    if components.get("catchment_area", 0.0) >= 0.7:
        positive.append("Catchment area is sufficient for sustained pond storage.")
    elif components.get("catchment_area", 0.0) >= 0.4:
        negative.append("Catchment area is moderate; may limit storage volume.")
    else:
        negative.append("Catchment area is small; limited natural inflow.")

    if components.get("rainfall", 0.0) >= 0.6:
        positive.append("Adequate rainfall supports consistent runoff generation.")
    elif components.get("rainfall", 0.0) >= 0.3:
        negative.append("Moderate rainfall; supplemental water may be needed.")
    else:
        negative.append("Low rainfall; unreliable natural recharge.")

    if components.get("hydrology", 0.0) >= 0.6:
        positive.append("Strong hydrological connectivity with good upstream drainage.")
    elif components.get("hydrology", 0.0) >= 0.3:
        negative.append("Moderate hydrological connectivity; limited upstream area.")
    else:
        negative.append("Weak hydrological connectivity; minimal upstream contribution.")

    if components.get("terrain", 0.0) >= 0.6:
        positive.append("Favorable terrain for pond construction and stability.")
    elif components.get("terrain", 0.0) >= 0.3:
        negative.append("Moderate terrain constraints; requires careful design.")
    else:
        negative.append("Challenging terrain; significant earthworks likely.")

    if components.get("geography", 0.0) >= 0.6:
        positive.append("Geographically feasible; adequate clearance from infrastructure.")
    elif components.get("geography", 0.0) >= 0.3:
        negative.append("Moderate geographic constraints; some clearance concerns.")
    else:
        negative.append("Geographic constraints; proximity to infrastructure or water bodies.")

    if components.get("runoff", 0.0) >= 0.6:
        positive.append("High estimated annual runoff; strong water availability.")
    elif components.get("runoff", 0.0) >= 0.3:
        negative.append("Moderate runoff; storage capacity may be limited.")
    else:
        negative.append("Low estimated runoff; water availability may be insufficient.")

    if components.get("storage", 0.0) >= 0.6:
        positive.append("High indicative storage potential; good pond capacity.")
    elif components.get("storage", 0.0) >= 0.3:
        negative.append("Moderate storage potential; may require larger footprint.")
    else:
        negative.append("Low indicative storage potential; small effective volume.")

    return {"positive_factors": positive, "negative_factors": negative}


def build_explainability(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Generate legacy-format explainability from the new recommendation engine output.

    Falls back to legacy logic if new engine data is not present.
    """
    suitability = payload.get("suitability") or {}
    component_scores = suitability.get("component_scores") or {}
    overall_score = _safe_float(suitability.get("overall_score", 0.0), 0.0)

    # Try to use new engine's explanation_details if available
    pond_candidate = payload.get("pond_candidate") or {}
    explanation_details = pond_candidate.get("explanation_details")
    if explanation_details:
        return {
            "overall_score": _safe_float(explanation_details.get("overall_score", overall_score)),
            "recommendation": explanation_details.get("recommendation", "marginal"),
            "positive_factors": explanation_details.get("strengths", []),
            "negative_factors": explanation_details.get("concerns", []),
        }

    # Fallback to legacy logic
    components = {k: _safe_float(v) for k, v in (payload.get("suitability") or {}).get("component_scores", {}).items()}
    factors = _derive_positive_negative_factors(payload)

    recommendation = "highly_suitable" if overall_score >= 0.7 else "moderately_suitable" if overall_score >= 0.4 else "marginal"

    return {
        "overall_score": overall_score,
        "recommendation": recommendation,
        "positive_factors": factors["positive_factors"],
        "negative_factors": factors["negative_factors"],
    }

# Backward-compatible exports
__all__ = ["build_explainability"]
