from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_component_scores(payload: Mapping[str, Any]) -> Dict[str, float]:
    suitability = payload.get("suitability") or {}
    component_scores = suitability.get("component_scores") or {}
    result: Dict[str, float] = {}
    for key in [
        "slope",
        "catchment_area",
        "rainfall",
        "water_distance",
        "road_distance",
        "building_distance",
        "road_building_distance",
        "land_use",
    ]:
        result[key] = _safe_float(component_scores.get(key, 0.0), 0.0)
    return result


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _derive_positive_negative_factors(payload: Mapping[str, Any]) -> Dict[str, List[str]]:
    positive: List[str] = []
    negative: List[str] = []

    suitability = payload.get("suitability") or {}
    overall_score = _safe_float(suitability.get("overall_score", 0.0), 0.0)
    components = _coerce_component_scores(payload)

    if components.get("slope", 0.0) >= 0.6:
        positive.append("Slope remains within the preferred pond embankment range.")
    else:
        negative.append("Slope is outside the preferred range for a pond embankment.")

    if components.get("catchment_area", 0.0) >= 0.5:
        positive.append("Catchment area is large enough to collect and retain runoff.")
    else:
        negative.append("Catchment area is too small for a robust pond storage profile.")

    if components.get("rainfall", 0.0) >= 0.5:
        positive.append("Rainfall contribution is adequate for runoff inflow.")
    else:
        negative.append("Rainfall contribution is weak relative to expected pond refill requirements.")

    if components.get("land_use", 0.0) >= 0.6:
        positive.append("Land-use constraints are manageable and do not heavily penalize the site.")
    else:
        negative.append("Land-use or infrastructure constraints reduce suitability.")

    if overall_score >= 0.7:
        positive.append("Overall suitability score is strong enough to recommend the site.")
    elif overall_score >= 0.4:
        positive.append("Overall suitability is acceptable with moderate trade-offs.")
    else:
        negative.append("Overall suitability is marginal and should be treated cautiously.")

    if not positive:
        positive.append("The pipeline identified no critical hard-fail conditions in the available inputs.")
    if not negative:
        negative.append("No major constraints were identified in the derived site metrics.")

    return {"positive_factors": positive, "negative_factors": negative}


def _build_human_readable_recommendation(payload: Mapping[str, Any]) -> str:
    suitability = payload.get("suitability") or {}
    overall_score = _safe_float(suitability.get("overall_score", 0.0), 0.0)
    components = _coerce_component_scores(payload)

    if overall_score >= 0.7:
        headline = "Recommend this pond site as a high-confidence candidate."
    elif overall_score >= 0.4:
        headline = "This site is a moderate candidate and may be viable with safeguards."
    else:
        headline = "This site is marginal and should be reviewed carefully before approval."

    factor_summary = []
    if components.get("slope", 0.0) >= 0.6:
        factor_summary.append("slope is within an acceptable range")
    if components.get("catchment_area", 0.0) >= 0.5:
        factor_summary.append("catchment area supports runoff collection")
    if components.get("rainfall", 0.0) >= 0.5:
        factor_summary.append("rainfall is supportive of water capture")
    if components.get("land_use", 0.0) >= 0.6:
        factor_summary.append("land-use constraints are moderate and manageable")
    if not factor_summary:
        factor_summary.append("the available metrics do not show clear site advantages")

    return f"{headline} Key factors: {', '.join(factor_summary)}. Overall suitability score: {overall_score:.3f}."


def build_explainability(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Create explanation metadata from the final suitability/recommendation payload."""
    suitability = payload.get("suitability") or payload.get("recommended", {}).get("suitability") or {}
    overall_score = _safe_float(suitability.get("overall_score", 0.0), 0.0)
    component_scores = _coerce_component_scores(payload)
    factors = _derive_positive_negative_factors(payload)

    explanation = {
        "overall_score": float(overall_score),
        "recommendation": _build_human_readable_recommendation(payload),
        "positive_factors": factors["positive_factors"],
        "negative_factors": factors["negative_factors"],
        "contributing_factors": {
            "slope": float(component_scores.get("slope", 0.0)),
            "flow_accumulation": float(component_scores.get("catchment_area", 0.0)),
            "catchment_area": float(component_scores.get("catchment_area", 0.0)),
            "rainfall": float(component_scores.get("rainfall", 0.0)),
            "land_constraints": float(component_scores.get("land_use", 0.0)),
            "dem_hydrology_quality": float(min(1.0, max(0.0, overall_score + 0.15))),
            "suitability_score": float(overall_score),
        },
        "sources_used": {
            "dem_validation": bool(payload.get("dem_validation") or payload.get("terrain")),
            "hydrology": bool(payload.get("hydrology") or payload.get("terrain")),
            "external_services": bool(payload.get("enrichment") or payload.get("rainfall") or payload.get("land_context")),
        },
    }
    return explanation
