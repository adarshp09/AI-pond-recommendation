from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping, Optional

from dotenv import load_dotenv

from suitability import DEFAULT_WEIGHTS, evaluate_pond_suitability

load_dotenv()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


DEFAULT_REJECTION_CONSTRAINTS = {
    "max_slope_degrees": _env_float("POND_MAX_SLOPE_DEGREES", 35.0),
    "min_catchment_area_m2": _env_float("POND_MIN_CATCHMENT_AREA_M2", 0.0),
    "min_rainfall_mm": _env_float("POND_MIN_RAINFALL_MM", 0.0),
    "min_water_distance_m": _env_float("POND_MIN_WATER_DISTANCE_M", 25.0),
    "min_road_distance_m": _env_float("POND_MIN_ROAD_DISTANCE_M", 25.0),
    "min_building_distance_m": _env_float("POND_MIN_BUILDING_DISTANCE_M", 25.0),
}


def _candidate_factors(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    factors = candidate.get("factors", {}) or {}
    return {
        "slope_degrees": float(factors.get("slope_degrees", 0.0) or 0.0),
        "catchment_area_m2": float(factors.get("catchment_area_m2", 0.0) or 0.0),
        "rainfall_mm": float(factors.get("rainfall_mm", 0.0) or 0.0),
        "water_distance_m": float(factors.get("water_distance_m", 0.0) or 0.0),
        "road_distance_m": float(factors.get("road_distance_m", 0.0) or 0.0),
        "building_distance_m": float(factors.get("building_distance_m", 0.0) or 0.0),
    }


def _candidate_catchment_area(candidate: Mapping[str, Any]) -> Dict[str, float]:
    area = candidate.get("catchment_area") or {}
    if isinstance(area, Mapping):
        return {
            "area_m2": float(area.get("area_m2", 0.0) or 0.0),
            "area_hectares": float(area.get("area_hectares", 0.0) or 0.0),
            "area_km2": float(area.get("area_km2", 0.0) or 0.0),
        }
    return {"area_m2": 0.0, "area_hectares": 0.0, "area_km2": 0.0}


def _build_explanation(candidate: Mapping[str, Any], score: Mapping[str, Any]) -> str:
    detailed = _build_explanation_details(candidate, score)
    return detailed["summary"]


def _build_explanation_details(candidate: Mapping[str, Any], score: Mapping[str, Any]) -> Dict[str, Any]:
    factors = _candidate_factors(candidate)
    component_scores = score.get("component_scores", {})
    summary_parts = []
    positive_factors: List[str] = []
    negative_factors: List[str] = []

    slope_score = float(component_scores.get("slope", 0.0) or 0.0)
    catchment_score = float(component_scores.get("catchment_area", 0.0) or 0.0)
    rainfall_score = float(component_scores.get("rainfall", 0.0) or 0.0)
    land_score = float(component_scores.get("land_use", 0.0) or 0.0)
    flow_score = max(float(component_scores.get("water_distance", 0.0) or 0.0), rainfall_score)

    if factors["slope_degrees"] <= DEFAULT_REJECTION_CONSTRAINTS["max_slope_degrees"]:
        positive_factors.append("slope is within the recommended operating range")
    else:
        negative_factors.append("slope exceeds the preferred limit")
    summary_parts.append(f"slope score={slope_score:.2f}")

    if factors["catchment_area_m2"] >= DEFAULT_REJECTION_CONSTRAINTS["min_catchment_area_m2"]:
        positive_factors.append("catchment area is sufficient for pond storage")
    else:
        negative_factors.append("catchment area is below the preferred minimum")
    summary_parts.append(f"catchment score={catchment_score:.2f}")

    if factors["rainfall_mm"] >= DEFAULT_REJECTION_CONSTRAINTS["min_rainfall_mm"]:
        positive_factors.append("rainfall supports runoff contribution")
    else:
        negative_factors.append("rainfall is too low for reliable runoff generation")
    summary_parts.append(f"rainfall score={rainfall_score:.2f}")

    if factors["water_distance_m"] >= DEFAULT_REJECTION_CONSTRAINTS["min_water_distance_m"]:
        positive_factors.append("distance to water features is acceptable")
    else:
        negative_factors.append("site is too close to an existing water body")

    if factors["road_distance_m"] >= DEFAULT_REJECTION_CONSTRAINTS["min_road_distance_m"] and factors["building_distance_m"] >= DEFAULT_REJECTION_CONSTRAINTS["min_building_distance_m"]:
        positive_factors.append("distance from roads and buildings is acceptable")
    else:
        negative_factors.append("site is close to infrastructure")

    if not positive_factors:
        positive_factors.append("basic terrain and contour conditions were detected")

    if not negative_factors:
        negative_factors.append("no major engineering constraints were observed")

    overall_score = float(score.get("overall_score", 0.0))
    confidence = min(1.0, max(0.0, 0.55 + overall_score * 0.45))
    recommendation = "highly_suitable" if overall_score >= 0.7 else "moderately_suitable" if overall_score >= 0.4 else "marginal"

    summary = "; ".join(summary_parts) + f"; overall suitability={overall_score:.3f}; confidence={confidence:.3f}"
    return {
        "recommendation": recommendation,
        "summary": summary,
        "positive_factors": positive_factors,
        "negative_factors": negative_factors,
        "score_breakdown": {
            "slope": float(slope_score),
            "catchment": float(catchment_score),
            "flow_accumulation": float(flow_score),
            "rainfall": float(rainfall_score),
            "land": float(land_score),
        },
        "confidence": float(confidence),
    }


def reject_unsuitable_candidates(
    candidates: Iterable[Mapping[str, Any]],
    constraints: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Filter out candidates that fail the minimum operating constraints."""
    active_constraints = dict(DEFAULT_REJECTION_CONSTRAINTS)
    if constraints:
        active_constraints.update(constraints)

    accepted: List[Dict[str, Any]] = []

    for candidate in candidates:
        factors = _candidate_factors(candidate)
        if factors["slope_degrees"] > active_constraints["max_slope_degrees"]:
            continue
        if factors["catchment_area_m2"] < active_constraints["min_catchment_area_m2"]:
            continue
        if factors["rainfall_mm"] < active_constraints["min_rainfall_mm"]:
            continue
        if factors["water_distance_m"] < active_constraints["min_water_distance_m"]:
            continue
        if factors["road_distance_m"] < active_constraints["min_road_distance_m"]:
            continue
        if factors["building_distance_m"] < active_constraints["min_building_distance_m"]:
            continue

        accepted.append(dict(candidate))

    return accepted


def rank_candidates(
    candidates: Iterable[Mapping[str, Any]],
    constraints: Optional[Mapping[str, float]] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Return candidates sorted from best to worst with a suitability score and explanation."""
    filtered = reject_unsuitable_candidates(candidates, constraints)

    ranked: List[Dict[str, Any]] = []
    for candidate in filtered:
        factors = _candidate_factors(candidate)
        land_context = candidate.get("land_context") or {}
        score = evaluate_pond_suitability(
            slope_degrees=factors["slope_degrees"],
            catchment_area_m2=factors["catchment_area_m2"],
            total_precipitation_mm=factors["rainfall_mm"],
            water_distance_m=factors["water_distance_m"],
            road_distance_m=factors["road_distance_m"],
            building_distance_m=factors["building_distance_m"],
            land_context=land_context,
            weights=weights or DEFAULT_WEIGHTS,
        )

        explanation_details = _build_explanation_details(candidate, score)
        entry = {
            "id": candidate.get("id", "candidate"),
            "latitude": float(candidate.get("latitude", 0.0) or 0.0),
            "longitude": float(candidate.get("longitude", 0.0) or 0.0),
            "factors": factors,
            "catchment_area": _candidate_catchment_area(candidate),
            "land_context": land_context,
            "suitability": score,
            "explanation": _build_explanation(candidate, score),
            "explanation_details": explanation_details,
            "confidence": float(explanation_details["confidence"]),
        }
        ranked.append(entry)

    ranked.sort(key=lambda item: float(item["suitability"]["overall_score"]), reverse=True)
    return ranked
