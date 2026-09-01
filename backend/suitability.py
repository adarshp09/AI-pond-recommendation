"""Multi-factor pond suitability scoring.

The scoring model combines normalized component metrics into an overall
suitability score in the range [0, 1]. The weights are configurable and are
kept separate from the API clients and catchment logic so the same scoring
code can be reused for different catchments or map sources.

Overall score:
    score = sum(weight_i * component_score_i)

with each component score normalized to [0, 1].
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional


DEFAULT_WEIGHTS = {
    "slope": 0.25,
    "catchment_area": 0.20,
    "rainfall": 0.20,
    "water_distance": 0.15,
    "road_distance": 0.10,
    "land_use": 0.10,
}


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def _normalize(value: float, minimum: float, maximum: float, *, ideal_min: Optional[float] = None, ideal_max: Optional[float] = None, invert: bool = False) -> float:
    if maximum <= minimum:
        return 1.0 if (value >= maximum) else 0.0

    if ideal_min is not None and ideal_max is not None:
        if value < ideal_min:
            ratio = (value - minimum) / (ideal_min - minimum)
            score = _clamp(ratio)
            return 1.0 - score if invert else score
        if value > ideal_max:
            ratio = (value - ideal_max) / (maximum - ideal_max)
            score = _clamp(ratio)
            return 1.0 - score if not invert else score
        return 1.0

    ratio = (value - minimum) / (maximum - minimum)
    ratio = _clamp(ratio)
    return 1.0 - ratio if invert else ratio


def slope_suitability(slope_degrees: float, *, ideal_min_deg: float = 2.0, ideal_max_deg: float = 12.0, max_deg: float = 35.0) -> float:
    """Higher is better for moderate slopes with a safe pond embankment."""
    slope_pct = max(0.0, math.tan(math.radians(slope_degrees)) * 100.0)
    return _normalize(slope_pct, 0.0, max_deg, ideal_min=ideal_min_deg, ideal_max=ideal_max_deg)


def catchment_area_suitability(area_m2: float, *, min_area_m2: float = 10_000.0, max_area_m2: float = 3_000_000.0) -> float:
    """Reward catchments large enough to sustain water storage without being excessive."""
    if area_m2 <= 0:
        return 0.0
    return _normalize(area_m2, min_area_m2, max_area_m2, ideal_min=min_area_m2 * 1.5, ideal_max=max_area_m2 * 0.5)


def rainfall_suitability(total_precipitation_mm: float, *, min_mm: float = 200.0, max_mm: float = 2500.0, ideal_min_mm: float = 600.0, ideal_max_mm: float = 1800.0) -> float:
    """Higher rainfall is helpful up to a moderate limit, then it plateaus."""
    if total_precipitation_mm <= 0:
        return 0.0
    return _normalize(total_precipitation_mm, min_mm, max_mm, ideal_min=ideal_min_mm, ideal_max=ideal_max_mm)


def water_distance_suitability(distance_m: Optional[float], *, max_distance_m: float = 300.0) -> float:
    """Prefer sites with a nearby water source, but not directly on the feature."""
    if distance_m is None:
        return 0.5
    if distance_m <= 0:
        return 1.0
    return _clamp(1.0 - (distance_m / max_distance_m))


def road_or_building_distance_suitability(distance_m: Optional[float], *, min_distance_m: float = 25.0, max_distance_m: float = 250.0) -> float:
    """Prefer locations away from roads and building footprints."""
    if distance_m is None:
        return 0.5
    if distance_m <= 0:
        return 0.0
    if distance_m < min_distance_m:
        return 0.0
    if distance_m >= max_distance_m:
        return 1.0
    return _clamp((distance_m - min_distance_m) / (max_distance_m - min_distance_m))


def land_use_suitability(land_context: Optional[Mapping[str, Any]], *, water_penalty: float = 0.25, road_penalty: float = 0.15, building_penalty: float = 0.20) -> float:
    """Penalize sites with major infrastructure or permanent water constraints."""
    if not land_context:
        return 0.75

    score = 1.0
    water_features = land_context.get("water_bodies") or []
    roads = land_context.get("roads") or []
    buildings = land_context.get("buildings") or []

    if water_features:
        score -= min(water_penalty, 0.9)
    if roads:
        score -= min(road_penalty * min(len(roads), 3) / 3.0, 0.6)
    if buildings:
        score -= min(building_penalty * min(len(buildings), 3) / 3.0, 0.6)

    return _clamp(score)


def evaluate_pond_suitability(
    *,
    slope_degrees: float,
    catchment_area_m2: float,
    total_precipitation_mm: Optional[float] = None,
    water_distance_m: Optional[float] = None,
    road_distance_m: Optional[float] = None,
    building_distance_m: Optional[float] = None,
    land_context: Optional[Mapping[str, Any]] = None,
    weights: Optional[Mapping[str, float]] = None,
    **_: Any,
) -> Dict[str, Any]:
    """Return component and overall suitability scores in the range [0, 1]."""
    active_weights = dict(DEFAULT_WEIGHTS)
    if weights:
        active_weights.update(weights)

    total_weight = sum(active_weights.values())
    if total_weight <= 0:
        raise ValueError("Component weights must sum to a positive value.")

    rainfall_mm = float(total_precipitation_mm) if total_precipitation_mm is not None else 0.0

    component_scores = {
        "slope": slope_suitability(slope_degrees),
        "catchment_area": catchment_area_suitability(catchment_area_m2),
        "rainfall": rainfall_suitability(rainfall_mm),
        "water_distance": water_distance_suitability(water_distance_m),
        "road_distance": road_or_building_distance_suitability(road_distance_m),
        "building_distance": road_or_building_distance_suitability(building_distance_m),
        "land_use": land_use_suitability(land_context),
    }

    road_score = component_scores["road_distance"]
    building_score = component_scores["building_distance"]
    component_scores["road_building_distance"] = min(road_score, building_score)

    overall_score = 0.0
    for key, weight in active_weights.items():
        if key == "road_distance" and "road_building_distance" in component_scores:
            score = component_scores["road_building_distance"]
        elif key in component_scores:
            score = component_scores[key]
        else:
            score = 0.0
        overall_score += (weight / total_weight) * score

    overall_score = _clamp(overall_score)
    return {
        "component_scores": component_scores,
        "weights": {key: float(value) for key, value in active_weights.items()},
        "overall_score": float(overall_score),
    }
