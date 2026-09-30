from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from backend.geographic_context import GeographicContextConfig, decide_feasibility
from backend.suitability import DEFAULT_WEIGHTS, evaluate_pond_suitability


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


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name, str(int(default))).lower()
    return val in {"1", "true", "yes", "on"}


def _env_dict_float(name: str, default: Dict[str, float]) -> Dict[str, float]:
    val = os.getenv(name)
    if not val:
        return default
    try:
        result = {}
        for pair in val.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                result[k.strip()] = float(v.strip())
        return result if result else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RecommendationEngineConfig:
    """Centralized, validated configuration for the recommendation engine."""

    # Hard eligibility constraints (Phase 2-4)
    max_slope_degrees: float = field(
        default_factory=lambda: _env_float("POND_MAX_SLOPE_DEGREES", 35.0)
    )
    min_catchment_area_m2: float = field(
        default_factory=lambda: _env_float("POND_MIN_CATCHMENT_AREA_M2", 0.0)
    )
    min_rainfall_mm: float = field(
        default_factory=lambda: _env_float("POND_MIN_RAINFALL_MM", 0.0)
    )
    min_water_distance_m: float = field(
        default_factory=lambda: _env_float("POND_MIN_WATER_DISTANCE_M", 25.0)
    )
    min_road_distance_m: float = field(
        default_factory=lambda: _env_float("POND_MIN_ROAD_DISTANCE_M", 25.0)
    )
    min_building_distance_m: float = field(
        default_factory=lambda: _env_float("POND_MIN_BUILDING_DISTANCE_M", 25.0)
    )
    reject_inside_water: bool = field(
        default_factory=lambda: _env_bool("POND_REJECT_INSIDE_WATER", True)
    )

    # Feature normalization bounds
    slope_min_deg: float = 0.0
    slope_max_deg: float = 45.0
    slope_ideal_min_deg: float = 2.0
    slope_ideal_max_deg: float = 12.0

    catchment_min_m2: float = 1_000.0
    catchment_max_m2: float = 10_000_000.0
    catchment_ideal_min_m2: float = 50_000.0
    catchment_ideal_max_m2: float = 2_000_000.0

    rainfall_min_mm: float = 0.0
    rainfall_max_mm: float = 3_000.0
    rainfall_ideal_min_mm: float = 500.0
    rainfall_ideal_max_mm: float = 2_000.0

    runoff_min_m3: float = 0.0
    runoff_max_m3: float = 500_000.0
    runoff_ideal_min_m3: float = 5_000.0
    runoff_ideal_max_m3: float = 100_000.0

    storage_min_m3: float = 0.0
    storage_max_m3: float = 250_000.0
    storage_ideal_min_m3: float = 2_500.0
    storage_ideal_max_m3: float = 50_000.0

    water_dist_max_m: float = 500.0
    road_dist_min_m: float = 25.0
    road_dist_max_m: float = 500.0
    building_dist_min_m: float = 25.0
    building_dist_max_m: float = 500.0

    # Component weights (must sum to > 0, normalized internally)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "hydrology": 0.25,
        "terrain": 0.20,
        "geography": 0.15,
        "rainfall": 0.15,
        "runoff": 0.15,
        "storage": 0.10,
    })

    # Confidence weights
    confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        "dem_quality": 0.30,
        "hydrology_quality": 0.25,
        "geographic_availability": 0.20,
        "rainfall_availability": 0.15,
        "runoff_storage_availability": 0.10,
    })

    # Geographic context config
    geographic_config: Optional[GeographicContextConfig] = None

    def __post_init__(self):
        # Validate weights
        for key, value in self.weights.items():
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"Weight '{key}' must be a non-negative number")
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Weights must sum to a positive value")

        # Validate confidence weights
        for key, value in self.confidence_weights.items():
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"Confidence weight '{key}' must be a non-negative number")
        conf_total = sum(self.confidence_weights.values())
        if conf_total <= 0:
            raise ValueError("Confidence weights must sum to a positive value")

        # Validate bounds
        if self.slope_min_deg >= self.slope_max_deg:
            raise ValueError("slope_min_deg must be < slope_max_deg")
        if self.catchment_min_m2 >= self.catchment_max_m2:
            raise ValueError("catchment_min_m2 must be < catchment_max_m2")
        if self.rainfall_min_mm >= self.rainfall_max_mm:
            raise ValueError("rainfall_min_mm must be < rainfall_max_mm")
        if self.runoff_min_m3 >= self.runoff_max_m3:
            raise ValueError("runoff_min_m3 must be < runoff_max_m3")
        if self.storage_min_m3 >= self.storage_max_m3:
            raise ValueError("storage_min_m3 must be < storage_max_m3")

    @classmethod
    def from_env(cls) -> "RecommendationEngineConfig":
        return cls(
            max_slope_degrees=_env_float("POND_MAX_SLOPE_DEGREES", 35.0),
            min_catchment_area_m2=_env_float("POND_MIN_CATCHMENT_AREA_M2", 0.0),
            min_rainfall_mm=_env_float("POND_MIN_RAINFALL_MM", 0.0),
            min_water_distance_m=_env_float("POND_MIN_WATER_DISTANCE_M", 25.0),
            min_road_distance_m=_env_float("POND_MIN_ROAD_DISTANCE_M", 25.0),
            min_building_distance_m=_env_float("POND_MIN_BUILDING_DISTANCE_M", 25.0),
            reject_inside_water=_env_bool("POND_REJECT_INSIDE_WATER", True),
            weights=_env_dict_float("POND_REC_WEIGHTS", {
                "hydrology": 0.25,
                "terrain": 0.20,
                "geography": 0.15,
                "rainfall": 0.15,
                "runoff": 0.15,
                "storage": 0.10,
            }),
            confidence_weights=_env_dict_float("POND_REC_CONFIDENCE_WEIGHTS", {
                "dem_quality": 0.30,
                "hydrology_quality": 0.25,
                "geographic_availability": 0.20,
                "rainfall_availability": 0.15,
                "runoff_storage_availability": 0.10,
            }),
            geographic_config=GeographicContextConfig.from_env(),
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert to float, handling None, NaN, inf."""
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _safe_optional_float(value: Any) -> Optional[float]:
    """Safely convert to Optional[float], returning None for missing/invalid."""
    if value is None:
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _normalize_with_bounds(
    value: Optional[float],
    min_val: float,
    max_val: float,
    ideal_min: Optional[float] = None,
    ideal_max: Optional[float] = None,
    invert: bool = False,
    missing_score: float = 0.5,
) -> float:
    """Robust normalization handling None, NaN, inf, constant values, outliers."""
    if value is None:
        return missing_score
    if math.isnan(value) or math.isinf(value):
        return missing_score

    if max_val <= min_val:
        return 1.0 if value >= max_val else 0.0

    # Clamp to bounds
    clamped = max(min_val, min(max_val, value))

    if ideal_min is not None and ideal_max is not None:
        if clamped < ideal_min:
            if ideal_min <= min_val:
                return 0.0 if not invert else 1.0
            ratio = (clamped - min_val) / (ideal_min - min_val)
            ratio = max(0.0, min(1.0, ratio))
            return 1.0 - ratio if invert else ratio
        if clamped > ideal_max:
            if ideal_max >= max_val:
                return 1.0 if not invert else 0.0
            ratio = (clamped - ideal_max) / (max_val - ideal_max)
            ratio = max(0.0, min(1.0, ratio))
            return ratio if not invert else (1.0 - ratio)
        return 1.0

    ratio = (clamped - min_val) / (max_val - min_val)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 - ratio if invert else ratio


def _normalize_distance(
    value: Optional[float],
    min_dist: float,
    max_dist: float,
    missing_score: float = 0.5,
) -> float:
    """Normalize distance where closer is better (inverted), with minimum threshold."""
    if value is None:
        return missing_score
    if math.isnan(value) or math.isinf(value):
        return missing_score

    if value <= 0:
        return 0.0
    if value < min_dist:
        return 0.0
    if value >= max_dist:
        return 1.0
    return max(0.0, min(1.0, (value - min_dist) / (max_dist - min_dist)))


def _compute_correlation_penalty(
    scores: Dict[str, float],
    max_correlation: float = 0.8,
) -> Dict[str, float]:
    """Apply penalty for highly correlated features to avoid double-counting.

    Known correlations:
    - catchment_area <-> flow_accumulation <-> runoff <-> storage

    This reduces the weight of redundant signals.
    """
    # Simple approach: if multiple correlated features are high, reduce slightly
    correlated = ["catchment_area", "runoff", "storage"]
    high_correlated = [k for k in correlated if k in scores and scores[k] > 0.7]

    if len(high_correlated) >= 2:
        penalty = 0.9  # 10% penalty
        result = dict(scores)
        for k in high_correlated:
            result[k] = max(0.0, scores[k] * penalty)
        return result
    return scores


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

@dataclass
class EligibilityResult:
    eligible: bool
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


def check_hard_eligibility(
    candidate: Mapping[str, Any],
    config: RecommendationEngineConfig,
    dem_validation: Optional[Mapping[str, Any]] = None,
    hydrology_validation: Optional[Mapping[str, Any]] = None,
) -> EligibilityResult:
    """Check all hard eligibility constraints.

    Returns EligibilityResult with eligible=False and reason if rejected.
    Missing external data (None) does NOT cause rejection.
    """
    details = {}

    # 1. Coordinates
    lat = _safe_optional_float(candidate.get("latitude"))
    lon = _safe_optional_float(candidate.get("longitude"))
    if lat is None or lon is None:
        return EligibilityResult(False, "missing_coordinates", {"missing": ["latitude", "longitude"]})
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return EligibilityResult(False, "invalid_coordinates", {"latitude": lat, "longitude": lon})
    details["coordinates"] = {"latitude": lat, "longitude": lon}

    # 2. Slope
    slope = _safe_float(candidate.get("slope_degrees"))
    details["slope_degrees"] = slope
    if slope > config.max_slope_degrees:
        return EligibilityResult(False, "slope_exceeds_limit", {"slope": slope, "limit": config.max_slope_degrees})

    # 3. Catchment area
    catchment_area = _safe_float(candidate.get("catchment_area_m2"))
    details["catchment_area_m2"] = catchment_area
    if catchment_area < config.min_catchment_area_m2:
        return EligibilityResult(False, "catchment_too_small", {"area_m2": catchment_area, "minimum": config.min_catchment_area_m2})

    # 4. Rainfall (only reject if available and below minimum)
    rainfall = _safe_optional_float(candidate.get("rainfall_mm"))
    details["rainfall_mm"] = rainfall
    if rainfall is not None and rainfall < config.min_rainfall_mm:
        return EligibilityResult(False, "insufficient_rainfall", {"rainfall_mm": rainfall, "minimum": config.min_rainfall_mm})

    # 5. Geographic feasibility (Phase 4)
    feasibility = candidate.get("feasibility", "unknown")
    feasibility_reason = candidate.get("feasibility_reason")
    details["feasibility"] = feasibility
    details["feasibility_reason"] = feasibility_reason

    if feasibility == "rejected":
        return EligibilityResult(False, feasibility_reason or "geographic_rejection", details)

    # 6. DEM validation (if available)
    if dem_validation:
        dem_status = dem_validation.get("status", "unknown")
        if dem_status == "invalid":
            return EligibilityResult(False, "dem_invalid", {"dem_status": dem_status, "details": dem_validation})
        details["dem_status"] = dem_status

    # 7. Hydrology validation (if available)
    if hydrology_validation:
        hydro_status = hydrology_validation.get("status", "unknown")
        if hydro_status == "invalid":
            return EligibilityResult(False, "hydrology_invalid", {"hydrology_status": hydro_status, "details": hydrology_validation})
        details["hydrology_status"] = hydro_status

    return EligibilityResult(True, None, details)


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

@dataclass
class CandidateFeatures:
    """Extracted and validated features for scoring."""

    # Core features
    slope_degrees: float
    catchment_area_m2: float
    rainfall_mm: Optional[float]
    runoff_m3: Optional[float]
    storage_m3: Optional[float]

    # Geographic distances
    water_distance_m: Optional[float]
    road_distance_m: Optional[float]
    building_distance_m: Optional[float]
    feasibility: str
    land_context: Dict[str, Any]

    # Quality indicators
    dem_validation_score: Optional[float] = None
    hydrology_validation_score: Optional[float] = None
    flow_accumulation_cells: Optional[int] = None
    provider_ok: bool = True

    # Raw candidate data
    candidate_id: str = ""
    latitude: float = 0.0
    longitude: float = 0.0


def extract_features(
    candidate: Mapping[str, Any],
    dem_validation: Optional[Mapping[str, Any]] = None,
    hydrology_validation: Optional[Mapping[str, Any]] = None,
) -> CandidateFeatures:
    """Extract and validate all features from a candidate."""
    factors = candidate.get("factors", {}) or {}

    # Core features
    slope = _safe_float(factors.get("slope_degrees"))
    catchment = _safe_float(factors.get("catchment_area_m2"))
    rainfall = _safe_optional_float(factors.get("rainfall_mm"))

    # Runoff/Storage from Phase 5
    runoff = _safe_optional_float(candidate.get("estimated_runoff_m3"))
    storage = _safe_optional_float(candidate.get("estimated_storage_m3"))

    # Geographic distances
    water_dist = _safe_optional_float(factors.get("water_distance_m"))
    road_dist = _safe_optional_float(factors.get("road_distance_m"))
    building_dist = _safe_optional_float(factors.get("building_distance_m"))

    # Feasibility
    feasibility = candidate.get("feasibility", "unknown")
    land_context = candidate.get("land_context") or {}

    # Quality
    dem_score = _safe_optional_float(dem_validation.get("score")) if dem_validation else None
    hydro_score = _safe_optional_float(hydrology_validation.get("score")) if hydrology_validation else None
    flow_acc = None
    if candidate.get("flow_accumulation_cells") is not None:
        flow_acc = _safe_optional_float(candidate.get("flow_accumulation_cells"))
        if flow_acc is not None:
            flow_acc = int(flow_acc)

    return CandidateFeatures(
        slope_degrees=slope,
        catchment_area_m2=catchment,
        rainfall_mm=rainfall,
        runoff_m3=runoff,
        storage_m3=storage,
        water_distance_m=water_dist,
        road_distance_m=road_dist,
        building_distance_m=building_dist,
        feasibility=feasibility,
        land_context=land_context,
        dem_validation_score=dem_score,
        hydrology_validation_score=hydro_score,
        flow_accumulation_cells=flow_acc,
        provider_ok=feasibility != "unknown" or True,
        candidate_id=str(candidate.get("id", "candidate")),
        latitude=_safe_float(candidate.get("latitude")),
        longitude=_safe_float(candidate.get("longitude")),
    )


# ---------------------------------------------------------------------------
# Scoring Components
# ---------------------------------------------------------------------------

def score_hydrology(features: CandidateFeatures, config: RecommendationEngineConfig) -> float:
    """Hydrology score: based on flow accumulation and catchment quality."""
    # Use flow accumulation as primary hydrology indicator
    if features.flow_accumulation_cells is not None and features.flow_accumulation_cells > 0:
        # Normalize flow accumulation (higher is better)
        max_cells = 100_000  # Typical max for large catchments
        score = min(1.0, features.flow_accumulation_cells / max_cells)
        return score

    # Fallback to catchment area
    return _normalize_with_bounds(
        features.catchment_area_m2,
        config.catchment_min_m2,
        config.catchment_max_m2,
        config.catchment_ideal_min_m2,
        config.catchment_ideal_max_m2,
    )


def score_terrain(features: CandidateFeatures, config: RecommendationEngineConfig) -> float:
    """Terrain score: based on slope suitability."""
    return _normalize_with_bounds(
        features.slope_degrees,
        config.slope_min_deg,
        config.slope_max_deg,
        config.slope_ideal_min_deg,
        config.slope_ideal_max_deg,
        invert=True,  # Lower slope (percentage) is better within ideal range
    )


def score_geography(features: CandidateFeatures, config: RecommendationEngineConfig) -> float:
    """Geography score: based on distances to water, roads, buildings."""
    scores = []

    # Water distance (closer is better, but not on top of it)
    water_score = _normalize_distance(
        features.water_distance_m,
        0.0,
        config.water_dist_max_m,
        missing_score=0.5,
    )
    scores.append(water_score)

    # Road distance (farther is better beyond minimum)
    road_score = _normalize_distance(
        features.road_distance_m,
        config.road_dist_min_m,
        config.road_dist_max_m,
        missing_score=0.5,
    )
    scores.append(road_score)

    # Building distance (farther is better beyond minimum)
    building_score = _normalize_distance(
        features.building_distance_m,
        config.building_dist_min_m,
        config.building_dist_max_m,
        missing_score=0.5,
    )
    scores.append(building_score)

    # Average of available distance scores
    if scores:
        return sum(scores) / len(scores)
    return 0.5


def score_rainfall(features: CandidateFeatures, config: RecommendationEngineConfig) -> float:
    """Rainfall score: based on mean annual seasonal rainfall."""
    if features.rainfall_mm is None:
        return 0.5  # Neutral when unknown

    return _normalize_with_bounds(
        features.rainfall_mm,
        config.rainfall_min_mm,
        config.rainfall_max_mm,
        config.rainfall_ideal_min_mm,
        config.rainfall_ideal_max_mm,
    )


def score_runoff(features: CandidateFeatures, config: RecommendationEngineConfig) -> float:
    """Runoff score: based on estimated runoff volume."""
    if features.runoff_m3 is None:
        return 0.5  # Neutral when unknown

    return _normalize_with_bounds(
        features.runoff_m3,
        config.runoff_min_m3,
        config.runoff_max_m3,
        config.runoff_ideal_min_m3,
        config.runoff_ideal_max_m3,
    )


def score_storage(features: CandidateFeatures, config: RecommendationEngineConfig) -> float:
    """Storage score: based on estimated storage volume."""
    if features.storage_m3 is None:
        return 0.5  # Neutral when unknown

    return _normalize_with_bounds(
        features.storage_m3,
        config.storage_min_m3,
        config.storage_max_m3,
        config.storage_ideal_min_m3,
        config.storage_ideal_max_m3,
    )


# ---------------------------------------------------------------------------
# Confidence Calculation
# ---------------------------------------------------------------------------

def calculate_confidence(
    features: CandidateFeatures,
    component_scores: Dict[str, float],
    config: RecommendationEngineConfig,
) -> Tuple[float, Dict[str, float]]:
    """Calculate confidence score separate from suitability.

    Confidence is based on data quality and availability, not the score itself.
    """
    confidence_components = {}

    # DEM quality (0-1)
    if features.dem_validation_score is not None:
        confidence_components["dem_quality"] = max(0.0, min(1.0, features.dem_validation_score))
    else:
        confidence_components["dem_quality"] = 0.5

    # Hydrology quality (0-1)
    if features.hydrology_validation_score is not None:
        confidence_components["hydrology_quality"] = max(0.0, min(1.0, features.hydrology_validation_score))
    else:
        confidence_components["hydrology_quality"] = 0.5

    # Geographic availability
    if features.feasibility == "feasible":
        confidence_components["geographic_availability"] = 1.0
    elif features.feasibility == "rejected":
        confidence_components["geographic_availability"] = 0.0
    else:  # unknown
        confidence_components["geographic_availability"] = 0.5

    # Rainfall availability
    if features.rainfall_mm is not None:
        confidence_components["rainfall_availability"] = 1.0
    else:
        confidence_components["rainfall_availability"] = 0.3

    # Runoff/Storage availability
    if features.runoff_m3 is not None:
        confidence_components["runoff_storage_availability"] = 1.0
    else:
        confidence_components["runoff_storage_availability"] = 0.3

    # Weighted confidence
    total_weight = sum(config.confidence_weights.values())
    confidence = 0.0
    for key, weight in config.confidence_weights.items():
        score = confidence_components.get(key, 0.5)
        confidence += (weight / total_weight) * score

    return max(0.0, min(1.0, confidence)), confidence_components


# ---------------------------------------------------------------------------
# Explanation Generation
# ---------------------------------------------------------------------------

def generate_explanation(
    features: CandidateFeatures,
    component_scores: Dict[str, float],
    overall_score: float,
    confidence: float,
    eligibility: EligibilityResult,
    config: RecommendationEngineConfig,
) -> Dict[str, Any]:
    """Generate detailed, evidence-based explanation."""
    strengths: List[str] = []
    concerns: List[str] = []
    missing_evidence: List[str] = []

    # Hydrology
    if component_scores.get("hydrology", 0) >= 0.7:
        strengths.append("Strong hydrological connectivity with good upstream drainage area")
    elif component_scores.get("hydrology", 0) >= 0.4:
        strengths.append("Adequate hydrological connectivity")
    else:
        concerns.append("Limited upstream drainage contribution")

    # Terrain
    if features.slope_degrees <= config.slope_ideal_max_deg:
        strengths.append(f"Favorable slope ({features.slope_degrees:.1f}°) for embankment construction")
    elif features.slope_degrees <= config.max_slope_degrees:
        concerns.append(f"Moderate slope ({features.slope_degrees:.1f}°) requires careful design")
    else:
        concerns.append(f"Steep slope ({features.slope_degrees:.1f}°) exceeds recommended limit")

    # Geography
    if features.feasibility == "feasible":
        strengths.append("Geographically feasible: meets road/building clearance and water body separation")
    elif features.feasibility == "rejected":
        concerns.append(f"Geographic rejection: {eligibility.reason}")
    else:
        missing_evidence.append("Geographic context unavailable (OSM data not fetched)")

    if features.road_distance_m is not None:
        if features.road_distance_m >= config.road_dist_max_m:
            strengths.append(f"Well separated from roads ({features.road_distance_m:.0f} m)")
        elif features.road_distance_m >= config.road_dist_min_m:
            pass  # Acceptable
        else:
            concerns.append(f"Close to road ({features.road_distance_m:.0f} m)")

    if features.building_distance_m is not None:
        if features.building_distance_m >= config.building_dist_max_m:
            strengths.append(f"Well separated from buildings ({features.building_distance_m:.0f} m)")
        elif features.building_distance_m >= config.building_dist_min_m:
            pass
        else:
            concerns.append(f"Close to buildings ({features.building_distance_m:.0f} m)")

    # Rainfall
    if features.rainfall_mm is not None:
        if features.rainfall_mm >= config.rainfall_ideal_min_mm:
            strengths.append(f"Good seasonal rainfall ({features.rainfall_mm:.0f} mm/year) supports runoff")
        elif features.rainfall_mm >= config.rainfall_min_mm:
            concerns.append(f"Moderate rainfall ({features.rainfall_mm:.0f} mm/year)")
        else:
            concerns.append(f"Low rainfall ({features.rainfall_mm:.0f} mm/year)")
    else:
        missing_evidence.append("Rainfall data unavailable for this location")

    # Runoff
    if features.runoff_m3 is not None:
        if features.runoff_m3 >= config.runoff_ideal_min_m3:
            strengths.append(f"Estimated annual runoff: {features.runoff_m3:,.0f} m³")
        else:
            concerns.append(f"Low estimated runoff: {features.runoff_m3:,.0f} m³/year")
    else:
        missing_evidence.append("Runoff estimate unavailable")

    # Storage
    if features.storage_m3 is not None:
        if features.storage_m3 >= config.storage_ideal_min_m3:
            strengths.append(f"Indicative storage potential: {features.storage_m3:,.0f} m³")
        else:
            concerns.append(f"Limited storage potential: {features.storage_m3:,.0f} m³")
    else:
        missing_evidence.append("Storage estimate unavailable (requires terrain support)")

    # Quality indicators
    if features.dem_validation_score is not None and features.dem_validation_score < 0.6:
        concerns.append("DEM quality below preferred threshold")
    if features.hydrology_validation_score is not None and features.hydrology_validation_score < 0.6:
        concerns.append("Hydrology validation score below preferred threshold")

    # Default fallback
    if not strengths:
        strengths.append("Basic terrain and contour conditions detected")
    if not concerns:
        concerns.append("No major constraints identified")

    # Summary
    summary_parts = [
        f"overall={overall_score:.3f}",
        f"confidence={confidence:.3f}",
        f"hydrology={component_scores.get('hydrology', 0):.2f}",
        f"terrain={component_scores.get('terrain', 0):.2f}",
        f"geography={component_scores.get('geography', 0):.2f}",
        f"rainfall={component_scores.get('rainfall', 0):.2f}",
        f"runoff={component_scores.get('runoff', 0):.2f}",
        f"storage={component_scores.get('storage', 0):.2f}",
    ]
    summary = "; ".join(summary_parts)

    recommendation = "highly_suitable" if overall_score >= 0.7 else "moderately_suitable" if overall_score >= 0.4 else "marginal"

    return {
        "recommendation": recommendation,
        "summary": summary,
        "strengths": strengths,
        "concerns": concerns,
        "missing_evidence": missing_evidence,
        "score_components": {k: round(v, 4) for k, v in component_scores.items()},
        "confidence": round(confidence, 4),
    }


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------

@dataclass
class RankedCandidate:
    """Final ranked candidate with all computed fields."""
    id: str
    latitude: float
    longitude: float
    factors: Dict[str, Any]
    catchment_area: Dict[str, float]
    land_context: Dict[str, Any]
    suitability: Dict[str, Any]
    explanation: str
    explanation_details: Dict[str, Any]
    confidence: float
    eligibility: EligibilityResult


def run_recommendation_engine(
    candidates: Sequence[Mapping[str, Any]],
    *,
    config: Optional[RecommendationEngineConfig] = None,
    dem_validation: Optional[Mapping[str, Any]] = None,
    hydrology_validation: Optional[Mapping[str, Any]] = None,
) -> List[RankedCandidate]:
    """Run the complete recommendation pipeline.

    Pipeline:
    1. Hard eligibility filtering
    2. Feature extraction
    3. Component scoring
    4. Correlation penalty
    5. Weighted overall score
    6. Confidence calculation
    7. Explanation generation
    8. Deterministic ranking
    """
    active_config = config or RecommendationEngineConfig.from_env()

    # Step 1: Eligibility filtering
    eligible_candidates = []
    for candidate in candidates:
        eligibility = check_hard_eligibility(candidate, active_config, dem_validation, hydrology_validation)
        if eligibility.eligible:
            eligible_candidates.append((candidate, eligibility))

    if not eligible_candidates:
        return []

    # Step 2-7: Score each eligible candidate
    ranked: List[RankedCandidate] = []
    for candidate, eligibility in eligible_candidates:
        features = extract_features(candidate, dem_validation, hydrology_validation)

        # Component scores
        component_scores = {
            "hydrology": score_hydrology(features, active_config),
            "terrain": score_terrain(features, active_config),
            "geography": score_geography(features, active_config),
            "rainfall": score_rainfall(features, active_config),
            "runoff": score_runoff(features, active_config),
            "storage": score_storage(features, active_config),
        }

        # Correlation penalty
        component_scores = _compute_correlation_penalty(component_scores)

        # Weighted overall score
        total_weight = sum(active_config.weights.values())
        overall_score = 0.0
        for key, weight in active_config.weights.items():
            score = component_scores.get(key, 0.0)
            overall_score += (weight / total_weight) * score
        overall_score = max(0.0, min(1.0, overall_score))

        # Confidence
        confidence, confidence_components = calculate_confidence(features, component_scores, active_config)

        # Explanation
        explanation_details = generate_explanation(
            features, component_scores, overall_score, confidence, eligibility, active_config
        )

        # Build factors dict for response compatibility
        factors = {
            "slope_degrees": features.slope_degrees,
            "catchment_area_m2": features.catchment_area_m2,
            "rainfall_mm": features.rainfall_mm,
            "water_distance_m": features.water_distance_m,
            "road_distance_m": features.road_distance_m,
            "building_distance_m": features.building_distance_m,
        }

        catchment_area = {
            "area_m2": features.catchment_area_m2,
            "area_hectares": features.catchment_area_m2 / 10_000.0,
            "area_km2": features.catchment_area_m2 / 1_000_000.0,
        }

        # Suitability object matching existing format
        suitability_obj = {
            "component_scores": {k: round(v, 4) for k, v in component_scores.items()},
            "weights": {k: round(v, 4) for k, v in active_config.weights.items()},
            "overall_score": round(overall_score, 4),
        }

        ranked_candidate = RankedCandidate(
            id=features.candidate_id,
            latitude=features.latitude,
            longitude=features.longitude,
            factors=factors,
            catchment_area=catchment_area,
            land_context=features.land_context,
            suitability=suitability_obj,
            explanation=explanation_details["summary"],
            explanation_details=explanation_details,
            confidence=confidence,
            eligibility=eligibility,
        )
        ranked.append(ranked_candidate)

    # Step 8: Deterministic ranking
    # score DESC -> confidence DESC -> flow accumulation DESC -> row/col/id
    def rank_key(c: RankedCandidate) -> Tuple:
        flow_acc = c.eligibility.details.get("flow_accumulation_cells", 0) or 0
        return (
            -c.suitability["overall_score"],
            -c.confidence,
            -flow_acc,
            c.latitude,
            c.longitude,
            c.id,
        )

    ranked.sort(key=rank_key)
    return ranked


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_candidates(
    candidates: Sequence[Mapping[str, Any]],
    constraints: Optional[Mapping[str, float]] = None,
    weights: Optional[Mapping[str, float]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Public API compatible with existing recommendation.rank_candidates.

    This wraps the new engine for backward compatibility.
    """
    # Build config from constraints/weights if provided
    config = RecommendationEngineConfig.from_env()
    if constraints:
        config.max_slope_degrees = constraints.get("max_slope_degrees", config.max_slope_degrees)
        config.min_catchment_area_m2 = constraints.get("min_catchment_area_m2", config.min_catchment_area_m2)
        config.min_rainfall_mm = constraints.get("min_rainfall_mm", config.min_rainfall_mm)
        config.min_water_distance_m = constraints.get("min_water_distance_m", config.min_water_distance_m)
        config.min_road_distance_m = constraints.get("min_road_distance_m", config.min_road_distance_m)
        config.min_building_distance_m = constraints.get("min_building_distance_m", config.min_building_distance_m)
    if weights:
        config.weights.update(weights)

    # Extract validation data from kwargs
    dem_validation = kwargs.get("dem_validation")
    hydrology_validation = kwargs.get("hydrology_validation")

    ranked = run_recommendation_engine(
        candidates,
        config=config,
        dem_validation=dem_validation,
        hydrology_validation=hydrology_validation,
    )

    # Convert to dict format compatible with existing API
    result = []
    for rc in ranked:
        result.append({
            "id": rc.id,
            "latitude": rc.latitude,
            "longitude": rc.longitude,
            "factors": rc.factors,
            "catchment_area": rc.catchment_area,
            "land_context": rc.land_context,
            "suitability": rc.suitability,
            "explanation": rc.explanation,
            "explanation_details": rc.explanation_details,
            "confidence": rc.confidence,
            "feasibility": rc.eligibility.details.get("feasibility", "feasible"),
            "feasibility_reason": rc.eligibility.details.get("feasibility_reason"),
        })
    return result


__all__ = [
    "RecommendationEngineConfig",
    "CandidateFeatures",
    "EligibilityResult",
    "RankedCandidate",
    "check_hard_eligibility",
    "extract_features",
    "score_hydrology",
    "score_terrain",
    "score_geography",
    "score_rainfall",
    "score_runoff",
    "score_storage",
    "calculate_confidence",
    "generate_explanation",
    "run_recommendation_engine",
    "rank_candidates",
]