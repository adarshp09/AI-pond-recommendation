"""Tests for Phase 6: Final Recommendation & Evidence-Based Ranking."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from backend.recommendation_engine import (
    RecommendationEngineConfig,
    CandidateFeatures,
    EligibilityResult,
    check_hard_eligibility,
    extract_features,
    score_hydrology,
    score_terrain,
    score_geography,
    score_rainfall,
    score_runoff,
    score_storage,
    calculate_confidence,
    generate_explanation,
    run_recommendation_engine,
    rank_candidates,
    _normalize_with_bounds,
    _normalize_distance,
    _compute_correlation_penalty,
    _safe_float,
    _safe_optional_float,
)


class TestSafeConversions:
    """Test safe float conversion utilities."""

    def test_safe_float_valid(self):
        assert _safe_float(123) == 123.0
        assert _safe_float("456.78") == 456.78
        assert _safe_float(0) == 0.0

    def test_safe_float_none(self):
        assert _safe_float(None) == 0.0

    def test_safe_float_nan(self):
        assert _safe_float(float("nan")) == 0.0

    def test_safe_float_inf(self):
        assert _safe_float(float("inf")) == 0.0
        assert _safe_float(float("-inf")) == 0.0

    def test_safe_float_invalid(self):
        assert _safe_float("invalid") == 0.0
        assert _safe_float([1, 2, 3]) == 0.0

    def test_safe_optional_float_valid(self):
        assert _safe_optional_float(123) == 123.0
        assert _safe_optional_float("456.78") == 456.78

    def test_safe_optional_float_none(self):
        assert _safe_optional_float(None) is None

    def test_safe_optional_float_nan(self):
        assert _safe_optional_float(float("nan")) is None

    def test_safe_optional_float_inf(self):
        assert _safe_optional_float(float("inf")) is None


class TestNormalization:
    """Test feature normalization functions."""

    def test_normalize_with_bounds_basic(self):
        """Test basic normalization within bounds."""
        result = _normalize_with_bounds(50.0, 0.0, 100.0)
        assert result == 0.5

    def test_normalize_with_bounds_clamping(self):
        """Test clamping for out-of-bounds values."""
        assert _normalize_with_bounds(-10.0, 0.0, 100.0) == 0.0
        assert _normalize_with_bounds(150.0, 0.0, 100.0) == 1.0

    def test_normalize_with_ideal_range(self):
        """Test normalization with ideal range."""
        # Below ideal min
        result = _normalize_with_bounds(10.0, 0.0, 100.0, ideal_min=20.0, ideal_max=80.0)
        assert result == 0.5  # (10-0)/(20-0) = 0.5

        # Within ideal range
        result = _normalize_with_bounds(50.0, 0.0, 100.0, ideal_min=20.0, ideal_max=80.0)
        assert result == 1.0

        # Above ideal max
        result = _normalize_with_bounds(90.0, 0.0, 100.0, ideal_min=20.0, ideal_max=80.0)
        assert result == 0.5  # (90-80)/(100-80) = 0.5

    def test_normalize_with_invert(self):
        """Test inverted normalization (lower is better)."""
        result = _normalize_with_bounds(10.0, 0.0, 100.0, invert=True)
        assert result == pytest.approx(0.9)

        result = _normalize_with_bounds(90.0, 0.0, 100.0, invert=True)
        assert result == pytest.approx(0.1)

    def test_normalize_missing_value(self):
        """Test handling of None/NaN/inf values."""
        assert _normalize_with_bounds(None, 0.0, 100.0) == 0.5
        assert _normalize_with_bounds(float("nan"), 0.0, 100.0) == 0.5
        assert _normalize_with_bounds(float("inf"), 0.0, 100.0) == 0.5

    def test_normalize_constant_range(self):
        """Test when min == max."""
        assert _normalize_with_bounds(50.0, 10.0, 10.0) == 1.0
        assert _normalize_with_bounds(5.0, 10.0, 10.0) == 0.0

    def test_normalize_distance(self):
        """Test distance normalization (closer is better with minimum)."""
        # At minimum distance
        assert _normalize_distance(25.0, 25.0, 250.0) == 0.0

        # Between min and max
        assert _normalize_distance(125.0, 25.0, 225.0) == 0.5

        # At or beyond max
        assert _normalize_distance(250.0, 25.0, 250.0) == 1.0
        assert _normalize_distance(300.0, 25.0, 250.0) == 1.0

        # Below minimum
        assert _normalize_distance(10.0, 25.0, 250.0) == 0.0
        assert _normalize_distance(0.0, 25.0, 250.0) == 0.0

        # Missing value
        assert _normalize_distance(None, 25.0, 250.0) == 0.5


class TestCorrelationPenalty:
    """Test correlation penalty for redundant features."""

    def test_no_penalty_when_few_high(self):
        scores = {"hydrology": 0.8, "terrain": 0.3, "geography": 0.5}
        result = _compute_correlation_penalty(scores)
        assert result == scores

    def test_penalty_when_multiple_correlated_high(self):
        scores = {"catchment_area": 0.9, "runoff": 0.85, "storage": 0.8, "hydrology": 0.5}
        result = _compute_correlation_penalty(scores)
        assert result["catchment_area"] == pytest.approx(0.81)
        assert result["runoff"] == pytest.approx(0.765)
        assert result["storage"] == pytest.approx(0.72)
        assert result["hydrology"] == 0.5  # Unchanged

    def test_penalty_threshold(self):
        scores = {"catchment_area": 0.7, "runoff": 0.6, "storage": 0.5}
        result = _compute_correlation_penalty(scores)
        # Only catchment_area > 0.7, so no penalty (need >= 2)
        assert result == scores


class TestEligibility:
    """Test hard eligibility checking."""

    def create_valid_candidate(self, **overrides):
        base = {
            "id": "test-candidate",
            "latitude": 21.0,
            "longitude": 81.0,
            "slope_degrees": 5.0,
            "catchment_area_m2": 50000.0,
            "rainfall_mm": 1000.0,
            "estimated_runoff_m3": 15000.0,
            "estimated_storage_m3": 7500.0,
            "factors": {
                "slope_degrees": 5.0,
                "catchment_area_m2": 50000.0,
                "rainfall_mm": 1000.0,
                "water_distance_m": 100.0,
                "road_distance_m": 200.0,
                "building_distance_m": 200.0,
            },
            "feasibility": "feasible",
            "feasibility_reason": None,
            "land_context": {},
        }
        base.update(overrides)
        return base

    def test_valid_candidate_passes(self):
        config = RecommendationEngineConfig.from_env()
        candidate = self.create_valid_candidate()
        result = check_hard_eligibility(candidate, config)
        assert result.eligible is True
        assert result.reason is None

    def test_missing_coordinates_rejected(self):
        config = RecommendationEngineConfig.from_env()
        candidate = self.create_valid_candidate(latitude=None)
        result = check_hard_eligibility(candidate, config)
        assert result.eligible is False
        assert result.reason == "missing_coordinates"

    def test_invalid_coordinates_rejected(self):
        config = RecommendationEngineConfig.from_env()
        candidate = self.create_valid_candidate(latitude=100.0)
        result = check_hard_eligibility(candidate, config)
        assert result.eligible is False
        assert result.reason == "invalid_coordinates"

    def test_slope_exceeds_limit_rejected(self):
        config = RecommendationEngineConfig.from_env()
        candidate = self.create_valid_candidate(slope_degrees=40.0, factors={"slope_degrees": 40.0})
        result = check_hard_eligibility(candidate, config)
        assert result.eligible is False
        assert result.reason == "slope_exceeds_limit"

    def test_catchment_too_small_rejected(self):
        config = RecommendationEngineConfig(min_catchment_area_m2=5000.0)
        candidate = self.create_valid_candidate(
            catchment_area_m2=1000.0,
            factors={"catchment_area_m2": 1000.0}
        )
        result = check_hard_eligibility(candidate, config)
        assert result.eligible is False
        assert result.reason == "catchment_too_small"

    def test_insufficient_rainfall_rejected(self):
        config = RecommendationEngineConfig(min_rainfall_mm=500.0)
        candidate = self.create_valid_candidate(
            rainfall_mm=200.0,
            factors={"rainfall_mm": 200.0}
        )
        result = check_hard_eligibility(candidate, config)
        assert result.eligible is False
        assert result.reason == "insufficient_rainfall"

    def test_rainfall_none_not_rejected(self):
        config = RecommendationEngineConfig(min_rainfall_mm=500.0)
        candidate = self.create_valid_candidate(
            rainfall_mm=None,
            factors={"rainfall_mm": None}
        )
        result = check_hard_eligibility(candidate, config)
        assert result.eligible is True  # None should not cause rejection

    def test_geographic_rejected(self):
        config = RecommendationEngineConfig.from_env()
        candidate = self.create_valid_candidate(
            feasibility="rejected",
            feasibility_reason="insufficient_road_clearance"
        )
        result = check_hard_eligibility(candidate, config)
        assert result.eligible is False
        assert result.reason == "insufficient_road_clearance"

    def test_geographic_unknown_not_rejected(self):
        config = RecommendationEngineConfig.from_env()
        candidate = self.create_valid_candidate(feasibility="unknown")
        result = check_hard_eligibility(candidate, config)
        assert result.eligible is True

    def test_dem_invalid_rejected(self):
        config = RecommendationEngineConfig.from_env()
        candidate = self.create_valid_candidate()
        dem_validation = {"status": "invalid"}
        result = check_hard_eligibility(candidate, config, dem_validation=dem_validation)
        assert result.eligible is False
        assert result.reason == "dem_invalid"

    def test_hydrology_invalid_rejected(self):
        config = RecommendationEngineConfig.from_env()
        candidate = self.create_valid_candidate()
        hydro_validation = {"status": "invalid"}
        result = check_hard_eligibility(candidate, config, hydrology_validation=hydro_validation)
        assert result.eligible is False
        assert result.reason == "hydrology_invalid"


class TestFeatureExtraction:
    """Test feature extraction from backend.candidates."""

    def test_extract_all_features(self):
        candidate = {
            "id": "test-1",
            "latitude": 21.0,
            "longitude": 81.0,
            "slope_degrees": 5.0,
            "catchment_area_m2": 50000.0,
            "rainfall_mm": 1000.0,
            "estimated_runoff_m3": 15000.0,
            "estimated_storage_m3": 7500.0,
            "factors": {
                "slope_degrees": 5.0,
                "catchment_area_m2": 50000.0,
                "rainfall_mm": 1000.0,
                "water_distance_m": 100.0,
                "road_distance_m": 200.0,
                "building_distance_m": 200.0,
            },
            "feasibility": "feasible",
            "land_context": {"water_bodies": [], "roads": [], "buildings": []},
            "flow_accumulation_cells": 5000,
        }
        dem_val = {"score": 0.8}
        hydro_val = {"score": 0.9}

        features = extract_features(candidate, dem_val, hydro_val)

        assert features.candidate_id == "test-1"
        assert features.latitude == 21.0
        assert features.longitude == 81.0
        assert features.slope_degrees == 5.0
        assert features.catchment_area_m2 == 50000.0
        assert features.rainfall_mm == 1000.0
        assert features.runoff_m3 == 15000.0
        assert features.storage_m3 == 7500.0
        assert features.water_distance_m == 100.0
        assert features.road_distance_m == 200.0
        assert features.building_distance_m == 200.0
        assert features.feasibility == "feasible"
        assert features.dem_validation_score == 0.8
        assert features.hydrology_validation_score == 0.9
        assert features.flow_accumulation_cells == 5000

    def test_extract_with_missing_optional(self):
        candidate = {
            "id": "test-2",
            "latitude": 21.0,
            "longitude": 81.0,
            "slope_degrees": 5.0,
            "catchment_area_m2": 50000.0,
            "factors": {
                "slope_degrees": 5.0,
                "catchment_area_m2": 50000.0,
            },
            "feasibility": "unknown",
            "land_context": {},
        }

        features = extract_features(candidate, None, None)

        assert features.rainfall_mm is None
        assert features.runoff_m3 is None
        assert features.storage_m3 is None
        assert features.water_distance_m is None
        assert features.road_distance_m is None
        assert features.building_distance_m is None
        assert features.dem_validation_score is None
        assert features.hydrology_validation_score is None
        assert features.flow_accumulation_cells is None


class TestScoring:
    """Test individual component scoring."""

    def test_score_hydrology_with_flow_acc(self):
        config = RecommendationEngineConfig.from_env()
        features = CandidateFeatures(
            slope_degrees=5.0,
            catchment_area_m2=50000.0,
            rainfall_mm=1000.0,
            runoff_m3=15000.0,
            storage_m3=7500.0,
            water_distance_m=100.0,
            road_distance_m=200.0,
            building_distance_m=200.0,
            feasibility="feasible",
            land_context={},
            flow_accumulation_cells=50000,
        )
        score = score_hydrology(features, config)
        assert score > 0.0
        assert score <= 1.0

    def test_score_hydrology_fallback_catchment(self):
        config = RecommendationEngineConfig.from_env()
        features = CandidateFeatures(
            slope_degrees=5.0,
            catchment_area_m2=50000.0,
            rainfall_mm=1000.0,
            runoff_m3=15000.0,
            storage_m3=7500.0,
            water_distance_m=100.0,
            road_distance_m=200.0,
            building_distance_m=200.0,
            feasibility="feasible",
            land_context={},
            flow_accumulation_cells=None,
        )
        score = score_hydrology(features, config)
        assert score > 0.0
        assert score <= 1.0

    def test_score_terrain(self):
        config = RecommendationEngineConfig.from_env()
        features = CandidateFeatures(
            slope_degrees=5.0,
            catchment_area_m2=50000.0,
            rainfall_mm=1000.0,
            runoff_m3=15000.0,
            storage_m3=7500.0,
            water_distance_m=100.0,
            road_distance_m=200.0,
            building_distance_m=200.0,
            feasibility="feasible",
            land_context={},
        )
        score = score_terrain(features, config)
        # 5 degrees is in ideal range [2, 12], so should be high
        assert score > 0.7

    def test_score_terrain_steep(self):
        config = RecommendationEngineConfig.from_env()
        features = CandidateFeatures(
            slope_degrees=30.0,
            catchment_area_m2=50000.0,
            rainfall_mm=1000.0,
            runoff_m3=15000.0,
            storage_m3=7500.0,
            water_distance_m=100.0,
            road_distance_m=200.0,
            building_distance_m=200.0,
            feasibility="feasible",
            land_context={},
        )
        score = score_terrain(features, config)
        assert score < 0.5  # Steep slope should score lower

    def test_score_geography(self):
        config = RecommendationEngineConfig.from_env()
        features = CandidateFeatures(
            slope_degrees=5.0,
            catchment_area_m2=50000.0,
            rainfall_mm=1000.0,
            runoff_m3=15000.0,
            storage_m3=7500.0,
            water_distance_m=400.0,  # Good water distance
            road_distance_m=400.0,   # Good road distance
            building_distance_m=400.0,  # Good building distance
            feasibility="feasible",
            land_context={},
        )
        score = score_geography(features, config)
        # With all distances at 400m (well above 25m min, close to 500m max), score should be high
        assert score > 0.7

    def test_score_geography_missing(self):
        config = RecommendationEngineConfig.from_env()
        features = CandidateFeatures(
            slope_degrees=5.0,
            catchment_area_m2=50000.0,
            rainfall_mm=1000.0,
            runoff_m3=15000.0,
            storage_m3=7500.0,
            water_distance_m=None,
            road_distance_m=None,
            building_distance_m=None,
            feasibility="unknown",
            land_context={},
        )
        score = score_geography(features, config)
        assert score == 0.5  # Neutral when all missing

    def test_score_rainfall(self):
        config = RecommendationEngineConfig.from_env()
        features = CandidateFeatures(
            slope_degrees=5.0,
            catchment_area_m2=50000.0,
            rainfall_mm=1000.0,
            runoff_m3=15000.0,
            storage_m3=7500.0,
            water_distance_m=100.0,
            road_distance_m=200.0,
            building_distance_m=200.0,
            feasibility="feasible",
            land_context={},
        )
        score = score_rainfall(features, config)
        # 1000mm is in ideal range [500, 2000]
        assert score > 0.7

    def test_score_rainfall_missing(self):
        config = RecommendationEngineConfig.from_env()
        features = CandidateFeatures(
            slope_degrees=5.0,
            catchment_area_m2=50000.0,
            rainfall_mm=None,
            runoff_m3=15000.0,
            storage_m3=7500.0,
            water_distance_m=100.0,
            road_distance_m=200.0,
            building_distance_m=200.0,
            feasibility="feasible",
            land_context={},
        )
        score = score_rainfall(features, config)
        assert score == 0.5  # Neutral when missing

    def test_score_runoff(self):
        config = RecommendationEngineConfig.from_env()
        features = CandidateFeatures(
            slope_degrees=5.0,
            catchment_area_m2=50000.0,
            rainfall_mm=1000.0,
            runoff_m3=15000.0,
            storage_m3=7500.0,
            water_distance_m=100.0,
            road_distance_m=200.0,
            building_distance_m=200.0,
            feasibility="feasible",
            land_context={},
        )
        score = score_runoff(features, config)
        # 15000 is in ideal range [5000, 100000]
        assert score > 0.7

    def test_score_storage(self):
        config = RecommendationEngineConfig.from_env()
        features = CandidateFeatures(
            slope_degrees=5.0,
            catchment_area_m2=50000.0,
            rainfall_mm=1000.0,
            runoff_m3=15000.0,
            storage_m3=7500.0,
            water_distance_m=100.0,
            road_distance_m=200.0,
            building_distance_m=200.0,
            feasibility="feasible",
            land_context={},
        )
        score = score_storage(features, config)
        # 7500 is in ideal range [2500, 50000]
        assert score > 0.7


class TestConfidence:
    """Test confidence calculation."""

    def test_confidence_full_data(self):
        config = RecommendationEngineConfig.from_env()
        features = CandidateFeatures(
            slope_degrees=5.0,
            catchment_area_m2=50000.0,
            rainfall_mm=1000.0,
            runoff_m3=15000.0,
            storage_m3=7500.0,
            water_distance_m=100.0,
            road_distance_m=200.0,
            building_distance_m=200.0,
            feasibility="feasible",
            land_context={},
            dem_validation_score=0.9,
            hydrology_validation_score=0.9,
            flow_accumulation_cells=5000,
        )
        component_scores = {
            "hydrology": 0.8, "terrain": 0.8, "geography": 0.8,
            "rainfall": 0.8, "runoff": 0.8, "storage": 0.8,
        }
        confidence, components = calculate_confidence(features, component_scores, config)
        assert confidence > 0.8
        assert all(k in components for k in [
            "dem_quality", "hydrology_quality", "geographic_availability",
            "rainfall_availability", "runoff_storage_availability"
        ])

    def test_confidence_missing_data(self):
        config = RecommendationEngineConfig.from_env()
        features = CandidateFeatures(
            slope_degrees=5.0,
            catchment_area_m2=50000.0,
            rainfall_mm=None,
            runoff_m3=None,
            storage_m3=None,
            water_distance_m=None,
            road_distance_m=None,
            building_distance_m=None,
            feasibility="unknown",
            land_context={},
            dem_validation_score=None,
            hydrology_validation_score=None,
        )
        component_scores = {
            "hydrology": 0.5, "terrain": 0.5, "geography": 0.5,
            "rainfall": 0.5, "runoff": 0.5, "storage": 0.5,
        }
        confidence, components = calculate_confidence(features, component_scores, config)
        assert confidence < 0.6
        assert components["geographic_availability"] == 0.5
        assert components["rainfall_availability"] == 0.3
        assert components["runoff_storage_availability"] == 0.3


class TestExplanation:
    """Test explanation generation."""

    def test_generate_explanation(self):
        config = RecommendationEngineConfig.from_env()
        features = CandidateFeatures(
            slope_degrees=5.0,
            catchment_area_m2=50000.0,
            rainfall_mm=1000.0,
            runoff_m3=15000.0,
            storage_m3=7500.0,
            water_distance_m=100.0,
            road_distance_m=200.0,
            building_distance_m=200.0,
            feasibility="feasible",
            land_context={},
        )
        component_scores = {
            "hydrology": 0.8, "terrain": 0.8, "geography": 0.8,
            "rainfall": 0.8, "runoff": 0.8, "storage": 0.8,
        }
        eligibility = EligibilityResult(True, None, {})

        explanation = generate_explanation(
            features, component_scores, 0.8, 0.9, eligibility, config
        )

        assert "recommendation" in explanation
        assert "summary" in explanation
        assert "strengths" in explanation
        assert "concerns" in explanation
        assert "missing_evidence" in explanation
        assert "score_components" in explanation
        assert "confidence" in explanation
        assert explanation["recommendation"] == "highly_suitable"
        assert len(explanation["strengths"]) > 0


class TestEngineIntegration:
    """Test full engine pipeline."""

    def create_test_candidates(self, count=3):
        candidates = []
        for i in range(count):
            candidates.append({
                "id": f"candidate-{i}",
                "latitude": 21.0 + i * 0.001,
                "longitude": 81.0 + i * 0.001,
                "slope_degrees": 5.0,
                "catchment_area_m2": 50000.0 + i * 10000,
                "rainfall_mm": 1000.0,
                "estimated_runoff_m3": 15000.0 + i * 5000,
                "estimated_storage_m3": 7500.0 + i * 2500,
                "factors": {
                    "slope_degrees": 5.0,
                    "catchment_area_m2": 50000.0 + i * 10000,
                    "rainfall_mm": 1000.0,
                    "water_distance_m": 100.0,
                    "road_distance_m": 200.0,
                    "building_distance_m": 200.0,
                },
                "feasibility": "feasible",
                "feasibility_reason": None,
                "land_context": {},
                "flow_accumulation_cells": 5000 + i * 1000,
            })
        return candidates

    def test_run_engine_basic(self):
        config = RecommendationEngineConfig.from_env()
        candidates = self.create_test_candidates(3)

        ranked = run_recommendation_engine(candidates, config=config)

        assert len(ranked) == 3
        # Should be sorted by score DESC, then confidence DESC, then flow_acc DESC
        for i in range(len(ranked) - 1):
            assert ranked[i].suitability["overall_score"] >= ranked[i + 1].suitability["overall_score"]

    def test_run_engine_filters_ineligible(self):
        config = RecommendationEngineConfig(min_catchment_area_m2=100000.0)
        candidates = self.create_test_candidates(3)  # All have 50k-70k catchment

        ranked = run_recommendation_engine(candidates, config=config)

        assert len(ranked) == 0  # All filtered out

    def test_run_engine_deterministic(self):
        config = RecommendationEngineConfig.from_env()
        candidates = self.create_test_candidates(3)

        ranked1 = run_recommendation_engine(candidates, config=config)
        ranked2 = run_recommendation_engine(candidates, config=config)

        assert len(ranked1) == len(ranked2)
        for r1, r2 in zip(ranked1, ranked2):
            assert r1.id == r2.id
            assert r1.suitability["overall_score"] == r2.suitability["overall_score"]
            assert r1.confidence == r2.confidence

    def test_ranking_tiebreaker_confidence(self):
        """Test that confidence breaks score ties."""
        config = RecommendationEngineConfig.from_env()
        # Create candidates with same score but different confidence factors
        candidates = [
            {
                "id": "high-conf",
                "latitude": 21.0,
                "longitude": 81.0,
                "slope_degrees": 5.0,
                "catchment_area_m2": 50000.0,
                "rainfall_mm": 1000.0,
                "estimated_runoff_m3": 15000.0,
                "estimated_storage_m3": 7500.0,
                "factors": {"slope_degrees": 5.0, "catchment_area_m2": 50000.0, "rainfall_mm": 1000.0,
                           "water_distance_m": 100.0, "road_distance_m": 200.0, "building_distance_m": 200.0},
                "feasibility": "feasible",
                "land_context": {},
                "flow_accumulation_cells": 5000,
            },
            {
                "id": "low-conf",
                "latitude": 21.001,
                "longitude": 81.001,
                "slope_degrees": 5.0,
                "catchment_area_m2": 50000.0,
                "rainfall_mm": None,  # Missing rainfall -> lower confidence
                "estimated_runoff_m3": None,
                "estimated_storage_m3": None,
                "factors": {"slope_degrees": 5.0, "catchment_area_m2": 50000.0, "rainfall_mm": None,
                           "water_distance_m": None, "road_distance_m": None, "building_distance_m": None},
                "feasibility": "unknown",
                "land_context": {},
                "flow_accumulation_cells": 5000,
            },
        ]

        ranked = run_recommendation_engine(candidates, config=config)

        assert len(ranked) == 2
        # Both should have similar scores, but high-conf should rank higher due to confidence
        assert ranked[0].id == "high-conf"
        assert ranked[1].id == "low-conf"

    def test_ranking_tiebreaker_flow_accumulation(self):
        """Test flow accumulation tiebreaker."""
        config = RecommendationEngineConfig.from_env()
        candidates = [
            {
                "id": "high-flow",
                "latitude": 21.0,
                "longitude": 81.0,
                "slope_degrees": 5.0,
                "catchment_area_m2": 50000.0,
                "rainfall_mm": 1000.0,
                "estimated_runoff_m3": 15000.0,
                "estimated_storage_m3": 7500.0,
                "factors": {"slope_degrees": 5.0, "catchment_area_m2": 50000.0, "rainfall_mm": 1000.0,
                           "water_distance_m": 100.0, "road_distance_m": 200.0, "building_distance_m": 200.0},
                "feasibility": "feasible",
                "land_context": {},
                "flow_accumulation_cells": 10000,
            },
            {
                "id": "low-flow",
                "latitude": 21.001,
                "longitude": 81.001,
                "slope_degrees": 5.0,
                "catchment_area_m2": 50000.0,
                "rainfall_mm": 1000.0,
                "estimated_runoff_m3": 15000.0,
                "estimated_storage_m3": 7500.0,
                "factors": {"slope_degrees": 5.0, "catchment_area_m2": 50000.0, "rainfall_mm": 1000.0,
                           "water_distance_m": 100.0, "road_distance_m": 200.0, "building_distance_m": 200.0},
                "feasibility": "feasible",
                "land_context": {},
                "flow_accumulation_cells": 5000,
            },
        ]

        ranked = run_recommendation_engine(candidates, config=config)

        # Same score and confidence, but high-flow has more flow accumulation
        assert ranked[0].id == "high-flow"
        assert ranked[1].id == "low-flow"


class TestRankCandidatesAPI:
    """Test the public rank_candidates API."""

    def test_rank_candidates_compatibility(self):
        """Test that the API is compatible with existing code."""
        candidates = [{
            "id": "test-1",
            "latitude": 21.0,
            "longitude": 81.0,
            "slope_degrees": 5.0,
            "catchment_area_m2": 50000.0,
            "rainfall_mm": 1000.0,
            "estimated_runoff_m3": 15000.0,
            "estimated_storage_m3": 7500.0,
            "factors": {
                "slope_degrees": 5.0,
                "catchment_area_m2": 50000.0,
                "rainfall_mm": 1000.0,
                "water_distance_m": 100.0,
                "road_distance_m": 200.0,
                "building_distance_m": 200.0,
            },
            "feasibility": "feasible",
            "feasibility_reason": None,
            "land_context": {},
            "flow_accumulation_cells": 5000,
        }]

        # Call with old API style
        ranked = rank_candidates(
            candidates,
            constraints={"max_slope_degrees": 35.0},
            weights={"hydrology": 0.3, "terrain": 0.2},
            dem_validation={"score": 0.8},
            hydrology_validation={"score": 0.9},
        )

        assert len(ranked) == 1
        assert "id" in ranked[0]
        assert "suitability" in ranked[0]
        assert "explanation" in ranked[0]
        assert "confidence" in ranked[0]

    def test_rank_candidates_with_constraints(self):
        """Test that constraints are applied."""
        candidates = [{
            "id": "test-1",
            "latitude": 21.0,
            "longitude": 81.0,
            "slope_degrees": 40.0,  # Exceeds default 35
            "catchment_area_m2": 50000.0,
            "rainfall_mm": 1000.0,
            "factors": {"slope_degrees": 40.0, "catchment_area_m2": 50000.0, "rainfall_mm": 1000.0},
            "feasibility": "feasible",
            "land_context": {},
        }]

        # Default constraint (35 degrees) should reject
        ranked = rank_candidates(candidates)
        assert len(ranked) == 0

        # Relaxed constraint should accept
        ranked = rank_candidates(candidates, constraints={"max_slope_degrees": 45.0})
        assert len(ranked) == 1


class TestConfigValidation:
    """Test configuration validation."""

    def test_valid_config(self):
        config = RecommendationEngineConfig.from_env()
        assert config.max_slope_degrees == 35.0
        assert config.min_catchment_area_m2 == 0.0
        assert sum(config.weights.values()) > 0

    def test_invalid_weight_negative(self):
        with pytest.raises(ValueError):
            RecommendationEngineConfig(weights={"hydrology": -0.1})

    def test_invalid_weight_zero_sum(self):
        with pytest.raises(ValueError):
            RecommendationEngineConfig(weights={"hydrology": 0.0})

    def test_invalid_confidence_weight(self):
        with pytest.raises(ValueError):
            RecommendationEngineConfig(confidence_weights={"dem_quality": -0.1})

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            RecommendationEngineConfig(slope_min_deg=10.0, slope_max_deg=5.0)

    def test_custom_weights(self):
        config = RecommendationEngineConfig(
            weights={"hydrology": 0.5, "terrain": 0.5}
        )
        total = sum(config.weights.values())
        assert abs(total - 1.0) < 0.01  # Normalized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])