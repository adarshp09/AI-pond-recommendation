import math

from backend.suitability import DEFAULT_WEIGHTS, evaluate_pond_suitability


def test_score_is_between_zero_and_one():
    result = evaluate_pond_suitability(
        slope_degrees=8.0,
        catchment_area_m2=250000.0,
        total_precipitation_mm=1200.0,
        water_distance_m=120.0,
        road_distance_m=80.0,
        building_distance_m=150.0,
    )

    assert 0.0 <= result["overall_score"] <= 1.0
    for score in result["component_scores"].values():
        assert 0.0 <= score <= 1.0


def test_poor_slope_gets_lower_score():
    good = evaluate_pond_suitability(
        slope_degrees=8.0,
        catchment_area_m2=200000.0,
        total_precipitation_mm=1000.0,
        water_distance_m=60.0,
        road_distance_m=90.0,
        building_distance_m=110.0,
    )
    poor = evaluate_pond_suitability(
        slope_degrees=30.0,
        catchment_area_m2=200000.0,
        total_precipitation_mm=1000.0,
        water_distance_m=60.0,
        road_distance_m=90.0,
        building_distance_m=110.0,
    )

    assert good["component_scores"]["slope"] > poor["component_scores"]["slope"]
    assert good["overall_score"] > poor["overall_score"]


def test_adequate_catchment_gets_higher_score():
    low = evaluate_pond_suitability(
        slope_degrees=5.0,
        catchment_area_m2=10000.0,
        total_precipitation_mm=900.0,
        water_distance_m=50.0,
        road_distance_m=80.0,
        building_distance_m=120.0,
    )
    high = evaluate_pond_suitability(
        slope_degrees=5.0,
        catchment_area_m2=500000.0,
        total_precipitation_mm=900.0,
        water_distance_m=50.0,
        road_distance_m=80.0,
        building_distance_m=120.0,
    )

    assert low["component_scores"]["catchment_area"] < high["component_scores"]["catchment_area"]
    assert low["overall_score"] < high["overall_score"]


def test_component_weights_sum_to_one():
    total = sum(DEFAULT_WEIGHTS.values())
    assert math.isclose(total, 1.0)
