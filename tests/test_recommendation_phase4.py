from pathlib import Path

from fastapi.testclient import TestClient

from main import app
from backend.recommendation import rank_candidates, reject_unsuitable_candidates
from backend.suitability import evaluate_pond_suitability


client = TestClient(app)


def test_weighted_suitability_changes_with_weights():
    base = evaluate_pond_suitability(
        slope_degrees=8.0,
        catchment_area_m2=250000.0,
        total_precipitation_mm=1100.0,
        water_distance_m=80.0,
        road_distance_m=150.0,
        building_distance_m=200.0,
        land_context={"water_bodies": [], "roads": [], "buildings": []},
    )
    weighted = evaluate_pond_suitability(
        slope_degrees=8.0,
        catchment_area_m2=250000.0,
        total_precipitation_mm=1100.0,
        water_distance_m=80.0,
        road_distance_m=150.0,
        building_distance_m=200.0,
        land_context={"water_bodies": [], "roads": [], "buildings": []},
        weights={"slope": 0.7, "catchment_area": 0.1, "rainfall": 0.1, "water_distance": 0.05, "road_distance": 0.03, "land_use": 0.02},
    )

    assert 0.0 <= base["overall_score"] <= 1.0
    assert 0.0 <= weighted["overall_score"] <= 1.0
    assert base["overall_score"] != weighted["overall_score"]


def test_candidates_are_ranked_and_unsuitable_candidates_are_rejected():
    candidates = [
        {
            "id": "good",
            "latitude": 12.0,
            "longitude": 77.0,
            "factors": {
                "slope_degrees": 6.0,
                "catchment_area_m2": 300000.0,
                "rainfall_mm": 1200.0,
                "water_distance_m": 120.0,
                "road_distance_m": 180.0,
                "building_distance_m": 220.0,
            },
            "land_context": {"water_bodies": [], "roads": [], "buildings": []},
        },
        {
            "id": "poor",
            "latitude": 12.1,
            "longitude": 77.1,
            "factors": {
                "slope_degrees": 35.0,
                "catchment_area_m2": 2000.0,
                "rainfall_mm": 150.0,
                "water_distance_m": 10.0,
                "road_distance_m": 20.0,
                "building_distance_m": 15.0,
            },
            "land_context": {"water_bodies": [{"id": "river"}], "roads": [{"id": "road"}], "buildings": [{"id": "b"}]},
        },
    ]

    acceptable = reject_unsuitable_candidates(candidates)
    ranked = rank_candidates(acceptable)

    assert len(acceptable) == 1
    assert acceptable[0]["id"] == "good"
    assert ranked[0]["id"] == "good"
    assert ranked[0]["explanation"]
    assert ranked[0]["suitability"]["overall_score"] >= 0.0


def test_recommend_pond_with_sample_contours():
    sample_path = Path(__file__).resolve().parents[1] / "sample_contours.kml"
    content = sample_path.read_bytes()

    response = client.post(
        "/analyzeContour",
        files={"file": (sample_path.name, content, "application/xml")},
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "success"
    assert payload["pond_candidate"]["latitude"] is not None
    assert payload["pond_candidate"]["longitude"] is not None
    assert 0.0 <= payload["pond_candidate"]["suitability_score"] <= 1.0
    assert payload["pond_candidate"]["catchment_area"]["area_m2"] > 0
    assert isinstance(payload.get("alternative_candidates", []), list)
    assert len(payload["alternative_candidates"]) >= 0
