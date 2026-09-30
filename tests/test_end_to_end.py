from pathlib import Path

from fastapi.testclient import TestClient

from explainability import build_explainability
from main import app


client = TestClient(app)


def test_end_to_end_analysis_pipeline_success():
    sample_path = Path(__file__).resolve().parents[1] / "sample_contours.kml"
    response = client.post(
        "/analyzeContour",
        files={"file": (sample_path.name, sample_path.read_bytes(), "application/xml")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["suitability"]["overall_score"] >= 0.0
    assert payload["suitability"]["overall_score"] <= 1.0
    assert payload["pond_candidate"]["latitude"] is not None
    assert payload["pond_candidate"]["longitude"] is not None
    assert payload["catchment_area"]["area_m2"] > 0
    assert isinstance(payload.get("warnings", []), list)

    explanation = build_explainability(payload)
    assert explanation["overall_score"] == payload["suitability"]["overall_score"]
    assert explanation["recommendation"]
    assert explanation["positive_factors"] or explanation["negative_factors"]


def test_recommend_pond_returns_explainable_final_recommendation():
    sample_path = Path(__file__).resolve().parents[1] / "sample_contours.kml"
    response = client.post(
        "/analyzeContour",
        files={"file": (sample_path.name, sample_path.read_bytes(), "application/xml")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["pond_candidate"]["suitability_score"] >= 0.0
    assert payload["pond_candidate"]["suitability_score"] <= 1.0
    assert payload["recommendation"]["explanation"]

    explanation = build_explainability(payload)
    assert explanation["overall_score"] == payload["suitability"]["overall_score"]
    assert explanation["recommendation"]
