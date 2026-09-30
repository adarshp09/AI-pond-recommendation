from pathlib import Path

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_complete_pipeline_with_sample_contours():
    sample_path = Path(__file__).resolve().parents[1] / "sample_contours.kml"
    content = sample_path.read_bytes()

    response = client.post(
        "/analyzeContour",
        files={"file": (sample_path.name, content, "application/xml")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"

    assert "recommended_location" in payload
    assert payload["recommended_location"]["longitude"] is not None
    assert payload["recommended_location"]["latitude"] is not None
    assert payload["catchment_area"]["area_m2"] > 0
    assert 0.0 <= payload["suitability"]["overall_score"] <= 1.0


def test_enriched_pipeline_handles_external_failures_without_crashing(monkeypatch):
    sample_path = Path(__file__).resolve().parents[1] / "sample_contours.kml"
    content = sample_path.read_bytes()

    def raise_error(*args, **kwargs):
        raise RuntimeError("simulated external failure")

    monkeypatch.setattr("main.get_dem_for_bbox", raise_error)
    monkeypatch.setattr("main.fetch_historical_rainfall", raise_error)
    monkeypatch.setattr("main.fetch_land_context", raise_error)
    monkeypatch.setattr("main.search_places", raise_error)
    monkeypatch.setattr("main.geocode_place", raise_error)

    response = client.post(
        "/analyzeContour",
        files={"file": (sample_path.name, content, "application/xml")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert isinstance(payload.get("warnings", []), list)
