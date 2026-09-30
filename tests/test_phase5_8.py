from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


@pytest.fixture
def sample_kml_bytes():
    return (Path(__file__).resolve().parents[1] / "sample_contours.kml").read_bytes()


def test_analyze_returns_unified_payload(sample_kml_bytes):
    response = client.post(
        "/analyze",
        files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert "dem" in payload
    assert "terrain" in payload
    assert "hydrology" in payload
    assert "catchment" in payload
    assert "rainfall" in payload
    assert "land_features" in payload
    assert "recommendation" in payload
    assert payload["recommendation"]["best_location"]["latitude"] is not None
    assert payload["recommendation"]["best_location"]["longitude"] is not None
    assert isinstance(payload["warnings"], list)


def test_analyze_rejects_missing_or_invalid_kml():
    response = client.post(
        "/analyze",
        files={"file": ("invalid.txt", b"not a kml", "text/plain")},
    )

    assert response.status_code == 400
    payload = response.json()
    assert "detail" in payload
    assert payload["detail"]["code"] in {"UNSUPPORTED_FILE_TYPE", "INVALID_FILE"}


def test_analyze_handles_external_failures_gracefully(monkeypatch, sample_kml_bytes):
    def raise_error(*args, **kwargs):
        raise RuntimeError("simulated external failure")

    monkeypatch.setattr("main.get_dem_for_bbox", raise_error)
    monkeypatch.setattr("main.fetch_historical_rainfall", raise_error)
    monkeypatch.setattr("main.fetch_land_context", raise_error)

    response = client.post(
        "/analyze",
        files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert isinstance(payload["warnings"], list)
    assert any("fail" in warning.lower() or "unavailable" in warning.lower() for warning in payload["warnings"])


def test_constraints_config_and_explanation_are_available():
    from backend.recommendation import DEFAULT_REJECTION_CONSTRAINTS

    assert "max_slope_degrees" in DEFAULT_REJECTION_CONSTRAINTS
    assert "min_catchment_area_m2" in DEFAULT_REJECTION_CONSTRAINTS
    assert DEFAULT_REJECTION_CONSTRAINTS["max_slope_degrees"] > 0
