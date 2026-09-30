"""Baseline regression tests that lock in currently-working behaviour.

These tests intentionally assert on the *current* API contract and outputs so
that later phases which change algorithms cannot silently break the observable
behaviour established in Phase 1.
"""

from __future__ import annotations

import pytest

import fixtures


ANALYZE_CONTOUR_KEYS = {
    "status",
    "input",
    "contour_diagnostics",
    "terrain",
    "dem_validation",
    "pond_candidate",
    "alternative_candidates",
    "catchment",
    "recommended_location",
    "catchment_area",
    "suitability",
    "recommendation",
    "method",
    "warnings",
    "error_code",
    "message",
    "enrichment",
}

ANALYZE_LOCATION_KEYS = {
    "center",
    "min_latitude",
    "max_latitude",
    "min_longitude",
    "max_longitude",
    "radius_km",
    "status",
    "dem",
    "terrain",
    "hydrology",
    "catchment",
    "suitability",
    "pond_candidate",
    "alternative_candidates",
    "rainfall",
    "rainfall_runoff",
    "land_features",
    "recommendation",
    "warnings",
    "error_code",
    "message",
}


def test_health_and_root_endpoints(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["service"] == "AI Pond Recommendation API"
    assert "analyze_contour" in root.json()["routes"]
    assert "analyze_location" in root.json()["routes"]
    assert "export_kml" in root.json()["routes"]
    assert "export_kmz" in root.json()["routes"]
    assert "health" in root.json()["routes"]


def test_analyze_contour_schema_is_locked(client, sample_kml_bytes):
    payload = client.post(
        "/analyzeContour",
        files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
    ).json()
    assert payload["status"] == "success"
    assert set(payload.keys()) == ANALYZE_CONTOUR_KEYS


def test_analyze_location_schema_is_locked(client):
    payload = client.post(
        "/analyzeLocation",
        json={"latitude": 21.2, "longitude": 81.4, "radius_km": 10},
    ).json()
    assert set(payload.keys()) == ANALYZE_LOCATION_KEYS


def test_analyze_contour_is_deterministic(client, sample_kml_bytes):
    first = client.post(
        "/analyzeContour",
        files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
    ).json()
    second = client.post(
        "/analyzeContour",
        files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
    ).json()

    for field in ["latitude", "longitude", "elevation_m", "slope_degrees", "suitability_score"]:
        assert first["pond_candidate"][field] == second["pond_candidate"][field]
    assert first["terrain"] == second["terrain"]
    assert first["catchment"]["area_m2"] == second["catchment"]["area_m2"]


def test_complete_workflow_contour_to_export(client):
    """KML -> DEM -> terrain -> hydrology -> catchment -> suitability -> recommendation -> export."""

    filename, data = fixtures.build_all()["valid_kml"]

    analysis = client.post("/analyzeContour", files={"file": (filename, data, "application/xml")})
    assert analysis.status_code == 200
    payload = analysis.json()
    assert payload["status"] == "success"

    # Data flowed through every stage.
    assert payload["input"]["contours_detected"] > 0
    assert payload["terrain"]["grid_rows"] > 1
    assert payload["dem_validation"]["status"] in {"good", "acceptable", "poor"}
    assert payload["catchment"]["area_m2"] > 0
    assert 0.0 <= payload["suitability"]["overall_score"] <= 1.0
    assert payload["pond_candidate"]["latitude"] is not None

    kml = client.post("/exportLocationKml", json=payload)
    assert kml.status_code == 200
    assert kml.headers["content-type"].startswith("application/vnd.google-earth.kml")
    assert kml.content.lstrip().startswith(b"<?xml")

    kmz = client.post("/exportLocationKmz", json=payload)
    assert kmz.status_code == 200
    assert kmz.headers["content-type"].startswith("application/vnd.google-earth.kmz")
    assert kmz.content[:2] == b"PK"


def test_gis_land_context_endpoint(client):
    response = client.post("/gis/land-context", json={"bbox": {"south": 21.0, "west": 81.0, "north": 21.1, "east": 81.1}})
    assert response.status_code == 200
    assert "water_bodies" in response.json()


def test_gis_land_context_rejects_invalid_bbox(client):
    response = client.post("/gis/land-context", json={"bbox": {"south": "not-a-number"}})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_BBOX"


def test_suitability_weights_sum_to_one():
    from backend.suitability import DEFAULT_WEIGHTS

    assert pytest.approx(sum(DEFAULT_WEIGHTS.values()), rel=1e-9) == 1.0


def test_rejection_constraints_are_exposed():
    from backend.recommendation import DEFAULT_REJECTION_CONSTRAINTS

    assert DEFAULT_REJECTION_CONSTRAINTS["max_slope_degrees"] > 0
    assert "min_catchment_area_m2" in DEFAULT_REJECTION_CONSTRAINTS
