"""Phase 02 — tests for the opt-in recommendation-engine bridge.

Covers:

* the ``POND_RECOMMENDATION_ENGINE`` flag parsing (default OFF);
* the adapter (``backend.pipeline.engine_analysis``) mapping engine output onto
  the existing response blocks, including real distances / feasibility /
  confidence and the six engine components;
* the fallback (adapter returns ``None`` when there are no candidates);
* API behaviour through ``/analyzeContour`` and ``/analyzeLocation`` under BOTH
  flag states, with the top-level response schema unchanged.

Everything is offline: land context and rainfall are the shared conftest fakes /
an injected provider, and the engine's disk cache is disabled by the adapter.
"""

from __future__ import annotations

import numpy as np
import pytest

import fixtures
from backend import pipeline
from backend.candidates import CandidateConfig, CandidateSet, generate_candidates
from backend.hydrology import analyze_hydrology
from backend.models import DEM

# Locked top-level schemas (kept identical regardless of the flag).
CONTOUR_TOP_KEYS = {
    "status", "input", "contour_diagnostics", "terrain", "dem_validation",
    "pond_candidate", "alternative_candidates", "catchment", "recommended_location",
    "catchment_area", "suitability", "recommendation", "method", "warnings",
    "error_code", "message", "enrichment",
}
LOCATION_TOP_KEYS = {
    "center", "min_latitude", "max_latitude", "min_longitude", "max_longitude",
    "radius_km", "status", "dem", "terrain", "hydrology", "catchment", "suitability",
    "pond_candidate", "alternative_candidates", "rainfall", "rainfall_runoff",
    "land_features", "recommendation", "warnings", "error_code", "message",
}
LEGACY_COMPONENT_KEYS = {
    "slope", "catchment_area", "rainfall", "water_distance", "road_distance",
    "building_distance", "land_use", "road_building_distance",
}
ENGINE_FIELDS = {"confidence", "feasibility", "road_distance_m", "building_distance_m", "water_distance_m"}


# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------

def test_flag_defaults_to_off(monkeypatch):
    monkeypatch.delenv(pipeline.ENGINE_ENV, raising=False)
    assert pipeline.recommendation_engine_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "On", " true "])
def test_flag_truthy_values(monkeypatch, value):
    monkeypatch.setenv(pipeline.ENGINE_ENV, value)
    assert pipeline.recommendation_engine_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", "", "banana"])
def test_flag_falsy_values(monkeypatch, value):
    monkeypatch.setenv(pipeline.ENGINE_ENV, value)
    assert pipeline.recommendation_engine_enabled() is False


@pytest.mark.parametrize("value", ["1", "0"])
def test_flag_env_overrides_settings(monkeypatch, value):
    # An explicit env var wins over the (default-off) Settings value.
    monkeypatch.setenv(pipeline.ENGINE_ENV, value)
    assert pipeline.recommendation_engine_enabled() is (value == "1")


# ---------------------------------------------------------------------------
# Adapter (unit)
# ---------------------------------------------------------------------------

def _plane_dem(rows: int = 8, cols: int = 8, resolution_m: float = 30.0):
    elevation = 100.0 + np.arange(rows)[:, None] * 0.5 + np.arange(cols)[None, :] * 0.25
    valid = np.isfinite(elevation)
    hydrology = analyze_hydrology(elevation, resolution_m, valid)
    dem = DEM(
        elevation=elevation,
        x=np.arange(cols) * resolution_m,
        y=np.arange(rows) * resolution_m,
        resolution_m=resolution_m,
        crs="EPSG:32644",
        slope_degrees=hydrology.slope_degrees,
        valid_mask=valid,
    )
    return dem, hydrology


def test_adapter_returns_none_without_candidates(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "generate_candidates",
        lambda *a, **k: CandidateSet(
            candidates=[], rejected=[], seed_count=0, accepted_count=0, deduplicated_count=0
        ),
    )
    dem, hydrology = _plane_dem()
    assert pipeline.engine_analysis(dem, hydrology) is None


def test_adapter_maps_engine_output_onto_candidate(monkeypatch):
    dem, hydrology = _plane_dem()

    # A building ~44 m north of the strongest candidate => a real, finite distance.
    seed = generate_candidates(hydrology, dem, config=CandidateConfig()).candidates[0]
    lon0, lat0 = seed["longitude"], seed["latitude"]

    def provider(south, west, north, east):  # noqa: ARG001
        return {
            "bbox": {"south": south, "west": west, "north": north, "east": east},
            "water_bodies": [],
            "roads": [],
            "buildings": [
                {"type": "node", "id": 1, "tags": {"building": "yes"},
                 "lat": lat0 + 0.0004, "lon": lon0}
            ],
            "source": "fake-provider",
            "provider": "fake",
            "status": "available",
        }

    def fake_rainfall(latitude, longitude, start_date, end_date, timeout=30.0):  # noqa: ARG001
        return {
            "latitude": latitude, "longitude": longitude, "timezone": "UTC",
            "daily_rainfall_mm": [100.0, 50.0], "daily_dates": ["2023-07-01", "2023-08-01"],
            "total_precipitation_mm": 150.0, "source": "fake-rainfall",
        }

    import backend.rainfall_runoff as rainfall_runoff

    monkeypatch.setattr(rainfall_runoff, "fetch_historical_rainfall", fake_rainfall, raising=False)

    result = pipeline.engine_analysis(dem, hydrology, land_provider=provider)
    assert result is not None

    candidate = result["pond_candidate"]
    assert ENGINE_FIELDS.issubset(candidate.keys())
    assert 0.0 <= candidate["suitability_score"] <= 1.0
    assert candidate["feasibility"] in {"feasible", "rejected", "unknown"}

    # A real distance was computed (not a fabricated legacy constant).
    all_candidates = [candidate, *result["alternative_candidates"]]
    building_distances = [c["building_distance_m"] for c in all_candidates]
    assert any(d is not None for d in building_distances)

    # Engine six-component suitability.
    assert set(result["suitability"]["component_scores"]) == set(pipeline.ENGINE_COMPONENT_KEYS)
    assert set(result["recommendation"].keys()) == {
        "best_location", "alternatives", "suitability_score", "explanation"
    }
    assert result["diagnostics"]["candidate_count"] >= 1


def test_adapter_is_deterministic(monkeypatch):
    dem, hydrology = _plane_dem()

    def provider(south, west, north, east):  # noqa: ARG001
        return {
            "bbox": {"south": south, "west": west, "north": north, "east": east},
            "water_bodies": [], "roads": [], "buildings": [],
            "source": "fake", "provider": "fake", "status": "available",
        }

    first = pipeline.engine_analysis(dem, hydrology, land_provider=provider)
    second = pipeline.engine_analysis(dem, hydrology, land_provider=provider)
    assert first is not None and second is not None
    assert first["pond_candidate"] == second["pond_candidate"]
    assert first["suitability"] == second["suitability"]


# ---------------------------------------------------------------------------
# API integration — flag OFF (legacy) vs ON (engine)
# ---------------------------------------------------------------------------

def test_api_flag_off_keeps_legacy_contour(client, monkeypatch):
    monkeypatch.setenv(pipeline.ENGINE_ENV, "false")
    name, data = fixtures.build_all()["valid_kml"]
    payload = client.post("/analyzeContour", files={"file": (name, data, "application/xml")}).json()

    assert payload["status"] == "success"
    assert set(payload.keys()) == CONTOUR_TOP_KEYS
    assert not (ENGINE_FIELDS & payload["pond_candidate"].keys())
    assert set(payload["suitability"]["component_scores"]) == LEGACY_COMPONENT_KEYS
    # Legacy path does not annotate enrichment with the engine block.
    assert "recommendation_engine" not in payload["enrichment"]


def test_api_flag_off_keeps_legacy_location(client, monkeypatch):
    monkeypatch.setenv(pipeline.ENGINE_ENV, "false")
    payload = client.post(
        "/analyzeLocation", json={"latitude": 21.25, "longitude": 81.30, "radius_km": 3}
    ).json()

    assert payload["status"] == "success"
    assert set(payload.keys()) == LOCATION_TOP_KEYS
    assert not (ENGINE_FIELDS & payload["pond_candidate"].keys())
    assert set(payload["suitability"]["component_scores"]) == LEGACY_COMPONENT_KEYS


def test_api_flag_on_uses_engine_for_contour(client, monkeypatch):
    monkeypatch.setenv(pipeline.ENGINE_ENV, "true")
    name, data = fixtures.build_all()["valid_kml"]
    payload = client.post("/analyzeContour", files={"file": (name, data, "application/xml")}).json()

    assert payload["status"] == "success"
    # Top-level schema is unchanged by the flag.
    assert set(payload.keys()) == CONTOUR_TOP_KEYS
    assert ENGINE_FIELDS.issubset(payload["pond_candidate"].keys())
    assert set(payload["suitability"]["component_scores"]) == set(pipeline.ENGINE_COMPONENT_KEYS)
    assert 0.0 <= payload["suitability"]["overall_score"] <= 1.0
    assert payload["recommendation"]["best_location"]["source"] == "recommendation_engine"
    assert "recommendation_engine" in payload["enrichment"]


def test_api_flag_on_uses_engine_for_location(client, monkeypatch):
    monkeypatch.setenv(pipeline.ENGINE_ENV, "true")
    payload = client.post(
        "/analyzeLocation", json={"latitude": 21.25, "longitude": 81.30, "radius_km": 3}
    ).json()

    assert payload["status"] == "success"
    assert set(payload.keys()) == LOCATION_TOP_KEYS
    assert ENGINE_FIELDS.issubset(payload["pond_candidate"].keys())
    assert set(payload["suitability"]["component_scores"]) == set(pipeline.ENGINE_COMPONENT_KEYS)


def test_api_flag_on_is_deterministic(client, monkeypatch):
    monkeypatch.setenv(pipeline.ENGINE_ENV, "true")
    name, data = fixtures.build_all()["valid_kml"]
    first = client.post("/analyzeContour", files={"file": (name, data, "application/xml")}).json()
    second = client.post("/analyzeContour", files={"file": (name, data, "application/xml")}).json()
    assert first["pond_candidate"] == second["pond_candidate"]
    assert first["suitability"] == second["suitability"]


# ---------------------------------------------------------------------------
# Graceful degradation -> actionable warnings (Phase 05)
# ---------------------------------------------------------------------------

def test_adapter_reports_warning_when_geographic_fails(monkeypatch):
    dem, hydrology = _plane_dem()

    def boom(*_a, **_k):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(pipeline, "enrich_candidates", boom)
    result = pipeline.engine_analysis(dem, hydrology)

    assert result is not None
    assert any("Geographic context unavailable" in w for w in result["warnings"])


def test_adapter_reports_warning_when_rainfall_fails(monkeypatch):
    dem, hydrology = _plane_dem()

    def provider(south, west, north, east):  # noqa: ARG001
        return {
            "bbox": {"south": south, "west": west, "north": north, "east": east},
            "water_bodies": [], "roads": [], "buildings": [],
            "source": "fake", "provider": "fake", "status": "available",
        }

    def boom(*_a, **_k):
        raise RuntimeError("rain down")

    monkeypatch.setattr(pipeline, "_compute_runoff", boom)
    result = pipeline.engine_analysis(dem, hydrology, land_provider=provider)

    assert result is not None
    assert any("Rainfall/runoff data unavailable" in w for w in result["warnings"])


def test_api_flag_on_surfaces_engine_warnings(client, monkeypatch):
    monkeypatch.setenv(pipeline.ENGINE_ENV, "true")

    def boom(*_a, **_k):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(pipeline, "enrich_candidates", boom)

    name, data = fixtures.build_all()["valid_kml"]
    payload = client.post("/analyzeContour", files={"file": (name, data, "application/xml")}).json()

    assert payload["status"] == "success"
    assert isinstance(payload["warnings"], list)
    assert any("Geographic context unavailable" in w for w in payload["warnings"])
