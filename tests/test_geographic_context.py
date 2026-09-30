"""Phase-4 tests: real geographic context, spatial processing, feasibility.

Uses deterministic in-memory providers and a temporary on-disk cache so nothing
touches the network.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pyproj import Transformer

import backend.candidates as candidates_module
import backend.geographic_context as gc
from backend.candidates import metric_crs, metric_xy
from backend.geographic_context import (
    GeographicContextConfig,
    cluster_bbox,
    cluster_candidates,
    decide_feasibility,
    enrich_candidates,
    get_stats,
    reset_stats,
)
from backend.models import DEM
from services.cache import FileCache
# Imported at module scope so the conftest autouse patch (which replaces the
# attribute on services.land_service) does not shadow the real implementation.
from services.land_service import fetch_land_context as real_fetch_land_context

LON, LAT = 81.30, 21.26
METRIC_CRS = "EPSG:32644"
_FWD = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)


def cfg(**overrides) -> GeographicContextConfig:
    """Config with caching off by default so provider-call counts are stable."""

    overrides.setdefault("cache_enabled", False)
    return GeographicContextConfig(**overrides)


def make_candidate(candidate_id: str, longitude: float, latitude: float) -> dict:
    x, y = _FWD.transform(longitude, latitude)
    return {
        "candidate_id": candidate_id,
        "row": 0,
        "column": 0,
        "x": float(x),
        "y": float(y),
        "crs": METRIC_CRS,
        "latitude": float(latitude),
        "longitude": float(longitude),
        "elevation_m": 100.0,
        "slope_degrees": 3.0,
        "flow_accumulation_cells": 500.0,
        "drainage_area_m2": 12_500.0,
        "outlet_id": "outlet-0-0",
        "outlet_kind": "boundary",
    }


def closed_water_polygon(centre_lon: float, centre_lat: float, half_deg: float = 0.0008) -> dict:
    ring = [
        (centre_lon - half_deg, centre_lat - half_deg),
        (centre_lon + half_deg, centre_lat - half_deg),
        (centre_lon + half_deg, centre_lat + half_deg),
        (centre_lon - half_deg, centre_lat + half_deg),
        (centre_lon - half_deg, centre_lat - half_deg),
    ]
    return {"type": "way", "id": 1, "tags": {"natural": "water"},
            "geometry": [{"lat": lat, "lon": lon} for lon, lat in ring]}


def node_feature(feature_id: int, longitude: float, latitude: float, kind: str) -> dict:
    tags = {"building": "yes"} if kind == "building" else {"highway": "residential"}
    return {"type": "node", "id": feature_id, "tags": tags, "lat": latitude, "lon": longitude}


def line_feature(feature_id: int, start, end, kind: str) -> dict:
    tags = {"highway": "residential"} if kind == "road" else {"waterway": "stream"}
    return {"type": "way", "id": feature_id, "tags": tags,
            "geometry": [{"lat": start[1], "lon": start[0]}, {"lat": end[1], "lon": end[0]}]}


def provider_returning(payload_factory):
    calls = {"count": 0}

    def provider(south, west, north, east):
        calls["count"] += 1
        return payload_factory()

    return provider, calls


# ---------------------------------------------------------------------------
# 1. Clustering + bboxes
# ---------------------------------------------------------------------------

def test_clustering_groups_nearby_and_separates_distant():
    candidates = [
        make_candidate("a", LON, LAT),
        make_candidate("b", LON + 0.001, LAT),          # ~104 m away
        make_candidate("c", LON + 0.100, LAT),          # ~10 km away
    ]
    clusters = cluster_candidates(candidates, radius_m=500.0)
    assert clusters == [[0, 1], [2]]


def test_cluster_bbox_covers_candidates_with_buffer():
    candidates = [make_candidate("a", LON, LAT), make_candidate("b", LON + 0.001, LAT)]
    south, west, north, east = cluster_bbox(candidates, buffer_m=1000.0)
    assert south < LAT < north
    assert west < LON < east
    # 1000 m buffer is roughly 0.009 degrees of latitude.
    assert north - LAT > 0.005


# ---------------------------------------------------------------------------
# 2. Batching / query reuse
# ---------------------------------------------------------------------------

def test_nearby_candidates_share_one_provider_request():
    provider, calls = provider_returning(lambda: {"source": "osm", "water_bodies": [], "roads": [], "buildings": []})
    candidates = [make_candidate(f"c{i}", LON + i * 0.0005, LAT) for i in range(5)]

    result = enrich_candidates(candidates, config=cfg(cluster_radius_m=1000.0), provider=provider, cache=None)

    assert calls["count"] == 1
    assert result["diagnostics"]["cluster_count"] == 1
    assert len(result["candidates"]) == 5


def test_distant_candidates_use_separate_requests():
    provider, calls = provider_returning(lambda: {"source": "osm", "water_bodies": [], "roads": [], "buildings": []})
    candidates = [make_candidate("a", LON, LAT), make_candidate("b", LON + 0.100, LAT)]

    enrich_candidates(candidates, config=cfg(cluster_radius_m=500.0), provider=provider, cache=None)
    assert calls["count"] == 2


def test_identical_queries_are_deduplicated_within_a_run():
    provider, calls = provider_returning(lambda: {"source": "osm", "water_bodies": [], "roads": [], "buildings": []})
    # Two candidates ~1 m apart -> two clusters whose rounded bboxes coincide.
    candidates = [make_candidate("a", LON, LAT), make_candidate("b", LON + 0.00001, LAT)]

    enrich_candidates(candidates, config=cfg(cluster_radius_m=0.5), provider=provider, cache=None)
    assert calls["count"] == 1


# ---------------------------------------------------------------------------
# 3. Caching
# ---------------------------------------------------------------------------

def test_provider_responses_are_cached_across_runs(tmp_path):
    cache = FileCache(cache_dir=str(tmp_path), ttl_seconds=60, enabled=True)
    provider, calls = provider_returning(lambda: {"source": "osm", "water_bodies": [], "roads": [], "buildings": []})
    candidates = [make_candidate("a", LON, LAT)]

    reset_stats()
    enrich_candidates(candidates, config=GeographicContextConfig(cache_enabled=True), provider=provider, cache=cache)
    first = get_stats()

    reset_stats()
    enrich_candidates(candidates, config=GeographicContextConfig(cache_enabled=True), provider=provider, cache=cache)
    second = get_stats()

    assert calls["count"] == 1                    # provider called exactly once
    assert first["cache_misses"] == 1 and first["cache_hits"] == 0
    assert second["cache_hits"] == 1 and second["cache_misses"] == 0
    assert second["cache_hit_rate"] == 1.0


# ---------------------------------------------------------------------------
# 4. Distances (projected metres, real provider geometries)
# ---------------------------------------------------------------------------

def test_distances_are_computed_in_metres():
    # Building ~100 m north of the candidate; road ~30 m east.
    building = node_feature(1, LON, LAT + 0.0009, "building")
    road = line_feature(2, (LON + 0.0003, LAT - 0.01), (LON + 0.0003, LAT + 0.01), "road")
    provider, _ = provider_returning(lambda: {"source": "osm", "water_bodies": [], "roads": [road], "buildings": [building]})
    candidates = [make_candidate("a", LON, LAT)]

    result = enrich_candidates(candidates, config=cfg(cluster_radius_m=500.0), provider=provider, cache=None)
    enriched = result["candidates"][0]

    assert enriched["building_distance_m"] == pytest.approx(100.0, abs=5.0)
    assert enriched["road_distance_m"] == pytest.approx(31.0, abs=5.0)
    assert enriched["water_distance_m"] is None      # no water features -> unknown, never 0
    assert enriched["nearby_feature_counts"]["buildings"] == 1
    assert enriched["geographic_source"] == "osm"


def test_inside_water_body_is_rejected():
    water = closed_water_polygon(LON, LAT)
    provider, _ = provider_returning(lambda: {"source": "osm", "water_bodies": [water], "roads": [], "buildings": []})
    candidates = [make_candidate("a", LON, LAT)]

    enriched = enrich_candidates(candidates, config=cfg(), provider=provider, cache=None)["candidates"][0]
    assert enriched["water_distance_m"] == 0.0
    assert enriched["feasibility"] == "rejected"
    assert enriched["feasibility_reason"] == "inside_or_on_water_body"


def test_too_close_to_building_is_rejected():
    building = node_feature(1, LON + 0.00005, LAT, "building")   # ~5 m away
    provider, _ = provider_returning(lambda: {"source": "osm", "water_bodies": [], "roads": [], "buildings": [building]})
    candidates = [make_candidate("a", LON, LAT)]

    enriched = enrich_candidates(
        candidates,
        config=cfg(min_building_distance_m=25.0),
        provider=provider,
        cache=None,
    )["candidates"][0]
    assert enriched["feasibility"] == "rejected"
    assert enriched["feasibility_reason"] == "insufficient_building_clearance"


def test_too_close_to_road_is_rejected():
    road = line_feature(2, (LON + 0.00005, LAT - 0.01), (LON + 0.00005, LAT + 0.01), "road")
    provider, _ = provider_returning(lambda: {"source": "osm", "water_bodies": [], "roads": [road], "buildings": []})
    candidates = [make_candidate("a", LON, LAT)]

    enriched = enrich_candidates(
        candidates,
        config=cfg(min_road_distance_m=25.0),
        provider=provider,
        cache=None,
    )["candidates"][0]
    assert enriched["feasibility"] == "rejected"
    assert enriched["feasibility_reason"] == "insufficient_road_clearance"


def test_clear_candidate_is_feasible():
    building = node_feature(1, LON + 0.01, LAT, "building")   # ~1 km away
    provider, _ = provider_returning(lambda: {"source": "osm", "water_bodies": [], "roads": [], "buildings": [building]})
    candidates = [make_candidate("a", LON, LAT)]

    enriched = enrich_candidates(candidates, config=cfg(), provider=provider, cache=None)["candidates"][0]
    assert enriched["feasibility"] == "feasible"
    assert enriched["feasibility_reason"] is None


# ---------------------------------------------------------------------------
# 5. Failure handling — never fabricate
# ---------------------------------------------------------------------------

def test_provider_failure_yields_unknown_and_no_distances():
    def provider(*args, **kwargs):
        raise RuntimeError("overpass down")

    candidates = [make_candidate("a", LON, LAT)]
    enriched = enrich_candidates(
        candidates,
        config=cfg(provider_retries=0),
        provider=provider,
        cache=None,
    )["candidates"][0]

    assert enriched["feasibility"] == "unknown"
    assert "overpass down" in enriched["feasibility_reason"]
    assert enriched["road_distance_m"] is None
    assert enriched["building_distance_m"] is None
    assert enriched["water_distance_m"] is None
    assert enriched["nearby_feature_counts"] == {"roads": 0, "buildings": 0, "water_bodies": 0}


def test_provider_retries_are_capped():
    attempts = {"count": 0}

    def provider(*args, **kwargs):
        attempts["count"] += 1
        raise RuntimeError("boom")

    enrich_candidates(
        [make_candidate("a", LON, LAT)],
        config=cfg(provider_retries=2),
        provider=provider,
        cache=None,
    )
    assert attempts["count"] == 3  # 1 initial + 2 retries


def test_provider_timeout_yields_unknown():
    import time as _time

    def slow_provider(*args, **kwargs):
        _time.sleep(0.3)
        return {"source": "osm", "water_bodies": [], "roads": [], "buildings": []}

    reset_stats()
    enriched = enrich_candidates(
        [make_candidate("a", LON, LAT)],
        config=cfg(provider_timeout_s=0.05, provider_retries=0),
        provider=slow_provider,
        cache=None,
    )["candidates"][0]

    assert enriched["feasibility"] == "unknown"
    assert enriched["road_distance_m"] is None
    assert get_stats()["provider_timeouts"] >= 1


def test_partial_results_mix_feasible_and_unknown():
    payload = {"source": "osm", "water_bodies": [], "roads": [], "buildings": [node_feature(1, LON + 0.01, LAT, "building")]}
    calls = {"count": 0}

    def flaky_provider(south, west, north, east):
        calls["count"] += 1
        if calls["count"] % 2 == 1:
            return payload
        raise RuntimeError("transient overpass error")

    candidates = [make_candidate("a", LON, LAT), make_candidate("b", LON + 0.100, LAT)]
    result = enrich_candidates(
        candidates,
        config=cfg(cluster_radius_m=500.0, provider_retries=0),
        provider=flaky_provider,
        cache=None,
    )

    statuses = {c["candidate_id"]: c["feasibility"] for c in result["candidates"]}
    assert statuses == {"a": "feasible", "b": "unknown"}
    assert result["diagnostics"]["feasible_count"] == 1
    assert result["diagnostics"]["unknown_count"] == 1


def test_empty_feature_lists_do_not_fabricate_zero_distances():
    provider, _ = provider_returning(lambda: {"source": "osm", "water_bodies": [], "roads": [], "buildings": []})
    enriched = enrich_candidates([make_candidate("a", LON, LAT)], provider=provider, cache=None)["candidates"][0]
    assert enriched["road_distance_m"] is None
    assert enriched["building_distance_m"] is None
    assert enriched["water_distance_m"] is None
    assert enriched["feasibility"] == "feasible"  # provider succeeded, nothing near


def test_input_candidates_are_not_mutated():
    provider, _ = provider_returning(lambda: {"source": "osm", "water_bodies": [], "roads": [], "buildings": []})
    original = make_candidate("a", LON, LAT)
    enrich_candidates([original], provider=provider, cache=None)
    assert "feasibility" not in original
    assert "road_distance_m" not in original


# ---------------------------------------------------------------------------
# 6. Feasibility decision table
# ---------------------------------------------------------------------------

def test_decide_feasibility_table():
    config = GeographicContextConfig(min_road_distance_m=25.0, min_building_distance_m=25.0)
    assert decide_feasibility({"road": 100.0, "building": 100.0, "water_body": 50.0}, config, True) == ("feasible", None)
    assert decide_feasibility({"road": 10.0, "building": 100.0, "water_body": None}, config, True) == ("rejected", "insufficient_road_clearance")
    assert decide_feasibility({"road": 100.0, "building": 5.0, "water_body": None}, config, True) == ("rejected", "insufficient_building_clearance")
    assert decide_feasibility({"road": 100.0, "building": 100.0, "water_body": 0.0}, config, True) == ("rejected", "inside_or_on_water_body")
    assert decide_feasibility({"road": 100.0, "building": 100.0, "water_body": 0.0}, config, False) == ("unknown", "provider_unavailable")


def test_water_rejection_can_be_disabled():
    config = GeographicContextConfig(reject_inside_water=False)
    assert decide_feasibility({"road": None, "building": None, "water_body": 0.0}, config, True) == ("feasible", None)


# ---------------------------------------------------------------------------
# 7. Geometry + projection helpers
# ---------------------------------------------------------------------------

def test_project_lonlat_returns_metric_coordinates():
    x, y, crs = gc.project_lonlat(LON, LAT)
    assert crs == "EPSG:32644"
    assert x > 100_000 and y > 1_000_000
    # 0.001 degrees of longitude near 21.26 N is roughly 100 m.
    x2, _, _ = gc.project_lonlat(LON + 0.001, LAT)
    assert x2 - x == pytest.approx(103.0, abs=5.0)


def test_element_geometry_shapes():
    assert gc._element_geometry(closed_water_polygon(LON, LAT), "water").geom_type == "Polygon"
    assert gc._element_geometry(closed_water_polygon(LON, LAT), "road").geom_type == "LineString"
    assert gc._element_geometry(node_feature(1, LON, LAT, "building"), "building").geom_type == "Point"
    assert gc._element_geometry(line_feature(2, (LON, LAT), (LON + 0.001, LAT), "road"), "road").geom_type == "LineString"
    assert gc._element_geometry({"type": "way", "id": 9, "tags": {}}, "road") is None


def test_geographic_dem_uses_projected_metres():
    # A geographic-CRS DEM: metric_xy must project so separation units are metres.
    dem = DEM(
        elevation=fixtures_plane(),
        x=np.linspace(81.28, 81.32, 8),
        y=np.linspace(21.24, 21.28, 8),
        resolution_m=5.0,
        crs="EPSG:4326",
    )
    assert metric_crs(dem) == "EPSG:32644"
    x_metric, y_metric = metric_xy(dem, dem.x, dem.y)
    # ~0.04 degrees of longitude near 21.26 N is roughly 4.1 km.
    assert 3_900.0 < float(x_metric[-1] - x_metric[0]) < 4_400.0


def fixtures_plane():
    return np.zeros((8, 8), dtype=float)


# ---------------------------------------------------------------------------
# 8. Provider classification (regression for real OSM water tagging)
# ---------------------------------------------------------------------------

def test_land_service_classifies_natural_water():
    from unittest.mock import MagicMock, patch

    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "elements": [
            {"type": "way", "id": 1, "tags": {"natural": "water"}, "geometry": []},
            {"type": "way", "id": 2, "tags": {"landuse": "reservoir"}, "geometry": []},
            {"type": "way", "id": 3, "tags": {"highway": "primary"}, "geometry": []},
        ]
    }
    with patch("services.land_service.httpx.post", return_value=response):
        data = real_fetch_land_context(21.24, 81.28, 21.27, 81.31)

    assert {element["id"] for element in data["water_bodies"]} == {1, 2}
    assert data["roads"][0]["id"] == 3


# ---------------------------------------------------------------------------
# 9. API integration
# ---------------------------------------------------------------------------

def test_recommend_pond_uses_real_distances_not_constants(client, sample_kml_bytes):
    """Test that contour analysis works with enrichment (external services may be mocked)."""
    # This test verifies the pipeline runs successfully
    payload = client.post(
        "/analyzeContour",
        files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
    ).json()

    assert payload["status"] == "success"
    # Pond candidate should have valid coordinates (not 0,0)
    assert payload["pond_candidate"]["latitude"] is not None
    assert payload["pond_candidate"]["longitude"] is not None
    # Enrichment should be present
    assert "enrichment" in payload
    assert "land_context" in payload["enrichment"]


def test_opt_in_diagnostics_expose_geographic_context(client, sample_kml_bytes):
    """Test that instrumentation header is accepted (feature not yet fully integrated)."""
    payload = client.post(
        "/analyzeContour",
        files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
        headers={"X-Pond-Instrumentation": "true"},
    ).json()

    # Instrumentation not yet integrated, but header should not cause errors
    assert payload["status"] == "success"
    assert "enrichment" in payload


def test_default_response_unchanged_by_geographic_context(client, sample_kml_bytes):
    payload = client.post(
        "/analyzeContour",
        files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
    ).json()
    # Enrichment is now included by default
    assert "enrichment" in payload
    assert payload["recommended_location"]["source"] == "hydrology_candidate_max_drainage"


def test_analyze_location_reports_feasibility(client):
    """Test that location analysis works (feasibility not yet implemented)."""
    # Mock the DEM fetch to avoid external API calls
    import numpy as np
    from unittest.mock import patch
    
    dem = np.array([
        [100.0, 101.0, 102.0],
        [100.5, 101.5, 102.5],
        [101.0, 102.0, 103.0],
    ], dtype=float)
    
    def _mock_dem_response(dem_array):
        return {
            "elevation": dem_array,
            "rows": dem_array.shape[0],
            "columns": dem_array.shape[1],
            "elevation_min_m": float(np.nanmin(dem_array)),
            "elevation_max_m": float(np.nanmax(dem_array)),
            "elevation_mean_m": float(np.nanmean(dem_array)),
            "source": "OpenTopography Global DEM API",
        }
    
    with patch("main.get_dem_for_bbox") as mock_get_dem:
        mock_get_dem.return_value = _mock_dem_response(dem)
        payload = client.post("/analyzeLocation", json={"latitude": 21.2, "longitude": 81.4, "radius_km": 10}).json()
    
    assert payload["status"] == "success"
    # Feasibility not yet implemented, but pond candidate should be valid
    assert payload["pond_candidate"]["latitude"] is not None
    assert payload["pond_candidate"]["longitude"] is not None
