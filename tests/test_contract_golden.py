"""Golden contract & regression tests (Phase 1).

These lock the *externally observable* contract of the analysis and export
endpoints before the recommendation engine is wired in (Phase 2). They should
keep passing through later phases: if a change is intentional, update these
expectations deliberately rather than loosening them.

Everything runs offline through the shared `client` fixture and the fixtures in
``tests/fixtures.py`` (``tests/conftest.py`` replaces every external service with
a deterministic fake and disables the local OSM provider by default).

Documented inconsistencies in the CURRENT behaviour (locked as-is here, NOT
fixed — see ``docs/baseline.md``):

* ``suitability.component_scores`` includes ``building_distance`` and the derived
  ``road_building_distance``, but ``suitability.weights`` omits both.
* ``pond_candidate`` / ``alternative_candidates`` expose only
  ``suitability_score`` — they do NOT carry ``feasibility`` or real
  road/building/water distances (those live in the dormant engine, not the live
  path).
* ``/analyze``, ``/analyzeContour/enriched`` and ``/recommendPond`` still exist
  (the project docs claim they were removed); the latter two re-run
  ``analyze_contour``.
* The export endpoints accept any JSON object and degrade gracefully: they do
  NOT return ``400 INVALID_ANALYSIS_RESULT`` for a payload missing analysis
  fields.
"""

from __future__ import annotations

import io
import zipfile
import xml.etree.ElementTree as ET

import pytest

import fixtures

# ---------------------------------------------------------------------------
# Locked schema shapes
# ---------------------------------------------------------------------------

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

# Nested key sets that later phases must preserve.
POND_CANDIDATE_KEYS = {
    "id",
    "label",
    "latitude",
    "longitude",
    "elevation_m",
    "slope_degrees",
    "catchment_area_m2",
    "catchment_area",
    "suitability_score",
}
CATCHMENT_AREA_KEYS = {"area_m2", "area_hectares", "area_km2"}
CATCHMENT_CONTOUR_KEYS = {"area_m2", "area_hectares", "area_km2", "cell_count", "boundary"}
SUITABILITY_KEYS = {"component_scores", "weights", "overall_score"}
SUITABILITY_COMPONENT_KEYS = {
    "slope",
    "catchment_area",
    "rainfall",
    "water_distance",
    "road_distance",
    "building_distance",
    "land_use",
    "road_building_distance",
}
SUITABILITY_WEIGHT_KEYS = {
    "slope",
    "catchment_area",
    "rainfall",
    "water_distance",
    "road_distance",
    "land_use",
}
RECOMMENDATION_KEYS = {"best_location", "alternatives", "suitability_score", "explanation"}
ENRICHMENT_KEYS = {"status", "warnings", "dem", "rainfall", "land_context", "place_search", "geocoding"}
TERRAIN_KEYS = {
    "grid_resolution_m",
    "grid_rows",
    "grid_columns",
    "min_elevation_m",
    "max_elevation_m",
    "mean_elevation_m",
    "mean_slope_degrees",
    "max_slope_degrees",
    "valid_cell_fraction",
}
INPUT_KEYS = {
    "filename",
    "contours_detected",
    "elevation_min_m",
    "elevation_max_m",
    "coordinate_system",
}

# Current live route surface (documented in docs/baseline.md).
EXPECTED_ROUTE_PATHS = {
    "/",
    "/health",
    "/gis/land-context",
    "/analyze",
    "/analyzeLocation",
    "/analyzeContour",
    "/analyzeContour/enriched",
    "/exportLocationKml",
    "/exportLocationKmz",
    "/recommendPond",
}


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_kml():
    return fixtures.build_all()["valid_kml"]


def _post_contour(client, name, data):
    return client.post("/analyzeContour", files={"file": (name, data, "application/xml")})


def _post_location(client, **overrides):
    payload = {"latitude": 21.25, "longitude": 81.30, "radius_km": 3}
    payload.update(overrides)
    return client.post("/analyzeLocation", json=payload)


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------

def test_analyze_contour_top_level_schema_is_locked(client, valid_kml):
    payload = _post_contour(client, *valid_kml).json()
    assert set(payload.keys()) == ANALYZE_CONTOUR_KEYS


def test_analyze_contour_nested_schemas_are_locked(client, valid_kml):
    payload = _post_contour(client, *valid_kml).json()

    assert set(payload["input"].keys()) == INPUT_KEYS
    assert set(payload["terrain"].keys()) == TERRAIN_KEYS
    assert set(payload["pond_candidate"].keys()) == POND_CANDIDATE_KEYS
    assert set(payload["pond_candidate"]["catchment_area"].keys()) == CATCHMENT_AREA_KEYS
    assert set(payload["catchment"].keys()) == CATCHMENT_CONTOUR_KEYS
    assert set(payload["suitability"].keys()) == SUITABILITY_KEYS
    assert set(payload["recommendation"].keys()) == RECOMMENDATION_KEYS
    assert set(payload["enrichment"].keys()) == ENRICHMENT_KEYS
    assert isinstance(payload["alternative_candidates"], list)


def test_analyze_location_top_level_schema_is_locked(client):
    payload = _post_location(client).json()
    assert set(payload.keys()) == ANALYZE_LOCATION_KEYS


def test_analyze_location_nested_schemas_are_locked(client):
    payload = _post_location(client).json()

    assert set(payload["center"].keys()) == {"latitude", "longitude"}
    assert set(payload["pond_candidate"].keys()) == POND_CANDIDATE_KEYS
    assert set(payload["suitability"].keys()) == SUITABILITY_KEYS
    assert set(payload["recommendation"].keys()) == RECOMMENDATION_KEYS
    # Location catchment additionally carries validation + geometry.
    assert {"validation", "geometry"}.issubset(payload["catchment"].keys())
    assert {"preprocessing", "depressions", "outlet", "validation"}.issubset(
        payload["hydrology"].keys()
    )


# ---------------------------------------------------------------------------
# Documented quirks (locked intentionally)
# ---------------------------------------------------------------------------

def test_suitability_component_scores_superset_of_weights(client, valid_kml):
    # Locks the current quirk: component_scores has two keys weights does not.
    suitability = _post_contour(client, *valid_kml).json()["suitability"]
    assert set(suitability["component_scores"].keys()) == SUITABILITY_COMPONENT_KEYS
    assert set(suitability["weights"].keys()) == SUITABILITY_WEIGHT_KEYS
    assert set(suitability["weights"]).issubset(set(suitability["component_scores"]))
    assert {"building_distance", "road_building_distance"} == (
        set(suitability["component_scores"]) - set(suitability["weights"])
    )


def test_pond_candidate_does_not_expose_engine_fields(client, valid_kml):
    # The live path does not surface feasibility or real distances; the engine
    # rewrite (Phase 2) is expected to *add* these, at which point this test is
    # updated deliberately.
    candidate = _post_contour(client, *valid_kml).json()["pond_candidate"]
    for absent in ("feasibility", "feasibility_reason", "road_distance_m",
                   "building_distance_m", "water_distance_m", "confidence"):
        assert absent not in candidate


def test_legacy_duplicate_endpoints_still_registered(client):
    # Documents that the docs' "removed" endpoints are still present.
    paths = set(client.get("/openapi.json").json()["paths"].keys())
    assert {"/analyze", "/analyzeContour/enriched", "/recommendPond"}.issubset(paths)


def test_route_surface_contains_expected_paths(client):
    paths = set(client.get("/openapi.json").json()["paths"].keys())
    assert EXPECTED_ROUTE_PATHS.issubset(paths)


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

def test_analyze_contour_invariants(client, valid_kml):
    payload = _post_contour(client, *valid_kml).json()

    assert payload["status"] == "success"
    assert payload["error_code"] is None
    assert payload["message"] is None
    assert payload["input"]["contours_detected"] > 0
    assert payload["terrain"]["grid_rows"] > 1
    assert payload["terrain"]["grid_columns"] > 1
    assert payload["catchment"]["area_m2"] >= 0.0
    assert payload["catchment"]["cell_count"] >= 0
    assert payload["dem_validation"]["status"] in {"good", "acceptable", "poor", "invalid"}


def test_suitability_scores_are_normalized_and_weights_sum_to_one(client, valid_kml):
    suitability = _post_contour(client, *valid_kml).json()["suitability"]

    assert 0.0 <= suitability["overall_score"] <= 1.0
    for key, value in suitability["component_scores"].items():
        assert 0.0 <= value <= 1.0, f"component {key} out of range: {value}"
    assert sum(suitability["weights"].values()) == pytest.approx(1.0, abs=1e-9)


def test_pond_candidate_coordinates_are_finite(client, valid_kml):
    candidate = _post_contour(client, *valid_kml).json()["pond_candidate"]
    assert candidate["latitude"] is not None
    assert candidate["longitude"] is not None
    assert -90.0 <= candidate["latitude"] <= 90.0
    assert -180.0 <= candidate["longitude"] <= 180.0
    assert candidate["id"] == "primary"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_analyze_contour_is_fully_deterministic(client, valid_kml):
    first = _post_contour(client, *valid_kml).json()
    second = _post_contour(client, *valid_kml).json()
    # Mocked external services + deterministic pipeline => byte-equal JSON.
    assert first == second


def test_analyze_location_is_fully_deterministic(client):
    first = _post_location(client).json()
    second = _post_location(client).json()
    assert first == second


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_unsupported_extension_returns_400(client):
    response = client.post("/analyzeContour", files={"file": ("notes.txt", b"x", "text/plain")})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "UNSUPPORTED_FILE_TYPE"


def test_empty_file_returns_400(client):
    response = client.post("/analyzeContour", files={"file": ("empty.kml", b"", "application/xml")})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "EMPTY_FILE"


def test_malformed_kml_returns_422(client):
    name, data = fixtures.build_all()["malformed_kml"]
    response = _post_contour(client, name, data)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_CONTOUR_DATA"


def test_corrupt_kmz_returns_422(client):
    name, data = fixtures.build_all()["corrupt_kmz"]
    response = client.post("/analyzeContour", files={"file": (name, data, "application/zip")})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_CONTOUR_DATA"


def test_oversized_upload_returns_413(client, valid_kml, monkeypatch):
    import main

    monkeypatch.setattr(main, "MAX_FILE_SIZE_BYTES", 8, raising=True)
    response = _post_contour(client, *valid_kml)
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "FILE_TOO_LARGE"


def test_invalid_bbox_returns_400(client):
    response = client.post("/gis/land-context", json={"bbox": {"south": "not-a-number"}})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_BBOX"


def test_inverted_bbox_is_rejected_by_the_provider_not_the_route(client, monkeypatch):
    # Bbox validity is enforced by the land provider (land_service._validate_bbox),
    # NOT by the route: with the offline fake provider (which ignores the bbox)
    # POST /gis/land-context returns 200. The real provider raises LandError, which
    # the route maps to 400 INVALID_BBOX. This test locks both halves.
    import main
    from services.land_service import LandError, _validate_bbox

    with pytest.raises(LandError):
        _validate_bbox(south=22.0, north=21.0, west=81.0, east=81.1)

    def reject(*_args, **_kwargs):
        raise LandError("south must be smaller than north.")

    monkeypatch.setattr(main, "fetch_land_context", reject, raising=True)
    response = client.post(
        "/gis/land-context",
        json={"bbox": {"south": 22.0, "west": 81.0, "north": 21.0, "east": 81.1}},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_BBOX"


def test_invalid_location_returns_422(client):
    response = _post_location(client, latitude=999)
    assert response.status_code == 422


def test_dem_fetch_failure_returns_502(client, monkeypatch):
    import main
    from services.opentopography_service import OpenTopographyRetryableError

    def boom(*_args, **_kwargs):
        raise OpenTopographyRetryableError("simulated outage")

    monkeypatch.setattr(main, "get_dem_for_bbox", boom, raising=True)
    response = _post_location(client)
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "DEM_FETCH_TIMEOUT"


# ---------------------------------------------------------------------------
# GIS overlay contract
# ---------------------------------------------------------------------------

def test_gis_land_context_contract(client):
    response = client.post(
        "/gis/land-context",
        json={"bbox": {"south": 21.0, "west": 81.0, "north": 21.1, "east": 81.1}},
    )
    assert response.status_code == 200
    body = response.json()
    assert {"status", "water_bodies", "roads", "buildings", "source"}.issubset(body.keys())
    for key in ("water_bodies", "roads", "buildings"):
        assert isinstance(body[key], list)


# ---------------------------------------------------------------------------
# Service metadata
# ---------------------------------------------------------------------------

def test_health_contract(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_service_descriptor_contract(client):
    response = client.get("/", headers={"accept": "application/json"})
    assert response.status_code == 200
    body = response.json()
    assert {"service", "status", "version", "routes"} == set(body.keys())
    assert {"analyze_contour", "analyze_location", "export_kml", "export_kmz", "health"} == set(
        body["routes"].keys()
    )


def test_root_redirects_browsers_to_ui(client):
    response = client.get("/", headers={"accept": "text/html"}, follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"


# ---------------------------------------------------------------------------
# KML / KMZ export contracts
# ---------------------------------------------------------------------------

def _analysis_payload(client, valid_kml):
    return _post_contour(client, *valid_kml).json()


def test_export_kml_contract(client, valid_kml):
    payload = _analysis_payload(client, valid_kml)
    response = client.post("/exportLocationKml", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/vnd.google-earth.kml+xml")
    assert "attachment" in response.headers["content-disposition"]
    assert "pond-analysis.kml" in response.headers["content-disposition"]
    root = ET.fromstring(response.content)
    assert root.tag.endswith("}kml")
    assert root.findall(".//{http://www.opengis.net/kml/2.2}Placemark")


def test_export_kmz_contract(client, valid_kml):
    payload = _analysis_payload(client, valid_kml)
    response = client.post("/exportLocationKmz", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/vnd.google-earth.kmz")
    assert "pond-analysis.kmz" in response.headers["content-disposition"]
    assert response.content[:2] == b"PK"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert "doc.kml" in archive.namelist()


def test_kmz_contains_the_kml_document(client, valid_kml):
    payload = _analysis_payload(client, valid_kml)
    kmz = client.post("/exportLocationKmz", json=payload)
    with zipfile.ZipFile(io.BytesIO(kmz.content)) as archive:
        doc = archive.read("doc.kml")
    assert doc.lstrip().startswith(b"<?xml")


def test_export_accepts_arbitrary_object_gracefully(client):
    # Locks current behaviour: exports do NOT 400 on a payload without the
    # expected analysis fields; they return a minimal (empty) KML instead.
    response = client.post("/exportLocationKml", json={"unexpected": 1})
    assert response.status_code == 200
    root = ET.fromstring(response.content)
    assert root.tag.endswith("}kml")


# ---------------------------------------------------------------------------
# Legacy / duplicate endpoints
# ---------------------------------------------------------------------------
#
# These endpoints still exist (the project docs claim they were removed) and
# have their own response contracts. `/analyzeContour/enriched` and
# `/recommendPond` re-run the full `analyze_contour` pipeline and then overwrite
# `pond_candidate` with the contour-extent centroid `{latitude, longitude,
# source}` — a DIFFERENT (smaller) shape than `/analyzeContour`'s candidate.
# Locked as-is; see docs/baseline.md.

ANALYZE_KEYS = {
    "status",
    "input",
    "contour_diagnostics",
    "terrain",
    "dem",
    "hydrology",
    "catchment",
    "rainfall",
    "land_features",
    "recommendation",
    "pond_candidate",
    "alternative_candidates",
    "suitability",
    "metadata",
    "warnings",
    "error_code",
    "message",
}
ANALYZE_DEM_KEYS = {"quality", "shape", "resolution_m", "crs"}
ANALYZE_METADATA_KEYS = {"data_sources", "methods"}

RECOMMEND_POND_KEYS = {
    "status",
    "input_filename",
    "pond_candidate",
    "catchment",
    "recommended",
    "alternatives",
    "warnings",
    "enrichment",
}
RECOMMENDED_KEYS = {
    "id",
    "latitude",
    "longitude",
    "suitability",
    "catchment_area",
    "explanation",
    "alternatives",
}

RECOMMENDED_LOCATION_KEYS = {"latitude", "longitude", "source"}


def test_analyze_endpoint_contract(client, valid_kml):
    response = client.post(
        "/analyze", files={"file": (valid_kml[0], valid_kml[1], "application/xml")}
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "success"
    assert set(payload.keys()) == ANALYZE_KEYS
    assert set(payload["dem"].keys()) == ANALYZE_DEM_KEYS
    assert set(payload["metadata"].keys()) == ANALYZE_METADATA_KEYS
    assert set(payload["suitability"].keys()) == SUITABILITY_KEYS
    assert set(payload["recommendation"].keys()) == RECOMMENDATION_KEYS
    assert set(payload["pond_candidate"].keys()) == POND_CANDIDATE_KEYS


def test_analyze_contour_enriched_endpoint_contract(client, valid_kml):
    response = client.post(
        "/analyzeContour/enriched", files={"file": (valid_kml[0], valid_kml[1], "application/xml")}
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "success"
    # Same top-level shape as /analyzeContour...
    assert set(payload.keys()) == ANALYZE_CONTOUR_KEYS
    # ...but pond_candidate is the centroid recommended_location, not a candidate.
    assert set(payload["pond_candidate"].keys()) == RECOMMENDED_LOCATION_KEYS
    # ...and the catchment block has no boundary (unlike /analyzeContour).
    assert set(payload["catchment"].keys()) == {"area_m2", "area_hectares", "area_km2", "cell_count"}


def test_analyze_contour_and_enriched_have_different_pond_candidate_shapes(client, valid_kml):
    plain = _post_contour(client, *valid_kml).json()
    enriched = client.post(
        "/analyzeContour/enriched", files={"file": (valid_kml[0], valid_kml[1], "application/xml")}
    ).json()
    assert set(plain["pond_candidate"].keys()) == POND_CANDIDATE_KEYS
    assert set(enriched["pond_candidate"].keys()) == RECOMMENDED_LOCATION_KEYS
    assert plain["pond_candidate"] != enriched["pond_candidate"]


def test_recommend_pond_endpoint_contract(client, valid_kml):
    response = client.post(
        "/recommendPond", files={"file": (valid_kml[0], valid_kml[1], "application/xml")}
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "success"
    assert set(payload.keys()) == RECOMMEND_POND_KEYS
    assert set(payload["pond_candidate"].keys()) == RECOMMENDED_LOCATION_KEYS
    assert set(payload["recommended"].keys()) == RECOMMENDED_KEYS
    assert isinstance(payload["alternatives"], list)


def test_recommend_pond_is_deterministic(client, valid_kml):
    first = client.post(
        "/recommendPond", files={"file": (valid_kml[0], valid_kml[1], "application/xml")}
    ).json()
    second = client.post(
        "/recommendPond", files={"file": (valid_kml[0], valid_kml[1], "application/xml")}
    ).json()
    assert first == second


def test_analyze_endpoint_is_deterministic(client, valid_kml):
    first = client.post(
        "/analyze", files={"file": (valid_kml[0], valid_kml[1], "application/xml")}
    ).json()
    second = client.post(
        "/analyze", files={"file": (valid_kml[0], valid_kml[1], "application/xml")}
    ).json()
    assert first == second


def test_legacy_endpoints_share_contour_error_handling(client):
    name, data = fixtures.build_all()["malformed_kml"]
    for endpoint in ("/analyze", "/analyzeContour/enriched", "/recommendPond"):
        response = client.post(endpoint, files={"file": (name, data, "application/xml")})
        assert response.status_code == 422, endpoint
        assert response.json()["detail"]["code"] == "INVALID_CONTOUR_DATA", endpoint


def test_analyze_rejects_unsupported_extension(client):
    response = client.post("/analyze", files={"file": ("notes.txt", b"x", "text/plain")})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "UNSUPPORTED_FILE_TYPE"
