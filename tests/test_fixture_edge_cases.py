"""Edge-case coverage across the API and the hydrology/catchment stages.

All external services are stubbed deterministically by ``conftest``.
"""

from __future__ import annotations

import numpy as np
import pytest

import fixtures
from backend.catchment import catchment_area, delineate_catchment
from backend.hydrology import analyze_hydrology, calculate_d8_flow_direction, calculate_flow_accumulation


# ---------------------------------------------------------------------------
# API-level contour fixture coverage
# ---------------------------------------------------------------------------

# (fixture name, expected HTTP status, expected "status" field or None)
API_CASES = [
    ("valid_kml", 200, "success"),
    ("valid_kmz", 200, "success"),
    ("sparse_contours", 200, "success"),
    ("irregular_contours", 200, "success"),
    ("flat_terrain", 200, None),
    ("steep_terrain", 200, None),
    ("large_input", 200, None),
    ("duplicate_points", 200, None),
    ("missing_elevation", 200, "success"),
    ("missing_coordinates", 200, "success"),
    ("extreme_elevations", 200, None),
    ("empty_file", 400, None),
    ("malformed_kml", 422, None),
    ("no_coordinates_text", 422, None),
    ("invalid_geometry", 422, None),
    # Phase 2: insufficient/degenerate contour geometry is now a controlled 422.
    ("insufficient_contours", 422, None),
    ("kmz_without_kml", 422, None),
    # Phase 2: a structurally corrupt KMZ is now a controlled 422.
    ("corrupt_kmz", 422, None),
]


@pytest.mark.parametrize("name, expected_status, expected_status_field", API_CASES)
def test_contour_fixture_endpoint(client, name, expected_status, expected_status_field):
    filename, data = fixtures.build_all()[name]
    response = client.post(
        "/analyzeContour",
        files={"file": (filename, data, "application/octet-stream")},
    )

    assert response.status_code == expected_status, response.text

    if expected_status == 200:
        payload = response.json()
        if expected_status_field is not None:
            assert payload["status"] == expected_status_field
        else:
            assert payload["status"] in {"success", "failed"}
        # Every 200 response must still carry a well-formed contract.
        assert "pond_candidate" in payload
        assert isinstance(payload.get("warnings", []), list)
    else:
        assert "detail" in response.json()


def test_missing_elevation_fixture_reports_only_valid_contours(client):
    filename, data = fixtures.build_all()["missing_elevation"]
    payload = client.post("/analyzeContour", files={"file": (filename, data, "application/xml")}).json()
    assert payload["input"]["contours_detected"] == 3


def test_extreme_elevations_preserved_in_response(client):
    filename, data = fixtures.build_all()["extreme_elevations"]
    response = client.post("/analyzeContour", files={"file": (filename, data, "application/xml")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["input"]["elevation_min_m"] <= -500.0
    assert payload["input"]["elevation_max_m"] >= 9000.0


def test_duplicate_points_fixture_produces_valid_analysis(client):
    filename, data = fixtures.build_all()["duplicate_points"]
    response = client.post("/analyzeContour", files={"file": (filename, data, "application/xml")})
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["dem_validation"]["score"], float)


# ---------------------------------------------------------------------------
# Hydrology / catchment edge cases (synthetic DEMs)
# ---------------------------------------------------------------------------

def test_flat_dem_is_filled_and_drains_to_boundary():
    dem = fixtures.flat_dem((5, 5))
    hydrology = analyze_hydrology(dem, 5.0, np.isfinite(dem))
    # Priority-Flood removes the interior flats; only the boundary drains off-grid.
    assert hydrology.sink_cells == 0
    assert hydrology.depressions["original_sink_count"] == 9
    assert hydrology.depressions["resolved_sink_count"] == 9
    assert hydrology.depressions["remaining_sink_count"] == 0
    assert hydrology.outlet_selection["primary"] is not None
    assert np.nanmax(hydrology.flow_accumulation) > 1


def test_sink_dem_raw_sink_is_resolved_by_filling():
    dem = fixtures.sink_dem()
    valid = np.isfinite(dem)
    _, raw_sink_mask, _ = calculate_d8_flow_direction(dem, 5.0, valid)
    assert raw_sink_mask[2, 2]  # the pit is an internal sink in the raw DEM

    hydrology = analyze_hydrology(dem, 5.0, valid)
    assert hydrology.sink_cells == 0
    assert hydrology.depressions["original_sink_count"] == 1
    assert hydrology.depressions["resolved_sink_count"] == 1
    assert hydrology.depressions["max_fill_depth_m"] > 0


def test_basin_dem_fills_and_drains_to_boundary():
    dem = fixtures.basin_dem(7)
    valid = np.isfinite(dem)
    hydrology = analyze_hydrology(dem, 5.0, valid)
    assert hydrology.depressions["original_sink_count"] == 1
    assert hydrology.sink_cells == 0
    assert hydrology.depressions["max_fill_depth_m"] > 0
    outlet = hydrology.outlet_selection["primary"]
    assert outlet is not None
    mask = delineate_catchment(hydrology.flow_direction, hydrology.valid_mask, outlet)
    stats = catchment_area(mask, 5.0)
    assert stats["cell_count"] >= 1


def test_nan_cells_are_excluded_from_valid_mask():
    dem = fixtures.nan_dem(fraction_rows=2)
    hydrology = analyze_hydrology(dem, 5.0, np.isfinite(dem))
    expected_valid = int(np.isfinite(dem).sum())
    assert hydrology.valid_cells == expected_valid
    assert hydrology.valid_cells < dem.size


def test_disconnected_dem_catchment_does_not_cross_invalid_barrier():
    dem = fixtures.disconnected_dem()
    valid = np.isfinite(dem)
    hydrology = analyze_hydrology(dem, 5.0, valid)

    # Outlet on the left side (columns 0-1 only).
    left_outlet = (4, 0)
    mask = delineate_catchment(hydrology.flow_direction, hydrology.valid_mask, left_outlet)
    assert not np.any(mask[:, 3:])
    assert not np.any(mask[:, 2])


def test_single_valid_cell_dem_yields_single_cell_catchment():
    dem = fixtures.single_valid_cell_dem()
    valid = np.isfinite(dem)
    hydrology = analyze_hydrology(dem, 5.0, valid)
    mask = delineate_catchment(hydrology.flow_direction, hydrology.valid_mask, (1, 1))
    stats = catchment_area(mask, 5.0)
    assert stats["cell_count"] == 1
    assert stats["area_m2"] == 25.0


def test_flow_cycle_leaves_accumulation_unresolved():
    flow = fixtures.cycle_flow_direction()
    accumulation = calculate_flow_accumulation(flow)
    assert np.all(np.isnan(accumulation))


def test_delineate_catchment_rejects_invalid_outlet():
    dem = fixtures.plane_dem(4, 4)
    hydrology = analyze_hydrology(dem, 5.0, np.isfinite(dem))
    with pytest.raises(ValueError):
        delineate_catchment(hydrology.flow_direction, hydrology.valid_mask, (99, 99))


# ---------------------------------------------------------------------------
# External-service failure / timeout handling
# ---------------------------------------------------------------------------

def test_dem_fetch_failure_returns_502(client, monkeypatch):
    import main

    def raise_timeout(*args, **kwargs):
        raise TimeoutError("simulated OpenTopography timeout")

    monkeypatch.setattr(main, "get_dem_for_bbox", raise_timeout)
    response = client.post("/analyzeLocation", json={"latitude": 21.2, "longitude": 81.4, "radius_km": 10})
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "DEM_FETCH_ERROR"


def test_all_external_services_failing_keeps_contour_pipeline_alive(client, monkeypatch, sample_kml_bytes):
    import main

    def raise_error(*args, **kwargs):
        raise RuntimeError("simulated external failure")

    for name in ["get_dem_for_bbox", "fetch_historical_rainfall", "fetch_land_context", "search_places", "geocode_place"]:
        monkeypatch.setattr(main, name, raise_error)

    response = client.post(
        "/analyzeContour",
        files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["enrichment"]["status"] in {"partial", "success"}
