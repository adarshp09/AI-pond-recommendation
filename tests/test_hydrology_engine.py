"""Phase-2 tests for the corrected DEM/hydrology/catchment engine.

These assert mathematical invariants (conservation, monotonicity, routing
validity, area consistency) rather than only HTTP status codes.
"""

from __future__ import annotations

import numpy as np
import pytest

import fixtures
from backend.catchment import (
    catchment_area,
    delineate_catchment,
    mask_to_polygon,
    validate_catchment_geometry,
)
from backend.depression import (
    boundary_ring,
    fill_depressions,
    preprocess_dem,
)
from backend.hydrology import (
    D8_OFFSETS,
    analyze_hydrology,
    calculate_d8_flow,
    calculate_d8_flow_direction,
    calculate_flow_accumulation,
    count_internal_sinks,
    downstream_flat_index,
    find_outlets,
)
from backend.hydrology_validation import validate_hydrology
from backend.hydrology_validation import validate_outlet


# ---------------------------------------------------------------------------
# 1. DEM preprocessing
# ---------------------------------------------------------------------------

def test_preprocess_reports_valid_and_nan_cells():
    dem = fixtures.nan_dem(fraction_rows=2)
    info = preprocess_dem(dem, np.isfinite(dem), 5.0)

    assert info.rows == 6 and info.columns == 6
    assert info.total_cells == 36
    assert info.valid_cells == 24
    assert info.nan_count == 12
    assert info.resolution_m == 5.0
    assert info.elevation_range_m > 0
    assert info.warnings


def test_preprocess_rejects_empty_and_invalid():
    with pytest.raises(ValueError):
        preprocess_dem(np.array([]), None, 5.0)
    with pytest.raises(ValueError):
        preprocess_dem(np.full((3, 3), np.nan), np.zeros((3, 3), dtype=bool), 5.0)
    with pytest.raises(ValueError):
        preprocess_dem(fixtures.plane_dem(4, 4), None, 0.0)


# ---------------------------------------------------------------------------
# 2. Depression / sink handling
# ---------------------------------------------------------------------------

def test_plane_has_no_sinks_and_is_not_modified():
    dem = fixtures.plane_dem(6, 6)
    valid = np.isfinite(dem)
    result = fill_depressions(dem, valid, 5.0)

    assert count_internal_sinks(dem, valid) == 0
    assert result.modified_cell_count == 0
    assert result.max_fill_depth_m == 0.0
    assert np.allclose(result.filled_elevation[valid], dem[valid])


def test_single_pit_is_filled_and_reported():
    dem = fixtures.sink_dem()
    valid = np.isfinite(dem)
    result = fill_depressions(dem, valid, 5.0)

    assert result.modified_cell_count == 1
    assert result.modified_mask[2, 2]
    assert result.max_fill_depth_m > 0
    assert result.filled_elevation[2, 2] > dem[2, 2]


def test_boundary_cells_are_never_filled():
    dem = fixtures.flat_dem((5, 5))
    result = fill_depressions(dem, np.isfinite(dem), 5.0)
    ring = boundary_ring(dem.shape)
    assert np.allclose(result.filled_elevation[ring], dem[ring])
    assert not np.any(result.modified_mask & ring)


def test_nodata_cells_are_never_modified():
    dem = fixtures.nan_dem(fraction_rows=2)
    valid = np.isfinite(dem)
    result = fill_depressions(dem, valid, 5.0)
    assert np.all(np.isnan(result.filled_elevation[~valid]))


def test_enclosed_region_is_not_filled_and_reported():
    dem = fixtures.single_valid_cell_dem()
    valid = np.isfinite(dem)
    result = fill_depressions(dem, valid, 5.0)
    assert result.unreachable_cell_count == 1
    assert result.modified_cell_count == 0
    assert result.filled_elevation[1, 1] == dem[1, 1]


def test_flat_dem_interior_is_resolved_deterministically():
    dem = fixtures.flat_dem((5, 5))
    valid = np.isfinite(dem)
    first = fill_depressions(dem, valid, 5.0)
    second = fill_depressions(dem, valid, 5.0)
    assert np.allclose(first.filled_elevation[valid], second.filled_elevation[valid])
    assert first.modified_cell_count == 9


def test_epsilon_must_be_positive():
    with pytest.raises(ValueError):
        fill_depressions(fixtures.plane_dem(4, 4), None, 5.0, epsilon_m=0.0)


# ---------------------------------------------------------------------------
# 3. D8 flow direction
# ---------------------------------------------------------------------------

def test_d8_encoding_flows_east_on_eastward_slope():
    dem = np.tile(np.arange(10, 0, -1, dtype=float), (5, 1))  # decreases eastward
    flow_direction, _, _ = calculate_d8_flow_direction(dem, 5.0)
    assert np.all(flow_direction[:, :-1][1:-1, :] == 2)  # interior flows E


def test_d8_equal_slope_tie_breaks_to_lowest_index():
    dem = np.full((3, 3), 10.0)
    dem[0, 0] = 5.0  # NW of centre -> direction 7
    dem[0, 2] = 5.0  # NE of centre -> direction 1
    flow_direction, _, _ = calculate_d8_flow_direction(dem, 5.0)
    assert flow_direction[1, 1] == 1  # NE wins the tie deterministically


def test_d8_never_routes_through_nodata():
    dem = fixtures.plane_dem(5, 5)
    valid = np.isfinite(dem)
    valid[2, 2] = False
    flow_direction, _, _ = calculate_d8_flow_direction(dem, 5.0, valid)

    assert flow_direction[2, 2] == -1
    rows, cols = dem.shape
    for r in range(rows):
        for c in range(cols):
            if valid[r, c] and flow_direction[r, c] >= 0:
                dr, dc = D8_OFFSETS[flow_direction[r, c]]
                assert valid[r + dr, c + dc]


def test_d8_is_deterministic_on_flat_terrain():
    dem = fixtures.flat_dem((6, 6))
    valid = np.isfinite(dem)
    filled = fill_depressions(dem, valid, 5.0).filled_elevation
    first, _, _ = calculate_d8_flow_direction(filled, 5.0, valid)
    second, _, _ = calculate_d8_flow_direction(filled, 5.0, valid)
    assert np.array_equal(first, second)


def test_d8_boundary_detection():
    dem = fixtures.plane_dem(4, 4)
    _, _, edge = calculate_d8_flow_direction(dem, 5.0)
    assert np.array_equal(edge, boundary_ring(dem.shape))


def test_calculate_d8_flow_returns_sink_mask_not_direction():
    dem = fixtures.sink_dem()
    flow_direction, second = calculate_d8_flow(dem, 5.0)

    # The second value is a boolean mask, not the flow direction.
    assert second.dtype == bool
    assert not np.array_equal(second, flow_direction)
    assert second.shape == dem.shape


# ---------------------------------------------------------------------------
# 4. Flow accumulation
# ---------------------------------------------------------------------------

def test_accumulation_conservation_and_monotonicity():
    dem = fixtures.plane_dem(6, 6)
    valid = np.isfinite(dem)
    flow_direction, _, _ = calculate_d8_flow_direction(dem, 5.0, valid)
    accumulation = calculate_flow_accumulation(flow_direction, valid)

    flat_down = downstream_flat_index(flow_direction, valid)
    acc_flat = accumulation.ravel()
    valid_flat = valid.ravel()

    contributions = np.zeros(acc_flat.size)
    sources = np.flatnonzero(valid_flat & (flat_down >= 0))
    np.add.at(contributions, flat_down[sources], acc_flat[sources])

    expected = np.where(valid_flat, 1.0, 0.0) + contributions
    assert np.allclose(acc_flat[valid_flat], expected[valid_flat])

    # Monotonicity: downstream accumulation strictly exceeds upstream.
    downstream_cells = flat_down[sources]
    assert np.all(acc_flat[downstream_cells] > acc_flat[sources])


def test_accumulation_bounds_and_cycle_detection():
    dem = fixtures.plane_dem(6, 6)
    valid = np.isfinite(dem)
    flow_direction, _, _ = calculate_d8_flow_direction(dem, 5.0, valid)
    accumulation = calculate_flow_accumulation(flow_direction, valid)

    finite = accumulation[valid]
    assert np.all(finite >= 1.0)
    assert np.max(finite) <= int(np.sum(valid))

    cycle = fixtures.cycle_flow_direction()
    cycle_accumulation = calculate_flow_accumulation(cycle)
    assert np.all(np.isnan(cycle_accumulation))


# ---------------------------------------------------------------------------
# 5. Outlet detection
# ---------------------------------------------------------------------------

def test_outlet_prefers_boundary_max_accumulation():
    dem = fixtures.plane_dem(6, 6)
    hydrology = analyze_hydrology(dem, 5.0, np.isfinite(dem))
    selection = hydrology.outlet_selection

    assert selection["primary"] is not None
    assert selection["kind"] == "boundary"
    assert selection["reason"] == "max_upstream_accumulation_boundary_outlet"

    primary = selection["primary"]
    # Plane drains to its lowest corner; that corner must have max accumulation.
    assert hydrology.flow_accumulation[primary] == hydrology.max_accumulation_cells


def test_outlet_is_deterministic():
    dem = fixtures.plane_dem(6, 6)
    valid = np.isfinite(dem)
    hydrology = analyze_hydrology(dem, 5.0, valid)
    first = find_outlets(hydrology.flow_direction, valid, hydrology.flow_accumulation)
    second = find_outlets(hydrology.flow_direction, valid, hydrology.flow_accumulation)
    assert first["primary"] == second["primary"]
    assert [o["row"] for o in first["outlets"]] == [o["row"] for o in second["outlets"]]


def test_outlet_internal_endpoint_for_enclosed_region():
    dem = fixtures.single_valid_cell_dem()
    hydrology = analyze_hydrology(dem, 5.0, np.isfinite(dem))
    selection = hydrology.outlet_selection
    assert selection["primary"] == (1, 1)
    assert selection["kind"] == "internal"
    assert selection["reason"] == "internal_endpoint_no_boundary_outlet"


def test_outlet_multiple_drainage_points():
    dem = fixtures.disconnected_dem()
    hydrology = analyze_hydrology(dem, 5.0, np.isfinite(dem))
    outlets = hydrology.outlet_selection["outlets"]
    assert len(outlets) >= 2
    assert all(outlet["kind"] == "boundary" for outlet in outlets[:2])


def test_outlet_none_when_no_valid_cells():
    flow_direction = np.full((3, 3), -1, dtype=np.int8)
    valid = np.zeros((3, 3), dtype=bool)
    selection = find_outlets(flow_direction, valid)
    assert selection["primary"] is None
    assert selection["no_valid_outlet"] is True
    assert selection["reason"] == "no_valid_outlet"


# ---------------------------------------------------------------------------
# 6. Catchment
# ---------------------------------------------------------------------------

def test_catchment_area_equals_cells_times_cell_area():
    dem = fixtures.plane_dem(6, 6)
    hydrology = analyze_hydrology(dem, 5.0, np.isfinite(dem))
    outlet = hydrology.outlet_selection["primary"]
    mask = delineate_catchment(hydrology.flow_direction, hydrology.valid_mask, outlet)
    stats = catchment_area(mask, 5.0)

    assert stats["cell_count"] == int(np.sum(mask))
    assert stats["area_m2"] == pytest.approx(stats["cell_count"] * 25.0)
    # Plane drains entirely to one corner.
    assert stats["cell_count"] == 36


def test_catchment_polygon_area_matches_raster_area():
    dem = fixtures.plane_dem(6, 6)
    hydrology = analyze_hydrology(dem, 5.0, np.isfinite(dem))
    outlet = hydrology.outlet_selection["primary"]
    mask = delineate_catchment(hydrology.flow_direction, hydrology.valid_mask, outlet)

    x = np.arange(6) * 5.0
    y = np.arange(6) * 5.0
    geometry = mask_to_polygon(mask, x, y)

    assert geometry is not None
    assert geometry.area == pytest.approx(float(np.sum(mask)) * 25.0)
    assert validate_catchment_geometry(geometry)["valid"] is True


def test_catchment_preserves_holes():
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False  # a hole
    x = np.arange(5) * 5.0
    y = np.arange(5) * 5.0
    geometry = mask_to_polygon(mask, x, y)

    assert geometry is not None
    assert geometry.area == pytest.approx(24 * 25.0)
    assert geometry.geom_type == "Polygon"
    assert len(geometry.interiors) == 1


def test_catchment_disconnected_regions():
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = True
    mask[4, 4] = True
    x = np.arange(5) * 5.0
    y = np.arange(5) * 5.0
    geometry = mask_to_polygon(mask, x, y)
    assert geometry.geom_type == "MultiPolygon"
    assert geometry.area == pytest.approx(2 * 25.0)


def test_mask_to_polygon_empty_returns_none():
    mask = np.zeros((3, 3), dtype=bool)
    assert mask_to_polygon(mask, np.arange(3) * 5.0, np.arange(3) * 5.0) is None


def test_delineate_catchment_rejects_bad_outlet():
    dem = fixtures.plane_dem(4, 4)
    valid = np.isfinite(dem)
    hydrology = analyze_hydrology(dem, 5.0, valid)
    with pytest.raises(ValueError):
        delineate_catchment(hydrology.flow_direction, hydrology.valid_mask, (99, 99))

    valid[2, 2] = False
    with pytest.raises(ValueError):
        delineate_catchment(hydrology.flow_direction, valid, (2, 2))


# ---------------------------------------------------------------------------
# 7. Hydrology validation
# ---------------------------------------------------------------------------

def test_validation_status_good_on_plane():
    dem = fixtures.plane_dem(6, 6)
    hydrology = analyze_hydrology(dem, 5.0, np.isfinite(dem))
    result = validate_hydrology(
        hydrology.flow_direction,
        hydrology.flow_accumulation,
        hydrology.valid_mask,
        hydrology.sink_mask,
        hydrology.edge_outflow_mask,
        primary_outlet=hydrology.outlet_selection["primary"],
        cycle_count=hydrology.cycle_count,
    )
    assert result["status"] in {"good", "acceptable"}
    assert result["invalid_reasons"] == []


def test_validation_flags_missing_outlet():
    dem = fixtures.plane_dem(5, 5)
    hydrology = analyze_hydrology(dem, 5.0, np.isfinite(dem))
    result = validate_hydrology(
        hydrology.flow_direction,
        hydrology.flow_accumulation,
        hydrology.valid_mask,
        hydrology.sink_mask,
        hydrology.edge_outflow_mask,
        primary_outlet=None,
    )
    assert result["status"] == "invalid"
    assert "missing_outlet" in result["invalid_reasons"]


def test_validate_outlet_states():
    dem = fixtures.plane_dem(4, 4)
    hydrology = analyze_hydrology(dem, 5.0, np.isfinite(dem))
    ok = validate_outlet(hydrology.outlet_selection["primary"], hydrology.valid_mask, hydrology.flow_direction)
    assert ok["valid"] is True
    assert ok["terminal_cell"] is True

    missing = validate_outlet(None, hydrology.valid_mask, hydrology.flow_direction)
    assert missing["valid"] is False
    assert missing["reason"] == "no_outlet_selected"


# ---------------------------------------------------------------------------
# 8. Error handling / API compatibility
# ---------------------------------------------------------------------------

def test_two_point_contour_raises_controlled_value_error():
    from backend.parser import parse_contour_file
    from backend.terrain import analyze_terrain

    contour_data = parse_contour_file(fixtures.insufficient_contours_kml(), "insufficient.kml")
    with pytest.raises(ValueError):
        analyze_terrain(contour_data, resolution_m=5.0)


def test_corrupt_kmz_raises_controlled_value_error():
    from backend.parser import parse_contour_file

    with pytest.raises(ValueError):
        parse_contour_file(fixtures.corrupt_kmz(), "corrupt.kmz")


def test_insufficient_contour_endpoint_is_422(client):
    filename, data = fixtures.build_all()["insufficient_contours"]
    response = client.post("/analyzeContour", files={"file": (filename, data, "application/xml")})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_CONTOUR_DATA"


def test_corrupt_kmz_endpoint_is_422(client):
    filename, data = fixtures.build_all()["corrupt_kmz"]
    response = client.post("/analyzeContour", files={"file": (filename, data, "application/octet-stream")})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_CONTOUR_DATA"


def test_no_valid_outlet_returns_controlled_failure(client, monkeypatch, sample_kml_bytes):
    import main

    monkeypatch.setattr(
        main,
        "_select_outlet",
        lambda hydrology: {"primary": None, "kind": None, "outlets": [], "reason": "no_valid_outlet", "no_valid_outlet": True},
    )

    response = client.post(
        "/analyzeContour",
        files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failed"
    assert payload["error_code"] == "NO_VALID_OUTLET"
    assert payload["pond_candidate"] is None
    assert payload["recommended_location"]["latitude"] is None
    assert payload["recommended_location"]["longitude"] is None


def test_successful_response_never_contains_fabricated_origin(client, sample_kml_bytes):
    payload = client.post(
        "/analyzeContour",
        files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
    ).json()
    candidate = payload["pond_candidate"]
    assert not (candidate["latitude"] == 0.0 and candidate["longitude"] == 0.0)
