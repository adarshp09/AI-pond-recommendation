"""Tests for the baseline data-quality metric collectors."""

from __future__ import annotations

import numpy as np

import backend.data_quality as data_quality
from backend.hydrology import calculate_flow_accumulation
from fixtures import (
    cycle_flow_direction,
    flat_dem,
    nan_dem,
    plane_dem,
    sink_dem,
)
from backend.hydrology import analyze_hydrology
from backend.models import Contour, ContourData, DEM, TerrainResult
from backend.terrain import calculate_slope


def _contour_data() -> ContourData:
    contours = [
        Contour(elevation_m=100.0, coordinates=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]),
        Contour(elevation_m=110.0, coordinates=[(1.0, 1.0), (9.0, 1.0)]),
    ]
    return ContourData(
        contours=contours,
        source_crs="EPSG:4326",
        target_crs="EPSG:32644",
        min_x=0.0,
        max_x=10.0,
        min_y=0.0,
        max_y=10.0,
        min_elevation_m=100.0,
        max_elevation_m=110.0,
        longitude_min=81.0,
        longitude_max=81.1,
        latitude_min=21.0,
        latitude_max=21.1,
    )


def test_contour_metrics_counts_contours_and_coordinates():
    metrics = data_quality.contour_metrics(_contour_data())
    assert metrics["contour_count"] == 2
    assert metrics["coordinate_count"] == 5
    assert metrics["elevation_min_m"] == 100.0
    assert metrics["elevation_max_m"] == 110.0
    assert metrics["elevation_range_m"] == 10.0


def test_dem_metrics_reports_nan_percentage_and_slope_range():
    elevation = nan_dem(fraction_rows=2)  # 6x6 with 2 fully-NaN rows -> 12/36 NaN
    slope, gradient = calculate_slope(elevation, 5.0)
    dem = DEM(elevation, np.arange(6) * 5.0, np.arange(6) * 5.0, 5.0, "EPSG:32644", slope, gradient, np.isfinite(elevation))

    metrics = data_quality.dem_metrics(dem)
    assert metrics["dem_rows"] == 6
    assert metrics["dem_columns"] == 6
    assert metrics["dem_cells"] == 36
    assert metrics["dem_nan_percentage"] is not None
    assert metrics["dem_nan_percentage"] > 0.0
    assert metrics["slope_min_degrees"] is not None
    assert metrics["slope_max_degrees"] >= metrics["slope_min_degrees"]


def test_dem_metrics_handles_all_valid_plane():
    elevation = plane_dem(6, 6)
    slope, gradient = calculate_slope(elevation, 5.0)
    dem = DEM(elevation, np.arange(6) * 5.0, np.arange(6) * 5.0, 5.0, "EPSG:32644", slope, gradient, np.isfinite(elevation))

    metrics = data_quality.dem_metrics(dem)
    assert metrics["dem_nan_percentage"] == 0.0
    assert metrics["elevation_range_m"] is not None and metrics["elevation_range_m"] > 0.0


def test_hydrology_metrics_reports_valid_flow_direction_share():
    dem = plane_dem(6, 6)
    hydrology = analyze_hydrology(dem, 5.0, np.isfinite(dem))
    metrics = data_quality.hydrology_metrics(hydrology)

    assert 0.0 <= metrics["valid_flow_direction_percentage"] <= 100.0
    assert metrics["flow_accumulation_max_cells"] is not None
    assert metrics["flow_accumulation_max_cells"] >= metrics["flow_accumulation_mean_cells"]


def test_hydrology_metrics_detects_flow_cycles():
    flow = cycle_flow_direction()
    accumulation = calculate_flow_accumulation(flow)

    class _Result:
        pass

    result = _Result()
    result.flow_direction = flow
    result.valid_mask = np.ones_like(flow, dtype=bool)
    result.flow_accumulation = accumulation
    result.sink_cells = 0
    result.edge_outflow_cells = 0

    metrics = data_quality.hydrology_metrics(result)
    assert metrics["unresolved_cycle_cells"] == 4
    assert metrics["valid_flow_direction_percentage"] == 100.0


def test_hydrology_metrics_on_flat_terrain_after_filling():
    dem = flat_dem((5, 5))
    hydrology = analyze_hydrology(dem, 5.0, np.isfinite(dem))
    metrics = data_quality.hydrology_metrics(hydrology)
    # Interior flats are resolved by depression handling; only the boundary
    # drains off-grid, so there are no remaining internal sinks.
    assert metrics["sink_cells"] == 0
    assert metrics["valid_flow_direction_percentage"] > 0.0
    assert metrics["unresolved_cycle_cells"] == 0


def test_hydrology_metrics_on_sink_dem_after_filling():
    dem = sink_dem()
    hydrology = analyze_hydrology(dem, 5.0, np.isfinite(dem))
    metrics = data_quality.hydrology_metrics(hydrology)
    assert metrics["sink_cells"] == 0
    assert metrics["flow_accumulation_max_cells"] >= metrics["flow_accumulation_mean_cells"]


def test_candidate_and_catchment_metrics():
    candidates = [{"id": "a"}, {"id": "b"}]
    metrics = data_quality.candidate_metrics(candidates, rejected_count=1, recommendation_count=1)
    assert metrics["candidate_count"] == 2
    assert metrics["rejected_candidate_count"] == 1
    assert metrics["recommendation_count"] == 1

    catchment = data_quality.catchment_metrics({"area_m2": 10_000.0, "area_hectares": 1.0, "area_km2": 0.01, "cell_count": 400})
    assert catchment["catchment_area_m2"] == 10_000.0
    assert catchment["catchment_cell_count"] == 400


def test_external_availability_flags():
    availability = data_quality.external_availability(
        dem={"elevation": []},
        rainfall=None,
        land_context={"status": "unavailable"},
        place_search={"results": []},
        geocoding=None,
    )
    assert availability == {
        "dem": True,
        "rainfall": False,
        "land_context": False,
        "place_search": True,
        "geocoding": False,
    }


def test_build_data_quality_aggregates_sections():
    dem_array = plane_dem(5, 5)
    slope, gradient = calculate_slope(dem_array, 5.0)
    dem = DEM(dem_array, np.arange(5) * 5.0, np.arange(5) * 5.0, 5.0, "EPSG:32644", slope, gradient, np.isfinite(dem_array))
    terrain = TerrainResult(dem, 1.0, 2.0, 1.0)
    hydrology = analyze_hydrology(dem_array, 5.0, np.isfinite(dem_array))

    metrics = data_quality.build_data_quality(
        contour_data=_contour_data(),
        dem=dem,
        validation=None,
        hydrology=hydrology,
        catchment_stats={"area_m2": 25.0, "cell_count": 1},
        candidates=[{"id": "x"}],
        rejected_count=0,
        recommendation_count=1,
        external={"rainfall": True},
    )

    for key in ["contour_count", "dem_rows", "valid_flow_direction_percentage", "catchment_area_m2", "candidate_count", "external_data"]:
        assert key in metrics
    assert terrain.mean_slope_degrees == 1.0
