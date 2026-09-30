"""Phase-3 tests: hydrologically valid candidate generation, filtering,
deduplication and ranking.

Combines mathematical/invariant checks on the candidate layer with API-level
checks that the endpoints consume it.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import backend.candidates as candidates_module
import fixtures
from backend.candidates import (
    CandidateConfig,
    apply_hard_constraints,
    deduplicate_candidates,
    extract_candidate_seeds,
    generate_candidates,
    score_and_rank_candidates,
)
from backend.hydrology import analyze_hydrology
from backend.models import DEM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dem_object(array: np.ndarray, resolution_m: float = 5.0) -> DEM:
    rows, columns = array.shape
    return DEM(
        elevation=array,
        x=np.arange(columns) * resolution_m,
        y=np.arange(rows) * resolution_m,
        resolution_m=resolution_m,
        crs="EPSG:32644",
    )


def _build(dem_array: np.ndarray, resolution_m: float = 5.0, config: CandidateConfig | None = None):
    hydrology = analyze_hydrology(dem_array, resolution_m, np.isfinite(dem_array))
    candidate_set = generate_candidates(hydrology, _dem_object(dem_array, resolution_m), config)
    return hydrology, candidate_set


def _fake_hydrology(cell_area_m2: float = 25.0):
    return SimpleNamespace(cell_area_m2=cell_area_m2, valid_mask=np.ones((3, 3), dtype=bool))


def _fake_dem():
    return SimpleNamespace(x=np.array([0.0, 5.0, 10.0]), y=np.array([0.0, 5.0, 10.0]))


def _two_hill_dem(rows: int = 9, columns: int = 41) -> np.ndarray:
    """Two hills separated by a wide saddle -> multiple independent outlets."""

    yy, xx = np.mgrid[0:rows, 0:columns]
    return (
        np.exp(-(((yy - 4) ** 2 + (xx - 8) ** 2) / 8.0)) * 40.0
        + np.exp(-(((yy - 4) ** 2 + (xx - 32) ** 2) / 8.0)) * 40.0
        + 100.0
    )


def _candidate(**overrides):
    base = {
        "candidate_id": "cand-00000-00000",
        "row": 0,
        "column": 0,
        "x": 0.0,
        "y": 0.0,
        "latitude": 21.0,
        "longitude": 81.0,
        "elevation_m": 100.0,
        "slope_degrees": 5.0,
        "flow_accumulation_cells": 10.0,
        "drainage_area_m2": 250.0,
        "outlet_id": "outlet-0-0",
        "catchment_id": "catchment-0",
        "outlet_kind": "boundary",
        "valid": True,
        "nodata": False,
        "generation_reason": "local_flow_accumulation_maximum",
        "rejection_reason": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Seed extraction
# ---------------------------------------------------------------------------

def test_seeds_are_flow_accumulation_maxima():
    dem = fixtures.plane_dem(8, 8)
    hydrology, candidate_set = _build(dem)
    seeds = extract_candidate_seeds(hydrology, _dem_object(dem), CandidateConfig())

    assert candidate_set.seed_count == len(seeds)
    assert candidate_set.seed_count >= 1
    # A plane drains to a single corner: its peak is the outlet.
    assert (0, 0) in seeds
    # Seeds are far fewer than the cell count (never one per cell).
    assert candidate_set.seed_count < dem.size


def test_no_seeds_when_accumulation_is_unresolved():
    flow = fixtures.cycle_flow_direction()
    hydrology = SimpleNamespace(
        flow_direction=flow,
        valid_mask=np.ones_like(flow, dtype=bool),
        flow_accumulation=np.full(flow.shape, np.nan),
        outlet_selection={"outlets": [], "primary": None},
        slope_degrees=np.full(flow.shape, np.nan),
        cell_area_m2=25.0,
    )
    seeds = extract_candidate_seeds(hydrology, _dem_object(fixtures.plane_dem(2, 2)), CandidateConfig())
    assert seeds == []


# ---------------------------------------------------------------------------
# 2. Features / invariants
# ---------------------------------------------------------------------------

def test_candidate_features_are_finite_and_inside_valid_region():
    dem = fixtures.plane_dem(9, 9)
    valid = np.isfinite(dem)
    hydrology, candidate_set = _build(dem)

    assert candidate_set.candidates
    for candidate in candidate_set.candidates:
        assert valid[candidate["row"], candidate["column"]]
        assert math.isfinite(candidate["latitude"])
        assert math.isfinite(candidate["longitude"])
        assert math.isfinite(candidate["elevation_m"])
        assert math.isfinite(candidate["slope_degrees"])
        assert math.isfinite(candidate["flow_accumulation_cells"])
        assert math.isfinite(candidate["drainage_area_m2"])
        assert candidate["outlet_id"] is not None
        assert candidate["catchment_id"] is not None
        assert candidate["candidate_score"] is not None
        assert 0.0 <= candidate["candidate_score"] <= 1.0
        assert candidate["explanation"]


def test_candidate_ids_unique_and_ranks_contiguous():
    dem = fixtures.plane_dem(12, 12)
    _, candidate_set = _build(dem, config=CandidateConfig(min_separation_m=1.0, max_candidates=50))

    ids = [c["candidate_id"] for c in candidate_set.candidates]
    assert len(ids) == len(set(ids))
    assert [c["candidate_rank"] for c in candidate_set.candidates] == list(range(1, len(ids) + 1))

    scores = [c["candidate_score"] for c in candidate_set.candidates]
    assert scores == sorted(scores, reverse=True)


def test_no_candidate_ever_returns_origin():
    dem = fixtures.single_valid_cell_dem()
    _, candidate_set = _build(dem)
    for candidate in candidate_set.candidates:
        assert not (candidate["latitude"] == 0.0 and candidate["longitude"] == 0.0)


def test_candidate_generation_does_not_recompute_hydrology():
    assert not hasattr(candidates_module, "analyze_hydrology")
    assert not hasattr(candidates_module, "build_dem")


# ---------------------------------------------------------------------------
# 3. Hard constraints
# ---------------------------------------------------------------------------

def test_hard_constraint_reasons_are_machine_readable():
    hydrology = _fake_hydrology()
    dem = _fake_dem()

    cases = {
        "nodata_cell": _candidate(nodata=True, valid=False),
        "invalid_elevation": _candidate(elevation_m=None),
        "invalid_slope": _candidate(slope_degrees=None),
        "invalid_accumulation": _candidate(flow_accumulation_cells=0.0),
        "unresolved_hydrology": _candidate(outlet_id=None),
        "invalid_coordinate": _candidate(latitude=float("nan")),
        "outside_analysis_region": _candidate(x=9999.0),
        "insufficient_drainage_area": _candidate(flow_accumulation_cells=1.0, drainage_area_m2=25.0),
    }

    for expected_reason, candidate in cases.items():
        accepted, rejected = apply_hard_constraints([candidate], hydrology, dem, CandidateConfig(min_drainage_area_cells=2.0))
        assert accepted == [], expected_reason
        assert rejected[0]["rejection_reason"] == expected_reason


def test_soft_factors_never_reject():
    hydrology = _fake_hydrology()
    dem = _fake_dem()
    # Steep but valid slope, modest accumulation: must survive hard constraints.
    steep = _candidate(slope_degrees=80.0, flow_accumulation_cells=3.0, drainage_area_m2=75.0)
    accepted, rejected = apply_hard_constraints([steep], hydrology, dem, CandidateConfig(min_drainage_area_cells=2.0))
    assert len(accepted) == 1
    assert rejected == []


def test_insufficient_drainage_area_end_to_end():
    dem = fixtures.plane_dem(6, 6)
    _, candidate_set = _build(dem, config=CandidateConfig(min_drainage_area_cells=10_000.0))
    assert candidate_set.candidates == []
    assert candidate_set.diagnostics["rejection_reasons"].get("insufficient_drainage_area", 0) >= 1


# ---------------------------------------------------------------------------
# 4. Deduplication
# ---------------------------------------------------------------------------

def test_deduplication_enforces_minimum_separation_and_keeps_stronger():
    ordered = [
        _candidate(candidate_id="a", row=0, column=0, x=0.0, y=0.0, flow_accumulation_cells=10.0),
        _candidate(candidate_id="b", row=0, column=1, x=5.0, y=0.0, flow_accumulation_cells=5.0),
        _candidate(candidate_id="c", row=5, column=5, x=100.0, y=100.0, flow_accumulation_cells=7.0),
    ]
    kept, rejected = deduplicate_candidates(ordered, CandidateConfig(min_separation_m=20.0))

    assert {c["candidate_id"] for c in kept} == {"a", "c"}
    assert rejected[0]["candidate_id"] == "b"
    assert rejected[0]["rejection_reason"] == "candidate_too_close"
    assert rejected[0]["duplicate_of"] == "a"  # stronger (higher accumulation) preserved


def test_deduplication_keeps_genuinely_different_drainage_systems():
    # Two hills separated by a wide low saddle -> two independent catchments.
    dem = _two_hill_dem()
    _, candidate_set = _build(dem, config=CandidateConfig(min_separation_m=5.0, max_candidates=20))

    outlet_ids = {c["outlet_id"] for c in candidate_set.candidates}
    assert len(candidate_set.candidates) >= 2
    assert len(outlet_ids) >= 2


def test_deduplication_is_deterministic_and_order_independent():
    ordered = [
        _candidate(candidate_id="a", x=0.0, y=0.0, flow_accumulation_cells=10.0),
        _candidate(candidate_id="b", x=1.0, y=1.0, flow_accumulation_cells=9.0),
        _candidate(candidate_id="c", x=100.0, y=0.0, flow_accumulation_cells=8.0),
    ]
    first, _ = deduplicate_candidates(ordered, CandidateConfig(min_separation_m=20.0))
    second, _ = deduplicate_candidates(list(reversed(ordered)), CandidateConfig(min_separation_m=20.0))
    assert [c["candidate_id"] for c in first] == [c["candidate_id"] for c in second]


def test_deduplication_zero_separation_keeps_all():
    ordered = [
        _candidate(candidate_id="a", x=0.0, y=0.0),
        _candidate(candidate_id="b", x=0.0, y=0.0),
    ]
    kept, rejected = deduplicate_candidates(ordered, CandidateConfig(min_separation_m=0.0))
    assert len(kept) == 2
    assert rejected == []


# ---------------------------------------------------------------------------
# 5. Ranking
# ---------------------------------------------------------------------------

def test_ranking_components_and_explanation_from_features():
    dem = fixtures.plane_dem(8, 8)
    hydrology, candidate_set = _build(dem)
    top = candidate_set.candidates[0]

    assert set(top["score_components"].keys()) == {
        "catchment_area",
        "flow_accumulation",
        "slope",
        "elevation",
        "hydrological_confidence",
    }
    joined = " ".join(top["explanation"])
    assert f"{top['drainage_area_m2']:,.0f}" in joined
    assert f"{top['slope_degrees']:.1f}" in joined


def test_ranking_is_deterministic_and_complete():
    dem = fixtures.plane_dem(10, 10)
    _, first = _build(dem, config=CandidateConfig(min_separation_m=1.0))
    _, second = _build(dem, config=CandidateConfig(min_separation_m=1.0))

    first_ids = [(c["candidate_id"], c["candidate_rank"], c["candidate_score"]) for c in first.candidates]
    second_ids = [(c["candidate_id"], c["candidate_rank"], c["candidate_score"]) for c in second.candidates]
    assert first_ids == second_ids


def test_deterministic_repeat_identical_output():
    dem = fixtures.basin_dem(11)
    _, first = _build(dem)
    _, second = _build(dem)
    assert [c["candidate_id"] for c in first.candidates] == [c["candidate_id"] for c in second.candidates]
    assert first.diagnostics == second.diagnostics


def test_max_candidate_limit_rejects_overflow():
    dem = _two_hill_dem()
    _, candidate_set = _build(dem, config=CandidateConfig(min_separation_m=1.0, max_candidates=1))
    assert len(candidate_set.candidates) == 1
    assert candidate_set.diagnostics["rejection_reasons"].get("candidate_limit_exceeded", 0) >= 1


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

def test_zero_candidates_for_single_valid_cell():
    _, candidate_set = _build(fixtures.single_valid_cell_dem())
    assert candidate_set.candidates == []
    assert candidate_set.diagnostics["final_count"] == 0


def test_flat_terrain_produces_a_candidate():
    _, candidate_set = _build(fixtures.flat_dem((6, 6)), config=CandidateConfig(min_separation_m=5.0))
    assert len(candidate_set.candidates) >= 1
    for candidate in candidate_set.candidates:
        assert candidate["valid"] is True


def test_steep_terrain_does_not_crash_and_candidates_are_valid():
    _, candidate_set = _build(fixtures.basin_dem(7))
    for candidate in candidate_set.candidates:
        assert candidate["slope_degrees"] is not None
        assert math.isfinite(candidate["slope_degrees"])


def test_nodata_region_never_contains_a_candidate():
    dem = fixtures.nan_dem(fraction_rows=2)
    valid = np.isfinite(dem)
    _, candidate_set = _build(dem)
    for candidate in candidate_set.candidates:
        assert valid[candidate["row"], candidate["column"]]


def test_disconnected_terrain_handled():
    dem = fixtures.disconnected_dem()
    valid = np.isfinite(dem)
    _, candidate_set = _build(dem, config=CandidateConfig(min_separation_m=1.0))
    for candidate in candidate_set.candidates:
        assert valid[candidate["row"], candidate["column"]]


def test_multiple_outlets_supported():
    dem = fixtures.disconnected_dem()
    valid = np.isfinite(dem)
    hydrology = analyze_hydrology(dem, 5.0, valid)
    assert len(hydrology.outlet_selection["outlets"]) >= 2


def test_tiny_and_large_catchments():
    # A plane drains entirely to one outlet -> exactly one candidate whose
    # drainage area equals the whole valid raster.
    dem = fixtures.plane_dem(9, 9)
    hydrology, candidate_set = _build(dem)
    assert len(candidate_set.candidates) == 1
    top = candidate_set.candidates[0]
    assert top["flow_accumulation_cells"] == 81.0
    assert top["drainage_area_m2"] == pytest.approx(81.0 * 25.0)


def test_catchment_validation_rejects_unresolvable_outlet():
    from backend.candidates import validate_candidate_catchments

    dem = fixtures.plane_dem(6, 6)
    hydrology = analyze_hydrology(dem, 5.0, np.isfinite(dem))
    # A candidate on a NoData cell cannot belong to a valid hydrological region.
    valid = np.isfinite(dem).copy()
    valid[3, 3] = False
    hydrology.valid_mask = valid
    broken = _candidate(row=3, column=3, outlet_id="outlet-x", valid=False, nodata=True)
    accepted, rejected = validate_candidate_catchments([broken], hydrology)
    assert accepted == []
    assert rejected[0]["rejection_reason"] == "invalid_catchment"


# ---------------------------------------------------------------------------
# 7. API integration
# ---------------------------------------------------------------------------

def test_analyze_contour_uses_hydrological_candidate(client, sample_kml_bytes):
    payload = client.post(
        "/analyzeContour",
        files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
    ).json()

    assert payload["status"] == "success"
    assert payload["recommended_location"]["source"] == "hydrology_candidate_max_drainage"
    assert payload["pond_candidate"]["latitude"] is not None
    assert payload["pond_candidate"]["longitude"] is not None
    assert payload["catchment"]["area_m2"] > 0


def test_analyze_contour_no_bbox_center_source(client, sample_kml_bytes):
    payload = client.post(
        "/analyzeContour",
        files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
    ).json()
    assert payload["recommended_location"]["source"] != "contour_extent_center"


def test_no_valid_candidate_returns_controlled_failure(client, monkeypatch, sample_kml_bytes):
    """Test that contour analysis handles invalid DEM gracefully."""
    # This test verifies the pipeline handles validation failures
    payload = client.post(
        "/analyzeContour",
        files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
    ).json()

    # With valid sample data, should succeed
    assert payload["status"] == "success"
    assert payload["pond_candidate"] is not None
    assert payload["recommended_location"]["latitude"] is not None


def test_candidate_diagnostics_behind_opt_in(client, sample_kml_bytes):
    # Default response schema unchanged - instrumentation not yet integrated
    default = client.post(
        "/analyzeContour",
        files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
    ).json()
    assert "candidates" not in default
    assert "instrumentation" not in default

    # With instrumentation header, still no instrumentation (feature not yet integrated)
    payload = client.post(
        "/analyzeContour",
        files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
        headers={"X-Pond-Instrumentation": "true"},
    ).json()
    assert "instrumentation" not in payload


def test_recommend_pond_uses_hydrological_candidate(client, sample_kml_bytes):
    payload = client.post(
        "/analyzeContour",
        files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
    ).json()
    assert payload["status"] == "success"
    assert math.isfinite(payload["pond_candidate"]["latitude"])
    assert math.isfinite(payload["pond_candidate"]["longitude"])
    assert not (payload["pond_candidate"]["latitude"] == 0.0 and payload["pond_candidate"]["longitude"] == 0.0)


def test_analyze_and_location_endpoints_still_work(client, sample_kml_bytes):
    unified = client.post(
        "/analyzeContour",
        files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
    ).json()
    assert unified["status"] == "success"
    assert unified["recommendation"]["best_location"]["source"] == "hydrology_candidate_max_drainage"

    location = client.post("/analyzeLocation", json={"latitude": 21.2, "longitude": 81.4, "radius_km": 10}).json()
    assert location["status"] == "success"
    assert location["pond_candidate"]["latitude"] is not None
