"""Phase 09 — river/waterway exclusion tests.

Covers the spatial filter against a synthetic GeoPackage, the config defaults, and
the pipeline wiring (enabled vs disabled).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from backend import pipeline
from backend.candidates import CandidateSet
from backend.utils.spatial_filter import exclude_river_candidates
from config import settings

REPO_ROOT = Path(__file__).resolve().parents[1]
LON, LAT = 81.30, 21.26


def _write_water_gpkg(path: Path) -> Path:
    gpd = pytest.importorskip("geopandas")
    import pyogrio
    from shapely.geometry import LineString, Polygon

    square = Polygon([
        (LON - 0.0005, LAT - 0.0005),
        (LON + 0.0005, LAT - 0.0005),
        (LON + 0.0005, LAT + 0.0005),
        (LON - 0.0005, LAT + 0.0005),
        (LON - 0.0005, LAT - 0.0005),
    ])
    water = gpd.GeoDataFrame({"osm_id": ["1"], "osm_type": ["way"]}, geometry=[square], crs="EPSG:4326")
    # Horizontal waterway at lat 21.27, lon 81.295..81.305
    line = LineString([(81.295, 21.27), (81.305, 21.27)])
    waterways = gpd.GeoDataFrame({"osm_id": ["2"], "osm_type": ["way"]}, geometry=[line], crs="EPSG:4326")

    pyogrio.write_dataframe(water, path, layer="water_areas", driver="GPKG")
    pyogrio.write_dataframe(waterways, path, layer="waterways", driver="GPKG", append=True)
    return path


def _candidates():
    return [
        {"id": "inside", "longitude": LON, "latitude": LAT},
        # ~15 m north of the waterway line
        {"id": "near_line", "longitude": LON, "latitude": 21.270135},
        {"id": "far", "longitude": 81.35, "latitude": 21.35},
    ]


def test_excludes_points_inside_water_and_within_buffer(tmp_path):
    gpkg = _write_water_gpkg(tmp_path / "land.gpkg")

    kept = exclude_river_candidates(_candidates(), land_gpkg_path=str(gpkg), buffer_meters=30.0)
    kept_ids = {c["id"] for c in kept}

    assert kept_ids == {"far"}  # inside polygon + within 30 m of line excluded


def test_smaller_buffer_keeps_near_but_not_inside(tmp_path):
    gpkg = _write_water_gpkg(tmp_path / "land.gpkg")

    kept = exclude_river_candidates(_candidates(), land_gpkg_path=str(gpkg), buffer_meters=5.0)
    kept_ids = {c["id"] for c in kept}

    assert "near_line" in kept_ids   # 15 m > 5 m buffer
    assert "inside" not in kept_ids  # still inside the water polygon
    assert "far" in kept_ids


def test_zero_buffer_excludes_only_inside(tmp_path):
    gpkg = _write_water_gpkg(tmp_path / "land.gpkg")

    kept = exclude_river_candidates(_candidates(), land_gpkg_path=str(gpkg), buffer_meters=0.0)
    kept_ids = {c["id"] for c in kept}

    assert "inside" not in kept_ids
    assert {"near_line", "far"} <= kept_ids


def test_missing_dataset_returns_candidates_unchanged(tmp_path):
    candidates = _candidates()
    kept = exclude_river_candidates(
        candidates, land_gpkg_path=str(tmp_path / "does-not-exist.gpkg"), buffer_meters=30.0
    )
    assert kept == candidates


def test_empty_candidates_returns_empty(tmp_path):
    gpkg = _write_water_gpkg(tmp_path / "land.gpkg")
    assert exclude_river_candidates([], land_gpkg_path=str(gpkg)) == []


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_config_defaults_are_enabled_with_30m_buffer():
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"RIVER_EXCLUSION_ENABLED", "WATERBODY_BUFFER_METERS", "LOCAL_OSM_GPKG_PATH"}
    }
    code = (
        "from config import settings as s;"
        "print(s.RIVER_EXCLUSION_ENABLED, s.WATERBODY_BUFFER_METERS)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    enabled, buffer = result.stdout.strip().splitlines()[-1].split()
    assert enabled == "True"
    assert float(buffer) == 30.0


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------

def _two_candidate_set():
    base = {
        "x": 500000.0,
        "y": 2350000.0,
        "crs": "EPSG:32644",
        "latitude": LAT,
        "longitude": LON,
        "slope_degrees": 3.0,
        "drainage_area_m2": 50_000.0,
        "flow_accumulation_cells": 100.0,
        "elevation_m": 250.0,
    }
    return CandidateSet(
        candidates=[{**base, "candidate_id": "cand-1"}, {**base, "candidate_id": "cand-2"}],
        rejected=[],
        seed_count=2,
        accepted_count=2,
        deduplicated_count=2,
    )


def _fake_provider(south, west, north, east):  # noqa: ARG001
    return {
        "bbox": {"south": south, "west": west, "north": north, "east": east},
        "water_bodies": [], "roads": [], "buildings": [],
        "source": "fake", "provider": "fake", "status": "available",
    }


def test_pipeline_applies_exclusion_when_enabled(monkeypatch):
    monkeypatch.setitem(settings.__dict__, "RIVER_EXCLUSION_ENABLED", True)
    monkeypatch.setitem(settings.__dict__, "WATERBODY_BUFFER_METERS", 30.0)
    monkeypatch.setitem(settings.__dict__, "LOCAL_OSM_GPKG_PATH", "/tmp/fake-land.gpkg")

    candidate_set = _two_candidate_set()
    monkeypatch.setattr(pipeline, "generate_candidates", lambda *a, **k: candidate_set)

    calls = {}

    def fake_exclude(candidates, land_gpkg_path=None, buffer_meters=None):
        calls["n"] = len(candidates)
        calls["path"] = land_gpkg_path
        calls["buffer"] = buffer_meters
        return candidates[:1]  # drop one candidate

    monkeypatch.setattr(pipeline, "exclude_river_candidates", fake_exclude)

    hydrology = types.SimpleNamespace(cell_area_m2=25.0)
    result = pipeline.engine_analysis(None, hydrology, land_provider=_fake_provider)

    assert calls == {"n": 2, "path": "/tmp/fake-land.gpkg", "buffer": 30.0}
    assert result is not None
    assert any("excluded within 30 m" in warning for warning in result["warnings"])


def test_pipeline_skips_exclusion_when_disabled(monkeypatch):
    monkeypatch.setitem(settings.__dict__, "RIVER_EXCLUSION_ENABLED", False)

    candidate_set = _two_candidate_set()
    monkeypatch.setattr(pipeline, "generate_candidates", lambda *a, **k: candidate_set)

    def must_not_call(*_a, **_k):
        raise AssertionError("exclude_river_candidates must not be called when disabled")

    monkeypatch.setattr(pipeline, "exclude_river_candidates", must_not_call)

    hydrology = types.SimpleNamespace(cell_area_m2=25.0)
    result = pipeline.engine_analysis(None, hydrology, land_provider=_fake_provider)

    assert result is not None
    assert not any("excluded" in warning for warning in result["warnings"])


# ---------------------------------------------------------------------------
# Shared helper + legacy path
# ---------------------------------------------------------------------------

def test_apply_river_exclusion_disabled_returns_unchanged(monkeypatch):
    monkeypatch.setitem(settings.__dict__, "RIVER_EXCLUSION_ENABLED", False)
    candidates = _candidates()
    warnings: list[str] = []

    result = pipeline.apply_river_exclusion(candidates, warnings)

    assert result == candidates
    assert warnings == []


def test_apply_river_exclusion_emits_shared_warning_format(monkeypatch):
    monkeypatch.setitem(settings.__dict__, "RIVER_EXCLUSION_ENABLED", True)
    monkeypatch.setitem(settings.__dict__, "WATERBODY_BUFFER_METERS", 30.0)
    monkeypatch.setattr(
        pipeline, "exclude_river_candidates", lambda cands, **kw: list(cands[:1])
    )
    warnings: list[str] = []

    result = pipeline.apply_river_exclusion(_candidates(), warnings)

    assert len(result) == 1
    assert warnings == ["2 candidate(s) excluded within 30 m of rivers/water bodies."]


def test_legacy_helper_reselects_primary_and_warns(monkeypatch):
    import main as main_module

    def stub(candidates, warnings=None, **_kwargs):
        if warnings is not None:
            warnings.append("1 candidate(s) excluded within 30 m of rivers/water bodies.")
        return list(candidates[1:])

    monkeypatch.setattr(main_module, "apply_river_exclusion", stub)

    primary = {"id": "primary", "label": "Pond Candidate", "latitude": 1.0, "longitude": 2.0}
    alternatives = [{"id": "alternative-1", "latitude": 3.0, "longitude": 4.0}]
    recommendation = {
        "best_location": {"latitude": 1.0, "longitude": 2.0, "source": "hydrology_candidate_max_drainage"},
        "alternatives": alternatives,
        "suitability_score": 0.5,
        "explanation": "x",
    }
    warnings: list[str] = []

    new_primary, new_alts, new_rec = main_module._apply_river_exclusion_to_legacy_candidates(
        primary, alternatives, recommendation, warnings, "hydrology_candidate_max_drainage"
    )

    assert new_primary["id"] == "primary"
    assert new_primary["latitude"] == 3.0
    assert new_alts == []
    assert new_rec["best_location"]["latitude"] == 3.0
    assert new_rec["best_location"]["source"] == "river_exclusion_reselected"
    assert any("excluded within" in w for w in warnings)


def test_legacy_helper_keeps_original_when_all_excluded(monkeypatch):
    import main as main_module

    monkeypatch.setattr(main_module, "apply_river_exclusion", lambda cands, w=None, **k: [])

    primary = {"id": "primary", "latitude": 1.0, "longitude": 2.0}
    recommendation = {"best_location": {"latitude": 1.0, "longitude": 2.0, "source": "s"}}
    warnings: list[str] = []

    new_primary, _alts, _rec = main_module._apply_river_exclusion_to_legacy_candidates(
        primary, [], recommendation, warnings, "s"
    )

    assert new_primary is primary  # fail-open keeps the original
    assert any("All candidates" in w for w in warnings)


def test_legacy_helper_noop_when_nothing_excluded(monkeypatch):
    import main as main_module

    monkeypatch.setattr(
        main_module, "apply_river_exclusion", lambda cands, w=None, **k: list(cands)
    )
    primary = {"id": "primary", "latitude": 1.0, "longitude": 2.0}
    recommendation = {"best_location": {"latitude": 1.0, "longitude": 2.0, "source": "s"}}
    warnings: list[str] = []

    new_primary, _alts, new_rec = main_module._apply_river_exclusion_to_legacy_candidates(
        primary, [], recommendation, warnings, "s"
    )

    assert new_primary is primary
    assert new_rec is recommendation
    assert warnings == []


def test_legacy_endpoint_surfaces_exclusion_warning(client, monkeypatch):
    import fixtures
    import main as main_module

    monkeypatch.setitem(settings.__dict__, "RIVER_EXCLUSION_ENABLED", True)

    def stub(candidates, warnings=None, **_kwargs):
        if isinstance(warnings, list):
            warnings.append("1 candidate(s) excluded within 30 m of rivers/water bodies.")
        return list(candidates)

    monkeypatch.setattr(main_module, "apply_river_exclusion", stub)

    name, data = fixtures.build_all()["valid_kml"]
    payload = client.post(
        "/analyzeContour", files={"file": (name, data, "application/xml")}
    ).json()

    assert payload["status"] == "success"
    assert any("excluded within 30 m" in warning for warning in payload["warnings"])
