"""Determinism and structure checks for the fixture suite itself."""

from __future__ import annotations

import io
import zipfile

import pytest

import fixtures
from backend.parser import parse_contour_file


def test_all_fixtures_are_deterministic():
    first = fixtures.build_all()
    second = fixtures.build_all()

    assert set(first) == set(second)
    assert set(first) == set(fixtures.CONTOUR_FIXTURES)
    for name in first:
        assert first[name] == second[name], name


def test_valid_kmz_is_a_zip_containing_doc_kml():
    with zipfile.ZipFile(io.BytesIO(fixtures.valid_kmz())) as archive:
        assert "doc.kml" in archive.namelist()
        assert archive.read("doc.kml") == fixtures.valid_kml()


def test_valid_kml_parses_into_expected_contour_count():
    contour_data = parse_contour_file(fixtures.valid_kml(), "fixture_valid.kml")
    assert contour_data.contour_count == 5
    assert contour_data.min_elevation_m == 267.0
    assert contour_data.max_elevation_m == 271.0


def test_valid_kmz_parses_like_its_kml():
    parsed = parse_contour_file(fixtures.valid_kmz(), "fixture_valid.kmz")
    assert parsed.contour_count == 5


def test_empty_fixture_is_empty():
    assert fixtures.empty_file() == b""


@pytest.mark.parametrize("name", ["malformed_kml", "invalid_geometry", "no_coordinates_text"])
def test_invalid_kml_fixtures_raise_value_error(name):
    data = fixtures.CONTOUR_FIXTURES[name]()
    with pytest.raises(ValueError):
        parse_contour_file(data, f"{name}.kml")


def test_missing_elevation_fixture_drops_non_elevation_placemarks():
    parsed = parse_contour_file(fixtures.missing_elevation_kml(), "missing_elevation.kml")
    assert parsed.contour_count == 3


def test_missing_coordinates_fixture_drops_coordinate_less_placemarks():
    parsed = parse_contour_file(fixtures.missing_coordinates_kml(), "missing_coordinates.kml")
    assert parsed.contour_count == 3


def test_hydrology_dem_fixtures_have_expected_geometry():
    import numpy as np

    assert fixtures.flat_dem((4, 4)).shape == (4, 4)
    assert fixtures.plane_dem(5, 5).shape == (5, 5)
    assert fixtures.cycle_flow_direction().shape == (2, 2)

    single = fixtures.single_valid_cell_dem()
    assert int(np.isfinite(single).sum()) == 1

    disconnected = fixtures.disconnected_dem()
    assert bool(np.all(np.isnan(disconnected[:, 2])))

    with_nan = fixtures.nan_dem(fraction_rows=2)
    assert int(np.isnan(with_nan).sum()) == 2 * with_nan.shape[1]
