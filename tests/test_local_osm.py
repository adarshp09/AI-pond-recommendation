"""Phase 02 tests — local OSM GeoPackage provider.

Two layers of coverage:

* pure unit tests for the geometry → Overpass-element conversion (no I/O), and
* integration tests against a tiny synthetic GeoPackage built in ``tmp_path``.

Nothing here touches the network or the real regional dataset.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from services.land_providers import local_osm
from services.land_providers.base import (
    COVERAGE_FULL,
    COVERAGE_NONE,
    COVERAGE_PARTIAL,
    STATUS_AVAILABLE,
    STATUS_EMPTY,
    STATUS_MISSING,
    STATUS_OUT_OF_COVERAGE,
    STATUS_STALE,
    STATUS_UNREADABLE,
    ProviderUnavailable,
    bbox_relation,
)
from services.land_providers.local_osm import (
    LocalOSMProvider,
    _element_for_part,
    _frame_to_elements,
    _geometry_parts,
)

pytest.importorskip("shapely")
from shapely.geometry import LineString, MultiPolygon, Point, Polygon  # noqa: E402

LON, LAT = 81.30, 21.26
STUDY = (21.0, 81.05, 21.5, 81.55)  # south, west, north, east


# ---------------------------------------------------------------------------
# Fixture dataset
# ---------------------------------------------------------------------------

def _square(lon: float, lat: float, half: float = 0.001) -> Polygon:
    return Polygon([
        (lon - half, lat - half),
        (lon + half, lat - half),
        (lon + half, lat + half),
        (lon - half, lat + half),
        (lon - half, lat - half),
    ])


def _write_gpkg(path, *, generated_at: str | None = None, study_area=STUDY, layers=("water_areas", "waterways", "roads", "buildings"), water_geometry=None):
    gpd = pytest.importorskip("geopandas")
    import pyogrio

    for layer in layers:
        if layer == "water_areas":
            frame = gpd.GeoDataFrame(
                {"osm_id": ["1"], "osm_type": ["way"], "tags": [json.dumps({"natural": "water"})]},
                geometry=[water_geometry if water_geometry is not None else _square(LON, LAT)],
                crs="EPSG:4326",
            )
        elif layer == "waterways":
            frame = gpd.GeoDataFrame(
                {"osm_id": ["2"], "osm_type": ["way"], "tags": [json.dumps({"waterway": "stream"})]},
                geometry=[LineString([(LON - 0.01, LAT), (LON + 0.01, LAT)])],
                crs="EPSG:4326",
            )
        elif layer == "roads":
            frame = gpd.GeoDataFrame(
                {"osm_id": ["3"], "osm_type": ["way"], "tags": [json.dumps({"highway": "residential"})]},
                geometry=[LineString([(LON + 0.001, LAT - 0.01), (LON + 0.001, LAT + 0.01)])],
                crs="EPSG:4326",
            )
        else:
            frame = gpd.GeoDataFrame(
                {"osm_id": ["4"], "osm_type": ["way"], "tags": [json.dumps({"building": "yes"})]},
                geometry=[_square(LON, LAT + 0.001)],
                crs="EPSG:4326",
            )
        pyogrio.write_dataframe(frame, path, layer=layer, driver="GPKG", append=layer != layers[0])

    manifest = {
        "generated_at": generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "study_area": {"south": study_area[0], "west": study_area[1], "north": study_area[2], "east": study_area[3]},
    }
    path.with_name("manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.fixture
def dataset(tmp_path):
    gpkg = tmp_path / "land_context.gpkg"
    _write_gpkg(gpkg)
    return LocalOSMProvider(gpkg, manifest_path=tmp_path / "manifest.json", max_age_days=180)


# ---------------------------------------------------------------------------
# 1. Pure conversion unit tests (no I/O)
# ---------------------------------------------------------------------------

def test_element_for_polygon_emits_closed_ring():
    element = _element_for_part(_square(LON, LAT), "1", "way", {"natural": "water"}, 0)
    assert element["type"] == "way"
    assert element["tags"] == {"natural": "water"}
    coords = element["geometry"]
    assert coords[0] == coords[-1]
    assert len(coords) >= 4
    assert set(coords[0]) == {"lat", "lon"}


def test_element_for_line_and_point():
    line = _element_for_part(LineString([(LON, LAT), (LON + 0.001, LAT)]), "2", "way", {"highway": "x"}, 0)
    assert "geometry" in line and len(line["geometry"]) == 2
    point = _element_for_part(Point(LON, LAT), "5", "node", {"building": "yes"}, 0)
    assert point["lat"] == pytest.approx(LAT)
    assert point["lon"] == pytest.approx(LON)
    assert "geometry" not in point


def test_element_id_is_suffixed_for_multiparts():
    multi = MultiPolygon([_square(LON, LAT, 0.001), _square(LON + 0.01, LAT, 0.001)])
    parts = _geometry_parts(multi)
    assert len(parts) == 2
    assert _element_for_part(parts[0], "9", "way", {}, 0)["id"] == "9"
    assert _element_for_part(parts[1], "9", "way", {}, 1)["id"] == "9#1"


def test_element_for_degenerate_ring_is_dropped():
    assert _element_for_part(Polygon(), "1", "way", {}, 0) is None


def test_frame_to_elements_handles_empty_and_tags():
    assert _frame_to_elements(None, "roads") == []

    class _Frame:
        def __len__(self):
            return 0

    assert _frame_to_elements(_Frame(), "roads") == []


# ---------------------------------------------------------------------------
# 2. bbox relation helper
# ---------------------------------------------------------------------------

def test_bbox_relation_full_partial_none():
    assert bbox_relation((21.1, 81.1, 21.2, 81.2), STUDY) == COVERAGE_FULL
    assert bbox_relation((20.5, 81.1, 21.1, 81.2), STUDY) == COVERAGE_PARTIAL
    assert bbox_relation((10.0, 70.0, 10.1, 70.1), STUDY) == COVERAGE_NONE


# ---------------------------------------------------------------------------
# 3. Dataset descriptor / availability
# ---------------------------------------------------------------------------

def test_describe_reports_coverage_and_layers(dataset):
    info = dataset.describe()
    assert info["readable"] is True
    assert info["coverage"] == list(STUDY)
    assert set(info["layers"]) >= {"water_areas", "roads", "buildings"}
    assert info["age_days"] is not None


def test_available_and_covers(dataset):
    assert dataset.available() is True
    assert dataset.covers(21.24, 81.28, 21.27, 81.31) is True
    assert dataset.coverage_status(10.0, 70.0, 10.1, 70.1) == COVERAGE_NONE


def test_missing_dataset_is_unavailable(tmp_path):
    provider = LocalOSMProvider(tmp_path / "nope.gpkg", manifest_path=tmp_path / "manifest.json")
    assert provider.available() is False
    assert provider.describe()["exists"] is False
    with pytest.raises(ProviderUnavailable) as exc:
        provider.fetch(21.24, 81.28, 21.27, 81.31)
    assert exc.value.status == STATUS_MISSING


def test_empty_path_is_unavailable():
    provider = LocalOSMProvider("")
    assert provider.available() is False
    with pytest.raises(ProviderUnavailable) as exc:
        provider.fetch(21.24, 81.28, 21.27, 81.31)
    assert exc.value.status == STATUS_MISSING


def test_corrupt_dataset_is_unreadable(tmp_path):
    bad = tmp_path / "corrupt.gpkg"
    bad.write_bytes(b"this is not a geopackage at all")
    (tmp_path / "manifest.json").write_text(json.dumps({"study_area": {"south": 21.0, "west": 81.0, "north": 21.5, "east": 81.5}}))
    provider = LocalOSMProvider(bad, manifest_path=tmp_path / "manifest.json")
    assert provider.available() is False
    with pytest.raises(ProviderUnavailable) as exc:
        provider.fetch(21.24, 81.28, 21.27, 81.31)
    assert exc.value.status == STATUS_UNREADABLE


def test_stale_dataset_is_rejected(tmp_path):
    gpkg = tmp_path / "land_context.gpkg"
    old = (datetime.now(timezone.utc) - timedelta(days=400)).replace(microsecond=0).isoformat()
    _write_gpkg(gpkg, generated_at=old)
    provider = LocalOSMProvider(gpkg, manifest_path=tmp_path / "manifest.json", max_age_days=180)
    assert provider.is_stale() is True
    assert provider.available() is False
    with pytest.raises(ProviderUnavailable) as exc:
        provider.fetch(21.24, 81.28, 21.27, 81.31)
    assert exc.value.status == STATUS_STALE


def test_out_of_coverage_query_is_rejected(dataset):
    with pytest.raises(ProviderUnavailable) as exc:
        dataset.fetch(10.0, 70.0, 10.1, 70.1)
    assert exc.value.status == STATUS_OUT_OF_COVERAGE


def test_partially_covered_query_is_rejected(dataset):
    with pytest.raises(ProviderUnavailable) as exc:
        dataset.fetch(20.5, 81.1, 21.1, 81.2)  # crosses the study-area south edge
    assert exc.value.status == STATUS_OUT_OF_COVERAGE


# ---------------------------------------------------------------------------
# 4. Query results
# ---------------------------------------------------------------------------

def test_fetch_returns_normalised_elements(dataset):
    result = dataset.fetch(21.24, 81.28, 21.27, 81.31)
    assert result.status == STATUS_AVAILABLE
    assert result.provider == "local_osm"
    assert result.coverage == COVERAGE_FULL
    assert result.feature_counts["roads"] >= 1
    assert result.feature_counts["buildings"] >= 1
    assert result.water_bodies, "expected water features in the query bbox"
    assert all("tags" in element for element in result.roads)
    assert all("tags" in element for element in result.water_bodies)


def test_fetch_empty_bbox_is_valid_empty_not_error(tmp_path):
    # Dataset covers the study area, but query a covered corner with no features.
    gpkg = tmp_path / "land_context.gpkg"
    _write_gpkg(gpkg)
    provider = LocalOSMProvider(gpkg, manifest_path=tmp_path / "manifest.json", max_age_days=180)
    result = provider.fetch(21.40, 81.45, 21.42, 81.47)
    assert result.status == STATUS_EMPTY
    assert result.water_bodies == [] and result.roads == [] and result.buildings == []


def test_query_buffer_expands_the_read(dataset):
    wide = LocalOSMProvider(dataset.gpkg_path, manifest_path=dataset.manifest_path, query_buffer_m=500.0)
    assert wide._buffer(21.24, 81.28, 21.27, 81.31) != (21.24, 81.28, 21.27, 81.31)


def test_descriptor_cache_is_reused(dataset):
    first = dataset.describe()
    second = dataset.describe()
    assert first == second


def test_interior_rings_survive_gpkg_roundtrip(tmp_path):
    """A polygon hole in the GeoPackage is preserved on the element contract."""
    from shapely.geometry import Polygon

    gpkg = tmp_path / "holes.gpkg"
    holed = Polygon(
        [(LON - 0.002, LAT - 0.002), (LON + 0.002, LAT - 0.002), (LON + 0.002, LAT + 0.002), (LON - 0.002, LAT + 0.002), (LON - 0.002, LAT - 0.002)],
        [[(LON - 0.0005, LAT - 0.0005), (LON + 0.0005, LAT - 0.0005), (LON + 0.0005, LAT + 0.0005), (LON - 0.0005, LAT + 0.0005), (LON - 0.0005, LAT - 0.0005)]],
    )
    _write_gpkg(gpkg, water_geometry=holed)
    provider = LocalOSMProvider(gpkg, manifest_path=tmp_path / "manifest.json", max_age_days=180)

    result = provider.fetch(21.24, 81.28, 21.27, 81.31)
    water = result.water_bodies[0]
    assert "holes" in water and len(water["holes"]) == 1
    assert len(water["geometry"]) >= 4


def test_dataset_missing_required_layer_is_unreadable(tmp_path):
    gpkg = tmp_path / "partial.gpkg"
    _write_gpkg(gpkg, layers=("water_areas", "waterways", "roads"))  # no buildings layer
    provider = LocalOSMProvider(gpkg, manifest_path=tmp_path / "manifest.json")

    assert provider.available() is False
    assert "missing layers" in provider.describe()["error"]
    with pytest.raises(ProviderUnavailable) as exc:
        provider.fetch(21.24, 81.28, 21.27, 81.31)
    assert exc.value.status == STATUS_UNREADABLE


# ---------------------------------------------------------------------------
# 5. Downstream integration: real distances from local features
# ---------------------------------------------------------------------------

def test_local_features_drive_geographic_feasibility(dataset):
    import numpy as np
    from pyproj import Transformer

    from backend.geographic_context import GeographicContextConfig, enrich_candidates

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
    x, y = transformer.transform(LON, LAT)
    candidate = {
        "candidate_id": "a",
        "x": float(x),
        "y": float(y),
        "crs": "EPSG:32644",
        "latitude": LAT,
        "longitude": LON,
    }
    result = enrich_candidates(
        [candidate],
        config=GeographicContextConfig(cache_enabled=False, cluster_radius_m=500.0),
        provider=lambda s, w, n, e: dataset.fetch(s, w, n, e).to_payload(),
        cache=None,
    )
    enriched = result["candidates"][0]
    assert enriched["provider_ok"] is True
    assert enriched["water_distance_m"] == pytest.approx(0.0, abs=1.0)   # inside the water polygon
    assert enriched["feasibility"] == "rejected"
    assert enriched["feasibility_reason"] == "inside_or_on_water_body"
    assert enriched["geographic_source"].startswith("Local OSM GeoPackage")
