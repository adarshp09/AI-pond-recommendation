"""Phase 06 — OSM geometry correctness regression tests.

Covers polygons, polygons with holes, multipolygons, multiple outer/inner rings,
geometry clipping, CRS handling, and point containment / distance around polygon
boundaries and holes. Deterministic and fully offline.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from services.land_providers.local_osm import (
    _element_for_part,
    _frame_to_elements,
    _geometry_parts,
)

pytest.importorskip("shapely")

from pyproj import Transformer  # noqa: E402
from shapely.geometry import (  # noqa: E402
    GeometryCollection,
    MultiPolygon,
    Point,
    Polygon,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LON, LAT = 81.30, 21.26
METRIC_CRS = "EPSG:32644"
_FWD = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_osm_data", PROJECT_ROOT / "scripts" / "prepare_osm_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _load_prepare_module()


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def ring(lon: float, lat: float, half: float):
    return [(lon - half, lat - half), (lon + half, lat - half), (lon + half, lat + half), (lon - half, lat + half), (lon - half, lat - half)]


def ring_element(ring_coords):
    return [{"lat": lat, "lon": lon} for lon, lat in ring_coords]


def polygon_with_hole(centre_lon=LON, centre_lat=LAT, outer=0.001, inner=0.0004):
    return Polygon(ring(centre_lon, centre_lat, outer), [ring(centre_lon, centre_lat, inner)])


# ---------------------------------------------------------------------------
# 1. Interior rings survive the provider element contract
# ---------------------------------------------------------------------------

def test_element_preserves_interior_ring():
    element = _element_for_part(polygon_with_hole(), "42", "way", {"natural": "water"}, 0)
    assert "holes" in element and len(element["holes"]) == 1
    hole = element["holes"][0]
    assert hole[0] == hole[-1] and len(hole) >= 4
    assert element["geometry"][0] == element["geometry"][-1]


def test_element_without_hole_has_no_holes_key():
    element = _element_for_part(Polygon(ring(LON, LAT, 0.001)), "1", "way", {}, 0)
    assert "holes" not in element


def test_multiple_holes_preserved():
    multi = Polygon(
        ring(LON, LAT, 0.002),
        [ring(LON - 0.0008, LAT, 0.0003), ring(LON + 0.0008, LAT, 0.0003)],
    )
    element = _element_for_part(multi, "7", "way", {}, 0)
    assert len(element["holes"]) == 2


# ---------------------------------------------------------------------------
# 2. Multipolygons and nested collections flatten correctly
# ---------------------------------------------------------------------------

def test_multipolygon_flattens_to_one_element_per_outer():
    multi = MultiPolygon([polygon_with_hole(LON, LAT), Polygon(ring(LON + 0.01, LAT, 0.001))])
    parts = _geometry_parts(multi)
    assert len(parts) == 2
    elements = [_element_for_part(p, "9", "way", {}, i) for i, p in enumerate(parts)]
    assert elements[0]["id"] == "9" and elements[1]["id"] == "9#1"
    assert "holes" in elements[0]          # first outer carries its hole
    assert "holes" not in elements[1]


def test_nested_geometry_collection_flattens():
    nested = GeometryCollection([MultiPolygon([Polygon(ring(LON, LAT, 0.001))]), Point(LON, LAT)])
    parts = _geometry_parts(nested)
    assert len(parts) == 2
    assert {p.geom_type for p in parts} == {"Polygon", "Point"}


def test_frame_to_elements_emits_holes_from_frame():
    gpd = pytest.importorskip("geopandas")
    frame = gpd.GeoDataFrame(
        {"osm_id": ["1"], "osm_type": ["way"], "tags": ['{"natural": "water"}']},
        geometry=[polygon_with_hole()],
        crs="EPSG:4326",
    )
    elements = _frame_to_elements(frame, "water_bodies")
    assert len(elements) == 1
    assert len(elements[0]["holes"]) == 1


# ---------------------------------------------------------------------------
# 3. Containment / distance around boundaries and holes
# ---------------------------------------------------------------------------

def _enrich(payload_water, lon, lat):
    from backend.geographic_context import GeographicContextConfig, enrich_candidates

    x, y = _FWD.transform(lon, lat)
    candidate = {"candidate_id": "c", "x": float(x), "y": float(y), "crs": METRIC_CRS, "latitude": lat, "longitude": lon}
    payload = {
        "bbox": {"south": lat - 1, "west": lon - 1, "north": lat + 1, "east": lon + 1},
        "water_bodies": payload_water,
        "roads": [],
        "buildings": [],
        "source": "fixture",
        "provider": "local_osm",
        "status": "available",
    }
    return enrich_candidates(
        [candidate],
        config=GeographicContextConfig(cache_enabled=False, cluster_radius_m=500.0),
        provider=lambda s, w, n, e: payload,
        cache=None,
    )["candidates"][0]


def test_point_inside_hole_is_not_inside_water():
    element = _element_for_part(polygon_with_hole(), "1", "way", {"natural": "water"}, 0)
    enriched = _enrich([element], LON, LAT)  # centre of the hole
    assert enriched["provider_ok"] is True
    assert enriched["water_distance_m"] > 0.0            # hole is not water
    assert enriched["feasibility"] != "rejected"


def test_point_inside_shell_is_inside_water():
    element = _element_for_part(polygon_with_hole(), "1", "way", {"natural": "water"}, 0)
    # Inside the shell but outside the hole.
    enriched = _enrich([element], LON + 0.0007, LAT)
    assert enriched["water_distance_m"] == pytest.approx(0.0, abs=1e-6)
    assert enriched["feasibility"] == "rejected"
    assert enriched["feasibility_reason"] == "inside_or_on_water_body"


def test_distance_to_polygon_boundary_is_metric():
    element = _element_for_part(Polygon(ring(LON, LAT, 0.001)), "1", "way", {"natural": "water"}, 0)
    # ~0.003 deg east of the polygon edge (~2.1 km at this latitude) -> outside.
    enriched = _enrich([element], LON + 0.003, LAT)
    assert enriched["water_distance_m"] == pytest.approx(210.0, abs=40.0)
    assert enriched["feasibility"] == "feasible"


# ---------------------------------------------------------------------------
# 4. Preprocessing helpers: clipping, repair, rings
# ---------------------------------------------------------------------------

def test_prepare_bbox_parsing():
    assert prepare.parse_bbox("21.0,81.05,21.5,81.55") == (21.0, 81.05, 21.5, 81.55)
    with pytest.raises(Exception):
        prepare.parse_bbox("21.5,81.05,21.0,81.55")


def test_polygonal_normaliser_keeps_holes_and_drops_degenerate():
    multi = prepare._polygonal(polygon_with_hole())
    assert multi.geom_type == "MultiPolygon"
    assert len(multi.geoms) == 1
    assert len(multi.geoms[0].interiors) == 1
    assert prepare._polygonal(Polygon()) is None
    assert prepare._polygonal(Point(LON, LAT)) is None


def test_linear_normaliser():
    from shapely.geometry import LineString

    assert prepare._linear(LineString([(LON, LAT), (LON + 0.001, LAT)])).geom_type == "MultiLineString"
    assert prepare._linear(Polygon(ring(LON, LAT, 0.001))) is None


def test_interior_ring_count_recurses():
    multi = MultiPolygon([polygon_with_hole(LON, LAT), Polygon(ring(LON + 0.01, LAT, 0.001))])
    assert prepare._interior_ring_count(multi) == 1
    assert prepare._interior_ring_count(GeometryCollection([multi])) == 1
    assert prepare._interior_ring_count(Polygon(ring(LON, LAT, 0.001))) == 0


def test_repair_makes_invalid_polygon_valid_and_keeps_area():
    import shapely

    bowtie = Polygon([(LON, LAT), (LON + 0.002, LAT + 0.002), (LON + 0.002, LAT), (LON, LAT + 0.002), (LON, LAT)])
    assert not shapely.is_valid(bowtie)
    repaired = shapely.make_valid(bowtie)
    normalised = prepare._polygonal(repaired)
    assert normalised is not None and shapely.is_valid(normalised)
    # The bowtie's two lobes each have area 0.5*0.002*0.002 = 2e-6; the invalid
    # ring's own shoelace area cancels to 0, so compare to the lobes, not to it.
    assert normalised.area == pytest.approx(2e-6, rel=0.1)


def test_clipping_preserves_hole_and_clips_to_bbox():
    from shapely.geometry import box as shapely_box

    import shapely

    study = shapely_box(81.05, 21.0, 81.55, 21.5)
    # Outer ring straddles the study-area west edge; the hole sits fully inside,
    # so clipping the outer must not destroy the interior ring.
    straddling = polygon_with_hole(81.055, 21.25, outer=0.01, inner=0.002)
    assert straddling.intersects(study)
    assert not study.contains(straddling)  # genuinely straddling

    clipped = shapely.intersection(prepare._polygonal(straddling), study)
    normalised = prepare._polygonal(clipped)
    assert normalised is not None
    assert normalised.within(study.buffer(1e-6))
    assert prepare._interior_ring_count(normalised) >= 1  # hole survived the clip


def test_water_tag_rules():
    assert prepare._is_water({}, {"natural": "water"})
    assert prepare._is_water({"water": "reservoir"}, {})
    assert prepare._is_water({}, {"landuse": "basin"})
    assert not prepare._is_water({}, {"landuse": "forest"})


def test_classification_falls_back_to_other_tags_for_relations():
    """Multipolygon relations often expose tags only via other_tags."""
    # natural only present in other_tags (typical for a relation feature)
    assert prepare._is_water({"natural": "water"}, {})
    assert prepare._is_water({}, {"natural": "water"})
    assert prepare._is_water({}, {"water": "lake"})
    # building / highway / waterway via other_tags
    assert prepare._tag({}, {"building": "yes"}, "building") == "yes"
    assert prepare._tag({"building": "yes"}, {"building": "no"}, "building") == "yes"
    assert prepare._tag({}, {"highway": "primary"}, "highway") == "primary"
    assert prepare._tag({}, {}, "waterway") is None


def test_parse_other_tags():
    tags = prepare._parse_other_tags('"water"=>"reservoir","source"=>"survey"')
    assert tags == {"water": "reservoir", "source": "survey"}
    assert prepare._parse_other_tags(None) == {}


# ---------------------------------------------------------------------------
# 5. CRS handling
# ---------------------------------------------------------------------------

def test_element_coordinates_are_wgs84_and_project_to_metres():
    element = _element_for_part(Polygon(ring(LON, LAT, 0.001)), "1", "way", {}, 0)
    lon, lat = element["geometry"][0]["lon"], element["geometry"][0]["lat"]
    assert 81.0 < lon < 81.6 and 21.0 < lat < 21.6  # degrees, not metres

    enriched = _enrich([element], LON + 0.005, LAT)
    # 0.005 deg lon near 21.26 N ~= 519 m; must be reported in metres.
    assert 400.0 < enriched["water_distance_m"] < 700.0
