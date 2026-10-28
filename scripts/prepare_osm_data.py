#!/usr/bin/env python3
"""Phase 01/06 — turn a regional OSM extract into an indexed GeoPackage.

Reads a Geofabrik ``.osm.pbf`` extract, clips it to the configured study area
(``Durg area`` by default), extracts only the water / road / building features the
geographic-context engine needs, validates and repairs geometries, and writes a
spatially indexed GeoPackage plus a provenance ``manifest.json``.

Geometry correctness (Phase 06):

* The bounding-box clip keeps **relations** (multipolygons) together with their
  member ways and those ways' nodes, so GDAL can assemble relation-based areas
  instead of silently losing them.
* **Interior rings (holes)** are preserved end to end: repaired geometries keep
  holes, layers are written as MultiPolygon/MultiLineString, and the provider
  emits them as ``holes`` on the element contract.
* Repair is explicit and measurable: invalid / repaired / skipped / duplicate
  counts with reasons are recorded in the manifest and printed.

Output layers (EPSG:4326, MultiPolygon/MultiLineString, spatially indexed):

* ``water_areas`` — ``natural=water`` / ``landuse∈{reservoir,basin}`` / ``water=*``
* ``waterways``   — ``waterway=*``
* ``roads``       — ``highway=*``
* ``buildings``   — ``building=*``

This is an operator/preprocessing tool. It is NEVER invoked during API requests.

Examples
--------
    python scripts/prepare_osm_data.py
    python scripts/prepare_osm_data.py --bbox 21.0,81.05,21.5,81.55
    python scripts/prepare_osm_data.py --no-clip        # input already clipped
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "osm" / "raw"
DEFAULT_PREPARED_DIR = PROJECT_ROOT / "data" / "osm" / "prepared"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "osm" / "manifest.json"
DEFAULT_BBOX = os.getenv("OSM_STUDY_BBOX", "21.0,81.05,21.5,81.55")  # south,west,north,east
STUDY_AREA_NAME = os.getenv("OSM_STUDY_AREA_NAME", "Durg area (Chhattisgarh, India)")
PROCESSING_VERSION = "2.0.0"
# Bump when the clip algorithm changes so a stale cached clip is never reused.
CLIP_ALGORITHM_VERSION = 3
ATTRIBUTION = "© OpenStreetMap contributors (ODbL 1.0)"

# Water/road/building tag rules (kept in sync with the Overpass query contract).
WATER_LANDUSE = {"reservoir", "basin"}
_LAYER_GEOMETRY_TYPE = {
    "water_areas": "MultiPolygon",
    "waterways": "MultiLineString",
    "roads": "MultiLineString",
    "buildings": "MultiPolygon",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 256), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be 'south,west,north,east'")
    south, west, north, east = (float(p) for p in parts)
    if not (south < north and west < east):
        raise argparse.ArgumentTypeError("bbox must satisfy south<north and west<east")
    return south, west, north, east


def _require_geostack():
    try:
        import geopandas as gpd  # noqa: F401
        import pyogrio  # noqa: F401
        import shapely  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "prepare_osm_data.py requires geopandas + pyogrio (pip install -r requirements-preprocess.txt)"
        ) from exc
    return gpd


# ---------------------------------------------------------------------------
# Relation-aware bounding-box pre-clip (pyosmium)
# ---------------------------------------------------------------------------

def clip_pbf(source: Path, destination: Path, bbox: tuple[float, float, float, float]) -> dict:
    """Write a bbox-clipped copy of an OSM PBF, preserving relations.

    Three passes so that a relation kept for the study area is complete: its
    member ways are included, and those ways' nodes are included even when they
    fall outside the bbox (otherwise GDAL cannot assemble the multipolygon).
    """

    try:
        import osmium
    except ImportError as exc:
        raise SystemExit(
            "Clipping requires pyosmium (pip install osmium). Re-run with --no-clip "
            "if the input extract already covers only the study area."
        ) from exc

    south, west, north, east = bbox
    inside = lambda lon, lat: west <= lon <= east and south <= lat <= north  # noqa: E731

    keep_nodes: set[int] = set()
    keep_ways: set[int] = set()
    keep_relations: set[int] = set()

    # Pass 1 — nodes/ways intersecting the bbox, then relations referencing them.
    class _Scan(osmium.SimpleHandler):
        def node(self, node):
            if node.location.valid() and inside(node.location.lon, node.location.lat):
                keep_nodes.add(node.id)

        def way(self, way):
            if way.id in keep_ways:
                return
            for ref in way.nodes:
                if ref.location.valid() and inside(ref.location.lon, ref.location.lat):
                    keep_ways.add(way.id)
                    return

        def relation(self, relation):
            # Only multipolygon relations are assembled into areas. Keeping
            # admin ``boundary`` / ``route`` relations would drag in their entire
            # member ways (state-sized geometry), so they are intentionally skipped.
            if relation.tags.get("type") != "multipolygon":
                return
            hit = any(
                (member.type == "w" and member.ref in keep_ways)
                or (member.type == "n" and member.ref in keep_nodes)
                for member in relation.members
            )
            if not hit:
                return
            keep_relations.add(relation.id)
            for member in relation.members:
                if member.type == "w":
                    keep_ways.add(member.ref)
                elif member.type == "n":
                    keep_nodes.add(member.ref)

    # Pass 2 — pull in every node referenced by a kept way (complete geometry).
    class _NodeRefs(osmium.SimpleHandler):
        def way(self, way):
            if way.id in keep_ways:
                for ref in way.nodes:
                    keep_nodes.add(ref.ref)

    # Pass 3 — write the kept subset.
    counts = {"nodes": 0, "ways": 0, "relations": 0}
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name.replace(".osm.pbf", ".partial.osm.pbf"))
    writer = osmium.SimpleWriter(str(tmp))

    class _Write(osmium.SimpleHandler):
        def node(self, node):
            if node.id in keep_nodes:
                writer.add_node(node)
                counts["nodes"] += 1

        def way(self, way):
            if way.id in keep_ways:
                writer.add_way(way)
                counts["ways"] += 1

        def relation(self, relation):
            if relation.id in keep_relations:
                writer.add_relation(relation)
                counts["relations"] += 1

    try:
        _Scan().apply_file(str(source), locations=True)
        counts["bbox_nodes"] = len(keep_nodes)
        counts["bbox_ways"] = len(keep_ways)
        counts["multipolygon_relations"] = len(keep_relations)
        _NodeRefs().apply_file(str(source))
        _Write().apply_file(str(source))
    finally:
        writer.close()
    tmp.replace(destination)
    return counts


# ---------------------------------------------------------------------------
# Tag / geometry helpers
# ---------------------------------------------------------------------------

def _parse_other_tags(value: Any) -> dict:
    """GDAL OSM exposes remaining tags as ``"k"=>"v","k2"=>"v2"``."""

    if not isinstance(value, str) or not value:
        return {}
    result: dict[str, str] = {}
    for pair in value.split('","'):
        token = pair.strip().strip('"')
        if '"=>"' not in token:
            continue
        key, _, val = token.partition('"=>"')
        key = key.strip('"')
        val = val.strip('"')
        if key:
            result[key] = val
    return result


def _feature_tags(row: dict, other: dict) -> dict:
    tags = dict(other)
    for key in ("highway", "waterway", "water", "natural", "landuse", "building"):
        value = row.get(key)
        if isinstance(value, str) and value:
            tags[key] = value
    return tags


def _tag(row: dict, other: dict, key: str):
    """Value for a tag, preferring the GDAL column but falling back to other_tags.

    Multipolygon **relation** features often expose their tags only through
    ``other_tags``, so both sources must be consulted or relation-based water
    areas / buildings would be silently misclassified.
    """

    value = row.get(key)
    if isinstance(value, str) and value:
        return value
    value = other.get(key)
    return value if isinstance(value, str) and value else None


def _is_water(other: dict, row: dict) -> bool:
    if _tag(row, other, "natural") == "water":
        return True
    if _tag(row, other, "landuse") in WATER_LANDUSE:
        return True
    return bool(_tag(row, other, "water"))


def _polygonal(geometry):
    """Normalise any polygon-ish geometry (incl. repaired collections) to MultiPolygon."""

    from shapely.geometry import MultiPolygon

    parts: list[Any] = []

    def collect(geom) -> None:
        if geom is None or geom.is_empty:
            return
        geom_type = geom.geom_type
        if geom_type == "Polygon":
            parts.append(geom)
        elif geom_type == "MultiPolygon":
            parts.extend(geom.geoms)
        elif geom_type == "GeometryCollection":
            for part in geom.geoms:
                collect(part)

    collect(geometry)
    parts = [p for p in parts if not p.is_empty and len(p.exterior.coords) >= 4]
    return MultiPolygon(parts) if parts else None


def _linear(geometry):
    """Normalise any line-ish geometry to MultiLineString."""

    from shapely.geometry import MultiLineString

    parts: list[Any] = []

    def collect(geom) -> None:
        if geom is None or geom.is_empty:
            return
        geom_type = geom.geom_type
        if geom_type == "LineString":
            parts.append(geom)
        elif geom_type == "MultiLineString":
            parts.extend(geom.geoms)
        elif geom_type == "GeometryCollection":
            for part in geom.geoms:
                collect(part)

    collect(geometry)
    parts = [p for p in parts if not p.is_empty and len(p.coords) >= 2]
    return MultiLineString(parts) if parts else None


def _interior_ring_count(geometry) -> int:
    total = 0
    if geometry is None or geometry.is_empty:
        return 0
    geom_type = geometry.geom_type
    if geom_type == "Polygon":
        return len(geometry.interiors)
    if geom_type == "MultiPolygon":
        return sum(len(poly.interiors) for poly in geometry.geoms)
    if geom_type == "GeometryCollection":
        return sum(_interior_ring_count(part) for part in geometry.geoms)
    return 0


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build(input_pbf: Path, output_gpkg: Path, manifest_path: Path, bbox, *, clip: bool) -> dict:
    gpd = _require_geostack()
    import shapely
    from shapely.geometry import box as shapely_box

    started = time.perf_counter()
    south, west, north, east = bbox
    clip_info: dict[str, Any] = {"applied": False}
    read_source = input_pbf

    if clip:
        digest = hashlib.sha1(
            f"v{CLIP_ALGORITHM_VERSION}:{input_pbf.name}:{bbox}".encode()
        ).hexdigest()[:12]
        clipped = input_pbf.with_name(f"{input_pbf.stem}.clip-{digest}.osm.pbf")
        if not clipped.exists():
            print(f"[prepare] clipping {input_pbf.name} to bbox {bbox} …")
            counts = clip_pbf(input_pbf, clipped, bbox)
            clip_info = {"applied": True, "algorithm_version": CLIP_ALGORITHM_VERSION,
                         "clipped_file": clipped.name, **counts}
        else:
            clip_info = {"applied": True, "algorithm_version": CLIP_ALGORITHM_VERSION,
                         "clipped_file": clipped.name, "cached": True}
        read_source = clipped

    study_box = shapely_box(west, south, east, north)

    def read_layer(name: str):
        return gpd.read_file(read_source, layer=name, engine="pyogrio", on_invalid="ignore")

    frames: dict[str, Any] = {}
    for layer in ("points", "lines", "multipolygons"):
        try:
            frames[layer] = read_layer(layer)
            if "osm_type" not in frames[layer].columns:
                frames[layer] = frames[layer].assign(osm_type="node" if layer == "points" else "way")
        except Exception as exc:  # pragma: no cover - driver dependent
            print(f"[prepare] warning: could not read layer {layer!r}: {exc}", file=sys.stderr)
            frames[layer] = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")

    def build_frame(source_df, predicate, normaliser) -> tuple[Any, dict]:
        stats = {
            "candidates": 0,
            "invalid": 0,
            "repaired": 0,
            "duplicates": 0,
            "skipped": {},
            "interior_rings": 0,
        }
        rows: list[dict] = []
        geoms: list[Any] = []
        seen: set[bytes] = set()

        def skip(reason: str) -> None:
            stats["skipped"][reason] = stats["skipped"].get(reason, 0) + 1

        for record in source_df.itertuples(index=False):
            row = record._asdict()
            other = _parse_other_tags(row.get("other_tags"))
            if not predicate(other, row):
                continue
            stats["candidates"] += 1

            geometry = row.get("geometry")
            if geometry is None or geometry.is_empty:
                skip("empty_geometry")
                continue
            if not geometry.intersects(study_box):
                skip("outside_study_area")
                continue

            if not shapely.is_valid(geometry):
                stats["invalid"] += 1
                repaired = shapely.make_valid(geometry)
                if repaired is not None and not repaired.is_empty:
                    stats["repaired"] += 1
                    geometry = repaired
                else:
                    skip("repair_failed")
                    continue

            normalised = normaliser(geometry)
            if normalised is None:
                skip("no_supported_geometry")
                continue
            if not shapely.is_valid(normalised):
                normalised = shapely.make_valid(normalised)
                normalised = normaliser(normalised)
                if normalised is None:
                    skip("no_supported_geometry_after_repair")
                    continue

            key = shapely.to_wkb(normalised)
            if key in seen:
                stats["duplicates"] += 1
                continue
            seen.add(key)

            stats["interior_rings"] += _interior_ring_count(normalised)
            rows.append({
                "osm_id": str(row.get("osm_id") or row.get("osm_way_id") or ""),
                "osm_type": str(row.get("osm_type") or "way"),
                "tags": json.dumps(_feature_tags(row, other), sort_keys=True),
            })
            geoms.append(normalised)

        frame = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
        return frame, stats

    water_areas, water_stats = build_frame(
        frames["multipolygons"], lambda other, row: _is_water(other, row), _polygonal
    )
    waterways, waterway_stats = build_frame(
        frames["lines"], lambda other, row: bool(_tag(row, other, "waterway")), _linear
    )
    roads, road_stats = build_frame(
        frames["lines"], lambda other, row: bool(_tag(row, other, "highway")), _linear
    )
    buildings, building_stats = build_frame(
        frames["multipolygons"], lambda other, row: bool(_tag(row, other, "building")), _polygonal
    )

    layers = {
        "water_areas": (water_areas, water_stats),
        "waterways": (waterways, waterway_stats),
        "roads": (roads, road_stats),
        "buildings": (buildings, building_stats),
    }

    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    tmp_gpkg = output_gpkg.with_suffix(output_gpkg.suffix + ".tmp")
    if tmp_gpkg.exists():
        tmp_gpkg.unlink()

    import pyogrio

    for layer_name, (frame, _stats) in layers.items():
        frame = frame.reset_index(drop=True)
        pyogrio.write_dataframe(
            frame,
            tmp_gpkg,
            layer=layer_name,
            driver="GPKG",
            append=layer_name != "water_areas",
            geometry_type=_LAYER_GEOMETRY_TYPE[layer_name],
            layer_options={"SPATIAL_INDEX": "YES"},
        )

    tmp_gpkg.replace(output_gpkg)

    verified = _verify_gpkg(output_gpkg, list(layers))
    elapsed = time.perf_counter() - started

    geometry_stats = {
        name: {
            **stats,
            "written": verified["layers"][name]["count"],
            "geometry_type": verified["layers"][name]["geometry"],
            "interior_rings": verified["layers"][name]["interior_rings"],
        }
        for name, (_frame, stats) in layers.items()
    }

    manifest = {
        "schema_version": 2,
        "processing_version": PROCESSING_VERSION,
        "clip_algorithm_version": CLIP_ALGORITHM_VERSION,
        "generated_at": _now_iso(),
        "crs": "EPSG:4326",
        "attribution": ATTRIBUTION,
        "study_area": {"south": south, "west": west, "north": north, "east": east},
        "study_area_name": STUDY_AREA_NAME,
        "source": {
            "provider": "Geofabrik",
            "extract": input_pbf.name,
            "sha256": _sha256(input_pbf),
            "extract_last_modified": clip_info.get("extract_last_modified"),
        },
        "clip": clip_info,
        "gpkg": {
            "path": str(output_gpkg),
            "filename": output_gpkg.name,
            "bytes": output_gpkg.stat().st_size,
            "sha256": _sha256(output_gpkg),
            "layers": list(layers),
            "spatial_index": verified["spatial_index"],
        },
        "layers": verified["layers"],
        "geometry_stats": geometry_stats,
        "feature_counts": {
            "water": verified["layers"]["water_areas"]["count"] + verified["layers"]["waterways"]["count"],
            "roads": verified["layers"]["roads"]["count"],
            "buildings": verified["layers"]["buildings"]["count"],
        },
        "interior_rings_total": sum(v["interior_rings"] for v in verified["layers"].values()),
        "build_seconds": round(elapsed, 2),
    }
    sidecar = input_pbf.with_suffix(input_pbf.suffix + ".download.json")
    if sidecar.exists():
        try:
            manifest["download"] = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _verify_gpkg(path: Path, expected_layers: list[str]) -> dict:
    import pyogrio
    import shapely

    found = {name for name, _ in pyogrio.list_layers(path)}
    missing = [layer for layer in expected_layers if layer not in found]
    if missing:
        raise SystemExit(f"GeoPackage {path} is missing layers: {missing}")

    layers: dict[str, Any] = {}
    for layer in expected_layers:
        info = pyogrio.read_info(path, layer=layer)
        interior = 0
        try:
            frame = pyogrio.read_dataframe(path, layer=layer, on_invalid="ignore", columns=[])
            interior = int(sum(_interior_ring_count(geom) for geom in frame.geometry))
        except Exception:  # pragma: no cover - defensive
            pass
        layers[layer] = {
            "count": int(info.get("features") or 0),
            "geometry": info.get("geometry_type"),
            "interior_rings": interior,
        }
    spatial_index = _has_spatial_index(path, expected_layers)
    return {"layers": layers, "spatial_index": spatial_index}


def _has_spatial_index(path: Path, layers: list[str]) -> bool:
    import sqlite3

    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            connection.close()
    except sqlite3.Error:  # pragma: no cover
        return False
    return all(f"rtree_{layer}_geom" in tables for layer in layers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare an indexed local OSM GeoPackage.")
    parser.add_argument("--input", type=Path, default=None, help="source .osm.pbf")
    parser.add_argument("--output", type=Path, default=DEFAULT_PREPARED_DIR / "land_context.gpkg")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--bbox", type=parse_bbox, default=parse_bbox(DEFAULT_BBOX))
    parser.add_argument("--no-clip", dest="clip", action="store_false", help="input is already clipped")
    parser.set_defaults(clip=True)
    args = parser.parse_args(argv)

    input_pbf = args.input
    if input_pbf is None:
        candidates = sorted(DEFAULT_RAW_DIR.glob("*.osm.pbf"))
        candidates = [c for c in candidates if ".clip-" not in c.name] or candidates
        if not candidates:
            print(f"[prepare] FAILED: no .osm.pbf in {DEFAULT_RAW_DIR}; run download_osm_data.py", file=sys.stderr)
            return 1
        input_pbf = candidates[-1]
    if not input_pbf.exists():
        print(f"[prepare] FAILED: {input_pbf} does not exist", file=sys.stderr)
        return 1

    manifest = build(input_pbf, args.output, args.manifest, args.bbox, clip=args.clip)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
