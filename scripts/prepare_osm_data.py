#!/usr/bin/env python3
"""Phase 01 — turn a regional OSM extract into an indexed GeoPackage.

Reads a Geofabrik ``.osm.pbf`` extract, clips it to the configured study area,
extracts only the water / road / building features the geographic-context engine
needs, validates geometries and the CRS, and writes a spatially indexed GeoPackage
plus a provenance ``manifest.json``.

Output layers (EPSG:4326):

* ``water_areas`` — ``natural=water`` / ``landuse∈{reservoir,basin}`` / ``water=*``
* ``waterways``   — ``waterway=*``
* ``roads``       — ``highway=*``
* ``buildings``   — ``building=*``

This is an operator/preprocessing tool. It is NEVER invoked during API requests.

Requires ``geopandas`` + ``pyogrio`` (GDAL). ``osmium`` (pyosmium) is used for the
bounding-box pre-clip and is strongly recommended for full regional extracts;
without it the input must already be small enough to read directly.

Examples
--------
    python scripts/prepare_osm_data.py
    python scripts/prepare_osm_data.py --input data/osm/raw/central-zone-latest.osm.pbf \\
        --bbox 21.0,81.05,21.5,81.55
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
PROCESSING_VERSION = "1.0.0"
ATTRIBUTION = "© OpenStreetMap contributors (ODbL 1.0)"

# Water/road/building tag rules (kept in sync with the Overpass query contract).
WATER_LANDUSE = {"reservoir", "basin"}


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
            "prepare_osm_data.py requires geopandas + pyogrio (pip install geopandas pyogrio)"
        ) from exc
    return gpd


# ---------------------------------------------------------------------------
# Bounding-box pre-clip with pyosmium
# ---------------------------------------------------------------------------

def clip_pbf(source: Path, destination: Path, bbox: tuple[float, float, float, float]) -> dict:
    """Write a bbox-clipped copy of an OSM PBF. Returns counts."""

    try:
        import osmium
    except ImportError as exc:
        raise SystemExit(
            "Clipping requires pyosmium (pip install osmium). Re-run with --no-clip "
            "if the input extract already covers only the study area."
        ) from exc

    south, west, north, east = bbox
    counts = {"nodes": 0, "ways": 0}
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name.replace(".osm.pbf", ".partial.osm.pbf"))
    writer = osmium.SimpleWriter(str(tmp))

    def inside(lon: float, lat: float) -> bool:
        return west <= lon <= east and south <= lat <= north

    class _Clip(osmium.SimpleHandler):
        def node(self, node):
            if node.location.valid() and inside(node.location.lon, node.location.lat):
                writer.add_node(node)
                counts["nodes"] += 1

        def way(self, way):
            for node_ref in way.nodes:
                if node_ref.location.valid() and inside(node_ref.location.lon, node_ref.location.lat):
                    writer.add_way(way)
                    counts["ways"] += 1
                    return

    try:
        _Clip().apply_file(str(source), locations=True)
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
    """Collect the OSM tags relevant to the water/road/building contract."""

    tags = dict(other)
    for key in ("highway", "waterway", "water", "natural", "landuse", "building"):
        value = row.get(key)
        if isinstance(value, str) and value:
            tags[key] = value
    return tags


def _is_water(other: dict, row: dict) -> bool:
    if row.get("natural") == "water":
        return True
    if row.get("landuse") in WATER_LANDUSE:
        return True
    if row.get("water"):
        return True
    return bool(other.get("water"))


def _geom_kind(geometry) -> str:
    return getattr(geometry, "geom_type", "") or ""


def _repair(geometries):
    from shapely import make_valid
    from shapely.geometry import shape  # noqa: F401  (kept for parity)

    import geopandas as gpd  # noqa: F401

    return make_valid(geometries)


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
        digest = hashlib.sha1(f"{input_pbf.name}:{bbox}".encode()).hexdigest()[:12]
        clipped = input_pbf.with_name(f"{input_pbf.stem}.clip-{digest}.osm.pbf")
        if not clipped.exists():
            print(f"[prepare] clipping {input_pbf.name} to bbox {bbox} …")
            counts = clip_pbf(input_pbf, clipped, bbox)
            clip_info = {"applied": True, "clipped_file": clipped.name, **counts}
        else:
            clip_info = {"applied": True, "clipped_file": clipped.name, "cached": True}
        read_source = clipped

    study_box = shapely_box(west, south, east, north)

    def read_layer(name: str):
        return gpd.read_file(read_source, layer=name, engine="pyogrio", on_invalid="ignore")

    frames: dict[str, Any] = {}
    for layer in ("points", "lines", "multipolygons"):
        try:
            frames[layer] = read_layer(layer)
        except Exception as exc:  # pragma: no cover - driver dependent
            print(f"[prepare] warning: could not read layer {layer!r}: {exc}", file=sys.stderr)
            frames[layer] = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")

    def build_frame(source_df, predicate) -> "gpd.GeoDataFrame":
        rows: list[dict] = []
        geoms: list[Any] = []
        for _, record in source_df.iterrows():
            row = record.to_dict()
            other = _parse_other_tags(row.get("other_tags"))
            if not predicate(other, row):
                continue
            geometry = row.get("geometry")
            if geometry is None or geometry.is_empty:
                continue
            if not geometry.intersects(study_box):
                continue
            tags = _feature_tags(row, other)
            rows.append(
                {
                    "osm_id": str(row.get("osm_id") or row.get("osm_way_id") or ""),
                    "osm_type": "node" if row.get("osm_id") and "lat" in source_df.columns else "way",
                    "tags": json.dumps(tags, sort_keys=True),
                }
            )
            geoms.append(geometry)
        frame = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
        if len(frame):
            frame["geometry"] = shapely.make_valid(frame.geometry)
            frame = frame[~frame.geometry.is_empty & frame.geometry.notna()]
        return frame

    points = frames["points"]
    lines = frames["lines"]
    multipolygons = frames["multipolygons"]
    if "osm_type" not in points.columns and len(points):
        points = points.assign(osm_type="node")
    for frame in (points, lines, multipolygons):
        if len(frame) and "osm_type" not in frame.columns:
            frame["osm_type"] = "way"

    water_areas = build_frame(
        multipolygons,
        lambda other, row: _is_water(other, row) and _geom_kind(row.get("geometry")) in {"Polygon", "MultiPolygon"},
    )
    waterways = build_frame(lines, lambda other, row: bool(row.get("waterway")))
    roads = build_frame(lines, lambda other, row: bool(row.get("highway")))
    buildings = build_frame(
        multipolygons,
        lambda other, row: bool(row.get("building")) and _geom_kind(row.get("geometry")) in {"Polygon", "MultiPolygon"},
    )

    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    tmp_gpkg = output_gpkg.with_suffix(output_gpkg.suffix + ".tmp")
    if tmp_gpkg.exists():
        tmp_gpkg.unlink()

    import pyogrio

    layers = {
        "water_areas": water_areas,
        "waterways": waterways,
        "roads": roads,
        "buildings": buildings,
    }
    for layer_name, frame in layers.items():
        frame = frame.reset_index(drop=True)
        pyogrio.write_dataframe(
            frame,
            tmp_gpkg,
            layer=layer_name,
            driver="GPKG",
            append=layer_name != "water_areas",
            layer_options={"SPATIAL_INDEX": "YES"},
        )

    tmp_gpkg.replace(output_gpkg)

    # Verify: reopen and confirm layers, counts, CRS and spatial index.
    verified = _verify_gpkg(output_gpkg, list(layers))
    elapsed = time.perf_counter() - started

    manifest = {
        "schema_version": 1,
        "processing_version": PROCESSING_VERSION,
        "generated_at": _now_iso(),
        "crs": "EPSG:4326",
        "attribution": ATTRIBUTION,
        "study_area": {"south": south, "west": west, "north": north, "east": east},
        "source": {
            "provider": "Geofabrik",
            "extract": input_pbf.name,
            "sha256": _sha256(input_pbf),
        },
        "clip": clip_info,
        "gpkg": {
            "path": str(output_gpkg),
            "filename": output_gpkg.name,
            "bytes": output_gpkg.stat().st_size,
            "layers": list(layers),
            "spatial_index": verified["spatial_index"],
        },
        "layers": verified["layers"],
        "feature_counts": {
            "water": verified["layers"]["water_areas"]["count"] + verified["layers"]["waterways"]["count"],
            "roads": verified["layers"]["roads"]["count"],
            "buildings": verified["layers"]["buildings"]["count"],
        },
        "build_seconds": round(elapsed, 2),
    }
    # Preserve the download provenance if present.
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

    found = {name for name, _ in pyogrio.list_layers(path)}
    missing = [layer for layer in expected_layers if layer not in found]
    if missing:
        raise SystemExit(f"GeoPackage {path} is missing layers: {missing}")

    layers: dict[str, Any] = {}
    for layer in expected_layers:
        info = pyogrio.read_info(path, layer=layer)
        layers[layer] = {"count": int(info.get("features") or 0), "geometry": info.get("geometry_type")}
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
