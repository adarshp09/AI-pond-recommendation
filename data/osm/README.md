# Local OSM data (`data/osm`)

Local-first geographic data for the land-context engine (Phases 01–05). The API reads a
prepared, spatially indexed **GeoPackage**; nothing here is downloaded or processed during
a request.

## Layout

```
data/osm/
├── README.md           # this file (committed)
├── manifest.json       # generated provenance/coverage (committed)
├── raw/                # downloaded extracts            (git-ignored, large)
│   └── central-zone-latest.osm.pbf
└── prepared/           # generated GeoPackage            (git-ignored, large)
    └── land_context.gpkg
```

## Source & attribution

* Source: **Geofabrik** regional OpenStreetMap extracts (<https://download.geofabrik.de/>), default `asia/india/central-zone-latest.osm.pbf`.
* Data: © OpenStreetMap contributors, licensed **ODbL 1.0**.
* Coverage and extract date are recorded in `manifest.json`.

## GeoPackage layers (EPSG:4326, spatially indexed)

Default study area: **Durg area (Chhattisgarh, India)** — `south=21.0, west=81.05, north=21.5, east=81.55`.

| Layer | OSM selector | Geometry |
|---|---|---|
| `water_areas` | `natural=water`, `landuse∈{reservoir,basin}`, `water=*` | MultiPolygon |
| `waterways` | `waterway=*` | MultiLineString |
| `roads` | `highway=*` | MultiLineString |
| `buildings` | `building=*` | MultiPolygon |

Attribute columns: `osm_id`, `osm_type`, `tags` (JSON of the relevant OSM tags), `geometry`.

## Geometry correctness (Phase 06)

* The bbox clip is **relation-aware**: relations intersecting the study area are kept
  together with their member ways and those ways' nodes, so GDAL can assemble
  multipolygon areas instead of losing them.
* **Interior rings (holes)** are preserved through repair, clipping and writing
  (`MultiPolygon` layers keep holes), and are emitted by the provider as `holes` on the
  element contract so a point inside a hole is not treated as inside the feature.
* Geometry repair is explicit and measurable — `geometry_stats` in `manifest.json`
  records `candidates`, `invalid`, `repaired`, `duplicates`, `interior_rings` and
  `skipped` reasons (`empty_geometry`, `outside_study_area`, `repair_failed`,
  `no_supported_geometry`). Unsupported/failed geometries are dropped, never faked.

## Reproduce

```bash
# 1. Download (validates HTTP, size, PBF magic, and Geofabrik MD5)
python scripts/download_osm_data.py                       # default source
python scripts/download_osm_data.py --url <geofabrik-url> # explicit source
python scripts/download_osm_data.py --verify-only         # validate existing file

# 2. Prepare (bbox-clip → classify → validate → index → manifest)
python scripts/prepare_osm_data.py --bbox 21.0,81.05,21.5,81.55
python scripts/prepare_osm_data.py --no-clip              # input already clipped
```

Requirements: `geopandas`, `pyogrio` (GDAL), and `osmium` (pyosmium) for the bbox
pre-clip — install with `pip install -r requirements-preprocess.txt`. These are
preprocessing-only; the API never needs them.

## Rules

* Large extracts (`*.osm.pbf`) and GeoPackages (`*.gpkg`) are **never committed** (`.gitignore`).
* `manifest.json` and this README are committed.
* The dataset is machine-local: prepare it independently on each machine, or point
  `LOCAL_OSM_GPKG_PATH` at a verified local file.
