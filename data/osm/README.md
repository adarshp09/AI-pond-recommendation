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

| Layer | OSM selector | Geometry |
|---|---|---|
| `water_areas` | `natural=water`, `landuse∈{reservoir,basin}`, `water=*` | Polygon / MultiPolygon |
| `waterways` | `waterway=*` | LineString |
| `roads` | `highway=*` | LineString |
| `buildings` | `building=*` | Polygon / MultiPoint source → Polygon |

Attribute columns: `osm_id`, `osm_type`, `tags` (JSON of the relevant OSM tags), `geometry`.

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
