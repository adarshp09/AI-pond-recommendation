#!/usr/bin/env python3
"""Phase 06 — API smoke test for the land provider modes.

Exercises ``/analyzeContour`` and ``/analyzeLocation`` end to end with external
services stubbed, for one land-provider mode:

    python scripts/api_smoke_test.py --mode local_first
    python scripts/api_smoke_test.py --mode local_only
    python scripts/api_smoke_test.py --mode overpass_only

NOTE: ``--mode overpass_only`` is an OFFLINE TOOLING path only. The running
service never contacts Overpass (``LAND_PROVIDER_MODE=local_only``,
``OVERPASS_FALLBACK_ENABLED=false``); Overpass support is retained purely for
this kind of explicit, manual verification.

Prints a JSON summary. Exit code is non-zero if an endpoint does not return
``status == "success"``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="API smoke test for land provider modes.")
    parser.add_argument("--mode", default="local_first",
                        choices=["local_first", "local_only", "overpass_only"])
    parser.add_argument("--location", default="21.25,81.30", help="lat,lon inside coverage")
    args = parser.parse_args(argv)

    os.environ["LAND_PROVIDER_MODE"] = args.mode
    os.environ.setdefault("LOCAL_OSM_ENABLED", "true")
    os.environ.setdefault("OPENTOPOGRAPHY_API_KEY", "smoke-test")
    os.environ.setdefault("CACHE_ENABLED", "false")

    import numpy as np
    from fastapi.testclient import TestClient

    import main  # noqa: E402  (root shim -> backend.main)

    def fake_dem(*_a, **_k):
        yy, xx = np.mgrid[0:6, 0:6]
        elevation = 100.0 + yy * 0.5 + xx * 0.25
        return {
            "elevation": elevation, "rows": 6, "columns": 6,
            "elevation_min_m": float(elevation.min()), "elevation_max_m": float(elevation.max()),
            "elevation_mean_m": float(elevation.mean()), "source": "smoke-fake",
        }

    main.get_dem_for_bbox = fake_dem
    main.fetch_historical_rainfall = lambda *a, **k: {
        "status": "ok", "total_precipitation_mm": 120.0, "daily": [], "source": "smoke-fake"}
    main.search_places = lambda query, *a, **k: {"query": query, "results": [], "source": "smoke-fake"}
    main.geocode_place = lambda query, *a, **k: {"query": query, "results": [], "source": "smoke-fake"}

    client = TestClient(main.app)
    kml = (PROJECT_ROOT / "sample_contours.kml").read_bytes()
    lat, lon = (float(v) for v in args.location.split(","))

    contour = client.post(
        "/analyzeContour",
        files={"file": ("sample_contours.kml", kml, "application/xml")},
    )
    location = client.post("/analyzeLocation", json={"latitude": lat, "longitude": lon, "radius_km": 3})

    contour_body = contour.json()
    location_body = location.json()
    land = (contour_body.get("enrichment") or {}).get("land_context") or {}

    report = {
        "mode": args.mode,
        "analyzeContour": {
            "http": contour.status_code,
            "status": contour_body.get("status"),
            "land_provider": land.get("provider"),
            "land_status": land.get("status"),
            "land_counts": land.get("feature_counts"),
        },
        "analyzeLocation": {
            "http": location.status_code,
            "status": location_body.get("status"),
            "has_pond_candidate": location_body.get("pond_candidate") is not None,
        },
    }
    print(json.dumps(report, indent=2))
    ok = report["analyzeContour"]["status"] == "success" and report["analyzeLocation"]["status"] == "success"
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
