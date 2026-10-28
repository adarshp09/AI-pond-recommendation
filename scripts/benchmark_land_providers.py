#!/usr/bin/env python3
"""Phase 05 — benchmark local vs Overpass land providers on identical study areas.

Compares feature counts and latency for the same bounding boxes using:

* ``local_only``     — the prepared GeoPackage (offline, deterministic)
* ``overpass_only``  — the live Overpass provider chain (network)

NOTE: ``overpass_only`` is an OFFLINE TOOLING path only. The running service never
contacts Overpass (default ``LAND_PROVIDER_MODE=local_only``); the Overpass chain is
retained solely for explicit comparison/benchmark runs like this one.

Run after preparing a dataset (``scripts/prepare_osm_data.py``)::

    python scripts/benchmark_land_providers.py
    python scripts/benchmark_land_providers.py --repeat 10 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.land_service import MODE_LOCAL_ONLY, MODE_OVERPASS_ONLY, fetch_land_context  # noqa: E402

STUDY_AREAS = {
    "sample_contours": (21.24, 81.28, 21.27, 81.31),
    "wider_region": (21.20, 81.25, 21.30, 81.35),
}


def _measure(south, west, north, east, mode, repeat):
    latencies = []
    payload = None
    for _ in range(repeat):
        started = time.perf_counter()
        payload = fetch_land_context(south, west, north, east, mode=mode)
        latencies.append((time.perf_counter() - started) * 1000.0)
    return payload, latencies


def _summary(payload, latencies):
    counts = payload.get("feature_counts") or {}
    return {
        "provider": payload.get("provider"),
        "status": payload.get("status"),
        "source": payload.get("source"),
        "water_bodies": counts.get("water_bodies", len(payload.get("water_bodies", []))),
        "roads": counts.get("roads", len(payload.get("roads", []))),
        "buildings": counts.get("buildings", len(payload.get("buildings", []))),
        "latency_ms_min": round(min(latencies), 2),
        "latency_ms_mean": round(sum(latencies) / len(latencies), 2),
        "latency_ms_max": round(max(latencies), 2),
        "samples": len(latencies),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark local vs Overpass land providers.")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--skip-overpass", action="store_true")
    args = parser.parse_args(argv)

    report = {"study_areas": {}, "repeat": args.repeat}
    for name, (south, west, north, east) in STUDY_AREAS.items():
        entry = {"bbox": {"south": south, "west": west, "north": north, "east": east}}
        local_payload, local_latencies = _measure(south, west, north, east, MODE_LOCAL_ONLY, args.repeat)
        entry["local"] = _summary(local_payload, local_latencies)

        if not args.skip_overpass:
            overpass_payload, overpass_latencies = _measure(south, west, north, east, MODE_OVERPASS_ONLY, max(1, min(args.repeat, 3)))
            entry["overpass"] = _summary(overpass_payload, overpass_latencies)

        report["study_areas"][name] = entry

    print(json.dumps(report, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
