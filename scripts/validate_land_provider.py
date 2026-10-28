#!/usr/bin/env python3
"""Phase 06 — feasibility and performance validation on a fixed candidate set.

Runs the geographic-context engine over a deterministic 5x5 grid of candidate
points spanning the configured study area, using:

* ``local_only``     — the prepared GeoPackage (offline, deterministic)
* ``overpass_only``  — the live Overpass chain, when reachable

Reports distances, feasibility labels, unknown classifications, feature counts,
latency and peak RSS. Scoring/ranking formulas are unchanged by this script.

    python scripts/validate_land_provider.py
    python scripts/validate_land_provider.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyproj import Transformer  # noqa: E402

from backend.geographic_context import GeographicContextConfig, enrich_candidates  # noqa: E402
from services.land_service import MODE_LOCAL_ONLY, MODE_OVERPASS_ONLY, fetch_land_context  # noqa: E402

# Fixed, documented study area (Durg area) and candidate grid.
STUDY = (21.05, 81.10, 21.45, 81.50)  # south, west, north, east
GRID = 5
CRS = "EPSG:32644"


def fixed_candidates():
    south, west, north, east = STUDY
    lat_step = (north - south) / (GRID + 1)
    lon_step = (east - west) / (GRID + 1)
    transformer = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    candidates = []
    index = 0
    for row in range(1, GRID + 1):
        for col in range(1, GRID + 1):
            lat = south + row * lat_step
            lon = west + col * lon_step
            x, y = transformer.transform(lon, lat)
            candidates.append({
                "candidate_id": f"g{index:02d}",
                "x": float(x), "y": float(y), "crs": CRS,
                "latitude": lat, "longitude": lon,
            })
            index += 1
    return candidates


def _rss_mb() -> float:
    # ru_maxrss is bytes on macOS/Darwin and kibibytes on Linux.
    import platform

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    bytes_used = raw if platform.system() == "Darwin" else raw * 1024
    return bytes_used / (1024.0 * 1024.0)


def run_mode(mode: str, candidates: list[dict]) -> dict:
    config = GeographicContextConfig(cache_enabled=False, cluster_radius_m=1000.0, provider_retries=0)
    started = time.perf_counter()
    result = enrich_candidates(
        candidates,
        config=config,
        provider=lambda s, w, n, e: fetch_land_context(s, w, n, e, mode=mode),
        cache=None,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    diagnostics = result["diagnostics"]

    feasible = [c for c in result["candidates"] if c["feasibility"] == "feasible"]
    distances = [
        c["road_distance_m"] for c in result["candidates"] if c["road_distance_m"] is not None
    ]
    return {
        "mode": mode,
        "feasible": diagnostics["feasible_count"],
        "rejected": diagnostics["rejected_count"],
        "unknown": diagnostics["unknown_count"],
        "feasibility_reasons": diagnostics["feasibility_reasons"],
        "clusters": diagnostics["cluster_count"],
        "provider_calls": diagnostics["stats"]["provider_calls"],
        "road_distance_m": {
            "n": len(distances),
            "min": round(min(distances), 2) if distances else None,
            "mean": round(sum(distances) / len(distances), 2) if distances else None,
            "max": round(max(distances), 2) if distances else None,
        },
        "elapsed_ms": round(elapsed_ms, 1),
        "rss_mb": round(_rss_mb(), 1),
        "sample": feasible[0]["candidate_id"] if feasible else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate local vs Overpass land provider on a fixed grid.")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--skip-overpass", action="store_true")
    args = parser.parse_args(argv)

    candidates = fixed_candidates()
    report = {
        "study_area": {"south": STUDY[0], "west": STUDY[1], "north": STUDY[2], "east": STUDY[3]},
        "candidate_count": len(candidates),
        "candidate_ids": [c["candidate_id"] for c in candidates],
        "runs": {},
    }
    report["runs"]["local_only"] = run_mode(MODE_LOCAL_ONLY, candidates)

    if not args.skip_overpass:
        probe = fetch_land_context(*(STUDY[0], STUDY[1], STUDY[2], STUDY[3]), mode=MODE_OVERPASS_ONLY)
        if probe.get("status") == "available":
            report["runs"]["overpass_only"] = run_mode(MODE_OVERPASS_ONLY, candidates)
        else:
            report["runs"]["overpass_only"] = {
                "skipped": True,
                "status": probe.get("status"),
                "warning": probe.get("warning"),
            }

    print(json.dumps(report, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
