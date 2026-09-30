"""Phase-1 performance benchmark harness for the pond recommendation pipeline.

Measures, per concurrency level:

* throughput (requests/second) and total runtime,
* latency p50/p95/p99 (and mean/max),
* success/failure counts,
* process CPU time (user + sys) and peak RSS,
* Python peak allocation (tracemalloc),
* per-stage latency (from the pipeline instrumentation),
* external-API latency and external-data availability,
* baseline data-quality metrics.

By default every external service is replaced with a deterministic fake so the
benchmark is repeatable and offline. Pass ``--real`` to measure against the
live services (requires network and API keys); mocked and real numbers must be
reported separately because they are not comparable.

Usage::

    python benchmarks/benchmark_pipeline.py --fixture valid_kml
    python benchmarks/benchmark_pipeline.py --fixture valid_kml,large_input --levels 1,5,10,25,50,100
    python benchmarks/benchmark_pipeline.py --fixture valid_kml --real
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep the benchmark's provider cache separate from the application cache.
os.environ.setdefault("CACHE_DIR", str(PROJECT_ROOT / "benchmarks" / ".geocache"))

import httpx  # noqa: E402

from main import app  # noqa: E402
import instrumentation  # noqa: E402
import geographic_context  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "tests"))
import fixtures  # noqa: E402


# ---------------------------------------------------------------------------
# Deterministic external-service fakes (mocked mode)
# ---------------------------------------------------------------------------

def _fake_elevation(rows: int = 6, columns: int = 6) -> np.ndarray:
    yy, xx = np.mgrid[0:rows, 0:columns]
    return 100.0 + yy * 0.5 + xx * 0.25


def _fake_get_dem(*args, **kwargs) -> Dict[str, Any]:
    elevation = _fake_elevation()
    return {
        "elevation": elevation,
        "rows": int(elevation.shape[0]),
        "columns": int(elevation.shape[1]),
        "elevation_min_m": float(np.nanmin(elevation)),
        "elevation_max_m": float(np.nanmax(elevation)),
        "elevation_mean_m": float(np.nanmean(elevation)),
        "source": "OpenTopography Global DEM API",
    }


def _benchmark_land_context(south: float, west: float, north: float, east: float) -> Dict[str, Any]:
    """Deterministic OSM-shaped payload covering the queried bbox.

    Features are placed relative to the bbox centre so candidates land in a mix
    of feasible / rejected states.
    """

    centre_lat = (south + north) / 2.0
    centre_lon = (west + east) / 2.0
    lat_span = max(north - south, 1e-6)
    lon_span = max(east - west, 1e-6)

    buildings = [
        {"type": "node", "id": 1000 + index, "tags": {"building": "yes"},
         "lat": centre_lat + offset_lat * lat_span, "lon": centre_lon + offset_lon * lon_span}
        for index, (offset_lon, offset_lat) in enumerate([(0.0, 0.0), (0.2, -0.1), (-0.3, 0.25)])
    ]
    roads = [
        {"type": "way", "id": 2000, "tags": {"highway": "residential"},
         "geometry": [
             {"lat": centre_lat - lat_span, "lon": centre_lon + 0.1 * lon_span},
             {"lat": centre_lat + lat_span, "lon": centre_lon + 0.1 * lon_span},
         ]},
    ]
    water = [
        {"type": "way", "id": 3000, "tags": {"natural": "water"},
         "geometry": [
             {"lat": centre_lat - 0.02 * lat_span, "lon": centre_lon + 0.35 * lon_span},
             {"lat": centre_lat - 0.02 * lat_span, "lon": centre_lon + 0.40 * lon_span},
             {"lat": centre_lat + 0.02 * lat_span, "lon": centre_lon + 0.40 * lon_span},
             {"lat": centre_lat + 0.02 * lat_span, "lon": centre_lon + 0.35 * lon_span},
             {"lat": centre_lat - 0.02 * lat_span, "lon": centre_lon + 0.35 * lon_span},
         ]},
    ]

    return {
        "bbox": {"south": south, "west": west, "north": north, "east": east},
        "water_bodies": water,
        "roads": roads,
        "buildings": buildings,
        "source": "OpenStreetMap Overpass API (benchmark mock)",
    }


def install_mocks(external_latency_ms: float = 0.0, provider_latency_ms: float = 0.0) -> None:
    import services.land_service as land_service
    import main as main_module

    def with_latency(func, latency_ms: float):
        def wrapper(*args, **kwargs):
            if latency_ms > 0:
                time.sleep(latency_ms / 1000.0)
            return func(*args, **kwargs)

        return wrapper

    land_provider = with_latency(_benchmark_land_context, provider_latency_ms)

    main_module.get_dem_for_bbox = with_latency(_fake_get_dem, external_latency_ms)
    main_module.fetch_historical_rainfall = with_latency(
        lambda *a, **k: {"status": "ok", "total_precipitation_mm": 120.0, "source": "open-meteo-archive (mock)"},
        external_latency_ms,
    )
    main_module.fetch_land_context = land_provider
    main_module.search_places = with_latency(
        lambda query, *a, **k: {"query": query, "results": [], "source": "photon (mock)"}, external_latency_ms
    )
    main_module.geocode_place = with_latency(
        lambda query, *a, **k: {"query": query, "results": [], "source": "nominatim (mock)"}, external_latency_ms
    )
    # geographic_context resolves the provider through main's namespace, but
    # patch the service module too so no code path can reach the network.
    land_service.fetch_land_context = land_provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((percentile / 100.0) * (len(ordered) - 1))))
    return float(ordered[index])


def _rusage() -> Dict[str, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in bytes on macOS and in kilobytes on Linux.
    max_rss_kb = usage.ru_maxrss / 1024.0 if sys.platform == "darwin" else float(usage.ru_maxrss)
    return {
        "user_seconds": float(usage.ru_utime),
        "sys_seconds": float(usage.ru_stime),
        "max_rss_kb": max_rss_kb,
    }


def resolve_fixture(name: str) -> tuple:
    if name == "sample":
        path = PROJECT_ROOT / "sample_contours.kml"
        return path.name, path.read_bytes()
    registry = fixtures.build_all()
    if name not in registry:
        raise SystemExit(f"Unknown fixture '{name}'. Available: sample, {', '.join(sorted(registry))}")
    return registry[name]


async def _one_request(client: httpx.AsyncClient, endpoint: str, filename: str, data: bytes) -> tuple:
    started = time.perf_counter()
    try:
        response = await client.post(
            endpoint,
            files={"file": (filename, data, "application/xml")},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return response.status_code, elapsed_ms, None
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return None, elapsed_ms, f"{type(exc).__name__}: {exc}"


async def run_level(concurrency: int, requests: int, filename: str, data: bytes, endpoint: str) -> Dict[str, Any]:
    instrumentation.clear_traces()
    geographic_context.reset_stats()
    transport = httpx.ASGITransport(app=app)

    latencies: List[float] = []
    statuses: Dict[int, int] = {}
    failures = 0
    error_samples: List[str] = []

    tracemalloc.start()
    before = _rusage()
    started = time.perf_counter()

    async with httpx.AsyncClient(transport=transport, base_url="http://benchmark") as client:
        pending = requests
        while pending > 0:
            batch = min(concurrency, pending)
            results = await asyncio.gather(
                *[_one_request(client, endpoint, filename, data) for _ in range(batch)]
            )
            for status, elapsed_ms, error in results:
                latencies.append(elapsed_ms)
                if error is not None:
                    failures += 1
                    if len(error_samples) < 5:
                        error_samples.append(error)
                elif status is not None:
                    statuses[status] = statuses.get(status, 0) + 1
                    if status >= 500:
                        failures += 1
            pending -= batch

    total_seconds = time.perf_counter() - started
    after = _rusage()
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stage_totals: Dict[str, List[float]] = {}
    external_totals: List[float] = []
    data_quality: Dict[str, Any] = {}
    for trace in instrumentation.recent_traces():
        for record in trace.stages:
            stage_totals.setdefault(record.stage, []).append(record.duration_ms)
            if record.external_api_ms is not None:
                external_totals.append(record.external_api_ms)
        if trace.metrics and not data_quality:
            data_quality = dict(trace.metrics)

    cpu_seconds = (after["user_seconds"] - before["user_seconds"]) + (after["sys_seconds"] - before["sys_seconds"])

    return {
        "concurrency": concurrency,
        "requests": requests,
        "total_seconds": round(total_seconds, 4),
        "throughput_rps": round(requests / total_seconds, 3) if total_seconds > 0 else None,
        "success_count": sum(count for status, count in statuses.items() if status < 500),
        "failure_count": failures,
        "status_counts": {str(k): v for k, v in sorted(statuses.items())},
        "error_samples": error_samples,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3) if latencies else None,
            "p50": round(_percentile(latencies, 50), 3),
            "p95": round(_percentile(latencies, 95), 3),
            "p99": round(_percentile(latencies, 99), 3),
            "max": round(max(latencies), 3) if latencies else None,
        },
        "cpu_seconds": round(cpu_seconds, 4),
        "cpu_user_seconds": round(after["user_seconds"] - before["user_seconds"], 4),
        "cpu_sys_seconds": round(after["sys_seconds"] - before["sys_seconds"], 4),
        "cpu_utilization_percent": round((cpu_seconds / total_seconds) * 100.0, 2) if total_seconds > 0 else None,
        "peak_rss_kb": round(after["max_rss_kb"], 1),
        "tracemalloc_peak_kb": round(traced_peak / 1024.0, 2),
        "stages_ms_mean": {
            stage: round(statistics.fmean(values), 3) for stage, values in sorted(stage_totals.items())
        },
        "external_api_ms_mean": round(statistics.fmean(external_totals), 3) if external_totals else None,
        "data_quality": data_quality,
        "geographic": geographic_context.get_stats(),
    }


async def run_fixture(fixture_name: str, levels: List[int], requests_per_level: Optional[List[int]], endpoint: str) -> Dict[str, Any]:
    filename, data = resolve_fixture(fixture_name)

    # Warm-up so first-request import/JIT costs do not skew level 1.
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://benchmark") as client:
        await _one_request(client, endpoint, filename, data)

    results = []
    for index, concurrency in enumerate(levels):
        requests = requests_per_level[index] if requests_per_level else max(10, concurrency)
        result = await run_level(concurrency, requests, filename, data, endpoint)
        results.append(result)
        print(
            f"  [concurrency={concurrency:>3} requests={requests:>3}] "
            f"rps={result['throughput_rps']} p50={result['latency_ms']['p50']}ms "
            f"p95={result['latency_ms']['p95']}ms fail={result['failure_count']} "
            f"cpu={result['cpu_utilization_percent']}%",
            flush=True,
        )

    return {
        "fixture": fixture_name,
        "filename": filename,
        "input_bytes": len(data),
        "endpoint": endpoint,
        "levels": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pond pipeline benchmark harness")
    parser.add_argument("--fixture", default="valid_kml", help="Fixture name, 'sample', or comma-separated list")
    parser.add_argument("--levels", default="1,5,10,25,50,100", help="Comma-separated concurrency levels")
    parser.add_argument("--requests", default=None, help="Comma-separated request counts per level")
    parser.add_argument("--real", action="store_true", help="Use real external services instead of mocks")
    parser.add_argument("--external-latency-ms", type=float, default=0.0, help="Simulated per-call external latency (mocked mode)")
    parser.add_argument("--provider-latency-ms", type=float, default=0.0, help="Simulated GIS provider (Overpass) latency per query")
    parser.add_argument("--endpoint", default="/analyzeContour", help="Endpoint to benchmark (e.g. /analyzeContour/enriched)")
    parser.add_argument("--geo-cache", choices=["on", "off"], default="off",
                        help="Enable the geographic provider cache (default off, to isolate provider latency)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    mode = "real-services" if args.real else "mocked"
    if args.geo_cache == "off":
        os.environ["POND_GEO_CACHE_ENABLED"] = "0"
        cache_dir = Path(os.environ["CACHE_DIR"])
        if cache_dir.exists():
            for entry in cache_dir.glob("land_context_*.json"):
                entry.unlink()
    else:
        os.environ["POND_GEO_CACHE_ENABLED"] = "1"

    if not args.real:
        install_mocks(external_latency_ms=args.external_latency_ms, provider_latency_ms=args.provider_latency_ms)

    levels = [int(value) for value in args.levels.split(",") if value.strip()]
    requests_per_level = (
        [int(value) for value in args.requests.split(",")] if args.requests else None
    )
    fixture_names = [name.strip() for name in args.fixture.split(",") if name.strip()]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "endpoint": args.endpoint,
        "geo_cache": args.geo_cache,
        "external_latency_ms": args.external_latency_ms,
        "provider_latency_ms": args.provider_latency_ms,
        "levels_requested": levels,
        "results": [],
    }

    for fixture_name in fixture_names:
        print(f"Benchmarking fixture '{fixture_name}' ({mode})...", flush=True)
        report["results"].append(asyncio.run(run_fixture(fixture_name, levels, requests_per_level, args.endpoint)))

    output = Path(args.output) if args.output else PROJECT_ROOT / "benchmarks" / "results" / f"baseline-{mode}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
