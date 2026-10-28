"""Phase 04 — layered caching tests.

Covers the file cache primitives (get/set/TTL/bypass/clear/stats), the
single-flight ``get_or_set`` helper (including concurrency), per-provider caching
(rainfall, geocoding, place search, OpenTopography DEM bytes, local OSM query),
failure non-caching, and the ``/cache/*`` admin endpoints.

The shared suite disables caching (``conftest``); these tests opt in with an
isolated ``CACHE_DIR`` so they never touch the repo cache and never depend on the
network.
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock

import pytest

from config import settings
from services import cache as cache_mod
from services.cache import FileCache, cache_is_enabled, get_or_set


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """Enable caching into an isolated dir and redirect the singleton."""

    monkeypatch.setitem(settings.__dict__, "CACHE_ENABLED", True)
    monkeypatch.setitem(settings.__dict__, "CACHE_DIR", str(tmp_path))
    fresh = cache_mod.FileCache(cache_dir=str(tmp_path), enabled=True)
    monkeypatch.setattr(cache_mod, "_default_cache", fresh)
    return fresh


def _ok_response(payload):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    return response


# ---------------------------------------------------------------------------
# FileCache primitives
# ---------------------------------------------------------------------------

def test_filecache_roundtrip_and_miss(tmp_path):
    cache = FileCache(cache_dir=str(tmp_path), ttl_seconds=60, enabled=True)
    assert cache.get("p", {"a": 1}) is None
    cache.set("p", {"a": 1}, {"v": 42})
    assert cache.get("p", {"a": 1}) == {"v": 42}


def test_filecache_ttl_expiry(tmp_path):
    cache = FileCache(cache_dir=str(tmp_path), ttl_seconds=3600, enabled=True)
    cache.set("p", {"a": 1}, "value")
    # Per-call TTL override forces immediate expiry.
    assert cache.get("p", {"a": 1}, ttl_seconds=-1) is None
    # ...and a fresh set is readable again with the default TTL.
    cache.set("p", {"a": 2}, "value2")
    assert cache.get("p", {"a": 2}) == "value2"


def test_filecache_disabled_never_stores(tmp_path):
    cache = FileCache(cache_dir=str(tmp_path), ttl_seconds=60, enabled=False)
    cache.set("p", {"a": 1}, "value")
    assert cache.get("p", {"a": 1}) is None


def test_filecache_clear_and_stats(tmp_path):
    cache = FileCache(cache_dir=str(tmp_path), ttl_seconds=60, enabled=True)
    cache.set("p", {"a": 1}, "value")
    stats = cache.stats()
    assert stats["entries"] == 1 and stats["enabled"] is True
    cache.clear()
    assert cache.stats()["entries"] == 0
    assert cache.get("p", {"a": 1}) is None


def test_cache_is_enabled_honours_settings_and_bypass(monkeypatch):
    monkeypatch.setitem(settings.__dict__, "CACHE_ENABLED", True)
    monkeypatch.delenv("CACHE_BYPASS", raising=False)
    assert cache_is_enabled() is True

    monkeypatch.setenv("CACHE_BYPASS", "1")
    assert cache_is_enabled() is False

    monkeypatch.delenv("CACHE_BYPASS", raising=False)
    monkeypatch.setitem(settings.__dict__, "CACHE_ENABLED", False)
    assert cache_is_enabled() is False


# ---------------------------------------------------------------------------
# get_or_set
# ---------------------------------------------------------------------------

def test_get_or_set_caches_and_returns_value(tmp_cache):
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return {"x": 1}

    assert get_or_set("p", {"k": 1}, factory, ttl_seconds=60) == {"x": 1}
    assert get_or_set("p", {"k": 1}, factory, ttl_seconds=60) == {"x": 1}
    assert calls["n"] == 1


def test_get_or_set_single_flight_under_concurrency(tmp_cache):
    calls = {"n": 0}
    barrier = threading.Barrier(8)

    def factory():
        calls["n"] += 1
        time.sleep(0.1)
        return {"x": 1}

    results = []

    def worker():
        barrier.wait()
        results.append(get_or_set("sf", {"k": 1}, factory, ttl_seconds=60))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls["n"] == 1  # one computation shared by all callers
    assert results == [{"x": 1}] * 8


def test_get_or_set_does_not_cache_failures(tmp_cache):
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return "ok"

    with pytest.raises(RuntimeError):
        get_or_set("f", {"k": 1}, factory, ttl_seconds=60)
    assert get_or_set("f", {"k": 1}, factory, ttl_seconds=60) == "ok"
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Provider caching (no network: httpx is mocked)
# ---------------------------------------------------------------------------

def test_rainfall_cache_avoids_second_http_call(tmp_cache, monkeypatch):
    from services.rainfall_service import fetch_historical_rainfall

    calls = {"n": 0}

    def fake_get(*_a, **_k):
        calls["n"] += 1
        return _ok_response({"daily": {"time": ["2024-01-01"], "precipitation_sum": [5.0]}})

    monkeypatch.setattr("services.rainfall_service.httpx.get", fake_get)

    first = fetch_historical_rainfall(21.26, 81.29, "2024-01-01", "2024-01-01")
    second = fetch_historical_rainfall(21.26, 81.29, "2024-01-01", "2024-01-01")

    assert calls["n"] == 1
    assert first == second
    assert first["total_precipitation_mm"] == 5.0


def test_geocoding_cache_avoids_second_http_call(tmp_cache, monkeypatch):
    from services.geocoding_service import geocode_place

    calls = {"n": 0}

    def fake_get(*_a, **_k):
        calls["n"] += 1
        return _ok_response([{"display_name": "Nagpur", "lat": "21.1", "lon": "79.0", "address": {}}])

    monkeypatch.setattr("services.geocoding_service.httpx.get", fake_get)

    geocode_place("Nagpur")
    geocode_place("Nagpur")
    assert calls["n"] == 1

    geocode_place("Pune")
    assert calls["n"] == 2  # distinct key


def test_place_search_cache_avoids_second_http_call(tmp_cache, monkeypatch):
    from services.place_search_service import search_places

    calls = {"n": 0}

    def fake_get(*_a, **_k):
        calls["n"] += 1
        return _ok_response({"features": [{"properties": {"name": "X"}, "geometry": {"coordinates": [79.0, 21.0]}}]})

    monkeypatch.setattr("services.place_search_service.httpx.get", fake_get)

    search_places("X")
    search_places("X")
    assert calls["n"] == 1


def test_dem_cache_avoids_second_download(tmp_cache, monkeypatch):
    from services.opentopography_service import OpenTopographyService

    service = OpenTopographyService(api_key="test-key")
    response = MagicMock()
    response.content = b"geotiff-bytes"
    response.raise_for_status.return_value = None
    downloads = {"n": 0}

    def fake_get(*_a, **_k):
        downloads["n"] += 1
        return response

    monkeypatch.setattr("services.opentopography_service.httpx.Client", _ClientPatch(fake_get))

    parsed = {"n": 0}

    def fake_parse(content, south, north, west, east):  # noqa: ARG001
        parsed["n"] += 1
        return {"rows": 2, "columns": 2, "elevation_min_m": 1.0, "content": content}

    monkeypatch.setattr(service, "_parse_geotiff", fake_parse)

    first = service.get_dem(21.0, 22.0, 81.0, 82.0)
    second = service.get_dem(21.0, 22.0, 81.0, 82.0)

    assert downloads["n"] == 1  # second call served from the byte cache
    assert parsed["n"] == 2     # but re-parsed both times (bytes, not the dict)
    assert first["content"] == second["content"] == b"geotiff-bytes"


class _ClientPatch:
    """Patch helper: ``httpx.Client(...).__enter__().get`` delegates to fake_get."""

    def __init__(self, fake_get):
        self._fake_get = fake_get

    def __call__(self, *args, **kwargs):
        outer = MagicMock()
        inner = MagicMock()
        inner.get.side_effect = self._fake_get
        outer.__enter__.return_value = inner
        return outer


def test_local_osm_cache_avoids_second_read(tmp_cache, monkeypatch, tmp_path):
    gpd = pytest.importorskip("geopandas")
    import pyogrio
    from pathlib import Path

    from shapely.geometry import LineString, Polygon

    from services.land_providers.local_osm import LocalOSMProvider

    gpkg = Path(tmp_path) / "land_context.gpkg"
    square = Polygon([(81.3, 21.26), (81.301, 21.26), (81.301, 21.261), (81.3, 21.261), (81.3, 21.26)])
    layers = {
        "water_areas": gpd.GeoDataFrame({"osm_id": ["1"], "osm_type": ["way"], "tags": [json.dumps({"natural": "water"})]}, geometry=[square], crs="EPSG:4326"),
        "waterways": gpd.GeoDataFrame({"osm_id": ["2"], "osm_type": ["way"], "tags": [json.dumps({"waterway": "stream"})]}, geometry=[LineString([(81.29, 21.26), (81.31, 21.26)])], crs="EPSG:4326"),
        "roads": gpd.GeoDataFrame({"osm_id": ["3"], "osm_type": ["way"], "tags": [json.dumps({"highway": "residential"})]}, geometry=[LineString([(81.301, 21.25), (81.301, 21.27)])], crs="EPSG:4326"),
        "buildings": gpd.GeoDataFrame({"osm_id": ["4"], "osm_type": ["way"], "tags": [json.dumps({"building": "yes"})]}, geometry=[square], crs="EPSG:4326"),
    }
    for index, (layer, frame) in enumerate(layers.items()):
        pyogrio.write_dataframe(frame, gpkg, layer=layer, driver="GPKG", append=index != 0)
    manifest = Path(tmp_path) / "manifest.json"
    from datetime import datetime, timezone

    manifest.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "study_area": {"south": 21.0, "west": 81.05, "north": 21.5, "east": 81.55},
    }))

    provider = LocalOSMProvider(gpkg, manifest_path=manifest, max_age_days=180)

    original = pyogrio.read_dataframe
    reads = {"n": 0}

    def counting(*args, **kwargs):
        reads["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pyogrio, "read_dataframe", counting)

    first = provider.fetch(21.24, 81.28, 21.27, 81.31)
    reads_after_first = reads["n"]
    second = provider.fetch(21.24, 81.28, 21.27, 81.31)

    assert reads_after_first >= 1
    assert reads["n"] == reads_after_first  # second call served from cache
    assert first.feature_counts == second.feature_counts


# ---------------------------------------------------------------------------
# /cache admin endpoints
# ---------------------------------------------------------------------------

def test_cache_stats_endpoint(client, tmp_cache):
    response = client.get("/cache/stats")
    assert response.status_code == 200
    body = response.json()
    assert "entries" in body and "size_bytes" in body
    assert body["cache_enabled"] is True


def test_cache_clear_endpoint(client, tmp_cache):
    from services.cache import get_cache

    get_cache().set("p", {"a": 1}, "value")
    assert get_cache().stats()["entries"] == 1

    response = client.post("/cache/clear")
    assert response.status_code == 200
    assert response.json() == {"cleared": True}
    assert get_cache().stats()["entries"] == 0


def test_cache_clear_token_guard(client, tmp_cache, monkeypatch):
    monkeypatch.setitem(settings.__dict__, "CACHE_ADMIN_TOKEN", "s3cret")

    denied = client.post("/cache/clear")
    assert denied.status_code == 403
    assert denied.json()["detail"]["code"] == "CACHE_CLEAR_FORBIDDEN"

    allowed = client.post("/cache/clear", headers={"X-Cache-Admin-Token": "s3cret"})
    assert allowed.status_code == 200
    assert allowed.json() == {"cleared": True}
