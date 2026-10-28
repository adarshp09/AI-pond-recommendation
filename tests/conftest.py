"""Shared pytest configuration.

This suite is deterministic and offline: external services are replaced with
small, deterministic fakes by default so runs do not depend on the network or
on live API keys. Individual tests can still override a service (e.g. to
simulate a failure) using ``monkeypatch`` or ``unittest.mock.patch`` on the
``main`` module; those overrides win because they are applied later.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_KML = PROJECT_ROOT / "sample_contours.kml"


def _fake_dem_elevation(rows: int = 6, columns: int = 6) -> np.ndarray:
    """A deterministic, gently sloping plane."""

    yy, xx = np.mgrid[0:rows, 0:columns]
    return 100.0 + yy * 0.5 + xx * 0.25


def fake_get_dem_for_bbox(*args, **kwargs) -> dict:
    elevation = _fake_dem_elevation()
    return {
        "elevation": elevation,
        "rows": int(elevation.shape[0]),
        "columns": int(elevation.shape[1]),
        "elevation_min_m": float(np.nanmin(elevation)),
        "elevation_max_m": float(np.nanmax(elevation)),
        "elevation_mean_m": float(np.nanmean(elevation)),
        "source": "OpenTopography Global DEM API",
        "mocked": True,
    }


def fake_fetch_historical_rainfall(*args, **kwargs) -> dict:
    return {
        "status": "ok",
        "total_precipitation_mm": 120.0,
        "daily": [{"date": "2024-01-01", "precipitation_mm": 120.0}],
        "source": "open-meteo-archive (mock)",
        "mocked": True,
    }


def fake_fetch_land_context(*args, **kwargs) -> dict:
    return {
        "status": "ok",
        "water_bodies": [],
        "roads": [],
        "buildings": [],
        "source": "osm-overpass (mock)",
        "mocked": True,
    }


def fake_search_places(query: str, *args, **kwargs) -> dict:
    return {"query": query, "results": [], "source": "photon (mock)", "mocked": True}


def fake_geocode_place(query: str, *args, **kwargs) -> dict:
    return {"query": query, "results": [], "source": "nominatim (mock)", "mocked": True}


@pytest.fixture(autouse=True)
def mock_external_services(monkeypatch):
    """Replace every external dependency with a deterministic fake."""

    import main  # noqa: WPS433 (import inside fixture is intentional)
    import services.land_service as land_service  # noqa: WPS433
    from config import settings  # noqa: WPS433

    # Keep the default suite deterministic and offline: the local OSM dataset is
    # only exercised by tests that explicitly opt in (tests/test_local_osm.py and
    # tests/test_land_provider_fallback.py).
    monkeypatch.setitem(settings.__dict__, "LOCAL_OSM_ENABLED", False)

    # Phase 04 — keep provider caching out of the shared suite run. Several tests
    # reuse identical query keys ("Nagpur", the same rainfall window, ...) with
    # different mocked responses, so caching must be OFF by default; caching is
    # exercised explicitly (with an isolated CACHE_DIR) in tests/test_caching.py.
    monkeypatch.setitem(settings.__dict__, "CACHE_ENABLED", False)
    monkeypatch.setitem(settings.__dict__, "GEOCODING_MIN_INTERVAL_S", 0.0)

    # Phase 09 — keep the shared suite independent of the on-disk GeoPackage;
    # river exclusion is exercised explicitly in tests/test_spatial_filter.py.
    monkeypatch.setitem(settings.__dict__, "RIVER_EXCLUSION_ENABLED", False)

    # Phase 05 — cache hit/miss counters are process-global; reset per test.
    from services.cache import reset_cache_metrics

    reset_cache_metrics()

    monkeypatch.setattr(main, "get_dem_for_bbox", fake_get_dem_for_bbox, raising=False)
    monkeypatch.setattr(main, "fetch_historical_rainfall", fake_fetch_historical_rainfall, raising=False)
    monkeypatch.setattr(main, "fetch_land_context", fake_fetch_land_context, raising=False)
    monkeypatch.setattr(main, "search_places", fake_search_places, raising=False)
    monkeypatch.setattr(main, "geocode_place", fake_geocode_place, raising=False)
    # Also patch the service module directly so any code path importing it
    # (e.g. geographic_context) never reaches the real provider in tests.
    monkeypatch.setattr(land_service, "fetch_land_context", fake_fetch_land_context, raising=False)

    # Phase 02 — the opt-in engine bridge (backend.pipeline) resolves the land
    # provider dynamically from services.land_service (patched above) and calls
    # rainfall_runoff's module-level fetch_historical_rainfall; patch that too so
    # the engine path is fully offline in tests.
    try:
        import backend.rainfall_runoff as rainfall_runoff

        monkeypatch.setattr(
            rainfall_runoff,
            "fetch_historical_rainfall",
            fake_fetch_historical_rainfall,
            raising=False,
        )
    except Exception:  # pragma: no cover - import guard
        pass
    yield


@pytest.fixture
def sample_kml_path() -> Path:
    return SAMPLE_KML


@pytest.fixture
def sample_kml_bytes() -> bytes:
    return SAMPLE_KML.read_bytes()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from main import app

    return TestClient(app)
