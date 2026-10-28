"""Phase 03 — Overpass removed from runtime; local-only geographic enrichment.

Verifies that:

* the default provider mode is ``local_only`` with Overpass fallback disabled;
* ``fetch_land_context`` (default args) never contacts Overpass and fails closed
  to an explicit ``unavailable`` state (never a silent empty layer / zero
  distance);
* out-of-coverage and missing datasets are reported explicitly;
* Overpass is still reachable *explicitly* (``overpass_only``) for offline tools;
* the API surfaces the degraded state and never calls Overpass;
* ``/health/datasets`` reports coverage/staleness without raising;
* the prepared local GeoPackage actually serves features (when present).

All offline: the land provider is either the real selector with the suite's
local-OSM-disabled settings, an injected fake, or the on-disk GeoPackage; HTTP is
patched to raise if Overpass is ever attempted.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import fixtures
import main
from config import settings
from services.land_providers.base import (
    STATUS_MISSING,
    STATUS_OUT_OF_COVERAGE,
    STATUS_UNAVAILABLE,
    ProviderUnavailable,
)
from services.land_service import (
    MODE_LOCAL_ONLY,
    MODE_OVERPASS_ONLY,
    fetch_land_context as real_fetch_land_context,
)

BBOX = (21.24, 81.28, 21.27, 81.31)  # south, west, north, east (inside coverage)
NO_OVERPASS = AssertionError("Overpass must not be called at runtime")


class _FakeLocal:
    def __init__(self, error=None, result=None):
        self.error = error
        self.result = result
        self.calls = 0

    def fetch(self, south, west, north, east):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_default_mode_is_local_only_and_fallback_disabled():
    assert settings.LAND_PROVIDER_MODE == "local_only"
    assert settings.OVERPASS_FALLBACK_ENABLED is False


# ---------------------------------------------------------------------------
# Runtime never uses Overpass
# ---------------------------------------------------------------------------

def test_default_fetch_never_calls_overpass():
    # Suite fixture disables the local provider -> explicit unavailable, no HTTP.
    with patch("services.land_service.httpx.post", side_effect=NO_OVERPASS):
        payload = real_fetch_land_context(*BBOX)

    assert payload["status"] == STATUS_UNAVAILABLE
    assert payload["provider"] == "local_osm"
    assert payload["water_bodies"] == []
    assert payload["roads"] == []
    assert payload["buildings"] == []
    assert "local_only" in payload["warning"]


def test_out_of_coverage_is_explicit_not_empty():
    fake = _FakeLocal(error=ProviderUnavailable(STATUS_OUT_OF_COVERAGE, "query outside coverage"))
    with patch("services.land_service.httpx.post", side_effect=NO_OVERPASS):
        payload = real_fetch_land_context(10.0, 70.0, 10.1, 70.1, local_provider=fake)

    assert fake.calls == 1
    assert payload["status"] == STATUS_UNAVAILABLE
    assert payload["provider"] == "local_osm"
    assert payload["local_status"] == STATUS_OUT_OF_COVERAGE
    # Explicit zero counts (never fabricated features, never omitted).
    assert set(payload["feature_counts"]) == {"water_bodies", "roads", "buildings"}
    assert all(count == 0 for count in payload["feature_counts"].values())


def test_missing_dataset_fails_closed():
    fake = _FakeLocal(error=ProviderUnavailable(STATUS_MISSING, "no dataset"))
    with patch("services.land_service.httpx.post", side_effect=NO_OVERPASS):
        payload = real_fetch_land_context(*BBOX, local_provider=fake)

    assert payload["status"] == STATUS_UNAVAILABLE
    assert payload["local_status"] == STATUS_MISSING
    assert payload["water_bodies"] == []
    assert payload["roads"] == []
    assert payload["buildings"] == []


# ---------------------------------------------------------------------------
# Overpass retained for explicit/offline use
# ---------------------------------------------------------------------------

def test_overpass_only_still_reachable_for_tools():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"elements": [{"type": "way", "id": 1, "tags": {"highway": "primary"}}]}

    fake = _FakeLocal(error=ProviderUnavailable(STATUS_MISSING, "no dataset"))
    with patch("services.land_service.httpx.post", return_value=response):
        payload = real_fetch_land_context(*BBOX, mode=MODE_OVERPASS_ONLY, local_provider=fake)

    assert fake.calls == 0
    assert payload["provider"] == "overpass"
    assert payload["roads"][0]["tags"]["highway"] == "primary"


# ---------------------------------------------------------------------------
# Local retrieval (offline) via the selector
# ---------------------------------------------------------------------------

def test_real_local_dataset_serves_features_offline():
    gpkg = Path(settings.LOCAL_OSM_GPKG_PATH)
    if not gpkg.exists():
        pytest.skip("local GeoPackage not present")
    pytest.importorskip("geopandas")

    from services.land_providers.local_osm import LocalOSMProvider

    provider = LocalOSMProvider(
        gpkg,
        manifest_path=settings.LOCAL_OSM_MANIFEST_PATH,
        max_age_days=settings.LOCAL_OSM_MAX_AGE_DAYS,
    )
    with patch("services.land_service.httpx.post", side_effect=NO_OVERPASS):
        payload = real_fetch_land_context(*BBOX, mode=MODE_LOCAL_ONLY, local_provider=provider)

    assert payload["provider"] == "local_osm"
    assert payload["status"] in {"available", "empty"}
    assert payload["coverage"] == "full"
    assert payload["feature_counts"]["roads"] >= 1


# ---------------------------------------------------------------------------
# API integration
# ---------------------------------------------------------------------------

def test_api_contour_never_calls_overpass_and_degrades_explicitly(client, monkeypatch):
    # Use the REAL selector (override the suite stub) and forbid Overpass.
    monkeypatch.setattr(main, "fetch_land_context", real_fetch_land_context, raising=True)
    name, data = fixtures.build_all()["valid_kml"]

    with patch("services.land_service.httpx.post", side_effect=NO_OVERPASS):
        response = client.post(
            "/analyzeContour", files={"file": (name, data, "application/xml")}
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    land = payload["enrichment"]["land_context"]
    assert land["status"] == STATUS_UNAVAILABLE
    assert land["provider"] == "local_osm"


# ---------------------------------------------------------------------------
# Readiness probe
# ---------------------------------------------------------------------------

def test_health_datasets_reports_readiness(client):
    response = client.get("/health/datasets")
    assert response.status_code == 200
    body = response.json()

    assert body["provider"] == "local_osm"
    assert body["mode"] == "local_only"
    assert body["overpass_fallback_enabled"] is False
    assert "available" in body
    assert "path" in body
    if body.get("available"):
        assert body.get("coverage")
        assert body.get("layers")
