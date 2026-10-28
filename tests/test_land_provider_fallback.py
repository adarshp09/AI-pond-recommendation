"""Phase 03 tests — local-first provider selection, fallback and degraded states.

Uses duck-typed fake providers and mocked HTTP so the suite stays offline and
deterministic. No dependency on geopandas or a real dataset.
"""

from __future__ import annotations

import httpx
import pytest
from unittest.mock import MagicMock, patch

from config import settings
from services.land_providers.base import (
    STATUS_MISSING,
    STATUS_OUT_OF_COVERAGE,
    STATUS_STALE,
    STATUS_UNAVAILABLE,
    LandContextResult,
    ProviderUnavailable,
    bbox_dict,
)
from services.land_service import (
    MODE_LOCAL_FIRST,
    MODE_LOCAL_ONLY,
    MODE_OVERPASS_ONLY,
    ProviderError,
    fetch_land_context,
)

BBOX = (21.24, 81.28, 21.27, 81.31)  # south, west, north, east


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeLocal:
    """Fake local provider: either returns a payload or raises ProviderUnavailable."""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def fetch(self, south, west, north, east):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def local_result(*, empty=False, status="available"):
    return LandContextResult(
        bbox=bbox_dict(*BBOX),
        water_bodies=[] if empty else [{"type": "way", "id": 1, "tags": {"natural": "water"}, "geometry": []}],
        roads=[],
        buildings=[],
        source="Local OSM GeoPackage (test.gpkg)",
        provider="local_osm",
        status=status,
    )


def overpass_response(elements=None):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"elements": elements or []}
    return response


@pytest.fixture(autouse=True)
def _enable_overpass(monkeypatch):
    monkeypatch.setitem(settings.__dict__, "OVERPASS_FALLBACK_ENABLED", True)


# ---------------------------------------------------------------------------
# 1. Local-first selection
# ---------------------------------------------------------------------------

def test_local_first_uses_local_and_never_calls_overpass():
    fake = FakeLocal(result=local_result())
    with patch("services.land_service.httpx.post", side_effect=AssertionError("Overpass must not be called")):
        payload = fetch_land_context(*BBOX, mode=MODE_LOCAL_FIRST, local_provider=fake)

    assert fake.calls == 1
    assert payload["provider"] == "local_osm"
    assert payload["status"] == "available"
    assert payload["source"].startswith("Local OSM GeoPackage")


def test_valid_empty_local_result_does_not_trigger_fallback():
    fake = FakeLocal(result=local_result(empty=True, status="empty"))
    with patch("services.land_service.httpx.post", side_effect=AssertionError("no fallback for empty result")):
        payload = fetch_land_context(*BBOX, local_provider=fake)

    assert payload["provider"] == "local_osm"
    assert payload["status"] == "empty"
    assert payload["water_bodies"] == []


# ---------------------------------------------------------------------------
# 2. Fallback policy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", [STATUS_MISSING, STATUS_OUT_OF_COVERAGE, STATUS_STALE])
def test_local_unusable_falls_back_to_overpass(status):
    fake = FakeLocal(error=ProviderUnavailable(status, f"local {status}"))
    with patch("services.land_service.httpx.post", return_value=overpass_response(
        [{"type": "way", "id": 7, "tags": {"highway": "primary"}}]
    )):
        payload = fetch_land_context(*BBOX, mode=MODE_LOCAL_FIRST, local_provider=fake)

    assert payload["provider"] == "overpass"
    assert payload["roads"][0]["tags"]["highway"] == "primary"
    assert "primary" in payload["source"]


def test_local_only_does_not_fall_back():
    fake = FakeLocal(error=ProviderUnavailable(STATUS_MISSING, "no dataset"))
    with patch("services.land_service.httpx.post", side_effect=AssertionError("fallback disabled")):
        payload = fetch_land_context(*BBOX, mode=MODE_LOCAL_ONLY, local_provider=fake)

    assert payload["status"] == STATUS_UNAVAILABLE
    assert payload["provider"] == "local_osm"
    assert "local_only" in payload["warning"]


def test_overpass_only_ignores_local():
    fake = FakeLocal(result=local_result())
    with patch("services.land_service.httpx.post", return_value=overpass_response()):
        payload = fetch_land_context(*BBOX, mode=MODE_OVERPASS_ONLY, local_provider=fake)

    assert fake.calls == 0
    assert payload["provider"] == "overpass"


def test_fallback_disabled_yields_degraded_state(monkeypatch):
    monkeypatch.setitem(settings.__dict__, "OVERPASS_FALLBACK_ENABLED", False)
    fake = FakeLocal(error=ProviderUnavailable(STATUS_MISSING, "no dataset"))
    with patch("services.land_service.httpx.post", side_effect=AssertionError("fallback disabled")):
        payload = fetch_land_context(*BBOX, mode=MODE_LOCAL_FIRST, local_provider=fake)

    assert payload["status"] == STATUS_UNAVAILABLE
    assert "warning" in payload
    assert payload["water_bodies"] == []


def test_no_provider_succeeds_returns_explicit_degraded_state():
    fake = FakeLocal(error=ProviderUnavailable(STATUS_MISSING, "no dataset"))

    def all_fail(url, **kwargs):
        raise httpx.NetworkError("unreachable")

    with patch("services.land_service.httpx.post", side_effect=all_fail):
        payload = fetch_land_context(*BBOX, mode=MODE_LOCAL_FIRST, local_provider=fake)

    assert payload["status"] == STATUS_UNAVAILABLE
    assert payload["provider"] == "overpass"
    assert "unavailable" in payload["source"].lower()
    assert payload["warning"]


# ---------------------------------------------------------------------------
# 3. Bounded external behaviour
# ---------------------------------------------------------------------------

def test_retry_exhaustion_is_bounded(monkeypatch):
    monkeypatch.setitem(settings.__dict__, "OVERPASS_RETRIES", 2)
    fake = FakeLocal(error=ProviderUnavailable(STATUS_MISSING, "no dataset"))
    attempts = {"n": 0}

    def fail(url, **kwargs):
        attempts["n"] += 1
        raise httpx.TimeoutException("timeout")

    with patch("services.land_service.httpx.post", side_effect=fail):
        payload = fetch_land_context(*BBOX, mode=MODE_LOCAL_FIRST, local_provider=fake)

    assert payload["status"] == STATUS_UNAVAILABLE
    # 3 providers deduped -> each retried (1 + OVERPASS_RETRIES)
    assert attempts["n"] == 3 * (1 + 2)


def test_connect_and_read_timeouts_are_applied():
    fake = FakeLocal(error=ProviderUnavailable(STATUS_MISSING, "no dataset"))
    with patch("services.land_service.httpx.post", return_value=overpass_response()) as mock:
        fetch_land_context(*BBOX, mode=MODE_LOCAL_FIRST, connect_timeout=7.0, read_timeout=11.0, local_provider=fake)

    timeout = mock.call_args[1]["timeout"]
    assert timeout.connect == 7.0
    assert timeout.read == 11.0


def test_invalid_bbox_still_raises():
    from services.land_service import LandError

    with pytest.raises(LandError):
        fetch_land_context(21.30, 81.28, 21.24, 81.31)


# ---------------------------------------------------------------------------
# 4. Source labelling / API regression
# ---------------------------------------------------------------------------

def test_local_payload_labelled_local_not_overpass(client, sample_kml_bytes):
    import main

    def local_payload(*args, **kwargs):
        return {
            "bbox": {"south": 21.0, "west": 81.0, "north": 21.5, "east": 81.5},
            "water_bodies": [],
            "roads": [],
            "buildings": [],
            "source": "Local OSM GeoPackage (land_context.gpkg)",
            "provider": "local_osm",
            "status": "available",
        }

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(main, "fetch_land_context", local_payload)
        payload = client.post(
            "/analyzeContour",
            files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
        ).json()

    land_context = payload["enrichment"]["land_context"]
    assert land_context["provider"] == "local_osm"
    assert land_context["source"].startswith("Local OSM GeoPackage")
    assert "Overpass" not in land_context["source"]


def test_analyze_unified_labels_local_data_source(client, sample_kml_bytes):
    import main
    import pytest as _pytest

    def local_payload(*args, **kwargs):
        return {
            "bbox": {"south": 21.0, "west": 81.0, "north": 21.5, "east": 81.5},
            "water_bodies": [],
            "roads": [],
            "buildings": [],
            "source": "Local OSM GeoPackage (land_context.gpkg)",
            "provider": "local_osm",
            "status": "available",
        }

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(main, "fetch_land_context", local_payload)
        payload = client.post(
            "/analyze",
            files={"file": ("sample_contours.kml", sample_kml_bytes, "application/xml")},
        ).json()

    sources = payload["metadata"]["data_sources"]
    assert "local_osm_geopackage" in sources
    assert "openstreetmap_overpass" not in sources


def test_gis_land_context_exposes_provider_status(client):
    import main
    import pytest as _pytest

    def unavailable_payload(south, west, north, east):
        return {
            "bbox": {"south": south, "west": west, "north": north, "east": east},
            "water_bodies": [],
            "roads": [],
            "buildings": [],
            "source": "Local OSM GeoPackage (unavailable)",
            "provider": "local_osm",
            "status": "unavailable",
            "warning": "dataset missing",
        }

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(main, "fetch_land_context", unavailable_payload)
        response = client.post("/gis/land-context", json={"bbox": {"south": 21.0, "west": 81.0, "north": 21.5, "east": 81.5}})

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "local_osm"
    assert body["status"] == "unavailable"
    assert body["warning"] == "dataset missing"


def test_unavailable_payload_is_not_treated_as_valid_features():
    """geographic_context must treat an unavailable payload as provider failure."""

    from backend.geographic_context import GeographicContextConfig, enrich_candidates
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    x, y = transformer.transform(81.30, 21.26)
    candidate = {"candidate_id": "a", "x": float(x), "y": float(y), "crs": "EPSG:32643", "latitude": 21.26, "longitude": 81.30}

    payload = {
        "bbox": {"south": 21.24, "west": 81.28, "north": 21.27, "east": 81.31},
        "water_bodies": [], "roads": [], "buildings": [],
        "source": "Local OSM GeoPackage (unavailable)",
        "provider": "local_osm",
        "status": "unavailable",
        "warning": "dataset missing",
    }

    enriched = enrich_candidates(
        [candidate],
        config=GeographicContextConfig(cache_enabled=False, cluster_radius_m=500.0, provider_retries=0),
        provider=lambda s, w, n, e: payload,
        cache=None,
    )["candidates"][0]

    assert enriched["provider_ok"] is False
    assert enriched["feasibility"] == "unknown"
    assert enriched["road_distance_m"] is None
    assert enriched["feasibility_reason"] == "dataset missing"
