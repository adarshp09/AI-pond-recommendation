"""Phase 04 — provider resilience tests (timeouts, retries, bounded failure).

All offline: HTTP clients are mocked. Caching is disabled by the shared conftest
fixture, so retry counts reflect real attempt counts.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from config import settings


def _ok(payload):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    return response


# ---------------------------------------------------------------------------
# Rainfall (Open-Meteo)
# ---------------------------------------------------------------------------

def test_rainfall_retries_then_succeeds(monkeypatch):
    monkeypatch.setitem(settings.__dict__, "RAINFALL_RETRIES", 2)
    from services.rainfall_service import fetch_historical_rainfall

    attempts = {"n": 0}

    def fake_get(*_a, **_k):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.NetworkError("flaky")
        return _ok({"daily": {"time": ["2024-01-01"], "precipitation_sum": [3.0]}})

    monkeypatch.setattr("services.rainfall_service.httpx.get", fake_get)
    data = fetch_historical_rainfall(21.26, 81.29, "2024-01-01", "2024-01-01")

    assert attempts["n"] == 3
    assert data["total_precipitation_mm"] == 3.0


def test_rainfall_exhausts_retries_and_raises(monkeypatch):
    monkeypatch.setitem(settings.__dict__, "RAINFALL_RETRIES", 2)
    from services.rainfall_service import RainfallError, fetch_historical_rainfall

    attempts = {"n": 0}

    def fake_get(*_a, **_k):
        attempts["n"] += 1
        raise httpx.ConnectTimeout("down")

    monkeypatch.setattr("services.rainfall_service.httpx.get", fake_get)
    with pytest.raises(RainfallError, match="Unable to connect"):
        fetch_historical_rainfall(21.26, 81.29, "2024-01-01", "2024-01-01")
    assert attempts["n"] == 3  # 1 + RAINFALL_RETRIES


# ---------------------------------------------------------------------------
# Geocoding (Nominatim)
# ---------------------------------------------------------------------------

def test_geocoding_retries_then_succeeds(monkeypatch):
    monkeypatch.setitem(settings.__dict__, "GEOCODING_RETRIES", 2)
    from services.geocoding_service import geocode_place

    attempts = {"n": 0}

    def fake_get(*_a, **_k):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise httpx.NetworkError("flaky")
        return _ok([{"display_name": "Nagpur", "lat": "21.1", "lon": "79.0", "address": {}}])

    monkeypatch.setattr("services.geocoding_service.httpx.get", fake_get)
    data = geocode_place("Nagpur")

    assert attempts["n"] == 2
    assert data["results"][0]["display_name"] == "Nagpur"


def test_geocoding_exhausts_retries_and_raises(monkeypatch):
    monkeypatch.setitem(settings.__dict__, "GEOCODING_RETRIES", 1)
    from services.geocoding_service import GeocodingError, geocode_place

    attempts = {"n": 0}

    def fake_get(*_a, **_k):
        attempts["n"] += 1
        raise httpx.ConnectTimeout("down")

    monkeypatch.setattr("services.geocoding_service.httpx.get", fake_get)
    with pytest.raises(GeocodingError, match="Unable to connect"):
        geocode_place("Nagpur")
    assert attempts["n"] == 2


def test_geocoding_throttle_spaces_requests(monkeypatch):
    import services.geocoding_service as geocoding

    monkeypatch.setitem(settings.__dict__, "GEOCODING_MIN_INTERVAL_S", 0.5)
    monkeypatch.setattr(geocoding, "_last_call_monotonic", 4.9)
    monkeypatch.setattr(geocoding.time, "monotonic", lambda: 5.0)

    slept = []
    monkeypatch.setattr(geocoding.time, "sleep", lambda seconds: slept.append(seconds))

    geocoding._throttle()

    assert slept and abs(slept[0] - 0.4) < 0.05


# ---------------------------------------------------------------------------
# Place search (Photon)
# ---------------------------------------------------------------------------

def test_place_search_retries_then_succeeds(monkeypatch):
    monkeypatch.setitem(settings.__dict__, "PLACE_SEARCH_RETRIES", 2)
    from services.place_search_service import search_places

    attempts = {"n": 0}

    def fake_get(*_a, **_k):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.NetworkError("flaky")
        return _ok({"features": [{"properties": {"name": "X"}, "geometry": {"coordinates": [79.0, 21.0]}}]})

    monkeypatch.setattr("services.place_search_service.httpx.get", fake_get)
    data = search_places("X")

    assert attempts["n"] == 3
    assert data["results"][0]["name"] == "X"


def test_place_search_exhausts_retries_and_raises(monkeypatch):
    monkeypatch.setitem(settings.__dict__, "PLACE_SEARCH_RETRIES", 1)
    from services.place_search_service import PlaceSearchError, search_places

    attempts = {"n": 0}

    def fake_get(*_a, **_k):
        attempts["n"] += 1
        raise httpx.ConnectTimeout("down")

    monkeypatch.setattr("services.place_search_service.httpx.get", fake_get)
    with pytest.raises(PlaceSearchError, match="Unable to connect"):
        search_places("X")
    assert attempts["n"] == 2


# ---------------------------------------------------------------------------
# DEM (OpenTopography) — retries are bounded and recover
# ---------------------------------------------------------------------------

def test_dem_retries_then_succeeds():
    from services.opentopography_service import OpenTopographyService

    service = OpenTopographyService(api_key="test-key", max_retries=2)
    response = MagicMock()
    response.content = b"geotiff-bytes"
    response.raise_for_status.return_value = None
    attempts = {"n": 0}

    def fake_get(*_a, **_k):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ReadTimeout("slow")
        return response

    with patch("services.opentopography_service.httpx.Client") as mock_client, patch.object(
        service, "_parse_geotiff", return_value={"rows": 2, "columns": 2, "elevation_min_m": 1.0}
    ):
        mock_client.return_value.__enter__.return_value.get.side_effect = fake_get
        result = service.get_dem(21.0, 22.0, 81.0, 82.0)

    assert attempts["n"] == 3
    assert result["rows"] == 2
