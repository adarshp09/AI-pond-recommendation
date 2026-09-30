from unittest.mock import MagicMock, patch

import pytest

from services.geocoding_service import GeocodingError, geocode_place


def test_geocode_place_parses_successful_response():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = [
        {
            "display_name": "Nagpur, Maharashtra, India",
            "lat": "21.1458",
            "lon": "79.0882",
            "address": {"city": "Nagpur"},
        }
    ]

    with patch("services.geocoding_service.httpx.get", return_value=response):
        data = geocode_place("Nagpur")

    assert data["results"][0]["display_name"] == "Nagpur, Maharashtra, India"
    assert data["results"][0]["latitude"] == 21.1458
    assert data["results"][0]["longitude"] == 79.0882


def test_geocode_place_rejects_empty_query():
    with pytest.raises(GeocodingError, match="query"):
        geocode_place("   ")


def test_geocode_place_handles_network_failure():
    with patch("services.geocoding_service.httpx.get", side_effect=TimeoutError("timeout")):
        with pytest.raises(GeocodingError, match="Unable to connect|timeout"):
            geocode_place("Nagpur")
