from unittest.mock import MagicMock, patch

import pytest

from services.place_search_service import PlaceSearchError, search_places


def test_search_places_parses_response():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "features": [
            {
                "properties": {
                    "name": "Nagpur",
                    "country": "India",
                },
                "geometry": {"coordinates": [79.0882, 21.1458]},
            }
        ]
    }

    with patch("services.place_search_service.httpx.get", return_value=response):
        data = search_places("Nagpur")

    assert data["results"][0]["name"] == "Nagpur"
    assert data["results"][0]["latitude"] == 21.1458
    assert data["results"][0]["longitude"] == 79.0882


def test_search_places_rejects_empty_query():
    with pytest.raises(PlaceSearchError, match="query"):
        search_places("  ")


def test_search_places_handles_network_failure():
    with patch("services.place_search_service.httpx.get", side_effect=TimeoutError("timeout")):
        with pytest.raises(PlaceSearchError, match="Unable to connect|timeout"):
            search_places("Nagpur")
