from unittest.mock import MagicMock, patch

import pytest

from services.rainfall_service import RainfallError, fetch_historical_rainfall


def test_fetch_historical_rainfall_parses_response():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "daily": {
            "time": ["2024-01-01", "2024-01-02"],
            "precipitation_sum": [0.0, 12.5],
        },
        "latitude": 21.26,
        "longitude": 81.29,
        "timezone": "GMT",
    }

    with patch("services.rainfall_service.httpx.get", return_value=response):
        data = fetch_historical_rainfall(21.26, 81.29, "2024-01-01", "2024-01-02")

    assert data["latitude"] == 21.26
    assert data["daily_rainfall_mm"][0] == 0.0
    assert data["daily_rainfall_mm"][1] == 12.5
    assert data["total_precipitation_mm"] == 12.5


def test_fetch_historical_rainfall_rejects_invalid_coordinates():
    with pytest.raises(RainfallError, match="latitude"):
        fetch_historical_rainfall(91, 81.29, "2024-01-01", "2024-01-02")


def test_fetch_historical_rainfall_handles_network_failure():
    with patch("services.rainfall_service.httpx.get", side_effect=TimeoutError("timeout")):
        with pytest.raises(RainfallError, match="Unable to connect|timeout"):
            fetch_historical_rainfall(21.26, 81.29, "2024-01-01", "2024-01-02")


def test_fetch_historical_rainfall_requires_dates():
    with pytest.raises(RainfallError, match="start_date"):
        fetch_historical_rainfall(21.26, 81.29, None, "2024-01-02")
