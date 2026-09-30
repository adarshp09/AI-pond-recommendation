from unittest.mock import MagicMock, patch

import pytest
import httpx

from services.land_service import LandError, fetch_land_context, ProviderError, PermanentError


def test_fetch_land_context_parses_overpass_response():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "elements": [
            {"type": "way", "id": 1, "tags": {"water": "pond"}, "geometry": [{"lat": 21.26, "lon": 81.29}]},
            {"type": "way", "id": 2, "tags": {"highway": "primary"}, "geometry": [{"lat": 21.27, "lon": 81.30}]},
            {"type": "node", "id": 3, "tags": {"building": "yes"}, "lat": 21.26, "lon": 81.29},
        ]
    }

    with patch("services.land_service.httpx.post", return_value=response):
        data = fetch_land_context(21.24, 81.28, 21.27, 81.31)

    assert data["water_bodies"][0]["tags"]["water"] == "pond"
    assert data["roads"][0]["tags"]["highway"] == "primary"
    assert data["buildings"][0]["tags"]["building"] == "yes"
    assert data["bbox"]["south"] == 21.24


def test_fetch_land_context_rejects_invalid_bbox():
    with pytest.raises(LandError, match="south must be smaller than north"):
        fetch_land_context(21.30, 81.28, 21.24, 81.31)


def test_fetch_land_context_handles_network_failure():
    with patch("services.land_service.httpx.post", side_effect=httpx.ConnectTimeout("timeout")):
        data = fetch_land_context(21.24, 81.28, 21.27, 81.31)
        assert data["water_bodies"] == []
        assert data["roads"] == []
        assert data["buildings"] == []
        assert "unavailable" in data["source"].lower() or "warning" in data


def test_fetch_land_context_primary_success():
    """Primary provider succeeds on first attempt."""
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"elements": [{"type": "way", "id": 1, "tags": {"water": "pond"}}]}

    with patch("services.land_service.httpx.post", return_value=response):
        data = fetch_land_context(21.24, 81.28, 21.27, 81.31)

    assert data["water_bodies"][0]["tags"]["water"] == "pond"
    assert "primary" in data["source"]


def test_fetch_land_context_primary_failure_fallback_success():
    """Primary fails (network error), fallback succeeds."""
    from config import settings
    from services.land_service import ProviderError

    primary_url = settings.OVERPASS_URL
    fallback_url = settings.OVERPASS_FALLBACK_URL

    def mock_post(url, **kwargs):
        if url == primary_url:
            raise httpx.NetworkError("Network unreachable")
        else:  # fallback URL
            response = MagicMock()
            response.raise_for_status.return_value = None
            response.json.return_value = {"elements": [{"type": "way", "id": 1, "tags": {"highway": "primary"}}]}
            return response

    with patch("services.land_service.httpx.post", side_effect=mock_post):
        data = fetch_land_context(21.24, 81.28, 21.27, 81.31)

    assert data["roads"][0]["tags"]["highway"] == "primary"
    assert "fallback" in data["source"]


def test_fetch_land_context_all_providers_fail():
    """All providers fail; returns empty land context with warning."""
    def mock_post(url, **kwargs):
        raise httpx.NetworkError("Network unreachable")

    with patch("services.land_service.httpx.post", side_effect=mock_post):
        data = fetch_land_context(21.24, 81.28, 21.27, 81.31)
        assert data["water_bodies"] == []
        assert data["roads"] == []
        assert data["buildings"] == []
        assert "unavailable" in data["source"].lower() or "warning" in data


def test_fetch_land_context_retries_then_success():
    """Provider fails twice then succeeds (bounded retries)."""
    attempts = [0]

    def mock_post(url, **kwargs):
        attempts[0] += 1
        if attempts[0] <= 2:
            raise httpx.TimeoutException("Timeout")
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"elements": [{"type": "way", "id": 1, "tags": {"water": "pond"}}]}
        return response

    with patch("services.land_service.httpx.post", side_effect=mock_post):
        data = fetch_land_context(21.24, 81.28, 21.27, 81.31)

    assert data["water_bodies"][0]["tags"]["water"] == "pond"
    assert attempts[0] == 3  # Initial + 2 retries


def test_fetch_land_context_4xx_no_retry():
    """4xx client errors are not retried (permanent failure)."""
    response = MagicMock()
    response.status_code = 400
    response.text = "Bad Request"
    response.raise_for_status.side_effect = httpx.HTTPStatusError("400", request=MagicMock(), response=response)

    with patch("services.land_service.httpx.post", return_value=response):
        data = fetch_land_context(21.24, 81.28, 21.27, 81.31)
        assert data["water_bodies"] == []
        assert data["roads"] == []
        assert data["buildings"] == []
        assert "unavailable" in data["source"].lower() or "warning" in data


def test_fetch_land_context_5xx_retries():
    """5xx server errors are retried and eventually fall back."""
    from config import settings

    attempts = [0]
    primary_url = settings.OVERPASS_URL
    fallback_url = settings.OVERPASS_FALLBACK_URL

    def mock_post(url, **kwargs):
        attempts[0] += 1
        if attempts[0] <= 2:
            response = MagicMock()
            response.status_code = 500
            response.text = "Internal Server Error"
            response.raise_for_status.side_effect = httpx.HTTPStatusError("500", request=MagicMock(), response=response)
            return response
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"elements": [{"type": "way", "id": 1, "tags": {"water": "pond"}}]}
        return response

    with patch("services.land_service.httpx.post", side_effect=mock_post):
        data = fetch_land_context(21.24, 81.28, 21.27, 81.31)

    assert data["water_bodies"][0]["tags"]["water"] == "pond"
    assert attempts[0] == 3  # Primary fails twice, fallback succeeds on second attempt


def test_fetch_land_context_timeout_configurable():
    """Custom timeout is respected."""
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"elements": []}

    with patch("services.land_service.httpx.post", return_value=response) as mock:
        fetch_land_context(21.24, 81.28, 21.27, 81.31, connect_timeout=10.0, read_timeout=15.0)

    # Check that the timeout was passed to httpx.post
    call_kwargs = mock.call_args[1]
    assert "timeout" in call_kwargs
    timeout_obj = call_kwargs["timeout"]
    assert timeout_obj.connect == 10.0
    assert timeout_obj.read == 15.0


def test_fetch_land_context_provider_deduplication():
    """Duplicate provider URLs are deduplicated."""
    from config import settings

    # Temporarily set same URL for both
    original_fallback = settings.OVERPASS_FALLBACK_URL
    settings.__dict__["OVERPASS_FALLBACK_URL"] = settings.OVERPASS_URL

    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"elements": []}

    with patch("services.land_service.httpx.post", return_value=response):
        data = fetch_land_context(21.24, 81.28, 21.27, 81.31)

    # Only one provider attempted
    assert "primary" in data["source"]

    # Restore
    settings.__dict__["OVERPASS_FALLBACK_URL"] = original_fallback