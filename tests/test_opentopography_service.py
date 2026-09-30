from unittest.mock import MagicMock, patch

import httpx
import pytest

from services.opentopography_service import (
    OpenTopographyError,
    OpenTopographyService,
    get_dem_for_bbox,
)


def test_get_dem_for_bbox_parses_response():
    service = OpenTopographyService(api_key="test-key")
    fake_response = MagicMock()
    fake_response.content = b"geotiff-bytes"
    fake_response.raise_for_status.return_value = None

    with patch("services.opentopography_service.httpx.Client") as mock_client, patch.object(
        service,
        "_parse_geotiff",
        return_value={
            "elevation": [[10.0, 11.0], [12.0, 13.0]],
            "rows": 2,
            "columns": 2,
            "elevation_min_m": 10.0,
            "elevation_max_m": 13.0,
            "elevation_mean_m": 11.5,
            "valid_cell_fraction": 1.0,
            "source": "OpenTopography Global DEM API",
        },
    ):
        mock_client.return_value.__enter__.return_value.get.return_value = fake_response
        result = service.get_dem(21.0, 22.0, 81.0, 82.0)

    assert result["rows"] == 2
    assert result["columns"] == 2
    assert result["elevation_min_m"] == 10.0
    assert result["source"] == "OpenTopography Global DEM API"


def test_get_dem_for_bbox_rejects_invalid_bbox():
    service = OpenTopographyService(api_key="test-key")

    with pytest.raises(OpenTopographyError, match="south must be smaller than north"):
        service.get_dem(20.0, 20.0, 81.0, 82.0)


def test_get_dem_for_bbox_handles_api_failure():
    service = OpenTopographyService(api_key="test-key")

    with patch("services.opentopography_service.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.get.side_effect = httpx.ReadTimeout("read timeout")

        with pytest.raises(OpenTopographyError, match="Unable to connect|timeout|retries"):
            service.get_dem(21.0, 22.0, 81.0, 82.0)


def test_get_dem_for_bbox_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENTOPOGRAPHY_API_KEY", raising=False)
    with pytest.raises(OpenTopographyError, match="API key"):
        OpenTopographyService(api_key=None)


def test_helper_function_wraps_service():
    with patch("services.opentopography_service.OpenTopographyService.get_dem") as mock_get:
        mock_get.return_value = {"rows": 1, "columns": 1, "elevation_min_m": 1.0}
        result = get_dem_for_bbox(81.0, 21.0, 82.0, 22.0)

    assert result["rows"] == 1
    assert result["columns"] == 1
