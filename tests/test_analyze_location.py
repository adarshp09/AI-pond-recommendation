import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch

from main import app

client = TestClient(app)


def test_analyze_location_allows_browser_preflight_from_ipv6_dev_server():
    response = client.options(
        "/analyzeLocation",
        headers={
            "Origin": "http://[::]:5500",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://[::]:5500"
    assert "POST" in response.headers["access-control-allow-methods"]


def _mock_dem_response(dem_array):
    """Helper to create a mock DEM response."""
    return {
        "elevation": dem_array,
        "rows": dem_array.shape[0],
        "columns": dem_array.shape[1],
        "elevation_min_m": float(np.nanmin(dem_array)),
        "elevation_max_m": float(np.nanmax(dem_array)),
        "elevation_mean_m": float(np.nanmean(dem_array)),
        "source": "OpenTopography Global DEM API",
    }


def test_valid_post_analyze_location():
    payload = {"latitude": 21.2, "longitude": 81.4, "radius_km": 12}
    dem = np.array([
        [100.0, 101.0, 102.0],
        [100.5, 101.5, 102.5],
        [101.0, 102.0, 103.0],
    ], dtype=float)

    with patch("main.get_dem_for_bbox") as mock_get_dem:
        mock_get_dem.return_value = _mock_dem_response(dem)
        response = client.post("/analyzeLocation", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["center"]["latitude"] == payload["latitude"]
    assert data["center"]["longitude"] == payload["longitude"]
    assert data["radius_km"] == payload["radius_km"]
    assert data["min_latitude"] <= payload["latitude"] <= data["max_latitude"]
    assert data["min_longitude"] <= payload["longitude"] <= data["max_longitude"]
    assert data["status"] == "success"
    assert "dem" in data
    assert "terrain" in data
    assert "hydrology" in data


def test_invalid_request_returns_4xx():
    response = client.post("/analyzeLocation", json={"latitude": 99, "longitude": 80, "radius_km": 5})
    assert response.status_code in {400, 422}

    response = client.post("/analyzeLocation", json={"latitude": 20, "longitude": 80, "radius_km": 0})
    assert response.status_code in {400, 422}


def test_response_contains_center_and_bounds():
    payload = {"latitude": 10, "longitude": 20, "radius_km": 5}
    dem = np.array([
        [100.0, 101.0, 102.0],
        [100.5, 101.5, 102.5],
        [101.0, 102.0, 103.0],
    ], dtype=float)

    with patch("main.get_dem_for_bbox") as mock_get_dem:
        mock_get_dem.return_value = _mock_dem_response(dem)
        response = client.post("/analyzeLocation", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert set(["center", "min_latitude", "max_latitude", "min_longitude", "max_longitude", "radius_km"]).issubset(data.keys())
    assert isinstance(data["center"], dict)
    assert data["status"] == "success"


def test_location_route_enriches_dem_and_hydrology_when_available():
    dem = np.array([
        [10.0, 11.0, 12.0],
        [10.5, 11.5, 12.5],
        [11.0, 12.0, 13.0],
    ], dtype=float)

    with patch("main.get_dem_for_bbox") as mock_get_dem:
        mock_get_dem.return_value = {
            "elevation": dem,
            "rows": dem.shape[0],
            "columns": dem.shape[1],
            "elevation_min_m": float(np.nanmin(dem)),
            "elevation_max_m": float(np.nanmax(dem)),
            "elevation_mean_m": float(np.nanmean(dem)),
            "source": "OpenTopography Global DEM API",
        }

        response = client.post("/analyzeLocation", json={"latitude": 21.2, "longitude": 81.4, "radius_km": 12})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["dem"]["source"] == "OpenTopography Global DEM API"
    assert payload["terrain"]["grid_rows"] == 3
    assert payload["hydrology"]["valid_cells"] > 0


def test_location_route_handles_opentopography_failure():
    with patch("main.get_dem_for_bbox", side_effect=Exception("DEM unavailable")):
        response = client.post("/analyzeLocation", json={"latitude": 21.2, "longitude": 81.4, "radius_km": 12})

    assert response.status_code == 502
    payload = response.json()
    assert payload["detail"]["code"] == "DEM_FETCH_ERROR"


def test_analyze_contour_still_works():
    response = client.get("/health")
    assert response.status_code == 200

    with open("sample_contours.kml", "rb") as fh:
        upload = client.post("/analyzeContour", files={"file": ("sample_contours.kml", fh, "application/vnd.google-earth.kml+xml")})

    assert upload.status_code == 200
    payload = upload.json()
    assert payload["status"] == "success"
