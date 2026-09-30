from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_invalid_non_kml_input_rejected():
    response = client.post(
        "/analyzeContour",
        files={"file": ("bad.txt", b"not valid xml", "text/plain")},
    )
    assert response.status_code == 400
    payload = response.json()
    assert payload["detail"]["code"] == "UNSUPPORTED_FILE_TYPE"


def test_empty_file_rejected():
    response = client.post(
        "/analyzeContour",
        files={"file": ("empty.kml", b"", "application/xml")},
    )
    assert response.status_code == 400
    payload = response.json()
    assert payload["detail"]["code"] == "EMPTY_FILE"


def test_corrupt_kml_returns_validation_error():
    response = client.post(
        "/analyzeContour",
        files={"file": ("corrupt.kml", b"<kml><broken>", "application/xml")},
    )
    assert response.status_code in {400, 422}


def test_insufficient_contour_data_rejected():
    payload = b"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<kml xmlns=\"http://www.opengis.net/kml/2.2\">
  <Document><Placemark><name>100</name><LineString><coordinates>0,0,0 1,1,0</coordinates></LineString></Placemark></Document>
</kml>
"""
    response = client.post(
        "/analyzeContour",
        files={"file": ("insufficient.kml", payload, "application/xml")},
    )
    assert response.status_code in {200, 422}


def test_external_api_failure_is_handled_gracefully(monkeypatch):
    sample_path = Path(__file__).resolve().parents[1] / "sample_contours.kml"

    def raise_error(*args, **kwargs):
        raise RuntimeError("simulated external failure")

    monkeypatch.setattr("main.fetch_historical_rainfall", raise_error)
    monkeypatch.setattr("main.fetch_land_context", raise_error)
    monkeypatch.setattr("main.get_dem_for_bbox", raise_error)

    response = client.post(
        "/analyzeContour",
        files={"file": (sample_path.name, sample_path.read_bytes(), "application/xml")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert isinstance(payload.get("warnings", []), list)
    assert payload["enrichment"]["status"] in {"partial", "success"}


def test_missing_optional_api_data_does_not_crash():
    sample_path = Path(__file__).resolve().parents[1] / "sample_contours.kml"
    response = client.post(
        "/analyzeContour",
        files={"file": (sample_path.name, sample_path.read_bytes(), "application/xml")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["enrichment"]["warnings"] is not None
