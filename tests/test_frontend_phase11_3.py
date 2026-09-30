from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from fastapi.testclient import TestClient
from main import app


def test_sample_contours_expose_phase_11_3_gis_fields():
    client = TestClient(app)
    sample_path = Path(__file__).resolve().parents[1] / "sample_contours.kml"

    response = client.post(
        "/analyzeContour",
        files={"file": (sample_path.name, sample_path.read_bytes(), "application/xml")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["pond_candidate"]["latitude"] is not None
    assert payload["pond_candidate"]["longitude"] is not None
    assert isinstance(payload.get("alternative_candidates"), list)
    assert payload["catchment"]["boundary"]["type"] in {"Polygon", "MultiPolygon"}


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for frontend tests")
def test_phase_11_3_leaflet_frontend_behaviour():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["node", "tests/frontend_phase11_3_runner.js"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
