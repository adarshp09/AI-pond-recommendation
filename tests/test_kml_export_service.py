from io import BytesIO
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from services.kml_export_service import generate_kml, generate_kmz


ANALYSIS = {
    "center": {"latitude": 21.2, "longitude": 81.4},
    "min_latitude": 21.1,
    "max_latitude": 21.3,
    "min_longitude": 81.2,
    "max_longitude": 81.6,
    "catchment": {"boundary": {"type": "Polygon", "coordinates": [[[81.3, 21.1], [81.5, 21.1], [81.5, 21.3], [81.3, 21.3], [81.3, 21.1]]]}},
    "pond_candidate": {"latitude": 21.21, "longitude": 81.41},
    "alternative_candidates": [{"latitude": 21.22, "longitude": 81.42}],
}


def test_generate_kml_contains_valid_wgs84_coordinates():
    content = generate_kml(ANALYSIS)
    root = ET.fromstring(content)
    text = content
    assert root.tag.endswith("kml")
    assert "81.4,21.2,0" in text
    assert "81.41,21.21,0" in text
    assert "Selected Location" in text
    assert "Catchment Boundary" in text


def test_generate_kmz_contains_doc_kml():
    archive = ZipFile(BytesIO(generate_kmz(ANALYSIS)))
    assert archive.namelist() == ["doc.kml"]
    ET.fromstring(archive.read("doc.kml"))


def test_missing_geometry_is_omitted_safely():
    content = generate_kml({"center": {"latitude": 21.2, "longitude": 81.4}})
    assert "Selected Location" in content
    assert "Analysis Boundary" not in content
    ET.fromstring(content)


def test_invalid_coordinates_are_not_exported():
    content = generate_kml({"pond_candidate": {"latitude": 999, "longitude": 999}})
    assert "999" not in content
    ET.fromstring(content)
