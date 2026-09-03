from __future__ import annotations

from io import BytesIO
from typing import Any, Iterable, Mapping
from zipfile import ZIP_DEFLATED, ZipFile
import math
import xml.etree.ElementTree as ET


KML_NAMESPACE = "http://www.opengis.net/kml/2.2"
ET.register_namespace("", KML_NAMESPACE)


def _tag(name: str) -> str:
    return f"{{{KML_NAMESPACE}}}{name}"


def _finite_pair(value: Any) -> tuple[float, float] | None:
    try:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        longitude, latitude = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(longitude) and math.isfinite(latitude)):
        return None
    if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
        return None
    return longitude, latitude


def _coordinates(values: Iterable[Any], *, close: bool = False) -> list[tuple[float, float]]:
    points = [point for point in (_finite_pair(value) for value in values) if point is not None]
    if close and len(points) >= 3 and points[0] != points[-1]:
        points.append(points[0])
    return points


def _coordinate_text(points: Iterable[tuple[float, float]]) -> str:
    return " ".join(f"{longitude:.12g},{latitude:.12g},0" for longitude, latitude in points)


def _name(parent: ET.Element, value: str) -> None:
    ET.SubElement(parent, _tag("name")).text = value


def _point_placemark(document: ET.Element, name: str, location: Any) -> bool:
    if not isinstance(location, Mapping):
        return False
    point = _finite_pair((location.get("longitude"), location.get("latitude")))
    if point is None:
        return False
    placemark = ET.SubElement(document, _tag("Placemark"))
    _name(placemark, name)
    point_element = ET.SubElement(placemark, _tag("Point"))
    ET.SubElement(point_element, _tag("coordinates")).text = _coordinate_text([point])
    return True


def _line_placemark(document: ET.Element, name: str, points: Iterable[Any]) -> bool:
    coordinates = _coordinates(points, close=False)
    if len(coordinates) < 2:
        return False
    placemark = ET.SubElement(document, _tag("Placemark"))
    _name(placemark, name)
    line = ET.SubElement(placemark, _tag("LineString"))
    ET.SubElement(line, _tag("tessellate")).text = "1"
    ET.SubElement(line, _tag("coordinates")).text = _coordinate_text(coordinates)
    return True


def _polygon_placemark(document: ET.Element, name: str, coordinates: Iterable[Any]) -> bool:
    ring = _coordinates(coordinates, close=True)
    if len(ring) < 4:
        return False
    placemark = ET.SubElement(document, _tag("Placemark"))
    _name(placemark, name)
    polygon = ET.SubElement(placemark, _tag("Polygon"))
    outer = ET.SubElement(ET.SubElement(polygon, _tag("outerBoundaryIs")), _tag("LinearRing"))
    ET.SubElement(outer, _tag("coordinates")).text = _coordinate_text(ring)
    return True


def _geojson_placemark(document: ET.Element, name: str, geometry: Any) -> bool:
    if not isinstance(geometry, Mapping):
        return False
    if geometry.get("type") == "Feature":
        geometry = geometry.get("geometry")
    if not isinstance(geometry, Mapping):
        return False
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon" and isinstance(coordinates, list):
        rings = [ring for ring in coordinates if isinstance(ring, list)]
        return bool(rings) and _polygon_placemark(document, name, rings[0])
    if geometry_type == "MultiPolygon" and isinstance(coordinates, list):
        created = False
        for index, polygon in enumerate(coordinates, start=1):
            if isinstance(polygon, list) and polygon:
                created = _polygon_placemark(document, f"{name} {index}", polygon[0]) or created
        return created
    if geometry_type == "LineString" and isinstance(coordinates, list):
        return _line_placemark(document, name, coordinates)
    return False


def _boundary_placemark(document: ET.Element, name: str, boundary: Any) -> bool:
    if _geojson_placemark(document, name, boundary):
        return True
    if isinstance(boundary, Mapping) and "coordinates" in boundary:
        return _polygon_placemark(document, name, boundary["coordinates"])
    if isinstance(boundary, list):
        return _polygon_placemark(document, name, boundary)
    return False


def _analysis_boundary(analysis_result: Mapping[str, Any]) -> list[tuple[float, float]]:
    try:
        south = float(analysis_result["min_latitude"])
        north = float(analysis_result["max_latitude"])
        west = float(analysis_result["min_longitude"])
        east = float(analysis_result["max_longitude"])
    except (KeyError, TypeError, ValueError):
        return []
    return _coordinates([(west, south), (east, south), (east, north), (west, north)], close=True)


def generate_kml(analysis_result: Mapping[str, Any]) -> str:
    """Generate a WGS84 KML document from an analysis response."""
    if not isinstance(analysis_result, Mapping):
        raise ValueError("Analysis result must be an object.")

    root = ET.Element(_tag("kml"))
    document = ET.SubElement(root, _tag("Document"))
    _name(document, "AI Pond Recommendation Analysis")

    center = analysis_result.get("center")
    _point_placemark(document, "Selected Location", center)
    _boundary_placemark(document, "Analysis Boundary", _analysis_boundary(analysis_result))

    catchment = analysis_result.get("catchment") or {}
    _boundary_placemark(document, "Catchment Boundary", catchment.get("boundary"))

    _point_placemark(document, "Recommended Pond", analysis_result.get("pond_candidate"))
    alternatives = analysis_result.get("alternative_candidates") or []
    for index, candidate in enumerate(alternatives, start=1):
        _point_placemark(document, f"Alternative Candidate {index}", candidate)

    contours = analysis_result.get("contours") or analysis_result.get("elevation_contours") or []
    for index, contour in enumerate(contours, start=1):
        if not isinstance(contour, Mapping):
            continue
        name = f"Elevation Contour {contour.get('elevation_m', index)}"
        _line_placemark(document, name, contour.get("coordinates") or [])

    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def generate_kmz(analysis_result: Mapping[str, Any]) -> bytes:
    """Package generated KML as a standards-compatible KMZ archive."""
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("doc.kml", generate_kml(analysis_result).encode("utf-8"))
    return output.getvalue()
