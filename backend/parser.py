from __future__ import annotations

import io
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pyproj import Transformer

from models import Contour, ContourData


KML_NS = {
    "kml": "http://www.opengis.net/kml/2.2"
}

DEFAULT_TARGET_CRS = "EPSG:32644"


# ---------------------------------------------------------------------------
# FILE READING
# ---------------------------------------------------------------------------

def _read_kml_file(file_path: Union[str, Path]) -> bytes:
    """
    Read either KML or KMZ and return the KML XML bytes.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".kml":
        return path.read_bytes()

    if suffix == ".kmz":
        with zipfile.ZipFile(path, "r") as z:
            kml_files = [
                name
                for name in z.namelist()
                if name.lower().endswith(".kml")
            ]

            if not kml_files:
                raise ValueError("KMZ file does not contain a KML file")

            selected = next(
                (
                    name
                    for name in kml_files
                    if Path(name).name.lower() == "doc.kml"
                ),
                kml_files[0],
            )

            return z.read(selected)

    raise ValueError(
        "Unsupported file format. Expected .kml or .kmz"
    )


def _read_kml_bytes(data: bytes, filename: str = "input.kml") -> bytes:
    """Read KML or KMZ bytes from memory."""

    suffix = Path(filename).suffix.lower()

    if suffix == ".kml":
        return data

    if suffix == ".kmz":
        with zipfile.ZipFile(io.BytesIO(data), "r") as z:
            kml_files = [
                name
                for name in z.namelist()
                if name.lower().endswith(".kml")
            ]

            if not kml_files:
                raise ValueError("KMZ file does not contain a KML file")

            selected = next(
                (
                    name
                    for name in kml_files
                    if Path(name).name.lower() == "doc.kml"
                ),
                kml_files[0],
            )

            return z.read(selected)

    raise ValueError(
        "Unsupported file format. Expected .kml or .kmz"
    )


# ---------------------------------------------------------------------------
# KML HELPERS
# ---------------------------------------------------------------------------

def _strip_namespace(tag: str) -> str:
    return tag.split("}")[-1]


def _extract_coordinates(element: ET.Element) -> List[Tuple[float, float]]:
    """
    Extract longitude/latitude coordinates from a KML coordinates element.
    """

    coordinate_element = element.find(".//kml:coordinates", KML_NS)

    if coordinate_element is None or not coordinate_element.text:
        return []

    result = []

    raw_values = coordinate_element.text.strip().split()

    for value in raw_values:
        parts = value.split(",")

        if len(parts) < 2:
            continue

        try:
            lon = float(parts[0])
            lat = float(parts[1])
        except ValueError:
            continue

        result.append((lon, lat))

    return result


def _extract_numeric(text: Optional[str]) -> Optional[float]:
    """
    Extract a numerical elevation from arbitrary text.

    Examples:
        '285' -> 285
        'Contour 285 m' -> 285
        'Elevation: 285.0m' -> 285
    """

    if not text:
        return None

    match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
        text,
    )

    if not match:
        return None

    try:
        return float(match.group())
    except ValueError:
        return None


def _extract_elevation(placemark: ET.Element) -> Optional[float]:
    """
    Try several common KML elevation conventions.
    """

    # 1. Look for ExtendedData fields.
    for element in placemark.iter():

        tag = _strip_namespace(element.tag).lower()

        if tag in {
            "elevation",
            "elevation_m",
            "height",
            "altitude",
            "contour",
            "contourelevation",
        }:
            value = _extract_numeric(element.text)

            if value is not None:
                return value

    # 2. Look at SimpleData.
    for element in placemark.iter():

        tag = _strip_namespace(element.tag).lower()

        if tag == "simpledata":

            name = (
                element.attrib.get("name", "")
                .lower()
            )

            if any(
                key in name
                for key in [
                    "elev",
                    "height",
                    "altitude",
                    "contour",
                ]
            ):
                value = _extract_numeric(element.text)

                if value is not None:
                    return value

    # 3. Try name.
    name_element = placemark.find("kml:name", KML_NS)

    if name_element is not None:

        value = _extract_numeric(name_element.text)

        if value is not None:
            return value

    # 4. Try description.
    description = placemark.find(
        "kml:description",
        KML_NS,
    )

    if description is not None:

        value = _extract_numeric(description.text)

        if value is not None:
            return value

    return None


# ---------------------------------------------------------------------------
# PARSER
# ---------------------------------------------------------------------------

def parse_contour_file(
    file_path: Union[str, Path, bytes, bytearray],
    filename_or_crs: Optional[str] = None,
    target_crs: Optional[str] = None,
) -> ContourData:
    """
    Parse KML/KMZ contour map from a path or raw bytes.

    The legacy call pattern used in the tests passes a filename as the
    second positional argument, while the API path passes a CRS string.
    We accept both forms for compatibility.
    """

    if target_crs is None and filename_or_crs is not None:
        candidate = str(filename_or_crs)
        if candidate.upper().startswith("EPSG:") or candidate.upper().startswith("4326"):
            target_crs = candidate
            filename_for_bytes = "input.kml"
        else:
            target_crs = DEFAULT_TARGET_CRS
            filename_for_bytes = candidate
    elif target_crs is None:
        target_crs = DEFAULT_TARGET_CRS
        filename_for_bytes = "input.kml"
    else:
        filename_for_bytes = "input.kml"

    if isinstance(file_path, (bytes, bytearray)):
        xml_bytes = _read_kml_bytes(
            bytes(file_path),
            filename_for_bytes,
        )
    else:
        xml_bytes = _read_kml_file(file_path)

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise ValueError(
            f"Invalid KML XML: {exc}"
        ) from exc

    transformer = Transformer.from_crs(
        "EPSG:4326",
        target_crs,
        always_xy=True,
    )

    contours: List[Contour] = []

    geographic_points: List[Tuple[float, float]] = []

    projected_points: List[Tuple[float, float]] = []

    placemarks = root.findall(
        ".//kml:Placemark",
        KML_NS,
    )

    for placemark in placemarks:

        elevation = _extract_elevation(placemark)

        if elevation is None:
            continue

        coordinates = _extract_coordinates(placemark)

        if len(coordinates) < 2:
            continue

        projected = []

        for lon, lat in coordinates:

            x, y = transformer.transform(
                lon,
                lat,
            )

            projected.append(
                (float(x), float(y))
            )

            geographic_points.append(
                (float(lon), float(lat))
            )

            projected_points.append(
                (float(x), float(y))
            )

        name_element = placemark.find(
            "kml:name",
            KML_NS,
        )

        name = (
            name_element.text.strip()
            if name_element is not None
            and name_element.text
            else None
        )

        contours.append(
            Contour(
                elevation_m=float(elevation),
                coordinates=projected,
                name=name,
            )
        )

    if not contours:
        raise ValueError(
            "No valid contour lines were detected in the input file"
        )

    if not projected_points:
        raise ValueError(
            "No valid contour coordinates were detected"
        )

    projected_array = np.asarray(
        projected_points,
        dtype=float,
    )

    geographic_array = np.asarray(
        geographic_points,
        dtype=float,
    )

    return ContourData(
        contours=contours,
        source_crs="EPSG:4326",
        target_crs=target_crs,
        min_x=float(np.min(projected_array[:, 0])),
        max_x=float(np.max(projected_array[:, 0])),
        min_y=float(np.min(projected_array[:, 1])),
        max_y=float(np.max(projected_array[:, 1])),
        min_elevation_m=float(
            min(c.elevation_m for c in contours)
        ),
        max_elevation_m=float(
            max(c.elevation_m for c in contours)
        ),
        longitude_min=float(
            np.min(geographic_array[:, 0])
        ),
        longitude_max=float(
            np.max(geographic_array[:, 0])
        ),
        latitude_min=float(
            np.min(geographic_array[:, 1])
        ),
        latitude_max=float(
            np.max(geographic_array[:, 1])
        ),
    )


# ---------------------------------------------------------------------------
# DIAGNOSTICS
# ---------------------------------------------------------------------------

def contour_diagnostics(
    data: ContourData,
) -> Dict:
    """
    Generate diagnostics used by the DEM validation stage.
    """

    elevations = data.elevations

    unique_elevations = np.unique(
        np.round(elevations, 6)
    )

    frequencies = {}

    for elevation in unique_elevations:

        count = int(
            np.sum(
                np.isclose(
                    elevations,
                    elevation,
                )
            )
        )

        frequencies[str(float(elevation))] = count

    contour_intervals = np.diff(
        unique_elevations
    )

    lengths = []

    for contour in data.contours:

        coords = np.asarray(
            contour.coordinates,
            dtype=float,
        )

        if len(coords) < 2:
            lengths.append(0.0)
            continue

        diffs = np.diff(
            coords,
            axis=0,
        )

        distances = np.sqrt(
            np.sum(
                diffs ** 2,
                axis=1,
            )
        )

        lengths.append(
            float(np.sum(distances))
        )

    return {
        "contour_count": data.contour_count,

        "unique_elevation_count": int(
            len(unique_elevations)
        ),

        "elevation_min_m": float(
            np.min(elevations)
        ),

        "elevation_max_m": float(
            np.max(elevations)
        ),

        "elevation_mean_m": float(
            np.mean(elevations)
        ),

        "elevation_median_m": float(
            np.median(elevations)
        ),

        "elevation_std_m": float(
            np.std(elevations)
        ),

        "elevation_levels_m": [
            float(x)
            for x in unique_elevations
        ],

        "elevation_frequency": frequencies,

        "contour_interval": {
            "min_m": (
                float(np.min(contour_intervals))
                if len(contour_intervals)
                else 0.0
            ),
            "mean_m": (
                float(np.mean(contour_intervals))
                if len(contour_intervals)
                else 0.0
            ),
            "max_m": (
                float(np.max(contour_intervals))
                if len(contour_intervals)
                else 0.0
            ),
        },

        "spatial_extent": {
            "min_longitude": data.longitude_min,
            "max_longitude": data.longitude_max,
            "min_latitude": data.latitude_min,
            "max_latitude": data.latitude_max,
        },

        "contour_length": {
            "min_m": float(np.min(lengths)),
            "mean_m": float(np.mean(lengths)),
            "max_m": float(np.max(lengths)),
        },
    }