"""Deterministic, lightweight test fixtures for the pond pipeline.

Two families of fixtures live here:

1. **Contour (KML/KMZ) fixtures** – byte payloads that exercise the upload ->
   parse -> DEM -> hydrology -> catchment -> recommendation path for valid,
   sparse, irregular, flat, steep, large, malformed, corrupt, empty, duplicate
   and extreme inputs.
2. **Synthetic DEM fixtures** – numpy arrays that exercise hydrology and
   catchment behaviour that cannot be produced from KML alone (sinks, NaN
   regions, disconnected areas, flow cycles).

Everything is generated in-process and is byte-for-byte deterministic so tests
and benchmarks are repeatable and no binary artefacts are committed.
"""

from __future__ import annotations

import io
import math
import zipfile
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

CENTER_LON = 81.30
CENTER_LAT = 21.26

KML_CONTENT_TYPE = "application/vnd.google-earth.kml+xml"
KMZ_CONTENT_TYPE = "application/vnd.google-earth.kmz"


# ---------------------------------------------------------------------------
# KML / KMZ construction
# ---------------------------------------------------------------------------

def _square_ring(lon: float, lat: float, half: float) -> List[Tuple[float, float]]:
    return [
        (lon - half, lat - half),
        (lon + half, lat - half),
        (lon + half, lat + half),
        (lon - half, lat + half),
        (lon - half, lat - half),
    ]


def _irregular_ring(lon: float, lat: float, half: float, points: int = 10) -> List[Tuple[float, float]]:
    ring: List[Tuple[float, float]] = []
    for index in range(points + 1):
        angle = 2.0 * math.pi * index / points
        radius = half * (1.0 + 0.35 * math.sin(3.0 * angle))
        ring.append((lon + radius * math.cos(angle), lat + radius * math.sin(angle)))
    return ring


def _coordinates(points: Sequence[Tuple[float, float]], with_altitude: bool = False) -> str:
    if with_altitude:
        return " ".join(f"{lon:.9f},{lat:.9f},0" for lon, lat in points)
    return " ".join(f"{lon:.9f},{lat:.9f}" for lon, lat in points)


def placemark(
    name: str,
    points: Sequence[Tuple[float, float]],
    *,
    with_altitude: bool = False,
    geometry: str = "LineString",
    include_coordinates: bool = True,
) -> str:
    coordinates = _coordinates(points, with_altitude=with_altitude) if include_coordinates else ""
    return (
        "<Placemark>"
        f"<name>{name}</name>"
        f"<{geometry}><coordinates>{coordinates}</coordinates></{geometry}>"
        "</Placemark>"
    )


def build_kml(placemarks: Iterable[str]) -> bytes:
    body = "".join(placemarks)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2">'
        "<Document>"
        f"{body}"
        "</Document>"
        "</kml>"
    )
    return xml.encode("utf-8")


def to_kmz(kml_bytes: bytes, inner_name: str = "doc.kml") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(inner_name, kml_bytes)
    return buffer.getvalue()


def concentric_kml(
    levels: Sequence[float],
    *,
    max_half: float = 0.0025,
    min_half: float = 0.0008,
    points: int = 12,
    irregular: bool = False,
    lon: float = CENTER_LON,
    lat: float = CENTER_LAT,
) -> bytes:
    levels = list(levels)
    placemarks = []
    for index, level in enumerate(levels):
        fraction = index / max(len(levels) - 1, 1)
        half = max_half - (max_half - min_half) * fraction
        ring = _irregular_ring(lon, lat, half, points) if irregular else _square_ring(lon, lat, half)
        placemarks.append(placemark(f"{float(level):.1f}", ring))
    return build_kml(placemarks)


# ---------------------------------------------------------------------------
# Contour (KML/KMZ) fixtures
# ---------------------------------------------------------------------------

_VALID_LEVELS = [267.0, 268.0, 269.0, 270.0, 271.0]


def valid_kml() -> bytes:
    return concentric_kml(_VALID_LEVELS, max_half=0.0025, min_half=0.0008, points=12)


def valid_kmz() -> bytes:
    return to_kmz(valid_kml())


def sparse_contours_kml() -> bytes:
    return concentric_kml([270.0, 275.0], max_half=0.0025, min_half=0.0010)


def irregular_contours_kml() -> bytes:
    return concentric_kml(_VALID_LEVELS, max_half=0.0025, min_half=0.0008, points=16, irregular=True)


def flat_terrain_kml() -> bytes:
    # Four rings that all report the same elevation -> zero elevation range.
    return concentric_kml([100.0, 100.0, 100.0, 100.0], max_half=0.0020, min_half=0.0006)


def steep_terrain_kml() -> bytes:
    # Many levels packed into a tiny extent -> extreme gradients.
    levels = [100.0 + 5.0 * index for index in range(13)]
    return concentric_kml(levels, max_half=0.0004, min_half=0.00008, points=8)


def large_input_kml() -> bytes:
    # 40 levels over a moderate extent -> a larger DEM than the other fixtures.
    levels = [200.0 + 2.0 * index for index in range(40)]
    return concentric_kml(levels, max_half=0.0040, min_half=0.0004, points=24)


def extreme_elevations_kml() -> bytes:
    return concentric_kml([-500.0, 0.0, 4000.0, 9000.0], max_half=0.0020, min_half=0.0006)


def malformed_kml() -> bytes:
    return b"<kml><broken>"


def corrupt_kmz() -> bytes:
    return b"PK\x03\x04this-is-not-a-valid-zip-archive"


def empty_file() -> bytes:
    return b""


def missing_elevation_kml() -> bytes:
    placemarks = [
        placemark("contour-without-elevation-a", _square_ring(CENTER_LON, CENTER_LAT, 0.0024)),
        placemark("no numeric elevation here", _square_ring(CENTER_LON, CENTER_LAT, 0.0022)),
        placemark("270", _square_ring(CENTER_LON, CENTER_LAT, 0.0020)),
        placemark("271", _square_ring(CENTER_LON, CENTER_LAT, 0.0016)),
        placemark("272", _square_ring(CENTER_LON, CENTER_LAT, 0.0012)),
    ]
    return build_kml(placemarks)


def missing_coordinates_kml() -> bytes:
    placemarks = [
        placemark("270", _square_ring(CENTER_LON, CENTER_LAT, 0.0020)),
        placemark("271", [], include_coordinates=False),
        placemark("272", _square_ring(CENTER_LON, CENTER_LAT, 0.0014)),
        placemark("273", [], include_coordinates=False),
        placemark("274", _square_ring(CENTER_LON, CENTER_LAT, 0.0010)),
    ]
    return build_kml(placemarks)


def duplicate_points_kml() -> bytes:
    base = _square_ring(CENTER_LON, CENTER_LAT, 0.0020)
    duplicated = [point for point in base for _ in (0, 1)]
    repeated = [
        placemark("270", duplicated),
        placemark("270", duplicated),
        placemark("275", _square_ring(CENTER_LON, CENTER_LAT, 0.0010)),
    ]
    return build_kml(repeated)


def kmz_without_kml() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("readme.txt", "no kml here")
    return buffer.getvalue()


def invalid_geometry_kml() -> bytes:
    # Every placemark has fewer than two coordinates -> parser must reject.
    placemarks = [
        placemark("270", [(CENTER_LON, CENTER_LAT)]),
        placemark("271", [(CENTER_LON + 0.001, CENTER_LAT + 0.001)]),
    ]
    return build_kml(placemarks)


def insufficient_contours_kml() -> bytes:
    return build_kml([placemark("270", [(CENTER_LON, CENTER_LAT), (CENTER_LON + 0.0005, CENTER_LAT)])])


def no_coordinates_text_kml() -> bytes:
    # coordinates element present but contains junk tokens.
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        "<Placemark><name>270</name><LineString><coordinates>nan,nan foo,bar</coordinates></LineString></Placemark>"
        "</Document></kml>"
    )
    return xml.encode("utf-8")


# ---------------------------------------------------------------------------
# Synthetic DEM fixtures
# ---------------------------------------------------------------------------

def plane_dem(rows: int = 6, columns: int = 6, base: float = 100.0, dy: float = 0.5, dx: float = 0.25) -> np.ndarray:
    yy, xx = np.mgrid[0:rows, 0:columns]
    return base + yy * dy + xx * dx


def flat_dem(shape: Tuple[int, int] = (5, 5), value: float = 10.0) -> np.ndarray:
    return np.full(shape, value, dtype=float)


def basin_dem(size: int = 7) -> np.ndarray:
    """A bowl that drains to a single central sink (local minimum)."""

    yy, xx = np.mgrid[0:size, 0:size]
    center = (size - 1) / 2.0
    return ((yy - center) ** 2 + (xx - center) ** 2).astype(float)


def sink_dem() -> np.ndarray:
    """Plane with a single artificial pit so exactly one sink is created."""

    dem = plane_dem(5, 5)
    dem[2, 2] = 0.0
    return dem


def nan_dem(fraction_rows: int = 2) -> np.ndarray:
    """Plane whose bottom rows are NaN (invalid DEM cells)."""

    dem = plane_dem(6, 6)
    dem[-fraction_rows:, :] = np.nan
    return dem


def disconnected_dem() -> np.ndarray:
    """Two valid regions separated by a fully-invalid NaN column."""

    dem = plane_dem(5, 5)
    dem[:, 2] = np.nan
    return dem


def single_valid_cell_dem() -> np.ndarray:
    dem = np.full((3, 3), np.nan)
    dem[1, 1] = 5.0
    return dem


def cycle_flow_direction() -> np.ndarray:
    """A hand-built D8 direction grid containing a 2x2 flow cycle.

    Directions: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW.
    Cells (0,0)->(0,1)->(1,1)->(1,0)->(0,0) form a closed cycle.
    """

    flow = np.full((2, 2), -1, dtype=np.int8)
    flow[0, 0] = 2  # east -> (0,1)
    flow[0, 1] = 4  # south -> (1,1)
    flow[1, 1] = 6  # west -> (1,0)
    flow[1, 0] = 0  # north -> (0,0)
    return flow


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

CONTOUR_FIXTURES: Dict[str, callable] = {
    "valid_kml": valid_kml,
    "valid_kmz": valid_kmz,
    "sparse_contours": sparse_contours_kml,
    "irregular_contours": irregular_contours_kml,
    "flat_terrain": flat_terrain_kml,
    "steep_terrain": steep_terrain_kml,
    "large_input": large_input_kml,
    "extreme_elevations": extreme_elevations_kml,
    "malformed_kml": malformed_kml,
    "corrupt_kmz": corrupt_kmz,
    "kmz_without_kml": kmz_without_kml,
    "empty_file": empty_file,
    "missing_elevation": missing_elevation_kml,
    "missing_coordinates": missing_coordinates_kml,
    "duplicate_points": duplicate_points_kml,
    "invalid_geometry": invalid_geometry_kml,
    "insufficient_contours": insufficient_contours_kml,
    "no_coordinates_text": no_coordinates_text_kml,
}

FILE_SUFFIXES = {
    "valid_kml": "fixture_valid.kml",
    "valid_kmz": "fixture_valid.kmz",
    "sparse_contours": "fixture_sparse.kml",
    "irregular_contours": "fixture_irregular.kml",
    "flat_terrain": "fixture_flat.kml",
    "steep_terrain": "fixture_steep.kml",
    "large_input": "fixture_large.kml",
    "extreme_elevations": "fixture_extreme.kml",
    "malformed_kml": "fixture_malformed.kml",
    "corrupt_kmz": "fixture_corrupt.kmz",
    "kmz_without_kml": "fixture_no_kml.kmz",
    "empty_file": "fixture_empty.kml",
    "missing_elevation": "fixture_missing_elevation.kml",
    "missing_coordinates": "fixture_missing_coordinates.kml",
    "duplicate_points": "fixture_duplicate_points.kml",
    "invalid_geometry": "fixture_invalid_geometry.kml",
    "insufficient_contours": "fixture_insufficient.kml",
    "no_coordinates_text": "fixture_no_coords.kml",
}


def build_all() -> Dict[str, Tuple[str, bytes]]:
    """Return ``{name: (filename, bytes)}`` for every contour fixture."""

    return {name: (FILE_SUFFIXES[name], factory()) for name, factory in CONTOUR_FIXTURES.items()}


__all__ = [
    "CONTOUR_FIXTURES",
    "CENTER_LAT",
    "CENTER_LON",
    "build_all",
    "build_kml",
    "concentric_kml",
    "corrupt_kmz",
    "cycle_flow_direction",
    "disconnected_dem",
    "flat_dem",
    "nan_dem",
    "placemark",
    "plane_dem",
    "single_valid_cell_dem",
    "sink_dem",
    "to_kmz",
]
