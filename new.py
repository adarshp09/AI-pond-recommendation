from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple

from services.geocoding_service import geocode_place


DEFAULT_CENTER = (81.30, 21.26)


def _resolve_center(location: str | None = None, latitude: float | None = None, longitude: float | None = None) -> Tuple[float, float]:
    if latitude is not None and longitude is not None:
        return float(longitude), float(latitude)

    if location and str(location).strip():
        try:
            result = geocode_place(str(location).strip())
            candidates = result.get("results") or []
            if candidates:
                lat = float(candidates[0].get("latitude", DEFAULT_CENTER[1]))
                lon = float(candidates[0].get("longitude", DEFAULT_CENTER[0]))
                return lon, lat
        except Exception:
            pass

    return DEFAULT_CENTER


def _ring_points(center_x: float, center_y: float, radius: float, levels: int = 12) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for i in range(levels):
        angle = (2.0 * math.pi * i) / levels
        points.append((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))
    return points


def _contour_line(contour_elevation: float, center_x: float, center_y: float, radius: float) -> str:
    pts = _ring_points(center_x, center_y, radius)
    coords = " ".join(f"{x},{y},0" for x, y in pts)
    return (
        "<Placemark>\n"
        f"  <name>{contour_elevation}</name>\n"
        "  <LineString>\n"
        "    <coordinates>\n"
        f"      {coords}\n"
        "    </coordinates>\n"
        "  </LineString>\n"
        "</Placemark>\n"
    )


def generate_synthetic_kml(
    scenario: str = "basin",
    output_path: str | Path = "synthetic_contours.kml",
    location: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    center_lon, center_lat = _resolve_center(location=location, latitude=latitude, longitude=longitude)

    scenarios = {
        "basin": {"base": 240.0, "step": 5.0, "radius": 0.0035},
        "valley": {"base": 210.0, "step": 6.0, "radius": 0.0040},
        "hill": {"base": 300.0, "step": 8.0, "radius": 0.0032},
        "flat": {"base": 280.0, "step": 1.0, "radius": 0.0025},
        "invalid": {"base": 50.0, "step": 2.0, "radius": 0.0004},
    }

    config = scenarios.get(str(scenario).lower(), scenarios["basin"])
    contours: List[str] = []
    for i in range(6):
        elevation = config["base"] + i * config["step"]
        radius = config["radius"] + i * 0.0002
        contours.append(_contour_line(elevation, center_lon, center_lat, radius))

    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        '  <Document>\n'
        f'    <name>synthetic_{str(scenario).lower()}_{str(location or "custom").replace(" ", "_")}</name>\n'
        + "".join(contours) +
        '  </Document>\n'
        '</kml>\n'
    )

    path.write_text(kml, encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic KML contour data for a location.")
    parser.add_argument("location", nargs="?", default=None, help="Place name, for example: 'Durg CG'")
    parser.add_argument("--scenario", choices=["basin", "valley", "hill", "flat", "invalid"], default="basin")
    parser.add_argument("--output", default="generated/custom_contours.kml", help="Output KML path")
    args = parser.parse_args()

    generate_synthetic_kml(
        scenario=args.scenario,
        output_path=args.output,
        location=args.location,
    )
    print(f"Generated {args.output} for location: {args.location or 'default center'}")
