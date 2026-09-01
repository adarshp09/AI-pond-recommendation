from __future__ import annotations

import math


MAX_RADIUS_KM = 100.0
EARTH_RADIUS_KM = 6371.0


def calculate_analysis_bounds(latitude: float, longitude: float, radius_km: float) -> dict:
    """Return a center plus bounding box for a circular analysis area."""
    try:
        lat = float(latitude)
        lon = float(longitude)
        radius = float(radius_km)
    except (TypeError, ValueError) as exc:
        raise ValueError("Latitude, longitude, and radius must be numeric values.") from exc

    if not -90.0 <= lat <= 90.0:
        raise ValueError("Latitude must be between -90 and 90 degrees.")
    if not -180.0 <= lon <= 180.0:
        raise ValueError("Longitude must be between -180 and 180 degrees.")
    if radius <= 0:
        raise ValueError("Radius must be greater than zero.")
    if radius > MAX_RADIUS_KM:
        raise ValueError(f"Radius must not exceed {MAX_RADIUS_KM} km.")

    lat_radians = math.radians(lat)
    lon_radians = math.radians(lon)

    lat_delta = (radius / EARTH_RADIUS_KM) * (180.0 / math.pi)
    lon_delta = lat_delta / math.cos(lat_radians) if abs(math.cos(lat_radians)) > 1e-12 else 180.0

    min_lat = lat - lat_delta
    max_lat = lat + lat_delta
    min_lon = lon - lon_delta
    max_lon = lon + lon_delta

    min_lat = max(-90.0, min_lat)
    max_lat = min(90.0, max_lat)
    min_lon = max(-180.0, min_lon)
    max_lon = min(180.0, max_lon)

    return {
        "center": {"latitude": lat, "longitude": lon},
        "min_latitude": min_lat,
        "max_latitude": max_lat,
        "min_longitude": min_lon,
        "max_longitude": max_lon,
        "radius_km": radius,
    }
