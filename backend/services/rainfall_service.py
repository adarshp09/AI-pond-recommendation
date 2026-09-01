from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()


class RainfallError(Exception):
    pass


def _validate_coordinates(latitude: float, longitude: float) -> None:
    if not isinstance(latitude, (int, float)) or not (-90 <= float(latitude) <= 90):
        raise RainfallError("latitude must be a numeric value between -90 and 90.")
    if not isinstance(longitude, (int, float)) or not (-180 <= float(longitude) <= 180):
        raise RainfallError("longitude must be a numeric value between -180 and 180.")


def fetch_historical_rainfall(
    latitude: float,
    longitude: float,
    start_date: Optional[str],
    end_date: Optional[str],
    timezone: str = "auto",
    timeout: float = 30.0,
) -> Dict[str, Any]:
    if start_date is None:
        raise RainfallError("start_date is required.")
    if end_date is None:
        raise RainfallError("end_date is required.")

    _validate_coordinates(latitude, longitude)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": timezone,
    }

    try:
        response = httpx.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
    except TimeoutError as exc:
        raise RainfallError(f"Unable to connect to Open-Meteo: timeout ({timeout}s)") from exc
    except httpx.HTTPError as exc:
        raise RainfallError(f"Unable to connect to Open-Meteo: {exc}") from exc

    payload = response.json()
    daily_times = payload.get("daily", {}).get("time", [])
    daily_values = payload.get("daily", {}).get("precipitation_sum", [])

    return {
        "latitude": float(payload.get("latitude", latitude)),
        "longitude": float(payload.get("longitude", longitude)),
        "timezone": payload.get("timezone", timezone),
        "daily_rainfall_mm": [float(v) if v is not None else 0.0 for v in daily_values],
        "daily_dates": list(daily_times),
        "total_precipitation_mm": float(sum(float(v) for v in daily_values if v is not None)),
        "source": "Open-Meteo Historical Weather API",
    }
