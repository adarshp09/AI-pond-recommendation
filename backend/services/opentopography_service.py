from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class OpenTopographyError(Exception):
    pass


class OpenTopographyService:
    BASE_URL = "https://portal.opentopography.org/API/globaldem"
    DEFAULT_DATASET = "SRTMGL1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        dataset: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("OPENTOPOGRAPHY_API_KEY")
        self.dataset = dataset or os.getenv("OPENTOPOGRAPHY_DATASET", self.DEFAULT_DATASET)
        self.timeout = timeout

        if not self.api_key:
            raise OpenTopographyError(
                "API key is not configured. Add OPENTOPOGRAPHY_API_KEY to your .env file."
            )

    def get_dem(
        self,
        south: float,
        north: float,
        west: float,
        east: float,
    ) -> Dict[str, Any]:
        self._validate_bbox(south, north, west, east)

        params = {
            "demtype": self.dataset,
            "south": south,
            "north": north,
            "west": west,
            "east": east,
            "outputFormat": "GTiff",
            "API_Key": self.api_key,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(self.BASE_URL, params=params)
                response.raise_for_status()
        except TimeoutError as exc:
            raise OpenTopographyError(f"Unable to connect to OpenTopography: timeout ({self.timeout}s)") from exc
        except httpx.HTTPError as exc:
            raise OpenTopographyError(f"Unable to connect to OpenTopography: {exc}") from exc

        if not response.content:
            raise OpenTopographyError("OpenTopography returned an empty response.")

        return self._parse_geotiff(response.content, south, north, west, east)

    @staticmethod
    def _validate_bbox(south: float, north: float, west: float, east: float) -> None:
        for name, value in {"south": south, "north": north, "west": west, "east": east}.items():
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise OpenTopographyError(f"{name} must be a finite number.")

        if south >= north:
            raise OpenTopographyError("south must be smaller than north.")
        if west >= east:
            raise OpenTopographyError("west must be smaller than east.")
        if not -90 <= south <= 90:
            raise OpenTopographyError("south latitude must be between -90 and 90.")
        if not -90 <= north <= 90:
            raise OpenTopographyError("north latitude must be between -90 and 90.")
        if not -180 <= west <= 180:
            raise OpenTopographyError("west longitude must be between -180 and 180.")
        if not -180 <= east <= 180:
            raise OpenTopographyError("east longitude must be between -180 and 180.")

    def _parse_geotiff(
        self,
        content: bytes,
        south: float,
        north: float,
        west: float,
        east: float,
    ) -> Dict[str, Any]:
        try:
            import rasterio
        except ImportError as exc:  # pragma: no cover
            raise OpenTopographyError(
                "rasterio is required to read OpenTopography GeoTIFF responses."
            ) from exc

        try:
            with rasterio.io.MemoryFile(content) as memfile:
                with memfile.open() as dataset:
                    elevation = dataset.read(1).astype(np.float64)
                    transform = dataset.transform
                    nodata = dataset.nodata
                    if nodata is not None:
                        elevation[np.isclose(elevation, nodata)] = np.nan
                    valid = np.isfinite(elevation)
                    if not np.any(valid):
                        raise OpenTopographyError("OpenTopography DEM contains no valid elevation cells.")
                    return {
                        "elevation": elevation,
                        "rows": int(elevation.shape[0]),
                        "columns": int(elevation.shape[1]),
                        "south": float(south),
                        "north": float(north),
                        "west": float(west),
                        "east": float(east),
                        "dataset": self.dataset,
                        "source": "OpenTopography Global DEM API",
                        "crs": str(dataset.crs) if dataset.crs else None,
                        "transform": tuple(transform),
                        "nodata": float(nodata) if nodata is not None else None,
                        "elevation_min_m": float(np.nanmin(elevation)),
                        "elevation_max_m": float(np.nanmax(elevation)),
                        "elevation_mean_m": float(np.nanmean(elevation)),
                        "valid_cell_fraction": float(np.mean(valid)),
                    }
        except OpenTopographyError:
            raise
        except Exception as exc:  # pragma: no cover
            raise OpenTopographyError(f"Unable to parse DEM response: {exc}") from exc


def get_dem_for_bbox(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> Dict[str, Any]:
    service = OpenTopographyService()
    return service.get_dem(
        south=min_lat,
        north=max_lat,
        west=min_lon,
        east=max_lon,
    )


def fetch_opentopography_dem(
    south: float,
    north: float,
    west: float,
    east: float,
) -> Dict[str, Any]:
    return get_dem_for_bbox(
        min_lon=west,
        min_lat=south,
        max_lon=east,
        max_lat=north,
    )
