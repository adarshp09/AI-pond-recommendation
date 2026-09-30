from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import httpx
import numpy as np
from dotenv import load_dotenv

from config import settings

load_dotenv()

logger = logging.getLogger(__name__)


class OpenTopographyError(Exception):
    pass


class OpenTopographyRetryableError(OpenTopographyError):
    """Retryable failure (network, timeout, 5xx)."""
    pass


class OpenTopographyPermanentError(OpenTopographyError):
    """Non-retryable failure (4xx, invalid query)."""
    pass


class OpenTopographyService:
    BASE_URL = "https://portal.opentopography.org/API/globaldem"
    DEFAULT_DATASET = "SRTMGL1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        dataset: Optional[str] = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        max_retries: int | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENTOPOGRAPHY_API_KEY")
        self.dataset = dataset or os.getenv("OPENTOPOGRAPHY_DATASET", self.DEFAULT_DATASET)
        self.connect_timeout = connect_timeout if connect_timeout is not None else settings.OPENTOPOGRAPHY_CONNECT_TIMEOUT
        self.read_timeout = read_timeout if read_timeout is not None else settings.OPENTOPOGRAPHY_READ_TIMEOUT
        self.max_retries = max_retries if max_retries is not None else settings.OPENTOPOGRAPHY_MAX_RETRIES

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

        timeout = httpx.Timeout(
            connect=self.connect_timeout,
            read=self.read_timeout,
            write=self.read_timeout,
            pool=self.connect_timeout,
        )

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.get(self.BASE_URL, params=params)
                    response.raise_for_status()
            except httpx.ConnectTimeout as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                last_error = exc
                logger.warning(
                    "opentopography_connect_timeout",
                    extra={"attempt": attempt + 1, "max_retries": self.max_retries, "elapsed_ms": round(elapsed_ms, 2)},
                )
                if attempt < self.max_retries:
                    time.sleep(min(0.5 * (2 ** attempt), 2.0))
                    logger.info(
                        "opentopography_retry",
                        extra={"attempt": attempt + 1, "reason": "connect_timeout"},
                    )
                continue
            except httpx.ReadTimeout as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                last_error = exc
                logger.warning(
                    "opentopography_read_timeout",
                    extra={"attempt": attempt + 1, "max_retries": self.max_retries, "elapsed_ms": round(elapsed_ms, 2)},
                )
                if attempt < self.max_retries:
                    time.sleep(min(0.5 * (2 ** attempt), 2.0))
                    logger.info(
                        "opentopography_retry",
                        extra={"attempt": attempt + 1, "reason": "read_timeout"},
                    )
                continue
            except httpx.TimeoutException as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                last_error = exc
                logger.warning(
                    "opentopography_timeout",
                    extra={"attempt": attempt + 1, "max_retries": self.max_retries, "elapsed_ms": round(elapsed_ms, 2)},
                )
                if attempt < self.max_retries:
                    time.sleep(min(0.5 * (2 ** attempt), 2.0))
                    logger.info(
                        "opentopography_retry",
                        extra={"attempt": attempt + 1, "reason": "timeout"},
                    )
                continue
            except httpx.NetworkError as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                last_error = exc
                logger.warning(
                    "opentopography_network_error",
                    extra={"attempt": attempt + 1, "max_retries": self.max_retries, "elapsed_ms": round(elapsed_ms, 2)},
                )
                if attempt < self.max_retries:
                    time.sleep(min(0.5 * (2 ** attempt), 2.0))
                    logger.info(
                        "opentopography_retry",
                        extra={"attempt": attempt + 1, "reason": "network_error"},
                    )
                continue
            except httpx.HTTPStatusError as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                # 5xx server errors are retryable, 4xx are not
                if 500 <= exc.response.status_code < 600:
                    last_error = exc
                    logger.warning(
                        "opentopography_server_error",
                        extra={"attempt": attempt + 1, "status_code": exc.response.status_code, "elapsed_ms": round(elapsed_ms, 2)},
                    )
                    if attempt < self.max_retries:
                        time.sleep(min(0.5 * (2 ** attempt), 2.0))
                        logger.info(
                            "opentopography_retry",
                            extra={"attempt": attempt + 1, "reason": f"http_{exc.response.status_code}"},
                        )
                    continue
                else:
                    # 4xx client errors are not retryable
                    logger.error(
                        "opentopography_client_error",
                        extra={"status_code": exc.response.status_code, "elapsed_ms": round(elapsed_ms, 2)},
                    )
                    raise OpenTopographyPermanentError(
                        f"OpenTopography returned {exc.response.status_code}: {exc.response.text[:200]}"
                    ) from exc
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                last_error = exc
                logger.warning(
                    "opentopography_error",
                    extra={"attempt": attempt + 1, "max_retries": self.max_retries, "elapsed_ms": round(elapsed_ms, 2), "error": type(exc).__name__},
                )
                if attempt < self.max_retries:
                    time.sleep(min(0.5 * (2 ** attempt), 2.0))
                    logger.info(
                        "opentopography_retry",
                        extra={"attempt": attempt + 1, "reason": type(exc).__name__},
                    )
                continue

            # Success
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.info(
                "opentopography_success",
                extra={"elapsed_ms": round(elapsed_ms, 2)},
            )
            return self._parse_geotiff(response.content, south, north, west, east)

        # All retries exhausted
        logger.error(
            "opentopography_retries_exhausted",
            extra={"max_retries": self.max_retries, "last_error": str(last_error) if last_error else "unknown"},
        )
        raise OpenTopographyRetryableError(f"OpenTopography failed after {self.max_retries} retries") from last_error

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
