from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)


class LandError(Exception):
    pass


class ProviderError(LandError):
    """Retryable provider failure (network, timeout, 5xx)."""
    pass


class PermanentError(LandError):
    """Non-retryable failure (4xx, invalid query)."""
    pass


class LandContextUnavailable(LandError):
    """Land context unavailable but core analysis can continue."""
    pass


def _validate_bbox(south: float, north: float, west: float, east: float) -> None:
    if not all(isinstance(v, (int, float)) for v in [south, north, west, east]):
        raise LandError("All bounding-box values must be numeric.")
    if south >= north:
        raise LandError("south must be smaller than north.")
    if west >= east:
        raise LandError("west must be smaller than east.")


def _build_overpass_query(south: float, west: float, north: float, east: float) -> str:
    return f"""
    [out:json][timeout:25];
    (
      way[water][bbox:{south},{west},{north},{east}];
      way[waterway][bbox:{south},{west},{north},{east}];
      way[natural=water][bbox:{south},{west},{north},{east}];
      way[highway][bbox:{south},{west},{north},{east}];
      node[building][bbox:{south},{west},{north},{east}];
      way[building][bbox:{south},{west},{north},{east}];
    );
    out center tags geom;
    """


def _fetch_from_provider(
    url: str,
    query: str,
    connect_timeout: float,
    read_timeout: float,
    provider_name: str,
) -> Dict[str, Any]:
    """Fetch from a single Overpass provider with retries.

    Raises ProviderError for retryable failures, PermanentError for non-retryable.
    """
    max_retries = settings.OVERPASS_RETRIES
    last_error: Optional[Exception] = None

    timeout = httpx.Timeout(
        connect=connect_timeout,
        read=read_timeout,
        write=read_timeout,
        pool=connect_timeout,
    )
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Accept": "application/json",
        "User-Agent": settings.OVERPASS_USER_AGENT,
    }

    for attempt in range(max_retries + 1):
        started = time.perf_counter()
        try:
            response = httpx.post(
                url,
                data={"data": query},
                timeout=timeout,
                headers=headers,
            )
            response.raise_for_status()
        except httpx.ConnectTimeout as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            last_error = exc
            logger.debug(
                "overpass_connect_timeout",
                extra={"provider": provider_name, "attempt": attempt + 1, "max_retries": max_retries, "elapsed_ms": round(elapsed_ms, 2)},
            )
            if attempt < max_retries:
                time.sleep(min(0.1 * (2 ** attempt), 0.5))
                continue
            continue
        except httpx.ReadTimeout as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            last_error = exc
            logger.warning(
                "overpass_read_timeout",
                extra={"provider": provider_name, "attempt": attempt + 1, "max_retries": max_retries, "elapsed_ms": round(elapsed_ms, 2)},
            )
            if attempt < max_retries:
                time.sleep(min(0.1 * (2 ** attempt), 0.5))
                continue
            continue
        except httpx.TimeoutException as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            last_error = exc
            logger.warning(
                "overpass_timeout",
                extra={"provider": provider_name, "attempt": attempt + 1, "max_retries": max_retries, "elapsed_ms": round(elapsed_ms, 2)},
            )
            if attempt < max_retries:
                time.sleep(min(0.1 * (2 ** attempt), 0.5))
                continue
            continue
        except httpx.NetworkError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            last_error = exc
            logger.debug(
                "overpass_network_error",
                extra={"provider": provider_name, "attempt": attempt + 1, "max_retries": max_retries, "elapsed_ms": round(elapsed_ms, 2)},
            )
            if attempt < max_retries:
                time.sleep(min(0.1 * (2 ** attempt), 0.5))
                continue
            continue
        except httpx.HTTPStatusError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            # 5xx server errors are retryable, 4xx are not
            if 500 <= exc.response.status_code < 600:
                last_error = exc
                logger.debug(
                    "overpass_server_error",
                    extra={"provider": provider_name, "attempt": attempt + 1, "status_code": exc.response.status_code, "elapsed_ms": round(elapsed_ms, 2)},
                )
                if attempt < max_retries:
                    time.sleep(min(0.1 * (2 ** attempt), 0.5))
                    continue
                continue
            elif exc.response.status_code in {408, 425, 429}:
                last_error = exc
                logger.warning(
                    "overpass_rate_limit_or_transient_client_error",
                    extra={"provider": provider_name, "status_code": exc.response.status_code, "elapsed_ms": round(elapsed_ms, 2)},
                )
                if attempt < max_retries:
                    time.sleep(min(0.5 * (2 ** attempt), 2.0))
                    continue
                continue
            else:
                # 4xx client errors are not retryable
                logger.warning(
                    "overpass_client_error",
                    extra={"provider": provider_name, "status_code": exc.response.status_code, "elapsed_ms": round(elapsed_ms, 2)},
                )
                raise PermanentError(
                    f"Overpass provider {provider_name} returned {exc.response.status_code}: {exc.response.text[:200]}"
                ) from exc
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            last_error = exc
            logger.debug(
                "overpass_error",
                extra={"provider": provider_name, "attempt": attempt + 1, "max_retries": max_retries, "elapsed_ms": round(elapsed_ms, 2), "error": type(exc).__name__},
            )
            if attempt < max_retries:
                time.sleep(min(0.1 * (2 ** attempt), 0.5))
                continue
            continue

        # Success
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "overpass_provider_success",
            extra={"provider": provider_name, "elapsed_ms": round(elapsed_ms, 2)},
        )
        return response.json()

    # All retries exhausted
    logger.warning(
        "overpass_provider_retries_exhausted",
        extra={"provider": provider_name, "max_retries": max_retries, "last_error": str(last_error) if last_error else "unknown"},
    )
    raise ProviderError(f"Overpass provider {provider_name} failed after {max_retries} retries") from last_error


def fetch_land_context(
    south: float,
    west: float,
    north: float,
    east: float,
    connect_timeout: float | None = None,
    read_timeout: float | None = None,
) -> Dict[str, Any]:
    """Fetch land context (water, roads, buildings) from OSM Overpass.

    Supports primary + fallback provider with bounded retries.
    Falls back only for retryable failures (network, timeout, 5xx).
    Returns empty land context on failure (never raises).
    """
    _validate_bbox(south, north, west, east)

    effective_connect_timeout = connect_timeout if connect_timeout is not None else settings.OVERPASS_CONNECT_TIMEOUT
    effective_read_timeout = read_timeout if read_timeout is not None else settings.OVERPASS_READ_TIMEOUT
    query = _build_overpass_query(south, west, north, east)

    primary_url = settings.OVERPASS_URL
    fallback_url = settings.OVERPASS_FALLBACK_URL

    providers = [
        ("primary", primary_url),
        ("fallback", fallback_url),
        ("secondary_fallback", settings.OVERPASS_SECONDARY_FALLBACK_URL),
    ]
    # Deduplicate providers
    seen = set()
    unique_providers = []
    for name, url in providers:
        if url not in seen:
            seen.add(url)
            unique_providers.append((name, url))

    def fetch_uncached() -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for provider_name, url in unique_providers:
            try:
                result = _fetch_from_provider(
                    url, query, effective_connect_timeout, effective_read_timeout, provider_name
                )
                water_bodies = []
                roads = []
                buildings = []

                for element in result.get("elements", []):
                    tags = element.get("tags", {})
                    if (
                        "water" in tags
                        or "waterway" in tags
                        or tags.get("natural") == "water"
                        or tags.get("landuse") in {"reservoir", "basin"}
                    ):
                        water_bodies.append(element)
                    elif "highway" in tags:
                        roads.append(element)
                    elif "building" in tags:
                        buildings.append(element)

                return {
                    "bbox": {"south": south, "west": west, "north": north, "east": east},
                    "water_bodies": water_bodies,
                    "roads": roads,
                    "buildings": buildings,
                    "source": f"OpenStreetMap Overpass API ({provider_name}: {url})",
                }
            except PermanentError as exc:
                logger.warning(
                    "overpass_permanent_error",
                    extra={"provider": provider_name, "error": str(exc)},
                )
                continue
            except ProviderError as exc:
                last_error = exc
                if provider_name != unique_providers[-1][0]:
                    next_provider = unique_providers[unique_providers.index((provider_name, url)) + 1][0]
                    logger.info(
                        "overpass_fallback_activated",
                        extra={"from_provider": provider_name, "to_provider": next_provider},
                    )
                continue

        logger.warning(
            "overpass_all_providers_failed",
            extra={"provider_count": len(unique_providers), "last_error": str(last_error) if last_error else "unknown"},
        )
        return {
            "bbox": {"south": south, "west": west, "north": north, "east": east},
            "water_bodies": [],
            "roads": [],
            "buildings": [],
            "source": "OpenStreetMap Overpass API (unavailable)",
            "warning": "Land context unavailable - all Overpass providers failed",
        }

    return fetch_uncached()