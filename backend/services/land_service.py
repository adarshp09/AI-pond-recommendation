from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config import settings

try:  # providers live next to this module
    from services.land_providers.local_osm import LocalOSMProvider
    from services.land_providers.base import (
        STATUS_AVAILABLE,
        STATUS_UNAVAILABLE,
        LandContextResult,
        ProviderUnavailable,
        bbox_dict,
    )
except ImportError:  # pragma: no cover - alternative import path
    from .land_providers.local_osm import LocalOSMProvider
    from .land_providers.base import (
        STATUS_AVAILABLE,
        STATUS_UNAVAILABLE,
        LandContextResult,
        ProviderUnavailable,
        bbox_dict,
    )

logger = logging.getLogger(__name__)

# Provider modes (Phase 03 selection policy).
#
# DEPRECATED FOR RUNTIME — the Overpass chain in this module is retained ONLY for
# offline extraction/benchmark tooling. The service runs local-only by default
# (`LAND_PROVIDER_MODE=local_only`, `OVERPASS_FALLBACK_ENABLED=false`), so Overpass
# is never queried during a request. Use `mode=MODE_OVERPASS_ONLY` explicitly from
# scripts (api_smoke_test.py / benchmark_land_providers.py) if you need it.
MODE_LOCAL_FIRST = "local_first"
MODE_LOCAL_ONLY = "local_only"
MODE_OVERPASS_ONLY = "overpass_only"
_VALID_MODES = {MODE_LOCAL_FIRST, MODE_LOCAL_ONLY, MODE_OVERPASS_ONLY}


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
            way[water]({south},{west},{north},{east});
            way[waterway]({south},{west},{north},{east});
            way[natural=water]({south},{west},{north},{east});
            way[highway]({south},{west},{north},{east});
            node[building]({south},{west},{north},{east});
            way[building]({south},{west},{north},{east});
    );
    out center tags geom;
    """


def _classify_elements(result: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    water_bodies: List[Dict[str, Any]] = []
    roads: List[Dict[str, Any]] = []
    buildings: List[Dict[str, Any]] = []

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
    return water_bodies, roads, buildings


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
            logger.debug(
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
            logger.debug(
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
    logger.debug(
        "overpass_provider_retries_exhausted",
        extra={"provider": provider_name, "max_retries": max_retries, "last_error": str(last_error) if last_error else "unknown"},
    )
    raise ProviderError(f"Overpass provider {provider_name} failed after {max_retries} retries") from last_error


def _dedupe_providers(providers: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen: set[str] = set()
    unique: List[Tuple[str, str]] = []
    for name, url in providers:
        if url not in seen:
            seen.add(url)
            unique.append((name, url))
    return unique


def _fetch_overpass(
    south: float,
    west: float,
    north: float,
    east: float,
    connect_timeout: float,
    read_timeout: float,
) -> LandContextResult:
    """Query the Overpass provider chain. Returns a normalised result.

    DEPRECATED FOR RUNTIME — only reached via ``mode="overpass_only"`` (offline
    tooling). With the default ``local_only`` + ``OVERPASS_FALLBACK_ENABLED=false``
    this is never called by the API.

    Never raises: an all-providers failure is returned as an ``unavailable`` result
    with an explicit warning (the core analysis continues).
    """

    query = _build_overpass_query(south, west, north, east)
    providers = _dedupe_providers([
        ("primary", settings.OVERPASS_URL),
        ("fallback", settings.OVERPASS_FALLBACK_URL),
        ("secondary_fallback", settings.OVERPASS_SECONDARY_FALLBACK_URL),
    ])

    bbox = bbox_dict(south, west, north, east)
    last_error: Optional[Exception] = None

    for index, (provider_name, url) in enumerate(providers):
        try:
            result = _fetch_from_provider(url, query, connect_timeout, read_timeout, provider_name)
        except PermanentError as exc:
            logger.warning("overpass_permanent_error", extra={"provider": provider_name, "error": str(exc)})
            continue
        except ProviderError as exc:
            last_error = exc
            if index + 1 < len(providers):
                logger.info(
                    "overpass_fallback_activated",
                    extra={"from_provider": provider_name, "to_provider": providers[index + 1][0]},
                )
            continue

        water_bodies, roads, buildings = _classify_elements(result)
        return LandContextResult(
            bbox=bbox,
            water_bodies=water_bodies,
            roads=roads,
            buildings=buildings,
            source=f"OpenStreetMap Overpass API ({provider_name}: {url})",
            provider="overpass",
            status=STATUS_AVAILABLE,
            feature_counts={
                "water_bodies": len(water_bodies),
                "roads": len(roads),
                "buildings": len(buildings),
            },
        )

    logger.debug(
        "overpass_all_providers_failed",
        extra={"provider_count": len(providers), "last_error": str(last_error) if last_error else "unknown"},
    )
    return LandContextResult(
        bbox=bbox,
        source="OpenStreetMap Overpass API (unavailable)",
        provider="overpass",
        status=STATUS_UNAVAILABLE,
        warning="Land context unavailable - all Overpass providers failed",
    )


def _resolve_mode(mode: str | None) -> str:
    # Default is local_only (Phase 03): runtime enrichment is served from the
    # local GeoPackage and never contacts Overpass unless explicitly requested.
    resolved = (mode or settings.LAND_PROVIDER_MODE or MODE_LOCAL_ONLY).strip().lower()
    return resolved if resolved in _VALID_MODES else MODE_LOCAL_ONLY


def _build_local_provider() -> Optional[LocalOSMProvider]:
    if not settings.LOCAL_OSM_ENABLED:
        return None
    try:
        return LocalOSMProvider(
            settings.LOCAL_OSM_GPKG_PATH,
            manifest_path=settings.LOCAL_OSM_MANIFEST_PATH,
            max_age_days=settings.LOCAL_OSM_MAX_AGE_DAYS,
            query_buffer_m=settings.LOCAL_OSM_QUERY_BUFFER_M,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("local_osm_init_failed", extra={"error": str(exc)})
        return None


def _unavailable_result(
    south: float,
    west: float,
    north: float,
    east: float,
    *,
    provider: str,
    source: str,
    reason: str,
    local_status: Optional[str] = None,
) -> LandContextResult:
    """Explicit degraded-data state — never a silent empty geographic result."""

    extra = {"local_status": local_status} if local_status is not None else {}
    return LandContextResult(
        bbox=bbox_dict(south, west, north, east),
        source=source,
        provider=provider,
        status=STATUS_UNAVAILABLE,
        warning=reason,
        feature_counts={},
        extra=extra,
    )


def fetch_land_context(
    south: float,
    west: float,
    north: float,
    east: float,
    connect_timeout: float | None = None,
    read_timeout: float | None = None,
    *,
    mode: str | None = None,
    local_provider: LocalOSMProvider | None = None,
) -> Dict[str, Any]:
    """Resolve land context (water, roads, buildings) for a WGS84 bbox.

    Selection (Phase 03):

    1. **Local first** — if a usable, fresh local dataset covers the query, use it and
       never contact Overpass. A *valid empty* local result is used as-is (no fallback).
    2. **Fallback** — if the local dataset is missing, unusable, stale, or outside
       coverage, follow the configured policy and (optionally) try the Overpass chain.
    3. **Degraded** — if nothing succeeds, return an explicit ``unavailable`` result with
       a warning rather than fabricating an empty geographic result.

    ``mode`` overrides ``settings.LAND_PROVIDER_MODE`` for a single call
    (``local_first`` | ``local_only`` | ``overpass_only``).

    Only an invalid bounding box raises (``LandError``); every provider failure is
    reported in the returned payload.
    """

    _validate_bbox(south, north, west, east)

    effective_connect_timeout = connect_timeout if connect_timeout is not None else settings.OVERPASS_CONNECT_TIMEOUT
    effective_read_timeout = read_timeout if read_timeout is not None else settings.OVERPASS_READ_TIMEOUT
    resolved_mode = _resolve_mode(mode)

    overpass_allowed = (
        resolved_mode in {MODE_LOCAL_FIRST, MODE_OVERPASS_ONLY}
        and settings.OVERPASS_FALLBACK_ENABLED
    )
    if resolved_mode == MODE_OVERPASS_ONLY:
        overpass_allowed = True  # explicit override ignores the fallback toggle

    local = local_provider if local_provider is not None else _build_local_provider()
    local_failure: Optional[ProviderUnavailable] = None

    if resolved_mode != MODE_OVERPASS_ONLY and local is not None:
        try:
            result = local.fetch(south, west, north, east)
        except ProviderUnavailable as exc:
            local_failure = exc
            logger.debug(
                "local_osm_unusable",
                extra={"provider": "local_osm", "status": exc.status, "reason": exc.message},
            )
        except LandError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            local_failure = ProviderUnavailable(STATUS_UNAVAILABLE, str(exc))
            logger.warning("local_osm_query_failed", extra={"error": str(exc)})
        else:
            logger.info(
                "land_context_provider_selected",
                extra={"provider": "local_osm", "status": result.status},
            )
            return result.to_payload()
    elif local is None:
        local_failure = ProviderUnavailable(STATUS_UNAVAILABLE, "local OSM provider disabled")

    if overpass_allowed:
        return _fetch_overpass(south, west, north, east, effective_connect_timeout, effective_read_timeout).to_payload()

    local_status = local_failure.status if local_failure is not None else None
    local_reason = local_failure.message if local_failure is not None else "unavailable"

    if resolved_mode == MODE_LOCAL_ONLY:
        return _unavailable_result(
            south, west, north, east,
            provider="local_osm",
            source="Local OSM GeoPackage (unavailable)",
            reason=f"Local OSM dataset unavailable ({local_status}: {local_reason}) and Overpass fallback is disabled (local_only).",
            local_status=local_status,
        ).to_payload()

    return _unavailable_result(
        south, west, north, east,
        provider="none",
        source="Land context (unavailable)",
        reason=f"No land provider available ({local_status}: {local_reason}) and Overpass fallback is disabled.",
        local_status=local_status,
    ).to_payload()
