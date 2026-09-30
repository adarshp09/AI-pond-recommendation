from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from services.cache import FileCache, get_cache
from services.rainfall_service import RainfallError, fetch_historical_rainfall

try:
    from config import settings
except ImportError:
    settings = None


DEFAULT_RUNOFF_COEFFICIENT = 0.3
DEFAULT_SEASONAL_MONTHS = [6, 7, 8, 9]
MIN_CATCHMENT_AREA_M2 = 100.0
MIN_RAINFALL_MM = 1.0

ENV_RUNOFF_COEFFICIENT = "POND_RUNOFF_COEFFICIENT"
ENV_SEASONAL_MONTHS = "POND_SEASONAL_MONTHS"
ENV_RAINFALL_YEARS = "POND_RAINFALL_YEARS"
ENV_CACHE_TTL_S = "POND_RAINFALL_CACHE_TTL_S"
ENV_CACHE_ENABLED = "POND_RAINFALL_CACHE_ENABLED"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name, str(int(default))).lower()
    return val in {"1", "true", "yes", "on"}


def _env_list_int(name: str, default: List[int]) -> List[int]:
    val = os.getenv(name)
    if not val:
        return default
    try:
        return [int(x.strip()) for x in val.split(",") if x.strip()]
    except (TypeError, ValueError):
        return default


@dataclass
class RainfallRunoffConfig:
    runoff_coefficient: float = field(
        default_factory=lambda: _env_float(ENV_RUNOFF_COEFFICIENT, DEFAULT_RUNOFF_COEFFICIENT)
    )
    seasonal_months: List[int] = field(
        default_factory=lambda: _env_list_int(ENV_SEASONAL_MONTHS, DEFAULT_SEASONAL_MONTHS)
    )
    rainfall_years: int = field(
        default_factory=lambda: _env_int(ENV_RAINFALL_YEARS, 5)
    )
    cache_ttl_seconds: int = field(
        default_factory=lambda: _env_int(ENV_CACHE_TTL_S, 86400 * 30)
    )
    cache_enabled: bool = field(
        default_factory=lambda: _env_bool(ENV_CACHE_ENABLED, True)
    )

    def __post_init__(self):
        if not 0.0 < self.runoff_coefficient <= 1.0:
            raise ValueError("runoff_coefficient must be in (0, 1]")
        if not self.seasonal_months:
            raise ValueError("seasonal_months cannot be empty")
        if any(not (1 <= m <= 12) for m in self.seasonal_months):
            raise ValueError("seasonal_months must be in 1..12")
        if self.rainfall_years < 1:
            raise ValueError("rainfall_years must be >= 1")

    @classmethod
    def from_env(cls) -> "RainfallRunoffConfig":
        return cls()


@dataclass
class RainfallRunoffStats:
    provider_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    provider_errors: int = 0
    provider_timeouts: int = 0
    provider_ms_total: float = 0.0
    requests_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        lookups = self.cache_hits + self.cache_misses
        return {
            "provider_calls": int(self.provider_calls),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "cache_hit_rate": float(self.cache_hits) / lookups if lookups else None,
            "provider_errors": int(self.provider_errors),
            "provider_timeouts": int(self.provider_timeouts),
            "provider_ms_total": round(float(self.provider_ms_total), 3),
            "requests_total": int(self.requests_total),
        }


_STATS = RainfallRunoffStats()
_STATS_LOCK = None


def _get_stats_lock():
    global _STATS_LOCK
    if _STATS_LOCK is None:
        import threading
        _STATS_LOCK = threading.Lock()
    return _STATS_LOCK


def _record(**deltas: float) -> None:
    lock = _get_stats_lock()
    with lock:
        for key, value in deltas.items():
            setattr(_STATS, key, getattr(_STATS, key) + value)


def reset_stats() -> None:
    lock = _get_stats_lock()
    with lock:
        for key in RainfallRunoffStats.__dataclass_fields__:
            setattr(_STATS, key, 0.0 if key == "provider_ms_total" else 0)


def get_stats() -> Dict[str, Any]:
    lock = _get_stats_lock()
    with lock:
        return _STATS.to_dict()


def _cache_key(latitude: float, longitude: float, start_date: str, end_date: str) -> Dict[str, Any]:
    return {
        "provider": "open_meteo_rainfall",
        "latitude": round(float(latitude), 6),
        "longitude": round(float(longitude), 6),
        "start_date": start_date,
        "end_date": end_date,
    }


def _fetch_rainfall_with_retries(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    config: RainfallRunoffConfig,
    cache: Optional[FileCache],
    recorder: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    last_error: Optional[str] = None
    max_retries = 2

    for attempt in range(max_retries + 1):
        started = time.perf_counter()
        try:
            payload = fetch_historical_rainfall(
                latitude, longitude, start_date, end_date, timeout=30.0
            )
            _record(provider_calls=1, provider_ms_total=(time.perf_counter() - started) * 1000.0)
            return payload, None
        except RainfallError as exc:
            _record(provider_errors=1, provider_ms_total=(time.perf_counter() - started) * 1000.0)
            last_error = f"RainfallError: {exc}"
        except TimeoutError as exc:
            _record(provider_timeouts=1, provider_ms_total=(time.perf_counter() - started) * 1000.0)
            last_error = f"TimeoutError: {exc}"
        except Exception as exc:
            _record(provider_errors=1, provider_ms_total=(time.perf_counter() - started) * 1000.0)
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < max_retries:
            time.sleep(min(0.1 * (2 ** attempt), 0.5))

    return None, last_error or "unknown error"


def _aggregate_seasonal_rainfall(
    daily_rainfall_mm: List[float],
    daily_dates: List[str],
    seasonal_months: List[int],
) -> Dict[str, Any]:
    if not daily_rainfall_mm or not daily_dates or len(daily_rainfall_mm) != len(daily_dates):
        return {
            "seasonal_total_mm": None,
            "seasonal_months": seasonal_months,
            "daily_values_used": 0,
            "date_range": None,
        }

    seasonal_values = []
    seasonal_dates = []

    for value, date_str in zip(daily_rainfall_mm, daily_dates):
        try:
            year, month, _ = map(int, date_str.split("-"))
            if month in seasonal_months:
                seasonal_values.append(float(value) if value is not None else 0.0)
                seasonal_dates.append(date_str)
        except (ValueError, AttributeError):
            continue

    if not seasonal_values:
        return {
            "seasonal_total_mm": None,
            "seasonal_months": seasonal_months,
            "daily_values_used": 0,
            "date_range": None,
        }

    return {
        "seasonal_total_mm": float(sum(seasonal_values)),
        "seasonal_months": seasonal_months,
        "daily_values_used": len(seasonal_values),
        "date_range": {
            "start": seasonal_dates[0] if seasonal_dates else None,
            "end": seasonal_dates[-1] if seasonal_dates else None,
        },
    }


def _compute_runoff_and_storage(
    rainfall_mm: Optional[float],
    catchment_area_m2: float,
    runoff_coefficient: float,
    has_terrain_support: bool,
) -> Dict[str, Any]:
    if rainfall_mm is None or rainfall_mm < MIN_RAINFALL_MM:
        return {
            "runoff_coefficient": float(runoff_coefficient),
            "estimated_runoff_m3": None,
            "estimated_storage_m3": None,
            "water_availability_status": "insufficient_rainfall_data",
            "calculation_notes": "Rainfall data unavailable or below minimum threshold",
        }

    if catchment_area_m2 < MIN_CATCHMENT_AREA_M2:
        return {
            "runoff_coefficient": float(runoff_coefficient),
            "estimated_runoff_m3": None,
            "estimated_storage_m3": None,
            "water_availability_status": "catchment_too_small",
            "calculation_notes": f"Catchment area {catchment_area_m2:.1f} m² below minimum {MIN_CATCHMENT_AREA_M2} m²",
        }

    rainfall_m = rainfall_mm / 1000.0
    runoff_m3 = rainfall_m * catchment_area_m2 * runoff_coefficient

    storage_m3 = None
    if has_terrain_support and runoff_m3 > 0:
        storage_m3 = runoff_m3 * 0.5

    if runoff_m3 < 10.0:
        status = "very_low"
    elif runoff_m3 < 1000.0:
        status = "low"
    elif runoff_m3 < 10000.0:
        status = "moderate"
    elif runoff_m3 < 100000.0:
        status = "good"
    else:
        status = "high"

    return {
        "runoff_coefficient": float(runoff_coefficient),
        "estimated_runoff_m3": float(runoff_m3),
        "estimated_storage_m3": float(storage_m3) if storage_m3 is not None else None,
        "water_availability_status": status,
        "calculation_notes": (
            f"V = P × A × C = {rainfall_m:.4f} m × {catchment_area_m2:.1f} m² × {runoff_coefficient:.2f}"
        ),
    }


def compute_rainfall_runoff(
    latitude: float,
    longitude: float,
    catchment_area_m2: float,
    has_terrain_support: bool,
    *,
    config: Optional[RainfallRunoffConfig] = None,
    cache: Optional[FileCache] = None,
    recorder: Any = None,
) -> Dict[str, Any]:
    active_config = config or RainfallRunoffConfig.from_env()
    active_recorder = recorder
    active_cache = cache

    if active_cache is None and active_config.cache_enabled:
        try:
            active_cache = get_cache(ttl_seconds=active_config.cache_ttl_seconds, enabled=True)
        except Exception:
            active_cache = None

    _record(requests_total=1)

    end_year = time.gmtime().tm_year - 1
    start_year = end_year - active_config.rainfall_years + 1
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    cache_key = _cache_key(latitude, longitude, start_date, end_date)
    cached_rainfall = None

    if active_cache and active_config.cache_enabled:
        cached_rainfall = active_cache.get("rainfall_runoff", cache_key)
        if cached_rainfall is not None:
            _record(cache_hits=1)
        else:
            _record(cache_misses=1)

    rainfall_payload = cached_rainfall
    if rainfall_payload is None:
        if active_recorder and hasattr(active_recorder, "stage"):
            with active_recorder.stage("rainfall_fetch") as handle:
                rainfall_payload, error = _fetch_rainfall_with_retries(
                    latitude, longitude, start_date, end_date, active_config, active_cache, active_recorder
                )
                if error:
                    handle.warn(f"Rainfall fetch failed: {error}")
        else:
            rainfall_payload, error = _fetch_rainfall_with_retries(
                latitude, longitude, start_date, end_date, active_config, active_cache, active_recorder
            )

        if rainfall_payload is not None and active_cache and active_config.cache_enabled:
            active_cache.set("rainfall_runoff", cache_key, rainfall_payload)

    if rainfall_payload is None:
        return {
            "rainfall_mm": None,
            "catchment_area_m2": float(catchment_area_m2),
            "runoff_coefficient": float(active_config.runoff_coefficient),
            "estimated_runoff_m3": None,
            "estimated_storage_m3": None,
            "water_availability_status": "provider_unavailable",
            "data_quality": {
                "source": "Open-Meteo Historical Weather API",
                "date_range": {"start": start_date, "end": end_date},
                "seasonal_months": active_config.seasonal_months,
                "rainfall_years": active_config.rainfall_years,
                "status": "failed",
                "error": "Unable to fetch rainfall data from provider",
            },
        }

    daily_rainfall = rainfall_payload.get("daily_rainfall_mm", [])
    daily_dates = rainfall_payload.get("daily_dates", [])

    seasonal = _aggregate_seasonal_rainfall(daily_rainfall, daily_dates, active_config.seasonal_months)
    seasonal_total = seasonal.get("seasonal_total_mm")

    years_covered = active_config.rainfall_years
    if seasonal_total is not None and years_covered > 0:
        mean_annual_seasonal = seasonal_total / years_covered
    else:
        mean_annual_seasonal = None

    runoff_result = _compute_runoff_and_storage(
        mean_annual_seasonal, catchment_area_m2, active_config.runoff_coefficient, has_terrain_support
    )

    data_quality = {
        "source": rainfall_payload.get("source", "Open-Meteo Historical Weather API"),
        "date_range": {"start": start_date, "end": end_date},
        "seasonal_months": active_config.seasonal_months,
        "rainfall_years": active_config.rainfall_years,
        "daily_values_count": len(daily_rainfall),
        "seasonal_values_count": seasonal.get("daily_values_used", 0),
        "seasonal_date_range": seasonal.get("date_range"),
        "status": "success" if seasonal_total is not None else "partial",
    }
    if seasonal_total is None:
        data_quality["warning"] = "No seasonal rainfall data in the requested months"

    return {
        "rainfall_mm": float(seasonal_total) if seasonal_total is not None else None,
        "mean_annual_seasonal_rainfall_mm": float(mean_annual_seasonal) if mean_annual_seasonal is not None else None,
        "catchment_area_m2": float(catchment_area_m2),
        **runoff_result,
        "data_quality": data_quality,
    }


def compute_rainfall_runoff_for_candidates(
    candidates: List[Dict[str, Any]],
    *,
    config: Optional[RainfallRunoffConfig] = None,
    cache: Optional[FileCache] = None,
    recorder: Any = None,
) -> List[Dict[str, Any]]:
    active_config = config or RainfallRunoffConfig.from_env()
    active_recorder = recorder
    active_cache = cache

    if active_cache is None and active_config.cache_enabled:
        try:
            active_cache = get_cache(ttl_seconds=active_config.cache_ttl_seconds, enabled=True)
        except Exception:
            active_cache = None

    end_year = time.gmtime().tm_year - 1
    start_year = end_year - active_config.rainfall_years + 1
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    unique_locations: Dict[Tuple[float, float], List[int]] = {}
    for idx, candidate in enumerate(candidates):
        lat = candidate.get("latitude")
        lon = candidate.get("longitude")
        if lat is not None and lon is not None:
            key = (round(float(lat), 6), round(float(lon), 6))
            unique_locations.setdefault(key, []).append(idx)

    location_results: Dict[Tuple[float, float], Dict[str, Any]] = {}

    for (lat, lon), indices in unique_locations.items():
        catchment_areas = [candidates[i].get("catchment_area_m2", 0.0) for i in indices]
        max_catchment = max(catchment_areas) if catchment_areas else 0.0
        has_terrain = any(candidates[i].get("has_terrain_support", True) for i in indices)

        cache_key = _cache_key(lat, lon, start_date, end_date)
        cached_rainfall = None

        if active_cache and active_config.cache_enabled:
            cached_rainfall = active_cache.get("rainfall_runoff", cache_key)
            if cached_rainfall is not None:
                _record(cache_hits=1)
            else:
                _record(cache_misses=1)

        rainfall_payload = cached_rainfall
        if rainfall_payload is None:
            if active_recorder and hasattr(active_recorder, "stage"):
                with active_recorder.stage("rainfall_fetch") as handle:
                    rainfall_payload, error = _fetch_rainfall_with_retries(
                        lat, lon, start_date, end_date, active_config, active_cache, active_recorder
                    )
                    if error:
                        handle.warn(f"Rainfall fetch failed for ({lat}, {lon}): {error}")
            else:
                rainfall_payload, error = _fetch_rainfall_with_retries(
                    lat, lon, start_date, end_date, active_config, active_cache, active_recorder
                )

            if rainfall_payload is not None and active_cache and active_config.cache_enabled:
                active_cache.set("rainfall_runoff", cache_key, rainfall_payload)

        location_results[(lat, lon)] = {
            "rainfall_payload": rainfall_payload,
            "catchment_area_m2": max_catchment,
            "has_terrain_support": has_terrain,
        }

    results = []
    for idx, candidate in enumerate(candidates):
        lat = candidate.get("latitude")
        lon = candidate.get("longitude")
        catchment_area = candidate.get("catchment_area_m2", 0.0)
        has_terrain = candidate.get("has_terrain_support", True)

        if lat is None or lon is None:
            results.append({
                "rainfall_mm": None,
                "mean_annual_seasonal_rainfall_mm": None,
                "catchment_area_m2": float(catchment_area),
                "runoff_coefficient": float(active_config.runoff_coefficient),
                "estimated_runoff_m3": None,
                "estimated_storage_m3": None,
                "water_availability_status": "missing_coordinates",
                "data_quality": {
                    "source": "Open-Meteo Historical Weather API",
                    "date_range": {"start": start_date, "end": end_date},
                    "seasonal_months": active_config.seasonal_months,
                    "rainfall_years": active_config.rainfall_years,
                    "status": "failed",
                    "error": "Candidate missing latitude/longitude",
                },
            })
            continue

        key = (round(float(lat), 6), round(float(lon), 6))
        loc_result = location_results.get(key)

        if loc_result is None or loc_result["rainfall_payload"] is None:
            results.append({
                "rainfall_mm": None,
                "mean_annual_seasonal_rainfall_mm": None,
                "catchment_area_m2": float(catchment_area),
                "runoff_coefficient": float(active_config.runoff_coefficient),
                "estimated_runoff_m3": None,
                "estimated_storage_m3": None,
                "water_availability_status": "provider_unavailable",
                "data_quality": {
                    "source": "Open-Meteo Historical Weather API",
                    "date_range": {"start": start_date, "end": end_date},
                    "seasonal_months": active_config.seasonal_months,
                    "rainfall_years": active_config.rainfall_years,
                    "status": "failed",
                    "error": "Unable to fetch rainfall data from provider",
                },
            })
            continue

        payload = loc_result["rainfall_payload"]
        daily_rainfall = payload.get("daily_rainfall_mm", [])
        daily_dates = payload.get("daily_dates", [])

        seasonal = _aggregate_seasonal_rainfall(daily_rainfall, daily_dates, active_config.seasonal_months)
        seasonal_total = seasonal.get("seasonal_total_mm")

        years_covered = active_config.rainfall_years
        if seasonal_total is not None and years_covered > 0:
            mean_annual_seasonal = seasonal_total / years_covered
        else:
            mean_annual_seasonal = None

        runoff_result = _compute_runoff_and_storage(
            mean_annual_seasonal, catchment_area, active_config.runoff_coefficient, has_terrain
        )

        data_quality = {
            "source": payload.get("source", "Open-Meteo Historical Weather API"),
            "date_range": {"start": start_date, "end": end_date},
            "seasonal_months": active_config.seasonal_months,
            "rainfall_years": active_config.rainfall_years,
            "daily_values_count": len(daily_rainfall),
            "seasonal_values_count": seasonal.get("daily_values_used", 0),
            "seasonal_date_range": seasonal.get("date_range"),
            "status": "success" if seasonal_total is not None else "partial",
        }
        if seasonal_total is None:
            data_quality["warning"] = "No seasonal rainfall data in the requested months"

        results.append({
            "rainfall_mm": float(seasonal_total) if seasonal_total is not None else None,
            "mean_annual_seasonal_rainfall_mm": float(mean_annual_seasonal) if mean_annual_seasonal is not None else None,
            "catchment_area_m2": float(catchment_area),
            **runoff_result,
            "data_quality": data_quality,
        })

    return results


__all__ = [
    "RainfallRunoffConfig",
    "RainfallRunoffStats",
    "compute_rainfall_runoff",
    "compute_rainfall_runoff_for_candidates",
    "get_stats",
    "reset_stats",
    "DEFAULT_RUNOFF_COEFFICIENT",
    "DEFAULT_SEASONAL_MONTHS",
]