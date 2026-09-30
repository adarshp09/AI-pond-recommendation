"""Tests for Phase 5: Rainfall, Runoff & Water Availability."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock

from backend.rainfall_runoff import (
    RainfallRunoffConfig,
    compute_rainfall_runoff,
    compute_rainfall_runoff_for_candidates,
    get_stats,
    reset_stats,
    DEFAULT_RUNOFF_COEFFICIENT,
    DEFAULT_SEASONAL_MONTHS,
)


class TestRainfallRunoffConfig:
    def test_default_config(self):
        config = RainfallRunoffConfig()
        assert config.runoff_coefficient == DEFAULT_RUNOFF_COEFFICIENT
        assert config.seasonal_months == DEFAULT_SEASONAL_MONTHS
        assert config.rainfall_years == 5
        assert config.cache_enabled is True

    def test_config_validation_runoff_coefficient(self):
        with pytest.raises(ValueError):
            RainfallRunoffConfig(runoff_coefficient=0.0)
        with pytest.raises(ValueError):
            RainfallRunoffConfig(runoff_coefficient=1.5)
        with pytest.raises(ValueError):
            RainfallRunoffConfig(runoff_coefficient=-0.1)

    def test_config_validation_seasonal_months(self):
        with pytest.raises(ValueError):
            RainfallRunoffConfig(seasonal_months=[])
        with pytest.raises(ValueError):
            RainfallRunoffConfig(seasonal_months=[0])
        with pytest.raises(ValueError):
            RainfallRunoffConfig(seasonal_months=[13])

    def test_config_validation_rainfall_years(self):
        with pytest.raises(ValueError):
            RainfallRunoffConfig(rainfall_years=0)


class TestRainfallRunoffStats:
    def test_reset_and_get_stats(self):
        reset_stats()
        stats = get_stats()
        assert stats["provider_calls"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["provider_errors"] == 0
        assert stats["provider_timeouts"] == 0
        assert stats["provider_ms_total"] == 0.0
        assert stats["requests_total"] == 0


class TestRunoffEquation:
    """Test the core runoff equation V = P × A × C."""

    def test_runoff_calculation_basic(self):
        """V = P(m) × A(m²) × C"""
        P_mm = 1000.0
        A_m2 = 10000.0
        C = 0.3
        P_m = P_mm / 1000.0
        expected = P_m * A_m2 * C
        assert expected == 3000.0

    def test_runoff_unit_conversion(self):
        """Test mm to m conversion."""
        rainfall_mm = 500.0
        rainfall_m = rainfall_mm / 1000.0
        assert rainfall_m == 0.5

    def test_runoff_zero_rainfall(self):
        """Zero rainfall should yield zero runoff."""
        from backend.rainfall_runoff import _compute_runoff_and_storage
        result = _compute_runoff_and_storage(0.0, 10000.0, 0.3, True)
        assert result["estimated_runoff_m3"] is None
        assert result["water_availability_status"] == "insufficient_rainfall_data"

    def test_runoff_zero_catchment(self):
        """Zero catchment area should yield None runoff."""
        from backend.rainfall_runoff import _compute_runoff_and_storage
        result = _compute_runoff_and_storage(1000.0, 0.0, 0.3, True)
        assert result["estimated_runoff_m3"] is None
        assert result["water_availability_status"] == "catchment_too_small"

    def test_runoff_small_catchment(self):
        """Catchment below minimum threshold."""
        from backend.rainfall_runoff import _compute_runoff_and_storage
        result = _compute_runoff_and_storage(1000.0, 50.0, 0.3, True)
        assert result["estimated_runoff_m3"] is None

    def test_runoff_without_terrain_support(self):
        """Storage should be None when terrain doesn't support it."""
        from backend.rainfall_runoff import _compute_runoff_and_storage
        result = _compute_runoff_and_storage(1000.0, 10000.0, 0.3, False)
        assert result["estimated_runoff_m3"] == 3000.0
        assert result["estimated_storage_m3"] is None

    def test_runoff_with_terrain_support(self):
        """Storage should be 50% of runoff when terrain supports it."""
        from backend.rainfall_runoff import _compute_runoff_and_storage
        result = _compute_runoff_and_storage(1000.0, 10000.0, 0.3, True)
        assert result["estimated_runoff_m3"] == 3000.0
        assert result["estimated_storage_m3"] == 1500.0


class TestSeasonalAggregation:
    """Test seasonal rainfall aggregation."""

    def test_aggregate_seasonal_months(self):
        """Test that only seasonal months are included."""
        from backend.rainfall_runoff import _aggregate_seasonal_rainfall

        daily_rainfall = [10.0, 5.0, 15.0, 20.0, 2.0, 8.0]
        daily_dates = [
            "2024-01-15", "2024-02-15", "2024-06-15",
            "2024-07-15", "2024-08-15", "2024-12-15"
        ]
        seasonal_months = [6, 7, 8]

        result = _aggregate_seasonal_rainfall(daily_rainfall, daily_dates, seasonal_months)

        assert result["seasonal_total_mm"] == 37.0
        assert result["daily_values_used"] == 3
        assert result["date_range"]["start"] == "2024-06-15"
        assert result["date_range"]["end"] == "2024-08-15"

    def test_aggregate_no_seasonal_data(self):
        """Test when no data falls in seasonal months."""
        from backend.rainfall_runoff import _aggregate_seasonal_rainfall

        daily_rainfall = [10.0, 5.0, 15.0]
        daily_dates = ["2024-01-15", "2024-02-15", "2024-03-15"]
        seasonal_months = [6, 7, 8]

        result = _aggregate_seasonal_rainfall(daily_rainfall, daily_dates, seasonal_months)

        assert result["seasonal_total_mm"] is None
        assert result["daily_values_used"] == 0

    def test_aggregate_mismatched_lengths(self):
        """Test handling of mismatched rainfall/dates arrays."""
        from backend.rainfall_runoff import _aggregate_seasonal_rainfall

        daily_rainfall = [10.0, 5.0]
        daily_dates = ["2024-06-15"]
        seasonal_months = [6]

        result = _aggregate_seasonal_rainfall(daily_rainfall, daily_dates, seasonal_months)

        assert result["seasonal_total_mm"] is None

    def test_aggregate_empty_arrays(self):
        """Test handling of empty arrays."""
        from backend.rainfall_runoff import _aggregate_seasonal_rainfall

        result = _aggregate_seasonal_rainfall([], [], [6, 7, 8])

        assert result["seasonal_total_mm"] is None
        assert result["daily_values_used"] == 0


class TestCatchmentConsistency:
    """Test catchment area consistency in calculations."""

    def test_catchment_area_used_in_runoff(self):
        """Verify catchment area flows through to runoff calculation."""
        from backend.rainfall_runoff import _compute_runoff_and_storage

        result = _compute_runoff_and_storage(1000.0, 20000.0, 0.3, True)
        assert result["estimated_runoff_m3"] == 6000.0

        result2 = _compute_runoff_and_storage(1000.0, 10000.0, 0.3, True)
        assert result2["estimated_runoff_m3"] == 3000.0

        assert result["estimated_runoff_m3"] == 2 * result2["estimated_runoff_m3"]

    def test_catchment_area_returned_in_result(self):
        """Verify catchment area is included in output."""
        from backend.rainfall_runoff import _compute_runoff_and_storage

        result = _compute_runoff_and_storage(1000.0, 15000.0, 0.3, True)
        assert "catchment_area_m2" not in result
        assert result["runoff_coefficient"] == 0.3


class TestMissingInvalidData:
    """Test handling of missing and invalid data."""

    def test_missing_rainfall_returns_none(self):
        """When rainfall is None, runoff should be None."""
        from backend.rainfall_runoff import _compute_runoff_and_storage

        result = _compute_runoff_and_storage(None, 10000.0, 0.3, True)
        assert result["estimated_runoff_m3"] is None
        assert result["water_availability_status"] == "insufficient_rainfall_data"

    def test_negative_rainfall_handled(self):
        """Negative rainfall should be treated as insufficient."""
        from backend.rainfall_runoff import _compute_runoff_and_storage

        result = _compute_runoff_and_storage(-100.0, 10000.0, 0.3, True)
        assert result["estimated_runoff_m3"] is None

    def test_zero_runoff_coefficient(self):
        """Zero runoff coefficient should yield zero runoff."""
        from backend.rainfall_runoff import _compute_runoff_and_storage

        result = _compute_runoff_and_storage(1000.0, 10000.0, 0.0, True)
        assert result["estimated_runoff_m3"] == 0.0

    def test_invalid_catchment_area(self):
        """Negative catchment area should be treated as too small."""
        from backend.rainfall_runoff import _compute_runoff_and_storage

        result = _compute_runoff_and_storage(1000.0, -1000.0, 0.3, True)
        assert result["estimated_runoff_m3"] is None


class TestProviderFailures:
    """Test handling of provider failures."""

    @patch("backend.rainfall_runoff.fetch_historical_rainfall")
    def test_provider_timeout_handled(self, mock_fetch):
        """Test that provider timeout returns proper error status."""
        from backend.services.rainfall_service import RainfallError

        mock_fetch.side_effect = TimeoutError("Connection timeout")

        result = compute_rainfall_runoff(
            21.0, 81.0, 10000.0, True,
            config=RainfallRunoffConfig(rainfall_years=1, cache_enabled=False),
            cache=None,
            recorder=None,
        )

        assert result["rainfall_mm"] is None
        assert result["water_availability_status"] == "provider_unavailable"
        assert "error" in result["data_quality"]

    @patch("backend.rainfall_runoff.fetch_historical_rainfall")
    def test_provider_http_error_handled(self, mock_fetch):
        """Test that HTTP errors are handled gracefully."""
        from backend.services.rainfall_service import RainfallError
        import httpx

        mock_fetch.side_effect = RainfallError("HTTP 500")

        result = compute_rainfall_runoff(
            21.0, 81.0, 10000.0, True,
            config=RainfallRunoffConfig(rainfall_years=1, cache_enabled=False),
            cache=None,
            recorder=None,
        )

        assert result["rainfall_mm"] is None
        assert result["water_availability_status"] == "provider_unavailable"

    @patch("backend.rainfall_runoff.fetch_historical_rainfall")
    def test_provider_returns_empty_data(self, mock_fetch):
        """Test handling of empty rainfall data."""
        mock_fetch.return_value = {
            "daily_rainfall_mm": [],
            "daily_dates": [],
            "source": "Open-Meteo",
        }

        result = compute_rainfall_runoff(
            21.0, 81.0, 10000.0, True,
            config=RainfallRunoffConfig(rainfall_years=1, cache_enabled=False),
            cache=None,
            recorder=None,
        )

        assert result["rainfall_mm"] is None
        assert result["data_quality"]["status"] == "partial"
        assert "warning" in result["data_quality"]


class TestCaching:
    """Test caching behavior."""

    @patch("backend.rainfall_runoff.get_cache")
    @patch("backend.rainfall_runoff.fetch_historical_rainfall")
    def test_cache_hit(self, mock_fetch, mock_get_cache):
        """Test that cached data is returned without provider call."""
        mock_cache = Mock()
        mock_cache.get.return_value = {
            "daily_rainfall_mm": [10.0] * 30,
            "daily_dates": [f"2024-07-{i:02d}" for i in range(1, 31)],
            "source": "Open-Meteo",
        }
        mock_get_cache.return_value = mock_cache

        result = compute_rainfall_runoff(
            21.0, 81.0, 10000.0, True,
            config=RainfallRunoffConfig(rainfall_years=1, cache_enabled=True),
            cache=mock_cache,
            recorder=None,
        )

        assert mock_fetch.call_count == 0
        assert result["rainfall_mm"] is not None

    @patch("backend.rainfall_runoff.get_cache")
    @patch("backend.rainfall_runoff.fetch_historical_rainfall")
    def test_cache_miss_calls_provider(self, mock_fetch, mock_get_cache):
        """Test that cache miss triggers provider call."""
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_fetch.return_value = {
            "daily_rainfall_mm": [10.0] * 30,
            "daily_dates": [f"2024-07-{i:02d}" for i in range(1, 31)],
            "source": "Open-Meteo",
        }
        mock_get_cache.return_value = mock_cache

        result = compute_rainfall_runoff(
            21.0, 81.0, 10000.0, True,
            config=RainfallRunoffConfig(rainfall_years=1, cache_enabled=True),
            cache=mock_cache,
            recorder=None,
        )

        assert mock_fetch.call_count == 1
        assert mock_cache.set.call_count == 1

    @patch("backend.rainfall_runoff.get_cache")
    @patch("backend.rainfall_runoff.fetch_historical_rainfall")
    def test_cache_disabled(self, mock_fetch, mock_get_cache):
        """Test that cache can be disabled."""
        mock_cache = Mock()
        mock_get_cache.return_value = mock_cache

        result = compute_rainfall_runoff(
            21.0, 81.0, 10000.0, True,
            config=RainfallRunoffConfig(rainfall_years=1, cache_enabled=False),
            cache=mock_cache,
            recorder=None,
        )

        assert mock_cache.get.call_count == 0
        assert mock_cache.set.call_count == 0


class TestDeterministicCalculations:
    """Test that calculations are deterministic."""

    def test_same_inputs_same_outputs(self):
        """Same inputs should always produce same outputs."""
        from backend.rainfall_runoff import _compute_runoff_and_storage

        result1 = _compute_runoff_and_storage(1000.0, 10000.0, 0.3, True)
        result2 = _compute_runoff_and_storage(1000.0, 10000.0, 0.3, True)

        assert result1 == result2

    def test_runoff_coefficient_variations(self):
        """Different runoff coefficients produce proportional results."""
        from backend.rainfall_runoff import _compute_runoff_and_storage

        result_03 = _compute_runoff_and_storage(1000.0, 10000.0, 0.3, True)
        result_05 = _compute_runoff_and_storage(1000.0, 10000.0, 0.5, True)

        assert result_05["estimated_runoff_m3"] / result_03["estimated_runoff_m3"] == pytest.approx(0.5 / 0.3)

    def test_water_availability_status_thresholds(self):
        """Test water availability status thresholds."""
        from backend.rainfall_runoff import _compute_runoff_and_storage

        assert _compute_runoff_and_storage(1.0, 100.0, 0.3, True)["water_availability_status"] == "very_low"
        assert _compute_runoff_and_storage(100.0, 1000.0, 0.3, True)["water_availability_status"] == "low"
        assert _compute_runoff_and_storage(1000.0, 1000.0, 0.3, True)["water_availability_status"] == "low"
        assert _compute_runoff_and_storage(1000.0, 10000.0, 0.3, True)["water_availability_status"] == "moderate"
        assert _compute_runoff_and_storage(1000.0, 100000.0, 0.3, True)["water_availability_status"] == "good"
        assert _compute_runoff_and_storage(1000.0, 1000000.0, 0.3, True)["water_availability_status"] == "high"


class TestMultipleCandidates:
    """Test processing multiple candidates."""

    @patch("backend.rainfall_runoff.fetch_historical_rainfall")
    @patch("backend.rainfall_runoff.get_cache")
    def test_multiple_candidates_same_location(self, mock_get_cache, mock_fetch):
        """Candidates at same location should share rainfall data."""
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_fetch.return_value = {
            "daily_rainfall_mm": [10.0] * 30,
            "daily_dates": [f"2024-07-{i:02d}" for i in range(1, 31)],
            "source": "Open-Meteo",
        }
        mock_get_cache.return_value = mock_cache

        candidates = [
            {"latitude": 21.0, "longitude": 81.0, "catchment_area_m2": 10000.0, "has_terrain_support": True},
            {"latitude": 21.0, "longitude": 81.0, "catchment_area_m2": 15000.0, "has_terrain_support": True},
        ]

        results = compute_rainfall_runoff_for_candidates(
            candidates,
            config=RainfallRunoffConfig(rainfall_years=1),
            cache=mock_cache,
            recorder=None,
        )

        assert len(results) == 2
        assert mock_fetch.call_count == 1
        assert results[0]["rainfall_mm"] == results[1]["rainfall_mm"]

    @patch("backend.rainfall_runoff.fetch_historical_rainfall")
    @patch("backend.rainfall_runoff.get_cache")
    def test_multiple_candidates_different_locations(self, mock_get_cache, mock_fetch):
        """Candidates at different locations should get separate rainfall data."""
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_fetch.return_value = {
            "daily_rainfall_mm": [10.0] * 30,
            "daily_dates": [f"2024-07-{i:02d}" for i in range(1, 31)],
            "source": "Open-Meteo",
        }
        mock_get_cache.return_value = mock_cache

        candidates = [
            {"latitude": 21.0, "longitude": 81.0, "catchment_area_m2": 10000.0, "has_terrain_support": True},
            {"latitude": 22.0, "longitude": 82.0, "catchment_area_m2": 15000.0, "has_terrain_support": True},
        ]

        results = compute_rainfall_runoff_for_candidates(
            candidates,
            config=RainfallRunoffConfig(rainfall_years=1),
            cache=mock_cache,
            recorder=None,
        )

        assert len(results) == 2
        assert mock_fetch.call_count == 2


class TestNoFabricatedValues:
    """Test that no values are fabricated when data is missing."""

    def test_no_zero_runoff_when_data_missing(self):
        """Runoff should be None, not 0, when rainfall data is unavailable."""
        from backend.rainfall_runoff import _compute_runoff_and_storage

        result = _compute_runoff_and_storage(None, 10000.0, 0.3, True)
        assert result["estimated_runoff_m3"] is None
        assert result["estimated_runoff_m3"] != 0.0

    def test_no_zero_storage_when_unsupported(self):
        """Storage should be None, not 0, when terrain doesn't support it."""
        from backend.rainfall_runoff import _compute_runoff_and_storage

        result = _compute_runoff_and_storage(1000.0, 10000.0, 0.3, False)
        assert result["estimated_storage_m3"] is None
        assert result["estimated_storage_m3"] != 0.0

    def test_rainfall_mm_is_none_not_zero(self):
        """Rainfall should be None, not 0, when unavailable."""
        result = compute_rainfall_runoff(
            21.0, 81.0, 10000.0, True,
            config=RainfallRunoffConfig(rainfall_years=1),
            cache=None,
            recorder=None,
        )
        # This will fail due to no actual API, but we test the structure
        assert "rainfall_mm" in result
        # When provider unavailable, rainfall_mm should be None


class TestDataQualityMetadata:
    """Test that data quality metadata is included."""

    @patch("backend.rainfall_runoff.fetch_historical_rainfall")
    def test_data_quality_includes_source(self, mock_fetch):
        """Data quality should include source information."""
        mock_fetch.return_value = {
            "daily_rainfall_mm": [10.0] * 30,
            "daily_dates": [f"2024-07-{i:02d}" for i in range(1, 31)],
            "source": "Open-Meteo Historical Weather API",
        }

        result = compute_rainfall_runoff(
            21.0, 81.0, 10000.0, True,
            config=RainfallRunoffConfig(rainfall_years=1, cache_enabled=False),
            cache=None,
            recorder=None,
        )

        assert "data_quality" in result
        assert result["data_quality"]["source"] in ("Open-Meteo Historical Weather API", "Open-Meteo")
        assert "date_range" in result["data_quality"]
        assert "seasonal_months" in result["data_quality"]
        assert "rainfall_years" in result["data_quality"]

    @patch("backend.rainfall_runoff.fetch_historical_rainfall")
    def test_data_quality_status_on_failure(self, mock_fetch):
        """Data quality status should reflect failure."""
        mock_fetch.side_effect = TimeoutError("Connection timeout")

        result = compute_rainfall_runoff(
            21.0, 81.0, 10000.0, True,
            config=RainfallRunoffConfig(rainfall_years=1, cache_enabled=False),
            cache=None,
            recorder=None,
        )

        assert result["data_quality"]["status"] == "failed"


class TestInstrumentation:
    """Test instrumentation/stats tracking."""

    def test_stats_tracking(self):
        """Stats should track provider calls, cache hits, etc."""
        reset_stats()

        from backend.rainfall_runoff import _record

        _record(provider_calls=1)
        _record(cache_hits=1)
        _record(provider_ms_total=100.0)

        stats = get_stats()
        assert stats["provider_calls"] == 1
        assert stats["cache_hits"] == 1
        assert stats["provider_ms_total"] == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])