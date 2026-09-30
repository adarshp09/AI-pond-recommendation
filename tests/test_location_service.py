import pytest

from services.location_service import calculate_analysis_bounds


def test_valid_coordinates_radius():
    result = calculate_analysis_bounds(12.5, 78.1, 10)

    assert result["center"]["latitude"] == pytest.approx(12.5)
    assert result["center"]["longitude"] == pytest.approx(78.1)
    assert result["radius_km"] == pytest.approx(10)
    assert result["min_latitude"] <= 12.5 <= result["max_latitude"]
    assert result["min_longitude"] <= 78.1 <= result["max_longitude"]


def test_radius_affects_bounds():
    small = calculate_analysis_bounds(20.0, 80.0, 5)
    large = calculate_analysis_bounds(20.0, 80.0, 25)

    assert (large["max_latitude"] - large["min_latitude"]) > (small["max_latitude"] - small["min_latitude"])
    assert (large["max_longitude"] - large["min_longitude"]) > (small["max_longitude"] - small["min_longitude"])


def test_invalid_latitude():
    with pytest.raises(ValueError):
        calculate_analysis_bounds(91, 80, 5)

    with pytest.raises(ValueError):
        calculate_analysis_bounds(-91, 80, 5)


def test_invalid_longitude():
    with pytest.raises(ValueError):
        calculate_analysis_bounds(20, 181, 5)

    with pytest.raises(ValueError):
        calculate_analysis_bounds(20, -181, 5)


def test_zero_or_negative_radius():
    with pytest.raises(ValueError):
        calculate_analysis_bounds(20, 80, 0)

    with pytest.raises(ValueError):
        calculate_analysis_bounds(20, 80, -2)


def test_excessive_radius():
    with pytest.raises(ValueError):
        calculate_analysis_bounds(20, 80, 101)
