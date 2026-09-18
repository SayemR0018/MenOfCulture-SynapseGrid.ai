"""Unit tests for ml/normalizer.py numeric parsing: solar factor semantics,
kWh literals, and percentage-of-capacity resolution.
"""
from ml.normalizer import (
    parse_kwh_value,
    parse_percentage_of_capacity,
    parse_solar_factor,
)


def test_25_percent_of_forecast_is_remaining_fraction():
    assert parse_solar_factor("usable solar treated as roughly 25% of the forecast") == 0.25


def test_around_25_percent():
    assert parse_solar_factor("solar output will be around 25% of forecast") == 0.25


def test_80_percent_reduction_means_factor_0_2():
    assert parse_solar_factor("solar is reduced by 80%") == 0.2


def test_20_percent_reduction_means_factor_0_8():
    assert parse_solar_factor("expect a 20% reduction in output") == 0.8


def test_drop_to_20_percent_is_remaining():
    assert parse_solar_factor("Solar output will drop to about 20% from 1 PM to 3 PM.") == 0.2


def test_one_fifth_fraction_word():
    assert parse_solar_factor("roughly one-fifth of normal solar output") == 0.2


def test_kwh_literal():
    assert parse_kwh_value("keep at least 100 kWh in reserve") == 100.0


def test_max_grid_kwh_literal():
    assert parse_kwh_value("maximum grid draw of 50 kWh") == 50.0


def test_percentage_of_known_capacity():
    assert parse_percentage_of_capacity("keep at least 50% of the 200 kWh battery", 200) == 100.0


def test_percentage_of_capacity_without_capacity_returns_none():
    assert parse_percentage_of_capacity("keep at least 50% of the battery", None) is None


def test_no_numeric_expression_returns_none():
    assert parse_solar_factor("Do not charge the battery from 2 AM until 5 AM.") is None
