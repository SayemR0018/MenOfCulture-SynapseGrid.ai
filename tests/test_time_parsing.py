"""Unit tests for ml/normalizer.py time-range parsing.

These pin down the exact start-inclusive/end-exclusive semantics mandated
by the problem statement, independent of any LLM call.
"""
from ml.normalizer import parse_time_range


def test_noon_until_2pm():
    assert parse_time_range("from noon until 2 PM") == [12, 13]


def test_between_6pm_and_9pm():
    assert parse_time_range("between 6 PM and 9 PM") == [18, 19, 20]


def test_2am_to_5am():
    assert parse_time_range("2 AM to 5 AM") == [2, 3, 4]


def test_overnight_10pm_until_midnight():
    assert parse_time_range("overnight from 10 PM until midnight") == [22, 23]


def test_1pm_to_3pm_dash_form():
    assert parse_time_range("1 PM - 3 PM") == [13, 14]


def test_hh_mm_24h_form():
    assert parse_time_range("between 13:00 and 15:00") == [13, 14]


def test_single_meridiem_inferred_for_end():
    # "2 to 5 AM" -> both should resolve to AM.
    assert parse_time_range("from 2 to 5 AM") == [2, 3, 4]


def test_no_range_returns_none():
    assert parse_time_range("The cafeteria menu changes tomorrow.") is None


def test_ambiguous_same_start_end_returns_none():
    assert parse_time_range("from 5 PM until 5 PM") is None


def test_through_keyword():
    assert parse_time_range("from noon through 2 PM") == [12, 13]
