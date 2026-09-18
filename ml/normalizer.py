"""Deterministic natural-language time & numeric parsing helpers.

These are NOT used to override the LLM's interpretation -- per the project
architecture, the LLM is the interpreter and deterministic code is only the
validator/authority on structure and bounds. This module exists for two
reasons:

1. It backs the offline ``mock`` LLM provider (see ``ml/providers.py``) so
   the whole pipeline is runnable and testable without any API key.
2. It is unit-tested directly (``tests/test_time_parsing.py`` and
   ``tests/test_numeric_parsing.py``) to pin down the exact semantics
   (start-inclusive/end-exclusive hours, "remaining fraction" percentages)
   that the LLM prompt also documents, so the prompt and the tests never
   drift apart.

Nothing in this file is ever trusted directly by the optimizer -- its
output still passes through ``ml/validator.py`` like any other candidate
directive.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_FRACTION_WORDS = {
    "a quarter": 0.25,
    "one quarter": 0.25,
    "one-quarter": 0.25,
    "quarter": 0.25,
    "a third": 1 / 3,
    "one third": 1 / 3,
    "half": 0.5,
    "a half": 0.5,
    "one-fifth": 0.2,
    "one fifth": 0.2,
    "three quarters": 0.75,
    "three-quarters": 0.75,
}

_HOUR_TOKEN = r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?|noon|midnight|[a-z]+)"


@dataclass(frozen=True)
class _TokenHour:
    hour24: int | None
    meridiem: str | None  # 'am' / 'pm' / None
    resolved: bool  # True if hour24 is already an unambiguous 24h value


def _parse_token(token: str) -> _TokenHour:
    token = token.strip().lower()
    if token == "noon":
        return _TokenHour(12, "pm", True)
    if token == "midnight":
        return _TokenHour(0, "am", True)

    m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", token)
    if not m:
        return _TokenHour(None, None, False)

    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    meridiem = m.group(3)
    if minute != 0:
        return _TokenHour(None, None, False)

    if meridiem is not None:
        base = 0 if hour == 12 else hour
        hour24 = base + (12 if meridiem == "pm" else 0)
        if not (0 <= hour24 <= 23):
            return _TokenHour(None, None, False)
        return _TokenHour(hour24, meridiem, True)

    # No explicit am/pm. Hours 13-23 and 0 are unambiguous 24h notation.
    if hour == 0 or 13 <= hour <= 23:
        return _TokenHour(hour, None, True)
    if 1 <= hour <= 12:
        # Ambiguous 12-hour value (e.g. bare "2") -- store the 1-12
        # representation and let the range resolver infer meridiem from
        # the companion token, or fail safely if it cannot.
        return _TokenHour(hour, None, False)

    return _TokenHour(None, None, False)


def _resolve_pair(start: _TokenHour, end: _TokenHour) -> tuple[int, int] | None:
    s_hour, s_mer, s_resolved = start.hour24, start.meridiem, start.resolved
    e_hour, e_mer, e_resolved = end.hour24, end.meridiem, end.resolved

    if s_hour is None or e_hour is None:
        return None

    if not s_resolved and e_mer is not None:
        s_hour = (s_hour % 12) + (12 if e_mer == "pm" else 0)
        s_resolved = True
    if not e_resolved and s_mer is not None:
        e_hour = (e_hour % 12) + (12 if s_mer == "pm" else 0)
        e_resolved = True

    if not s_resolved or not e_resolved:
        return None  # genuinely ambiguous -- do not guess

    return s_hour, e_hour


@dataclass(frozen=True)
class TimeRange:
    start_hour: int
    end_hour_exclusive: int

    def to_hours(self) -> list[int]:
        if self.end_hour_exclusive > self.start_hour:
            return list(range(self.start_hour, self.end_hour_exclusive))
        # Overnight wrap, e.g. "10 PM until 2 AM" -> [22, 23, 0, 1]
        return list(range(self.start_hour, 24)) + list(range(0, self.end_hour_exclusive))


_RANGE_PATTERNS = [
    re.compile(
        r"from\s+" + _HOUR_TOKEN + r"\s+(?:until|to|through|till)\s+" + _HOUR_TOKEN,
        re.IGNORECASE,
    ),
    re.compile(
        r"between\s+" + _HOUR_TOKEN + r"\s+and\s+" + _HOUR_TOKEN,
        re.IGNORECASE,
    ),
    re.compile(
        _HOUR_TOKEN + r"\s*(?:-|to|until|through|till)\s*" + _HOUR_TOKEN,
        re.IGNORECASE,
    ),
]


def parse_time_range(text: str) -> list[int] | None:
    """Parse a single natural-language time window into start-inclusive,
    end-exclusive integer hours (0-23).

    Returns ``None`` if the expression cannot be safely resolved -- callers
    must treat that as "do not guess" per the problem statement's ambiguity
    rule, not fall back to a default.
    """
    text = text.strip()

    for pattern in _RANGE_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        raw_start, raw_end = m.group(1), m.group(2)

        resolved = _resolve_pair(_parse_token(raw_start), _parse_token(raw_end))
        if resolved is None:
            continue
        start, end = resolved

        if start == end:
            continue

        return TimeRange(start, end).to_hours()

    return None


_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", re.IGNORECASE)
_KWH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*kwh", re.IGNORECASE)

# Patterns that unambiguously mean "the percentage is how much was CUT",
# as opposed to "the percentage is how much remains". Deliberately narrow --
# e.g. "drop to 20%" is remaining, but "drop by 80%" / "an 80% drop" /
# "80% reduction" is a cut amount.
_REDUCTION_PATTERNS = [
    re.compile(r"reduc\w*\s+by\s+\d", re.IGNORECASE),
    re.compile(r"\bcut\s+(?:by\s+)?\d", re.IGNORECASE),
    re.compile(r"\bdown\s+by\s+\d", re.IGNORECASE),
    re.compile(r"\bdecreas\w*\s+by\s+\d", re.IGNORECASE),
    re.compile(r"\bdrop\w*\s+by\s+\d", re.IGNORECASE),
    re.compile(r"\blose\s+\d", re.IGNORECASE),
    re.compile(r"\d+(?:\.\d+)?\s*%\s*(?:reduction|drop|decrease|cut)\b", re.IGNORECASE),
    re.compile(r"\ban?\s+\d+(?:\.\d+)?\s*%\s*(?:reduction|drop|decrease|cut)\b", re.IGNORECASE),
]


def parse_solar_factor(text: str) -> float | None:
    """Return the remaining-fraction ``factor`` in [0, 1] implied by text
    such as "25% of forecast" (-> 0.25) or "reduced by 80%" (-> 0.2).

    Returns ``None`` if no percentage/fraction expression is found.
    """
    lowered = text.lower()

    is_reduction = any(p.search(lowered) for p in _REDUCTION_PATTERNS)

    m = _PERCENT_RE.search(lowered)
    if m:
        pct = float(m.group(1)) / 100.0
        if is_reduction:
            return round(1.0 - pct, 6)
        return round(pct, 6)

    for phrase, frac in _FRACTION_WORDS.items():
        if phrase in lowered:
            if is_reduction:
                return round(1.0 - frac, 6)
            return round(frac, 6)

    return None


def parse_kwh_value(text: str) -> float | None:
    """Extract a plain kWh numeric literal, e.g. "at least 100 kWh" -> 100."""
    m = _KWH_RE.search(text)
    if m:
        return float(m.group(1))
    return None


def parse_percentage_of_capacity(text: str, capacity_kwh: float | None) -> float | None:
    """Resolve "50% of the battery" into kWh given a known capacity."""
    if capacity_kwh is None:
        return None
    m = _PERCENT_RE.search(text)
    if not m:
        return None
    pct = float(m.group(1)) / 100.0
    return round(pct * capacity_kwh, 6)
