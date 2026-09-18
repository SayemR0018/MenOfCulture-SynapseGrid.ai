"""Exceptions for the ML interpretation layer.

All exceptions here are expected to be caught at the API boundary and
converted into controlled HTTP error responses -- they must never surface
as raw stack traces.
"""


class GridWiseMLError(Exception):
    """Base class for all ML-layer errors."""


class LLMProviderError(GridWiseMLError):
    """The configured LLM provider failed to produce a response
    (network error, API error, timeout, etc.)."""


class LLMOutputParseError(GridWiseMLError):
    """The LLM response could not be parsed as JSON at all."""


class InterpretationValidationError(GridWiseMLError):
    """The LLM output failed deterministic validation after all retries
    were exhausted."""

    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors or []
