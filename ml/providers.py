"""LLM provider abstraction.

The rest of the system only ever calls ``provider.generate_structured(...)``.
Swapping models/providers is a matter of changing environment variables, not
application code:

    MODEL_PROVIDER=openai|anthropic|mock
    MODEL_NAME=<provider-specific model id>
    API_KEY=<secret, never hard-coded>

``mock`` is a deterministic, offline, rule-based provider (see
``ml/normalizer.py``) that requires no network access or API key. It exists
so the service is runnable and testable end-to-end without credentials; it
does NOT satisfy the competition's "LLM must be part of the interpretation
path" requirement and must not be used for the actual submission/deployment.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod

from ml.exceptions import LLMProviderError
from ml.normalizer import parse_percentage_of_capacity, parse_solar_factor, parse_time_range


class LLMProvider(ABC):
    """Abstract interface every model backend must implement."""

    @abstractmethod
    def generate_structured(self, system_prompt: str, user_prompt: str) -> str:
        """Send the prompts to the model and return its raw text response.

        Implementations should request JSON-only output where the provider
        supports it (e.g. response_format / JSON mode) but must always
        return the raw string -- parsing/validation happens elsewhere.
        """
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    def __init__(self, model_name: str, api_key: str, base_url: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMProviderError(
                "The 'openai' package is required for MODEL_PROVIDER=openai. "
                "Install it with `pip install openai`."
            ) from exc

        if not api_key:
            raise LLMProviderError("API_KEY is required for MODEL_PROVIDER=openai")

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model_name = model_name

    def generate_structured(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a controlled error
            raise LLMProviderError(f"OpenAI request failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMProviderError("OpenAI returned an empty response")
        return content


class AnthropicProvider(LLMProvider):
    def __init__(self, model_name: str, api_key: str):
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMProviderError(
                "The 'anthropic' package is required for MODEL_PROVIDER=anthropic. "
                "Install it with `pip install anthropic`."
            ) from exc

        if not api_key:
            raise LLMProviderError("API_KEY is required for MODEL_PROVIDER=anthropic")

        self._client = Anthropic(api_key=api_key)
        self._model_name = model_name

    def generate_structured(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.messages.create(
                model=self._model_name,
                max_tokens=2048,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(f"Anthropic request failed: {exc}") from exc

        text_parts = [block.text for block in response.content if block.type == "text"]
        content = "".join(text_parts).strip()
        if not content:
            raise LLMProviderError("Anthropic returned an empty response")
        return content


class MockProvider(LLMProvider):
    """Deterministic, offline, rule-based stand-in for a real LLM.

    Used as the default provider so the project runs out of the box without
    credentials, and so unit tests exercise the full interpreter pipeline
    without network access. Reuses the same time/numeric parsing rules
    documented in the system prompt (see ml/normalizer.py), so its behavior
    matches what a well-behaved LLM should produce for the paraphrases in
    tests/test_interpreter.py.
    """

    _NO_CHARGE_HINTS = (
        "no_charge_window", "do not charge", "not charge", "isolat",
        "charger will be", "charging is disabled", "charging is unavailable",
        "charging circuit", "disable charging", "charging will be",
    )
    _NO_DISCHARGE_HINTS = (
        "no_discharge_window", "do not discharge", "not discharge", "discharg",
    )
    _MAX_GRID_HINTS = (
        "max grid", "grid draw", "grid import", "grid intake", "limit grid",
        "grid cap", "grid to", "substation", "stay at or below",
    )
    _RESERVE_HINTS = (
        "reserve", "keep at least", "maintain at least", "battery capacity from",
        "at least", "must remain", "remain in the battery",
    )
    _SOLAR_HINTS = ("solar", "pv production", "rooftop array", "panel")

    def generate_structured(self, system_prompt: str, user_prompt: str) -> str:
        payload = self._extract_payload(user_prompt)
        notes: list[str] = payload.get("operator_notes", [])
        scenario_context = payload.get("scenario_context", {}) or {}
        battery = (scenario_context or {}).get("battery") or {}
        capacity = battery.get("capacity_kwh")

        entries = []
        for i, note in enumerate(notes):
            entries.append(self._interpret_note(i, note, capacity))

        return json.dumps({"directive_interpretation": entries})

    @staticmethod
    def _extract_payload(user_prompt: str) -> dict:
        marker = "CURRENT SCENARIO CONTEXT + CURRENT OPERATOR NOTES"
        idx = user_prompt.find(marker)
        if idx == -1:
            return {}
        chunk = user_prompt[idx:]
        start = chunk.find("{")
        depth = 0
        for i, ch in enumerate(chunk[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(chunk[start : i + 1])
                    except json.JSONDecodeError:
                        return {}
        return {}

    def _interpret_note(self, index: int, note: str, capacity: float | None) -> dict:
        lowered = note.lower()
        no_op = {
            "note_index": index,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": "The note does not specify a supported energy-management directive.",
        }

        hours = parse_time_range(note)

        if any(h in lowered for h in self._SOLAR_HINTS) and hours:
            factor = parse_solar_factor(note)
            if factor is not None:
                return {
                    "note_index": index,
                    "applies": True,
                    "directive_type": "solar_reduction",
                    "structured_adjustment": {"hours": hours, "factor": factor},
                    "explanation": f"Usable solar is scaled by {factor} during hours {hours}.",
                }

        if any(h in lowered for h in self._RESERVE_HINTS) and hours:
            kwh = self._extract_kwh_or_percentage(note, capacity)
            if kwh is not None:
                return {
                    "note_index": index,
                    "applies": True,
                    "directive_type": "minimum_battery_reserve",
                    "structured_adjustment": {"hours": hours, "minimum_energy_kwh": kwh},
                    "explanation": f"Battery must keep at least {kwh} kWh during hours {hours}.",
                }

        if any(h in lowered for h in self._MAX_GRID_HINTS) and hours:
            from ml.normalizer import parse_kwh_value

            kwh = parse_kwh_value(note)
            if kwh is not None:
                return {
                    "note_index": index,
                    "applies": True,
                    "directive_type": "max_grid_window",
                    "structured_adjustment": {"hours": hours, "max_grid_kwh": kwh},
                    "explanation": f"Grid import capped at {kwh} kWh during hours {hours}.",
                }

        if any(h in lowered for h in self._NO_CHARGE_HINTS) and hours:
            return {
                "note_index": index,
                "applies": True,
                "directive_type": "no_charge_window",
                "structured_adjustment": {"hours": hours},
                "explanation": f"Charging is disabled during hours {hours}.",
            }

        if any(h in lowered for h in self._NO_DISCHARGE_HINTS) and hours:
            return {
                "note_index": index,
                "applies": True,
                "directive_type": "no_discharge_window",
                "structured_adjustment": {"hours": hours},
                "explanation": f"Discharging is disabled during hours {hours}.",
            }

        return no_op

    @staticmethod
    def _extract_kwh_or_percentage(note: str, capacity: float | None) -> float | None:
        from ml.normalizer import parse_kwh_value

        kwh = parse_kwh_value(note)
        if kwh is not None:
            return kwh
        return parse_percentage_of_capacity(note, capacity)


def get_provider(
    provider_name: str,
    model_name: str,
    api_key: str | None,
    base_url: str | None = None,
) -> LLMProvider:
    """Factory used by the interpreter / app wiring."""
    provider_name = (provider_name or "mock").lower()

    if provider_name == "openai":
        return OpenAIProvider(model_name=model_name, api_key=api_key or "", base_url=base_url)
    if provider_name == "anthropic":
        return AnthropicProvider(model_name=model_name, api_key=api_key or "")
    if provider_name == "mock":
        return MockProvider()

    raise LLMProviderError(f"Unknown MODEL_PROVIDER '{provider_name}'")
