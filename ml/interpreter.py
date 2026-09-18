"""The operator-note interpreter: the orchestrator of the LLM interpretation
path described in the architecture diagram:

    Operator Notes -> LLM -> NLU -> Structured Directives -> Deterministic
    Validation -> (returned to caller; optimizer runs separately)

``DirectiveInterpreter.interpret`` never computes an energy schedule. It
only returns validated ``DirectiveInterpretation`` objects.
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging

from ml.exceptions import InterpretationValidationError, LLMOutputParseError, LLMProviderError
from ml.prompts import SYSTEM_PROMPT, build_correction_prompt, build_user_prompt
from ml.providers import LLMProvider
from ml.schemas import DirectiveInterpretation, DirectiveType, ScenarioContext
from ml.validator import ValidationResult, validate_interpretation

logger = logging.getLogger("synapsegrid.ml.interpreter")


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_llm_json(raw_text: str) -> list[dict]:
    """Parse the LLM's raw text into a list of raw interpretation dicts.

    Tolerant of markdown code fences and of the model returning either
    ``{"directive_interpretation": [...]}`` or a bare ``[...]``.
    """
    cleaned = _strip_code_fences(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMOutputParseError(f"LLM output is not valid JSON: {exc}") from exc

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        entries = data.get("directive_interpretation")
        if isinstance(entries, list):
            return entries
        raise LLMOutputParseError(
            "LLM output JSON object missing 'directive_interpretation' array"
        )
    raise LLMOutputParseError("LLM output JSON is neither an object nor an array")


def safe_fallback(num_notes: int) -> list[DirectiveInterpretation]:
    """The controlled failure path: every note becomes no_op.

    Used only after retries are exhausted, so the API never crashes or
    hangs on a misbehaving model -- it degrades to "apply nothing" rather
    than risking a hallucinated directive reaching the optimizer.
    """
    return [
        DirectiveInterpretation(
            note_index=i,
            applies=False,
            directive_type=DirectiveType.NO_OP,
            structured_adjustment=None,
            explanation=(
                "Interpretation unavailable: the model output could not be "
                "validated, so this note was safely treated as no_op."
            ),
        )
        for i in range(num_notes)
    ]


class DirectiveInterpreter:
    """Converts natural-language operator notes into validated directives."""

    def __init__(
        self,
        provider: LLMProvider,
        max_retries: int = 1,
        on_unsafe_fallback: str = "fallback",
    ):
        """
        Parameters
        ----------
        provider: the configured LLMProvider.
        max_retries: number of correction attempts after the first failed
            validation (Section 23: "use at most 1-2 retries").
        on_unsafe_fallback: "fallback" (default) returns an all-no_op
            interpretation if validation never succeeds; "raise" instead
            raises InterpretationValidationError so the caller can decide
            (e.g. return HTTP 422).
        """
        self._provider = provider
        self._max_retries = max_retries
        self._on_unsafe_fallback = on_unsafe_fallback

    def interpret(
        self,
        operator_notes: list[str],
        scenario_context: ScenarioContext | None = None,
        battery_capacity_kwh: float | None = None,
        request_id: str | None = None,
        scenario_id: str | None = None,
    ) -> list[DirectiveInterpretation]:
        num_notes = len(operator_notes)
        if num_notes == 0:
            return []

        system_prompt = SYSTEM_PROMPT
        user_prompt = build_user_prompt(operator_notes, scenario_context)

        last_raw_text = ""
        last_result: ValidationResult | None = None
        attempts = self._max_retries + 1

        for attempt in range(attempts):
            prompt_to_send = user_prompt if attempt == 0 else build_correction_prompt(
                last_result.errors if last_result else [], last_raw_text
            )

            try:
                raw_text = self._provider.generate_structured(system_prompt, prompt_to_send)
            except LLMProviderError as exc:
                logger.warning(
                    "llm_provider_error request_id=%s scenario_id=%s attempt=%d error=%s",
                    request_id, scenario_id, attempt, exc,
                )
                last_result = ValidationResult(False, [], [str(exc)])
                last_raw_text = ""
                continue

            last_raw_text = raw_text
            logger.info(
                "llm_response request_id=%s scenario_id=%s attempt=%d response=%s",
                request_id, scenario_id, attempt, _truncate(raw_text),
            )

            try:
                raw_entries = _parse_llm_json(raw_text)
            except LLMOutputParseError as exc:
                logger.warning(
                    "llm_parse_error request_id=%s scenario_id=%s attempt=%d error=%s",
                    request_id, scenario_id, attempt, exc,
                )
                last_result = ValidationResult(False, [], [str(exc)])
                continue

            result = validate_interpretation(raw_entries, num_notes, battery_capacity_kwh)
            last_result = result

            logger.info(
                "validation_result request_id=%s scenario_id=%s attempt=%d valid=%s errors=%s",
                request_id, scenario_id, attempt, result.valid, result.errors,
            )

            if result.valid:
                return result.entries

        # All attempts exhausted.
        if self._on_unsafe_fallback == "raise":
            raise InterpretationValidationError(
                "LLM output failed deterministic validation after retries",
                errors=last_result.errors if last_result else [],
            )

        logger.warning(
            "interpretation_fallback request_id=%s scenario_id=%s errors=%s",
            request_id, scenario_id, last_result.errors if last_result else [],
        )
        return safe_fallback(num_notes)

    async def interpret_async(
        self,
        operator_notes: list[str],
        scenario_context: ScenarioContext | None = None,
        battery_capacity_kwh: float | None = None,
        request_id: str | None = None,
        scenario_id: str | None = None,
        timeout_seconds: float = 20,
    ) -> list[DirectiveInterpretation]:
        """Runs the blocking ``interpret`` call off the event loop, bounded
        by ``timeout_seconds``. On timeout, follows ``on_unsafe_fallback``
        the same way an exhausted-retries validation failure does.
        """
        num_notes = len(operator_notes)
        if num_notes == 0:
            return []

        loop = asyncio.get_running_loop()
        call = functools.partial(
            self.interpret,
            operator_notes=operator_notes,
            scenario_context=scenario_context,
            battery_capacity_kwh=battery_capacity_kwh,
            request_id=request_id,
            scenario_id=scenario_id,
        )

        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, call), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.warning(
                "interpretation_timeout request_id=%s scenario_id=%s timeout_seconds=%s",
                request_id, scenario_id, timeout_seconds,
            )
            if self._on_unsafe_fallback == "raise":
                raise InterpretationValidationError(
                    f"LLM interpretation timed out after {timeout_seconds}s",
                    errors=[f"timeout after {timeout_seconds}s"],
                ) from None
            return safe_fallback(num_notes)


def _truncate(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"
