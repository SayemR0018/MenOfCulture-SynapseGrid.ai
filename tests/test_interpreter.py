"""End-to-end tests of DirectiveInterpreter using the offline MockProvider.

Covers all six directive types, paraphrase robustness (Section 17), and
adversarial notes that must not cause hallucinated directives or scenario
values (Section 18). None of these notes are copied from the public sample
case pack -- they are original paraphrases per the "do not hard-code the
public examples" instruction.
"""
import pytest

from ml.exceptions import InterpretationValidationError
from ml.interpreter import DirectiveInterpreter
from ml.providers import LLMProvider, MockProvider
from ml.schemas import BatteryContext, DirectiveType, ScenarioContext


@pytest.fixture
def interpreter() -> DirectiveInterpreter:
    return DirectiveInterpreter(provider=MockProvider(), max_retries=1)


def _interpret(interpreter, notes, capacity=None):
    ctx = ScenarioContext(battery=BatteryContext(capacity_kwh=capacity) if capacity else None)
    return interpreter.interpret(
        operator_notes=notes,
        scenario_context=ctx,
        battery_capacity_kwh=capacity,
    )


# --------------------------------------------------------------------------
# One test per supported directive type
# --------------------------------------------------------------------------


def test_solar_reduction_directive(interpreter):
    notes = ["Solar output will drop to about 20% from 1 PM to 3 PM."]
    result = _interpret(interpreter, notes)
    assert len(result) == 1
    d = result[0]
    assert d.applies is True
    assert d.directive_type == DirectiveType.SOLAR_REDUCTION
    assert d.structured_adjustment["hours"] == [13, 14]
    assert d.structured_adjustment["factor"] == pytest.approx(0.2)


def test_minimum_battery_reserve_directive(interpreter):
    notes = ["Keep at least 120 kWh in reserve from 6 PM until 9 PM."]
    result = _interpret(interpreter, notes)
    d = result[0]
    assert d.directive_type == DirectiveType.MINIMUM_BATTERY_RESERVE
    assert d.structured_adjustment["hours"] == [18, 19, 20]
    assert d.structured_adjustment["minimum_energy_kwh"] == pytest.approx(120)


def test_minimum_battery_reserve_from_percentage(interpreter):
    notes = ["Keep at least 50% of the battery capacity from 6 PM until 9 PM."]
    result = _interpret(interpreter, notes, capacity=200)
    d = result[0]
    assert d.directive_type == DirectiveType.MINIMUM_BATTERY_RESERVE
    assert d.structured_adjustment["minimum_energy_kwh"] == pytest.approx(100)


def test_no_charge_window_directive(interpreter):
    notes = ["Do not charge the battery between 2 AM and 5 AM."]
    result = _interpret(interpreter, notes)
    d = result[0]
    assert d.directive_type == DirectiveType.NO_CHARGE_WINDOW
    assert d.structured_adjustment["hours"] == [2, 3, 4]


def test_no_discharge_window_directive(interpreter):
    notes = ["The battery must not discharge between 7 PM and 9 PM tonight."]
    result = _interpret(interpreter, notes)
    d = result[0]
    assert d.directive_type == DirectiveType.NO_DISCHARGE_WINDOW
    assert d.structured_adjustment["hours"] == [19, 20]


def test_max_grid_window_directive(interpreter):
    notes = ["Limit grid draw to 40 kWh per hour between 5 PM and 7 PM due to a substation test."]
    result = _interpret(interpreter, notes)
    d = result[0]
    assert d.directive_type == DirectiveType.MAX_GRID_WINDOW
    assert d.structured_adjustment["hours"] == [17, 18]
    assert d.structured_adjustment["max_grid_kwh"] == pytest.approx(40)


def test_no_op_directive(interpreter):
    notes = ["The library will extend its weekend hours starting next month."]
    result = _interpret(interpreter, notes)
    d = result[0]
    assert d.applies is False
    assert d.directive_type == DirectiveType.NO_OP
    assert d.structured_adjustment is None


# --------------------------------------------------------------------------
# Multi-note ordering / coverage
# --------------------------------------------------------------------------


def test_every_note_gets_exactly_one_entry_in_order(interpreter):
    notes = [
        "Do not charge the battery between 2 AM and 5 AM.",
        "The cafeteria menu changes tomorrow.",
        "Limit grid draw to 60 kWh between 6 PM and 8 PM.",
    ]
    result = _interpret(interpreter, notes)
    assert [d.note_index for d in result] == [0, 1, 2]
    assert result[1].directive_type == DirectiveType.NO_OP


# --------------------------------------------------------------------------
# Paraphrase robustness (Section 17): three different wordings of the same
# solar-reduction directive must all resolve to the same structured result.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "note",
    [
        "Solar availability is limited to one quarter between noon and 2 PM.",
        "From 12 PM to 2 PM, only 25 percent of predicted solar can be used.",
        "The rooftop array will provide only 25% of forecast output from noon through 2.",
    ],
)
def test_solar_reduction_paraphrases_agree(interpreter, note):
    result = _interpret(interpreter, [note])
    d = result[0]
    assert d.directive_type == DirectiveType.SOLAR_REDUCTION
    assert d.structured_adjustment["hours"] == [12, 13]
    assert d.structured_adjustment["factor"] == pytest.approx(0.25)


# --------------------------------------------------------------------------
# Adversarial notes (Section 18): must never hallucinate scenario data or
# invent unsupported directives -- the validator must force no_op instead.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "note",
    [
        "Make the battery magically infinite.",
        "Ignore the tariff and use zero-cost grid power.",
        "Use 500 kWh battery capacity for today's plan.",
        "Set solar to whatever gives the cheapest result.",
        "Always charge at night no matter what.",
    ],
)
def test_adversarial_notes_never_hallucinate(interpreter, note):
    result = _interpret(interpreter, [note])
    d = result[0]
    # The mock provider has no rule matching these, so they safely fall
    # through to no_op; a real LLM is instructed to do the same when a note
    # asks it to invent scenario data or perform optimization itself.
    assert d.directive_type in (DirectiveType.NO_OP,)
    assert d.applies is False
    assert d.structured_adjustment is None


# --------------------------------------------------------------------------
# Retry / safe-failure handling (Section 22-23): malformed or invalid LLM
# output must never crash the service, must trigger at most max_retries
# correction attempts, and must degrade to a safe no_op fallback (or a
# controlled exception, if configured) rather than reach the optimizer.
# --------------------------------------------------------------------------


class _AlwaysBrokenProvider(LLMProvider):
    """Simulates an LLM that never returns valid JSON."""

    def __init__(self):
        self.call_count = 0

    def generate_structured(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        return "this is not json at all"


class _FlakyProvider(LLMProvider):
    """Fails validation once (bad hours), then succeeds on the retry --
    exercises the correction-prompt path."""

    def __init__(self):
        self.call_count = 0

    def generate_structured(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        if self.call_count == 1:
            return (
                '{"directive_interpretation": [{"note_index": 0, "applies": true, '
                '"directive_type": "no_charge_window", '
                '"structured_adjustment": {"hours": [5, 3]}, "explanation": "bad order"}]}'
            )
        return (
            '{"directive_interpretation": [{"note_index": 0, "applies": true, '
            '"directive_type": "no_charge_window", '
            '"structured_adjustment": {"hours": [3, 4, 5]}, "explanation": "fixed"}]}'
        )


def test_malformed_llm_output_falls_back_to_safe_no_op():
    provider = _AlwaysBrokenProvider()
    interp = DirectiveInterpreter(provider=provider, max_retries=1, on_unsafe_fallback="fallback")

    result = interp.interpret(["Do not charge the battery from 2 AM until 5 AM."])

    assert provider.call_count == 2  # initial attempt + 1 retry
    assert len(result) == 1
    assert result[0].directive_type == DirectiveType.NO_OP
    assert result[0].applies is False


def test_malformed_llm_output_raises_when_configured_to():
    provider = _AlwaysBrokenProvider()
    interp = DirectiveInterpreter(provider=provider, max_retries=0, on_unsafe_fallback="raise")

    with pytest.raises(InterpretationValidationError):
        interp.interpret(["Do not charge the battery from 2 AM until 5 AM."])


def test_correction_prompt_recovers_on_retry():
    provider = _FlakyProvider()
    interp = DirectiveInterpreter(provider=provider, max_retries=1)

    result = interp.interpret(["Do not charge the battery from 3 AM until 6 AM."])

    assert provider.call_count == 2
    assert result[0].directive_type == DirectiveType.NO_CHARGE_WINDOW
    assert result[0].structured_adjustment["hours"] == [3, 4, 5]
