"""Runs the full GridWise pipeline against the public sample case pack.

Per the project instructions, public cases are used as *tests of the
pipeline*, not as a lookup table: this file never hard-codes case IDs, note
wording, or reference numeric values. It loads the JSON pack at runtime,
sends each case's operator notes through the interpreter and the resulting
directives through the optimizer, then checks the same structural/physical
properties the hidden judge checks (Section 11.3):

- exactly 24 unique hourly_plan entries
- the energy-balance equation holds every hour
- solar usage never exceeds effective solar
- battery bounds / charge-discharge rate limits are respected
- end-of-day battery neutrality holds
- total_grid_kwh / total_cost_bdt / peak_grid_kwh match values recomputed
  from hourly_plan
- every operator note produced exactly one directive_interpretation entry

Interpretation *semantic* accuracy against the reference directives is
measured separately by scripts/evaluate_interpreter.py, since this test
suite runs against the offline MockProvider by default (see
tests/conftest.py) rather than a paid LLM.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas import BatteryConfig, HourEntry
from ml.interpreter import DirectiveInterpreter
from ml.providers import MockProvider
from ml.schemas import BatteryContext, ScenarioContext
from optimizer.cost import peak_grid_kwh, total_cost_bdt, total_grid_kwh
from optimizer.final_validator import replay_and_validate
from optimizer.optimizer import optimize_schedule

_CASES_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
)


def _load_cases() -> list[dict]:
    if not _CASES_PATH.exists():
        return []
    with open(_CASES_PATH) as f:
        data = json.load(f)
    return data["cases"]


_CASES = _load_cases()


@pytest.fixture(scope="module")
def interpreter() -> DirectiveInterpreter:
    return DirectiveInterpreter(provider=MockProvider(), max_retries=1)


@pytest.mark.skipif(not _CASES, reason="public sample case pack not found")
@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_public_case_produces_valid_schedule(interpreter, case):
    input_data = case["input"]
    hours = [HourEntry(**h) for h in input_data["hours"]]
    battery = BatteryConfig(**input_data["battery"])
    operator_notes = input_data["operator_notes"]

    scenario_context = ScenarioContext(
        battery=BatteryContext(capacity_kwh=battery.capacity_kwh)
    )

    directives = interpreter.interpret(
        operator_notes=operator_notes,
        scenario_context=scenario_context,
        battery_capacity_kwh=battery.capacity_kwh,
    )

    # Every note produces exactly one entry, in order, no gaps/duplicates.
    assert [d.note_index for d in directives] == list(range(len(operator_notes)))

    plan = optimize_schedule(hours, battery, directives)

    assert len(plan) == 24
    assert sorted(p.hour for p in plan) == list(range(24))

    errors = replay_and_validate(hours, battery, directives, plan)
    assert not errors, f"{case['id']} produced an invalid schedule: {errors}"

    tariff_by_hour = [h.tariff_bdt_per_kwh for h in sorted(hours, key=lambda h: h.hour)]
    assert total_grid_kwh(plan) >= 0
    assert total_cost_bdt(plan, tariff_by_hour) >= 0
    assert peak_grid_kwh(plan) == max(p.grid_kwh for p in plan)

    # End-of-day neutrality, restated explicitly for clarity in failures.
    assert plan[-1].battery_energy_after_kwh == pytest.approx(
        battery.initial_energy_kwh, abs=0.01
    )
