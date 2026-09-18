#!/usr/bin/env python3
"""Evaluation script for the GridWise operator-note interpreter (Section 20).

Runs the interpreter against a labeled case pack (by default the public
sample cases) and reports:

    - relevance accuracy            (applies True/False correct)
    - directive classification accuracy (directive_type correct)
    - hour extraction accuracy      (hours array exactly correct)
    - numeric parameter accuracy    (factor / minimum_energy_kwh / max_grid_kwh
                                      correct within tolerance)
    - no_op accuracy                (no_op cases correctly identified)
    - complete interpretation accuracy (every field correct for the note)

and a breakdown of error kinds: wrong_directive, wrong_hours, wrong_numeric,
missing_note, extra_note, invalid_schema, hallucinated_value.

Usage:
    python scripts/evaluate_interpreter.py
    python scripts/evaluate_interpreter.py --cases path/to/other_cases.json

The provider is selected the same way the running service selects it (env
vars MODEL_PROVIDER / MODEL_NAME / API_KEY); defaults to the offline mock
provider if unset, which is useful for smoke-testing this script itself but
does not reflect real LLM accuracy.

This script never modifies application logic based on the labeled cases --
it is read-only evaluation tooling.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from ml.interpreter import DirectiveInterpreter  # noqa: E402
from ml.providers import get_provider  # noqa: E402
from ml.schemas import BatteryContext, ScenarioContext  # noqa: E402

_NUMERIC_TOLERANCE = 0.01
_FACTOR_TOLERANCE = 0.005


@dataclass
class Metrics:
    total_notes: int = 0
    relevance_correct: int = 0
    directive_correct: int = 0
    hours_applicable: int = 0
    hours_correct: int = 0
    numeric_applicable: int = 0
    numeric_correct: int = 0
    no_op_total: int = 0
    no_op_correct: int = 0
    complete_correct: int = 0

    errors: dict = field(
        default_factory=lambda: {
            "wrong_directive": 0,
            "wrong_hours": 0,
            "wrong_numeric": 0,
            "missing_note": 0,
            "extra_note": 0,
            "invalid_schema": 0,
            "hallucinated_value": 0,
        }
    )

    per_case_notes: list = field(default_factory=list)


def _numeric_fields_for(directive_type: str) -> list[str]:
    return {
        "solar_reduction": ["factor"],
        "minimum_battery_reserve": ["minimum_energy_kwh"],
        "max_grid_window": ["max_grid_kwh"],
    }.get(directive_type, [])


def _numeric_close(a: float, b: float, field_name: str) -> bool:
    tol = _FACTOR_TOLERANCE if field_name == "factor" else _NUMERIC_TOLERANCE
    return abs(a - b) <= tol


def evaluate(cases: list[dict], interpreter: DirectiveInterpreter) -> Metrics:
    metrics = Metrics()

    for case in cases:
        input_data = case["input"]
        expected = {
            e["note_index"]: e for e in case["expected_output"]["directive_interpretation"]
        }
        notes = input_data["operator_notes"]
        battery = input_data.get("battery", {})
        capacity = battery.get("capacity_kwh")

        scenario_context = ScenarioContext(
            battery=BatteryContext(capacity_kwh=capacity) if capacity else None
        )

        try:
            actual = interpreter.interpret(
                operator_notes=notes,
                scenario_context=scenario_context,
                battery_capacity_kwh=capacity,
                scenario_id=case.get("id"),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{case.get('id')}] interpreter raised: {exc}", file=sys.stderr)
            metrics.errors["invalid_schema"] += len(notes)
            continue

        actual_by_index = {d.note_index: d for d in actual}

        for idx in range(len(notes)):
            metrics.total_notes += 1
            exp = expected.get(idx)
            act = actual_by_index.get(idx)

            if exp is None:
                continue  # case pack itself is missing a reference entry
            if act is None:
                metrics.errors["missing_note"] += 1
                metrics.per_case_notes.append((case["id"], idx, "missing_note"))
                continue

            exp_applies = exp["applies"]
            exp_type = exp["directive_type"]
            exp_adj = exp.get("structured_adjustment") or {}

            act_applies = act.applies
            act_type = act.directive_type.value
            act_adj = act.structured_adjustment or {}

            note_ok = True

            if act_applies == exp_applies:
                metrics.relevance_correct += 1
            else:
                note_ok = False

            if exp_type == "no_op":
                metrics.no_op_total += 1
                if act_type == "no_op":
                    metrics.no_op_correct += 1
                else:
                    note_ok = False

            if act_type == exp_type:
                metrics.directive_correct += 1
            else:
                metrics.errors["wrong_directive"] += 1
                note_ok = False

            if exp_type != "no_op":
                metrics.hours_applicable += 1
                if act_type == exp_type and act_adj.get("hours") == exp_adj.get("hours"):
                    metrics.hours_correct += 1
                else:
                    metrics.errors["wrong_hours"] += 1
                    note_ok = False

                numeric_fields = _numeric_fields_for(exp_type)
                for f in numeric_fields:
                    metrics.numeric_applicable += 1
                    exp_val = exp_adj.get(f)
                    act_val = act_adj.get(f)
                    if (
                        act_type == exp_type
                        and isinstance(act_val, (int, float))
                        and isinstance(exp_val, (int, float))
                        and _numeric_close(float(act_val), float(exp_val), f)
                    ):
                        metrics.numeric_correct += 1
                    else:
                        metrics.errors["wrong_numeric"] += 1
                        note_ok = False

                # Hallucination check: any key in the actual adjustment that
                # isn't part of the required shape for its directive type.
                allowed_keys = {"hours", *numeric_fields}
                extra_keys = set(act_adj.keys()) - allowed_keys
                if extra_keys:
                    metrics.errors["hallucinated_value"] += 1
                    note_ok = False

            if note_ok:
                metrics.complete_correct += 1
            else:
                metrics.per_case_notes.append((case["id"], idx, "incorrect"))

        for idx in actual_by_index:
            if idx not in expected:
                metrics.errors["extra_note"] += 1

    return metrics


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.1f}%"


def print_report(metrics: Metrics) -> None:
    print("=" * 60)
    print("GridWise Interpreter Evaluation")
    print("=" * 60)
    print(f"Total notes evaluated:            {metrics.total_notes}")
    print(
        f"Relevance accuracy:                "
        f"{_pct(metrics.relevance_correct, metrics.total_notes)} "
        f"({metrics.relevance_correct}/{metrics.total_notes})"
    )
    print(
        f"Directive classification accuracy: "
        f"{_pct(metrics.directive_correct, metrics.total_notes)} "
        f"({metrics.directive_correct}/{metrics.total_notes})"
    )
    print(
        f"Hour extraction accuracy:          "
        f"{_pct(metrics.hours_correct, metrics.hours_applicable)} "
        f"({metrics.hours_correct}/{metrics.hours_applicable})"
    )
    print(
        f"Numeric parameter accuracy:        "
        f"{_pct(metrics.numeric_correct, metrics.numeric_applicable)} "
        f"({metrics.numeric_correct}/{metrics.numeric_applicable})"
    )
    print(
        f"no_op accuracy:                    "
        f"{_pct(metrics.no_op_correct, metrics.no_op_total)} "
        f"({metrics.no_op_correct}/{metrics.no_op_total})"
    )
    print(
        f"Complete interpretation accuracy:  "
        f"{_pct(metrics.complete_correct, metrics.total_notes)} "
        f"({metrics.complete_correct}/{metrics.total_notes})"
    )
    print("-" * 60)
    print("Error breakdown:")
    for k, v in metrics.errors.items():
        print(f"  {k:20s} {v}")
    if metrics.per_case_notes:
        print("-" * 60)
        print("Incorrect notes:")
        for case_id, idx, kind in metrics.per_case_notes:
            print(f"  {case_id} note_index={idx} ({kind})")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default=str(
            Path(__file__).resolve().parent.parent
            / "docs"
            / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
        ),
        help="Path to a labeled case pack JSON (defaults to the public sample pack).",
    )
    args = parser.parse_args()

    with open(args.cases) as f:
        data = json.load(f)
    cases = data["cases"]

    settings = get_settings()
    provider = get_provider(
        settings.model_provider, settings.model_name, settings.api_key, settings.api_base_url
    )
    interpreter = DirectiveInterpreter(provider=provider, max_retries=settings.max_retries)

    print(f"Evaluating {len(cases)} case(s) from {args.cases}")
    print(f"Provider: {settings.model_provider} / model: {settings.model_name}\n")

    metrics = evaluate(cases, interpreter)
    print_report(metrics)


if __name__ == "__main__":
    main()
