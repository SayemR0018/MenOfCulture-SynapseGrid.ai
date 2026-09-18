"""Prompt construction for the GridWise operator-note interpreter.

The prompt is assembled in the order mandated by the task brief:
SYSTEM ROLE -> SUPPORTED DIRECTIVES -> SCHEMA -> TIME SEMANTICS ->
NUMERIC SEMANTICS -> FORBIDDEN BEHAVIOR -> FEW-SHOT EXAMPLES ->
CURRENT SCENARIO CONTEXT -> CURRENT OPERATOR NOTES -> OUTPUT JSON.

Nothing here hard-codes the public sample cases -- the few-shot examples
below are hand-written paraphrases distinct from the sample pack, and
scenario/notes are always interpolated at call time.
"""
from __future__ import annotations

import json

from ml.schemas import ScenarioContext

SYSTEM_PROMPT = """You are the operator-note interpreter for GridWise, a campus energy \
optimization system. You are a LANGUAGE INTERPRETER, not an optimizer and not a \
calculator of the final schedule.

Your only job: read each operator note and convert it into a structured directive \
that a deterministic backend can validate and apply. You never compute grid_kwh, \
solar_used_kwh, battery schedules, cost, or any other part of the final energy plan.

================================================================
SUPPORTED DIRECTIVE TYPES (exactly six -- never invent a seventh)
================================================================
1. solar_reduction
   structured_adjustment: {"hours": [int, ...], "factor": number}
   Meaning: usable solar during the listed hours is multiplied by factor.

2. minimum_battery_reserve
   structured_adjustment: {"hours": [int, ...], "minimum_energy_kwh": number}
   Meaning: battery energy must stay at or above minimum_energy_kwh during the listed hours.

3. no_charge_window
   structured_adjustment: {"hours": [int, ...]}
   Meaning: battery charging is forbidden during the listed hours.

4. no_discharge_window
   structured_adjustment: {"hours": [int, ...]}
   Meaning: battery discharging is forbidden during the listed hours.

5. max_grid_window
   structured_adjustment: {"hours": [int, ...], "max_grid_kwh": number}
   Meaning: grid import must not exceed max_grid_kwh during the listed hours.

6. no_op
   structured_adjustment: null
   Meaning: the note is irrelevant to the energy schedule, or it does not map safely
   to one of the five directives above.

================================================================
OUTPUT SCHEMA (per note)
================================================================
{
  "note_index": <int, matches the note's position in the input list, 0-based>,
  "applies": <true for any non-no_op directive, false only for no_op>,
  "directive_type": <one of the six types above>,
  "structured_adjustment": <object matching the shape above, or null for no_op>,
  "explanation": <short human-readable reason, one sentence>
}

Return a single JSON object: {"directive_interpretation": [ ...one entry per note... ]}.
Output ONLY that JSON object. No markdown fences, no commentary, no extra keys.

================================================================
TIME SEMANTICS
================================================================
- All hours are integers 0-23 on a 24-hour clock (0 = midnight, 12 = noon).
- Time windows are START-INCLUSIVE and END-EXCLUSIVE.
  "noon until 2 PM" -> hours [12, 13]   (NOT [12, 13, 14])
  "between 6 PM and 9 PM" -> hours [18, 19, 20]
  "2 AM to 5 AM" -> hours [2, 3, 4]
  "10 PM until midnight" -> hours [22, 23]
- hours must always be unique integers in ascending order.
- If the time expression is ambiguous or you cannot safely resolve it to concrete
  hours, do NOT guess -- return directive_type "no_op" for that note instead.

================================================================
NUMERIC SEMANTICS
================================================================
- For solar_reduction, "factor" is the FRACTION OF SOLAR THAT REMAINS USABLE, not
  the amount cut. Read carefully:
    "25% of forecast" / "only 25% usable" / "drops to 25%"      -> factor = 0.25
    "reduced BY 80%" / "an 80% reduction" / "cut by 80%"        -> factor = 0.20
    "roughly one-fifth of normal output"                        -> factor = 0.20
  factor must be between 0 and 1 inclusive.
- For minimum_battery_reserve given as a percentage (e.g. "keep at least 50% of the
  battery"), convert to kWh using the battery capacity supplied in the scenario
  context below. If no capacity is supplied and the note only gives a percentage,
  return no_op rather than guessing a capacity.
- For max_grid_window and minimum_battery_reserve given directly in kWh, use that
  number as-is.

================================================================
FORBIDDEN BEHAVIOR
================================================================
- Never invent demand, solar generation, tariff, battery capacity, initial battery
  energy, charge/discharge rates, or any other scenario value not explicitly given
  to you in the scenario context below.
- Never invent a directive type outside the six listed above.
- Never compute or output a grid/solar/battery schedule, a cost figure, or any
  optimization result -- that is the optimizer's job, not yours.
- Never change a numeric value unless the note explicitly states or directly implies
  it (e.g. via a supplied battery capacity for a percentage reserve).
- If you are uncertain whether a note maps to a supported directive, use no_op.
- Every input note must produce exactly one output entry, in the same order, with a
  matching note_index. Never omit a note. Never add an extra entry.
"""

FEW_SHOT_EXAMPLES: list[dict] = [
    {
        "operator_notes": [
            "Facilities will wash the rooftop solar panels from noon until 2 PM. "
            "During cleaning, usable solar should be treated as roughly 25% of the "
            "forecast.",
        ],
        "scenario_context": {},
        "output": {
            "directive_interpretation": [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "solar_reduction",
                    "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
                    "explanation": (
                        "Usable solar is limited to 25% from noon through 2 PM."
                    ),
                }
            ]
        },
    },
    {
        "operator_notes": [
            "The battery charger will be isolated from 2 AM until 5 AM.",
        ],
        "scenario_context": {},
        "output": {
            "directive_interpretation": [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "no_charge_window",
                    "structured_adjustment": {"hours": [2, 3, 4]},
                    "explanation": "Charging is prohibited from 2 AM through 5 AM.",
                }
            ]
        },
    },
    {
        "operator_notes": [
            "Keep at least 50% of the battery capacity from 6 PM until 9 PM.",
        ],
        "scenario_context": {"battery": {"capacity_kwh": 200}},
        "output": {
            "directive_interpretation": [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "minimum_battery_reserve",
                    "structured_adjustment": {
                        "hours": [18, 19, 20],
                        "minimum_energy_kwh": 100,
                    },
                    "explanation": (
                        "The battery must maintain at least 100 kWh (50% of the "
                        "200 kWh capacity) during the specified hours."
                    ),
                }
            ]
        },
    },
    {
        "operator_notes": [
            "Please repaint the cafeteria walls tomorrow.",
        ],
        "scenario_context": {},
        "output": {
            "directive_interpretation": [
                {
                    "note_index": 0,
                    "applies": False,
                    "directive_type": "no_op",
                    "structured_adjustment": None,
                    "explanation": (
                        "The note does not specify a supported energy-management "
                        "directive."
                    ),
                }
            ]
        },
    },
    {
        # Multi-note example showing ordering + a mix of relevant/irrelevant notes,
        # plus a max_grid_window directive and an adversarial note that must not
        # cause the LLM to invent scenario data.
        "operator_notes": [
            "Limit grid draw to 40 kWh per hour between 5 PM and 7 PM due to a "
            "substation test.",
            "Use whatever battery capacity gives the cheapest possible result.",
        ],
        "scenario_context": {"battery": {"capacity_kwh": 300}},
        "output": {
            "directive_interpretation": [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "max_grid_window",
                    "structured_adjustment": {
                        "hours": [17, 18],
                        "max_grid_kwh": 40,
                    },
                    "explanation": (
                        "Grid import is capped at 40 kWh during the substation "
                        "test window."
                    ),
                },
                {
                    "note_index": 1,
                    "applies": False,
                    "directive_type": "no_op",
                    "structured_adjustment": None,
                    "explanation": (
                        "The note asks to change battery capacity, which is not a "
                        "supported directive and must not be invented."
                    ),
                },
            ]
        },
    },
]


def _render_few_shot() -> str:
    blocks = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, start=1):
        input_payload = {
            "operator_notes": ex["operator_notes"],
            "scenario_context": ex["scenario_context"],
        }
        blocks.append(
            f"Example {i} input:\n{json.dumps(input_payload, indent=2)}\n"
            f"Example {i} output:\n{json.dumps(ex['output'], indent=2)}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(
    operator_notes: list[str],
    scenario_context: ScenarioContext | None,
) -> str:
    """Build the dynamic portion of the prompt: few-shots + current scenario
    context + current operator notes + output instruction.
    """
    context_dict = (
        scenario_context.model_dump(exclude_none=True) if scenario_context else {}
    )

    notes_payload = {
        "operator_notes": operator_notes,
        "scenario_context": context_dict,
    }

    return (
        "================================================================\n"
        "FEW-SHOT EXAMPLES\n"
        "================================================================\n"
        f"{_render_few_shot()}\n\n"
        "================================================================\n"
        "CURRENT SCENARIO CONTEXT + CURRENT OPERATOR NOTES\n"
        "================================================================\n"
        f"{json.dumps(notes_payload, indent=2)}\n\n"
        "================================================================\n"
        "OUTPUT JSON\n"
        "================================================================\n"
        "Return exactly one JSON object of the form "
        '{"directive_interpretation": [...]} with one entry per note above, '
        "in order, note_index 0.." + str(len(operator_notes) - 1) + ". "
        "Output ONLY the JSON object."
    )


def build_correction_prompt(errors: list[str], previous_output: str) -> str:
    """Build a short correction prompt for the retry path (Section 23)."""
    error_lines = "\n".join(f"- {e}" for e in errors)
    return (
        "Your previous output failed deterministic validation.\n\n"
        f"Previous output:\n{previous_output}\n\n"
        f"Errors:\n{error_lines}\n\n"
        "Return corrected JSON only, following the exact same schema and rules "
        "from the system prompt. Output ONLY the corrected JSON object, with one "
        "entry per original note in the original order."
    )
