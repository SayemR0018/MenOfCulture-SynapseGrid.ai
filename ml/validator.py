"""Deterministic validation of LLM-produced directive interpretations.

This module is the guardrail described in Section 08/09 of the problem
statement and in the task brief: LLM output is NEVER trusted directly. Every
rule enumerated in the brief's "DETERMINISTIC VALIDATION" section is checked
here, and the validator is written to never raise on malformed input -- it
always returns a (validated_entries_or_None, errors) pair.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from ml.schemas import (
    ADJUSTMENT_MODEL_BY_DIRECTIVE,
    DirectiveInterpretation,
    DirectiveType,
)

_ALLOWED_TYPES = {d.value for d in DirectiveType}


@dataclass
class ValidationResult:
    valid: bool
    entries: list[DirectiveInterpretation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def validate_interpretation(
    raw_entries: list[dict],
    num_notes: int,
    battery_capacity_kwh: float | None = None,
) -> ValidationResult:
    """Validate a list of raw (untrusted) interpretation entries.

    Parameters
    ----------
    raw_entries:
        Parsed JSON objects as returned by the LLM (already JSON-decoded,
        but otherwise untrusted).
    num_notes:
        Number of operator notes in the request -- every note must be
        represented exactly once.
    battery_capacity_kwh:
        If known, used to reject minimum_battery_reserve values that exceed
        physical battery capacity.

    Returns
    -------
    ValidationResult with either a fully validated ``entries`` list (in
    note_index order) or a non-empty ``errors`` list explaining why
    validation failed. Never raises.
    """
    errors: list[str] = []

    if not isinstance(raw_entries, list):
        return ValidationResult(False, [], ["directive_interpretation must be a list"])

    # --- Rule: note_index validity, no duplicates, full coverage ----------
    seen_indices: dict[int, dict] = {}
    for i, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            errors.append(f"entry at position {i} is not a JSON object")
            continue

        idx = entry.get("note_index")
        if not isinstance(idx, int) or isinstance(idx, bool):
            errors.append(f"entry at position {i} has missing/non-integer note_index")
            continue
        if idx < 0 or idx >= num_notes:
            errors.append(
                f"note_index {idx} is out of range for {num_notes} operator note(s)"
            )
            continue
        if idx in seen_indices:
            errors.append(f"duplicate note_index {idx}")
            continue
        seen_indices[idx] = entry

    missing = [i for i in range(num_notes) if i not in seen_indices]
    for i in missing:
        errors.append(f"missing interpretation entry for note_index {i}")

    if errors:
        return ValidationResult(False, [], errors)

    # --- Rule: per-entry structural + semantic validation ------------------
    validated: list[DirectiveInterpretation] = []
    for idx in range(num_notes):
        entry = seen_indices[idx]
        entry_errors = _validate_single_entry(entry, idx, battery_capacity_kwh)
        if entry_errors:
            errors.extend(entry_errors)
            continue
        validated.append(
            DirectiveInterpretation(
                note_index=idx,
                applies=bool(entry["applies"]),
                directive_type=DirectiveType(entry["directive_type"]),
                structured_adjustment=entry.get("structured_adjustment"),
                explanation=str(entry.get("explanation") or ""),
            )
        )

    if errors:
        return ValidationResult(False, [], errors)

    validated.sort(key=lambda e: e.note_index)
    return ValidationResult(True, validated, [])


def _validate_single_entry(
    entry: dict, idx: int, battery_capacity_kwh: float | None
) -> list[str]:
    errors: list[str] = []
    prefix = f"note_index {idx}:"

    directive_type = entry.get("directive_type")
    if not isinstance(directive_type, str) or directive_type not in _ALLOWED_TYPES:
        errors.append(
            f"{prefix} unsupported or missing directive_type {directive_type!r}"
        )
        return errors

    applies = entry.get("applies")
    if not isinstance(applies, bool):
        errors.append(f"{prefix} applies must be a boolean")
        return errors

    structured_adjustment = entry.get("structured_adjustment")

    if directive_type == DirectiveType.NO_OP.value:
        if applies is not False:
            errors.append(f"{prefix} no_op must have applies = false")
        if structured_adjustment is not None:
            errors.append(f"{prefix} no_op must have structured_adjustment = null")
        return errors

    # Every non-no_op directive must have applies = true.
    if applies is not True:
        errors.append(f"{prefix} non-no_op directive must have applies = true")
        return errors

    if not isinstance(structured_adjustment, dict):
        errors.append(
            f"{prefix} directive {directive_type} requires a structured_adjustment object"
        )
        return errors

    model_cls = ADJUSTMENT_MODEL_BY_DIRECTIVE[DirectiveType(directive_type)]
    try:
        adjustment = model_cls(**structured_adjustment)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            errors.append(f"{prefix} structured_adjustment.{loc}: {err['msg']}")
        return errors

    # Rule: reserve must not exceed battery capacity, if known.
    if directive_type == DirectiveType.MINIMUM_BATTERY_RESERVE.value:
        if (
            battery_capacity_kwh is not None
            and adjustment.minimum_energy_kwh > battery_capacity_kwh + 1e-9
        ):
            errors.append(
                f"{prefix} minimum_energy_kwh {adjustment.minimum_energy_kwh} exceeds "
                f"battery capacity {battery_capacity_kwh}"
            )

    return errors
