"""Unit tests for ml/validator.py -- the deterministic guardrail that must
never trust raw LLM output.
"""
from ml.validator import validate_interpretation


def _no_op(idx):
    return {
        "note_index": idx,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": "irrelevant",
    }


def test_valid_solar_reduction_passes():
    entries = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
            "explanation": "ok",
        }
    ]
    result = validate_interpretation(entries, num_notes=1)
    assert result.valid
    assert result.entries[0].directive_type.value == "solar_reduction"


def test_missing_note_is_rejected():
    result = validate_interpretation([_no_op(0)], num_notes=2)
    assert not result.valid
    assert any("missing interpretation entry for note_index 1" in e for e in result.errors)


def test_duplicate_note_index_is_rejected():
    result = validate_interpretation([_no_op(0), _no_op(0)], num_notes=1)
    assert not result.valid
    assert any("duplicate note_index" in e for e in result.errors)


def test_extra_note_index_out_of_range_is_rejected():
    result = validate_interpretation([_no_op(0), _no_op(5)], num_notes=1)
    assert not result.valid
    assert any("out of range" in e for e in result.errors)


def test_unsupported_directive_type_is_rejected():
    entries = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "set_tariff_to_zero",
            "structured_adjustment": {"hours": [1]},
            "explanation": "hallucinated",
        }
    ]
    result = validate_interpretation(entries, num_notes=1)
    assert not result.valid
    assert any("unsupported" in e for e in result.errors)


def test_non_ascending_hours_rejected():
    entries = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "no_charge_window",
            "structured_adjustment": {"hours": [4, 2, 3]},
            "explanation": "bad order",
        }
    ]
    result = validate_interpretation(entries, num_notes=1)
    assert not result.valid


def test_duplicate_hours_rejected():
    entries = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "no_charge_window",
            "structured_adjustment": {"hours": [2, 2, 3]},
            "explanation": "dup",
        }
    ]
    result = validate_interpretation(entries, num_notes=1)
    assert not result.valid


def test_hour_out_of_range_rejected():
    entries = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "no_charge_window",
            "structured_adjustment": {"hours": [24]},
            "explanation": "bad",
        }
    ]
    result = validate_interpretation(entries, num_notes=1)
    assert not result.valid


def test_solar_factor_out_of_bounds_rejected():
    entries = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [1], "factor": 1.5},
            "explanation": "bad",
        }
    ]
    result = validate_interpretation(entries, num_notes=1)
    assert not result.valid


def test_negative_reserve_rejected():
    entries = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "minimum_battery_reserve",
            "structured_adjustment": {"hours": [1], "minimum_energy_kwh": -5},
            "explanation": "bad",
        }
    ]
    result = validate_interpretation(entries, num_notes=1)
    assert not result.valid


def test_reserve_exceeding_capacity_rejected():
    entries = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "minimum_battery_reserve",
            "structured_adjustment": {"hours": [1], "minimum_energy_kwh": 500},
            "explanation": "too much",
        }
    ]
    result = validate_interpretation(entries, num_notes=1, battery_capacity_kwh=200)
    assert not result.valid
    assert any("exceeds battery capacity" in e for e in result.errors)


def test_negative_max_grid_rejected():
    entries = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "max_grid_window",
            "structured_adjustment": {"hours": [1], "max_grid_kwh": -1},
            "explanation": "bad",
        }
    ]
    result = validate_interpretation(entries, num_notes=1)
    assert not result.valid


def test_no_op_with_nonnull_adjustment_rejected():
    entries = [
        {
            "note_index": 0,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": {"hours": [1]},
            "explanation": "bad",
        }
    ]
    result = validate_interpretation(entries, num_notes=1)
    assert not result.valid


def test_relevant_directive_missing_adjustment_rejected():
    entries = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "no_charge_window",
            "structured_adjustment": None,
            "explanation": "bad",
        }
    ]
    result = validate_interpretation(entries, num_notes=1)
    assert not result.valid


def test_applies_false_on_non_no_op_rejected():
    entries = [
        {
            "note_index": 0,
            "applies": False,
            "directive_type": "no_charge_window",
            "structured_adjustment": {"hours": [1]},
            "explanation": "bad",
        }
    ]
    result = validate_interpretation(entries, num_notes=1)
    assert not result.valid


def test_hallucinated_extra_field_rejected():
    entries = [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {
                "hours": [1],
                "factor": 0.5,
                "battery_capacity_kwh": 999,  # invented scenario data
            },
            "explanation": "bad",
        }
    ]
    result = validate_interpretation(entries, num_notes=1)
    assert not result.valid


def test_malformed_entry_does_not_crash():
    result = validate_interpretation(["not-a-dict"], num_notes=1)  # type: ignore[list-item]
    assert not result.valid
    assert result.errors


def test_non_list_input_does_not_crash():
    result = validate_interpretation({"oops": True}, num_notes=1)  # type: ignore[arg-type]
    assert not result.valid
