"""Pydantic schemas for the SynapseGrid.ai LLM interpretation layer.

These models define the strict contract between the LLM output and the
deterministic validator/optimizer. Nothing downstream should ever consume
raw LLM JSON directly -- it must first be coerced into these models.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DirectiveType(str, Enum):
    SOLAR_REDUCTION = "solar_reduction"
    MINIMUM_BATTERY_RESERVE = "minimum_battery_reserve"
    NO_CHARGE_WINDOW = "no_charge_window"
    NO_DISCHARGE_WINDOW = "no_discharge_window"
    MAX_GRID_WINDOW = "max_grid_window"
    NO_OP = "no_op"


def _validate_hours(hours: list[int]) -> list[int]:
    if len(hours) == 0:
        raise ValueError("hours must not be empty")
    for h in hours:
        if not isinstance(h, int) or isinstance(h, bool):
            raise ValueError(f"hour {h!r} is not an integer")
        if h < 0 or h > 23:
            raise ValueError(f"hour {h} out of range 0-23")
    if len(set(hours)) != len(hours):
        raise ValueError("hours must be unique")
    if hours != sorted(hours):
        raise ValueError("hours must be ascending")
    return hours


class _HoursMixin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hours: list[int]

    @field_validator("hours")
    @classmethod
    def _check_hours(cls, v: list[int]) -> list[int]:
        return _validate_hours(v)


class SolarReductionAdjustment(_HoursMixin):
    factor: float = Field(..., ge=0.0, le=1.0)


class MinimumBatteryReserveAdjustment(_HoursMixin):
    minimum_energy_kwh: float = Field(..., ge=0.0)

    @field_validator("minimum_energy_kwh")
    @classmethod
    def _finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("minimum_energy_kwh must be finite")
        return v


class NoChargeWindowAdjustment(_HoursMixin):
    pass


class NoDischargeWindowAdjustment(_HoursMixin):
    pass


class MaxGridWindowAdjustment(_HoursMixin):
    max_grid_kwh: float = Field(..., ge=0.0)

    @field_validator("max_grid_kwh")
    @classmethod
    def _finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("max_grid_kwh must be finite")
        return v


StructuredAdjustment = Union[
    SolarReductionAdjustment,
    MinimumBatteryReserveAdjustment,
    NoChargeWindowAdjustment,
    NoDischargeWindowAdjustment,
    MaxGridWindowAdjustment,
]

ADJUSTMENT_MODEL_BY_DIRECTIVE: dict[DirectiveType, type[BaseModel]] = {
    DirectiveType.SOLAR_REDUCTION: SolarReductionAdjustment,
    DirectiveType.MINIMUM_BATTERY_RESERVE: MinimumBatteryReserveAdjustment,
    DirectiveType.NO_CHARGE_WINDOW: NoChargeWindowAdjustment,
    DirectiveType.NO_DISCHARGE_WINDOW: NoDischargeWindowAdjustment,
    DirectiveType.MAX_GRID_WINDOW: MaxGridWindowAdjustment,
}


class DirectiveInterpretation(BaseModel):
    """One machine-checkable interpretation entry for one operator note."""

    model_config = ConfigDict(extra="forbid")

    note_index: int = Field(..., ge=0)
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[dict] = None
    explanation: str = ""

    @field_validator("explanation")
    @classmethod
    def _default_explanation(cls, v: Optional[str]) -> str:
        return v or ""


class RawLLMInterpretationEntry(BaseModel):
    """Loosely-typed shape used to parse whatever the LLM returns before
    strict validation runs. Extra fields are ignored rather than rejected
    here -- the deterministic validator is responsible for rejecting bad
    output, not this parsing step, so a slightly noisy LLM response does not
    crash the service.
    """

    model_config = ConfigDict(extra="ignore")

    note_index: Optional[int] = None
    applies: Optional[bool] = None
    directive_type: Optional[str] = None
    structured_adjustment: Optional[dict] = None
    explanation: Optional[str] = None


class RawLLMResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    directive_interpretation: list[RawLLMInterpretationEntry] = Field(
        default_factory=list
    )


# --------------------------------------------------------------------------
# Scenario context (subset of the request forwarded to the LLM / used by the
# validator -- never the full request, and never mutated by the LLM).
# --------------------------------------------------------------------------


class BatteryContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    capacity_kwh: float
    initial_energy_kwh: Optional[float] = None
    minimum_energy_kwh: Optional[float] = None
    max_charge_kwh_per_hour: Optional[float] = None
    max_discharge_kwh_per_hour: Optional[float] = None


class ScenarioContext(BaseModel):
    """Minimal scenario information the LLM is allowed to see, e.g. battery
    capacity so it can convert "50% reserve" into kWh. Demand, solar, and
    tariff are intentionally excluded from what gets embedded in the prompt
    as facts the LLM should reason over numerically.
    """

    model_config = ConfigDict(extra="ignore")

    battery: Optional[BatteryContext] = None
