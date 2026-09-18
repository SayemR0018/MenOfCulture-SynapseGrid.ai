"""API request/response schemas -- the canonical contract from Section 06-10
of the problem statement. These are intentionally separate from
``ml/schemas.py`` (the LLM directive contract) to keep the ML layer and the
API/optimizer layer decoupled.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ml.schemas import DirectiveInterpretation


class HourEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hour: int = Field(..., ge=0, le=23)
    demand_kwh: float = Field(..., ge=0)
    solar_kwh: float = Field(..., ge=0)
    tariff_bdt_per_kwh: float = Field(..., ge=0)


class BatteryConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    capacity_kwh: float = Field(..., gt=0)
    initial_energy_kwh: float = Field(..., ge=0)
    minimum_energy_kwh: float = Field(..., ge=0)
    max_charge_kwh_per_hour: float = Field(..., ge=0)
    max_discharge_kwh_per_hour: float = Field(..., ge=0)

    @model_validator(mode="after")
    def _check_bounds(self):
        # Note: initial_energy_kwh is intentionally NOT required to be >=
        # minimum_energy_kwh here -- the optimizer can legitimately charge
        # during hour 0 to bring the battery up to the reserve level; that
        # is a feasibility question for the LP, not a request-shape error.
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh cannot exceed capacity_kwh")
        if self.initial_energy_kwh > self.capacity_kwh:
            raise ValueError("initial_energy_kwh cannot exceed capacity_kwh")
        return self


class OptimizeEnergyRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scenario_id: str = Field(..., min_length=1)
    operator_notes: list[str] = Field(..., min_length=1, max_length=3)
    hours: list[HourEntry] = Field(..., min_length=24, max_length=24)
    battery: BatteryConfig

    @field_validator("operator_notes")
    @classmethod
    def _notes_non_empty(cls, v: list[str]) -> list[str]:
        for note in v:
            if not note or not note.strip():
                raise ValueError("operator_notes entries must be non-empty strings")
        return v

    @field_validator("hours")
    @classmethod
    def _hours_cover_0_23(cls, v: list[HourEntry]) -> list[HourEntry]:
        hour_values = sorted(h.hour for h in v)
        if hour_values != list(range(24)):
            raise ValueError("hours must contain exactly one entry for each hour 0-23")
        return sorted(v, key=lambda h: h.hour)


class HourlyPlanEntry(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: Literal["charge", "discharge", "idle"]
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeEnergyResponse(BaseModel):
    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation]
    hourly_plan: list[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
