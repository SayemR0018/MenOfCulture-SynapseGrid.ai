"""Turns validated directives + the base scenario into the plain per-hour
arrays the optimizer's LP consumes.

This module is the ONLY place operator directives touch the math, and it
only accepts already-validated ``DirectiveInterpretation`` objects -- see
``ml/validator.py``. It never talks to the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.schemas import BatteryConfig, HourEntry
from ml.schemas import DirectiveInterpretation, DirectiveType

HOURS_PER_DAY = 24


@dataclass
class ScenarioMath:
    demand: list[float]
    tariff: list[float]
    effective_solar: list[float]
    min_reserve: list[float]
    max_grid: list[float | None]
    charge_blocked: list[bool]
    discharge_blocked: list[bool]
    max_charge_per_hour: list[float]
    max_discharge_per_hour: list[float]
    capacity: float
    initial_energy: float


def build_scenario_math(
    hours: list[HourEntry],
    battery: BatteryConfig,
    directives: list[DirectiveInterpretation],
) -> ScenarioMath:
    demand = [0.0] * HOURS_PER_DAY
    tariff = [0.0] * HOURS_PER_DAY
    base_solar = [0.0] * HOURS_PER_DAY
    for h in hours:
        demand[h.hour] = h.demand_kwh
        tariff[h.hour] = h.tariff_bdt_per_kwh
        base_solar[h.hour] = h.solar_kwh

    effective_solar = list(base_solar)
    min_reserve = [battery.minimum_energy_kwh] * HOURS_PER_DAY
    max_grid: list[float | None] = [None] * HOURS_PER_DAY
    charge_blocked = [False] * HOURS_PER_DAY
    discharge_blocked = [False] * HOURS_PER_DAY

    for d in directives:
        if not d.applies or d.directive_type == DirectiveType.NO_OP:
            continue
        adj = d.structured_adjustment or {}
        directive_hours = adj.get("hours", [])

        if d.directive_type == DirectiveType.SOLAR_REDUCTION:
            factor = adj["factor"]
            for h in directive_hours:
                # Multiply cumulatively so overlapping solar_reduction
                # directives on the same hour compound (0.5 and 0.5 -> 0.25),
                # rather than the later directive overwriting the earlier
                # one's reduction.
                effective_solar[h] *= factor

        elif d.directive_type == DirectiveType.MINIMUM_BATTERY_RESERVE:
            level = adj["minimum_energy_kwh"]
            for h in directive_hours:
                min_reserve[h] = max(min_reserve[h], level)

        elif d.directive_type == DirectiveType.NO_CHARGE_WINDOW:
            for h in directive_hours:
                charge_blocked[h] = True

        elif d.directive_type == DirectiveType.NO_DISCHARGE_WINDOW:
            for h in directive_hours:
                discharge_blocked[h] = True

        elif d.directive_type == DirectiveType.MAX_GRID_WINDOW:
            cap = adj["max_grid_kwh"]
            for h in directive_hours:
                max_grid[h] = cap if max_grid[h] is None else min(max_grid[h], cap)

    max_charge_per_hour = [
        0.0 if charge_blocked[h] else battery.max_charge_kwh_per_hour
        for h in range(HOURS_PER_DAY)
    ]
    max_discharge_per_hour = [
        0.0 if discharge_blocked[h] else battery.max_discharge_kwh_per_hour
        for h in range(HOURS_PER_DAY)
    ]

    return ScenarioMath(
        demand=demand,
        tariff=tariff,
        effective_solar=effective_solar,
        min_reserve=min_reserve,
        max_grid=max_grid,
        charge_blocked=charge_blocked,
        discharge_blocked=discharge_blocked,
        max_charge_per_hour=max_charge_per_hour,
        max_discharge_per_hour=max_discharge_per_hour,
        capacity=battery.capacity_kwh,
        initial_energy=battery.initial_energy_kwh,
    )
