"""Final validator: replays the optimizer's output hour-by-hour to confirm
every GridWise rule and every applied directive actually holds.

This mirrors what the hidden judge does (Section 11) and sits at the "Final
Validator" stage of the pipeline diagram, after the optimizer and before the
HTTP response. It is a safety net against optimizer bugs/numerical drift --
if it ever finds a violation, the API layer treats that as an internal
error rather than returning an invalid schedule.
"""
from __future__ import annotations

from app.schemas import BatteryConfig, HourEntry, HourlyPlanEntry
from ml.schemas import DirectiveInterpretation
from optimizer.constraints import HOURS_PER_DAY, build_scenario_math

_TOL = 0.01


def replay_and_validate(
    hours: list[HourEntry],
    battery: BatteryConfig,
    directives: list[DirectiveInterpretation],
    plan: list[HourlyPlanEntry],
) -> list[str]:
    errors: list[str] = []

    if len(plan) != HOURS_PER_DAY:
        return [f"hourly_plan must have exactly {HOURS_PER_DAY} entries, got {len(plan)}"]

    plan_hours = sorted(p.hour for p in plan)
    if plan_hours != list(range(HOURS_PER_DAY)):
        return ["hourly_plan must contain exactly one entry for each hour 0-23"]

    by_hour = {p.hour: p for p in plan}
    m = build_scenario_math(hours, battery, directives)

    prev_energy = m.initial_energy
    for h in range(HOURS_PER_DAY):
        p = by_hour[h]

        for field_name, value in (
            ("grid_kwh", p.grid_kwh),
            ("solar_used_kwh", p.solar_used_kwh),
            ("battery_kwh", p.battery_kwh),
        ):
            if value != value or value in (float("inf"), float("-inf")):
                errors.append(f"hour {h}: {field_name} is not finite")
            elif value < -_TOL:
                errors.append(f"hour {h}: {field_name} is negative ({value})")

        if p.battery_action == "idle" and abs(p.battery_kwh) > _TOL:
            errors.append(f"hour {h}: battery_action idle but battery_kwh={p.battery_kwh}")

        charge = p.battery_kwh if p.battery_action == "charge" else 0.0
        discharge = p.battery_kwh if p.battery_action == "discharge" else 0.0

        if charge > m.max_charge_per_hour[h] + _TOL:
            errors.append(
                f"hour {h}: charge {charge} exceeds allowed {m.max_charge_per_hour[h]}"
            )
        if discharge > m.max_discharge_per_hour[h] + _TOL:
            errors.append(
                f"hour {h}: discharge {discharge} exceeds allowed {m.max_discharge_per_hour[h]}"
            )

        if p.solar_used_kwh > m.effective_solar[h] + _TOL:
            errors.append(
                f"hour {h}: solar_used_kwh {p.solar_used_kwh} exceeds effective solar "
                f"{m.effective_solar[h]}"
            )

        if m.max_grid[h] is not None and p.grid_kwh > m.max_grid[h] + _TOL:
            errors.append(
                f"hour {h}: grid_kwh {p.grid_kwh} exceeds directive cap {m.max_grid[h]}"
            )

        expected_energy = prev_energy + charge - discharge
        if abs(expected_energy - p.battery_energy_after_kwh) > _TOL:
            errors.append(
                f"hour {h}: battery_energy_after_kwh {p.battery_energy_after_kwh} does not "
                f"match expected {expected_energy}"
            )

        if p.battery_energy_after_kwh < m.min_reserve[h] - _TOL:
            errors.append(
                f"hour {h}: battery_energy_after_kwh {p.battery_energy_after_kwh} below "
                f"required reserve {m.min_reserve[h]}"
            )
        if p.battery_energy_after_kwh > m.capacity + _TOL:
            errors.append(
                f"hour {h}: battery_energy_after_kwh {p.battery_energy_after_kwh} exceeds "
                f"capacity {m.capacity}"
            )

        balance_lhs = p.grid_kwh + p.solar_used_kwh + discharge
        balance_rhs = m.demand[h] + charge
        if abs(balance_lhs - balance_rhs) > _TOL:
            errors.append(
                f"hour {h}: energy balance violated ({balance_lhs} != {balance_rhs})"
            )

        prev_energy = p.battery_energy_after_kwh

    final_energy = by_hour[HOURS_PER_DAY - 1].battery_energy_after_kwh
    if abs(final_energy - m.initial_energy) > _TOL:
        errors.append(
            f"end-of-day battery neutrality violated: final={final_energy} "
            f"initial={m.initial_energy}"
        )

    return errors
