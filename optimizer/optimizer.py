"""The deterministic math optimizer.

This is the ONLY place that decides grid_kwh, solar_used_kwh, battery
charge/discharge, and cost. The LLM never touches this module. Directives
already validated by ``ml/validator.py`` are turned into per-hour arrays by
``optimizer/constraints.py`` and then fed into a linear program that
minimizes total grid cost subject to the GridWise energy/battery rules
(Section 09 of the problem statement).

Variables per hour h (0..23):
    grid[h]        >= 0                          grid energy purchased
    solar_used[h]  in [0, effective_solar[h]]     solar energy consumed
    charge[h]      in [0, max_charge[h]]          battery charge amount
    discharge[h]   in [0, max_discharge[h]]       battery discharge amount
    E[h]           in [min_reserve[h], capacity]  battery energy after hour h

Constraints:
    energy balance:      grid[h] + solar_used[h] + discharge[h]
                          - charge[h] = demand[h]
    battery recursion:   E[h] = E[h-1] + charge[h] - discharge[h]
                          (E[-1] = initial_energy_kwh)
    end-of-day neutrality: E[23] = initial_energy_kwh

Objective: minimize sum(grid[h] * tariff[h])

All constraints here are either simple bounds or linear equalities, so the
problem is solved exactly (to LP optimality / numerical tolerance) with
``scipy.optimize.linprog`` (HiGHS).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from app.schemas import BatteryConfig, HourEntry, HourlyPlanEntry
from ml.schemas import DirectiveInterpretation
from optimizer.constraints import HOURS_PER_DAY, ScenarioMath, build_scenario_math

_EPS = 1e-6


class OptimizerInfeasibleError(Exception):
    """Raised when the LP has no feasible solution.

    Per the problem statement, organizer-valid scenarios are always
    feasible; this is a controlled failure path for malformed/contradictory
    input rather than an expected outcome.
    """


def _var_index(block: int, hour: int) -> int:
    return block * HOURS_PER_DAY + hour


# Block indices.
_GRID, _SOLAR, _CHARGE, _DISCHARGE, _ENERGY = range(5)
_N_VARS = 5 * HOURS_PER_DAY


def _build_lp(m: ScenarioMath):
    c = np.zeros(_N_VARS)
    for h in range(HOURS_PER_DAY):
        c[_var_index(_GRID, h)] = m.tariff[h]

    bounds = [(0.0, None)] * _N_VARS
    for h in range(HOURS_PER_DAY):
        grid_upper = m.max_grid[h]
        bounds[_var_index(_GRID, h)] = (0.0, grid_upper)
        bounds[_var_index(_SOLAR, h)] = (0.0, max(m.effective_solar[h], 0.0))
        bounds[_var_index(_CHARGE, h)] = (0.0, max(m.max_charge_per_hour[h], 0.0))
        bounds[_var_index(_DISCHARGE, h)] = (0.0, max(m.max_discharge_per_hour[h], 0.0))
        bounds[_var_index(_ENERGY, h)] = (m.min_reserve[h], m.capacity)

    a_eq_rows: list[np.ndarray] = []
    b_eq: list[float] = []

    # (1) energy balance per hour.
    for h in range(HOURS_PER_DAY):
        row = np.zeros(_N_VARS)
        row[_var_index(_GRID, h)] = 1.0
        row[_var_index(_SOLAR, h)] = 1.0
        row[_var_index(_DISCHARGE, h)] = 1.0
        row[_var_index(_CHARGE, h)] = -1.0
        a_eq_rows.append(row)
        b_eq.append(m.demand[h])

    # (2) battery recursion.
    for h in range(HOURS_PER_DAY):
        row = np.zeros(_N_VARS)
        row[_var_index(_ENERGY, h)] = 1.0
        row[_var_index(_CHARGE, h)] = -1.0
        row[_var_index(_DISCHARGE, h)] = 1.0
        if h == 0:
            a_eq_rows.append(row)
            b_eq.append(m.initial_energy)
        else:
            row[_var_index(_ENERGY, h - 1)] = -1.0
            a_eq_rows.append(row)
            b_eq.append(0.0)

    # (3) end-of-day neutrality.
    row = np.zeros(_N_VARS)
    row[_var_index(_ENERGY, HOURS_PER_DAY - 1)] = 1.0
    a_eq_rows.append(row)
    b_eq.append(m.initial_energy)

    a_eq = np.vstack(a_eq_rows)
    b_eq_arr = np.array(b_eq)
    return c, a_eq, b_eq_arr, bounds


def optimize_schedule(
    hours: list[HourEntry],
    battery: BatteryConfig,
    directives: list[DirectiveInterpretation],
) -> list[HourlyPlanEntry]:
    m = build_scenario_math(hours, battery, directives)
    c, a_eq, b_eq, bounds = _build_lp(m)

    result = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not result.success:
        raise OptimizerInfeasibleError(
            f"No feasible schedule found: {result.message}"
        )

    x = result.x
    plan: list[HourlyPlanEntry] = []
    for h in range(HOURS_PER_DAY):
        grid = max(float(x[_var_index(_GRID, h)]), 0.0)
        solar_used = max(float(x[_var_index(_SOLAR, h)]), 0.0)
        charge = float(x[_var_index(_CHARGE, h)])
        discharge = float(x[_var_index(_DISCHARGE, h)])
        energy_after = float(x[_var_index(_ENERGY, h)])

        # Net out numerical noise where the LP assigns tiny simultaneous
        # charge+discharge; the net charge/discharge amount and every
        # resulting constraint (balance, recursion) is preserved exactly.
        net = charge - discharge
        if net > _EPS:
            action = "charge"
            magnitude = net
        elif net < -_EPS:
            action = "discharge"
            magnitude = -net
        else:
            action = "idle"
            magnitude = 0.0

        plan.append(
            HourlyPlanEntry(
                hour=h,
                grid_kwh=round(grid, 6),
                solar_used_kwh=round(solar_used, 6),
                battery_action=action,
                battery_kwh=round(magnitude, 6),
                battery_energy_after_kwh=round(energy_after, 6),
            )
        )

    return plan
