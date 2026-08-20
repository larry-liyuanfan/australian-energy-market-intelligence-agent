from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .schemas import BatterySpec


@dataclass(frozen=True)
class DispatchResult:
    charge_mw: list[float]
    discharge_mw: list[float]
    soc_mwh: list[float]
    gross_margin_aud: float
    equivalent_full_cycles: float
    capture_rate: float
    runtime_ms: float
    discharged_mwh: float = 0.0
    variable_degradation_cost_proxy_aud: float = 0.0
    net_operating_margin_proxy_aud: float = 0.0


@dataclass(frozen=True)
class RiskAwareDispatchResult:
    dispatch: DispatchResult
    scenario_net_margins_aud: list[float]
    expected_net_margin_aud: float
    lower_tail_cvar_aud: float
    alpha: float
    risk_aversion: float
    solver_mip_gap: float | None


def optimize_dispatch(
    prices: list[float],
    spec: BatterySpec,
    interval_hours: float = 5 / 60,
    *,
    variable_degradation_cost_aud_per_mwh_discharged: float = 0.0,
) -> DispatchResult:
    """Solve dispatch with SoC bounds and an optional variable cycling-cost proxy."""
    started = time.perf_counter()
    n = len(prices)
    if n == 0:
        raise ValueError("prices must not be empty")
    if variable_degradation_cost_aud_per_mwh_discharged < 0:
        raise ValueError("variable degradation cost must be non-negative")
    eta = math.sqrt(spec.round_trip_efficiency)
    # Variables: charge[n], discharge[n], soc[n+1], binary_mode[n].
    size = 4 * n + 1
    c = np.zeros(size)
    c[:n] = np.asarray(prices) * interval_hours
    c[n : 2 * n] = (
        -np.asarray(prices) + variable_degradation_cost_aud_per_mwh_discharged
    ) * interval_hours
    lower = np.zeros(size)
    upper = np.full(size, np.inf)
    upper[: 2 * n] = spec.power_mw
    lower[2 * n : 3 * n + 1] = spec.min_soc_fraction * spec.energy_mwh
    upper[2 * n : 3 * n + 1] = spec.max_soc_fraction * spec.energy_mwh
    upper[3 * n + 1 :] = 1
    lower[2 * n] = upper[2 * n] = spec.initial_soc_fraction * spec.energy_mwh
    lower[3 * n] = upper[3 * n] = spec.terminal_soc_fraction * spec.energy_mwh
    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []
    for t in range(n):
        row = np.zeros(size)
        row[2 * n + t + 1] = 1
        row[2 * n + t] = -1
        row[t] = -eta * interval_hours
        row[n + t] = interval_hours / eta
        rows.append(row)
        lb.append(0)
        ub.append(0)
        charge_mode = np.zeros(size)
        charge_mode[t] = 1
        charge_mode[3 * n + 1 + t] = -spec.power_mw
        rows.append(charge_mode)
        lb.append(-np.inf)
        ub.append(0)
        discharge_mode = np.zeros(size)
        discharge_mode[n + t] = 1
        discharge_mode[3 * n + 1 + t] = spec.power_mw
        rows.append(discharge_mode)
        lb.append(-np.inf)
        ub.append(spec.power_mw)
    integrality = np.zeros(size)
    integrality[3 * n + 1 :] = 1
    solved = milp(
        c, integrality=integrality, bounds=Bounds(lower, upper), constraints=LinearConstraint(np.vstack(rows), lb, ub)
    )
    if not solved.success or solved.x is None:
        raise RuntimeError(f"dispatch optimisation failed: {solved.message}")
    charge = solved.x[:n]
    discharge = solved.x[n : 2 * n]
    soc = solved.x[2 * n : 3 * n + 1]
    margin = float(np.sum((discharge - charge) * np.asarray(prices) * interval_hours))
    throughput = float(np.sum((charge + discharge) * interval_hours))
    discharged_mwh = float(np.sum(discharge * interval_hours))
    degradation_proxy = variable_degradation_cost_aud_per_mwh_discharged * discharged_mwh
    spread_value = float(np.sum(np.maximum(np.asarray(prices), 0) * spec.power_mw * interval_hours))
    return DispatchResult(
        charge.tolist(),
        discharge.tolist(),
        soc.tolist(),
        margin,
        throughput / (2 * spec.energy_mwh),
        margin / spread_value if spread_value else 0.0,
        (time.perf_counter() - started) * 1000,
        discharged_mwh,
        degradation_proxy,
        margin - degradation_proxy,
    )


def optimize_dispatch_cvar(
    price_scenarios: list[list[float]],
    spec: BatterySpec,
    interval_hours: float = 5 / 60,
    *,
    alpha: float = 0.8,
    risk_aversion: float = 0.5,
    variable_degradation_cost_aud_per_mwh_discharged: float = 0.0,
) -> RiskAwareDispatchResult:
    """Choose one dispatch against equally weighted price scenarios.

    The MILP maximises expected net operating margin plus ``risk_aversion``
    times the empirical lower-tail CVaR. All scenarios must be generated before
    the dispatch horizon; realised settlement prices are not accepted here.
    """

    started = time.perf_counter()
    if not price_scenarios or not price_scenarios[0]:
        raise ValueError("price_scenarios must not be empty")
    n = len(price_scenarios[0])
    if any(len(scenario) != n for scenario in price_scenarios):
        raise ValueError("all price scenarios must have identical horizons")
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1)")
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be non-negative")
    if variable_degradation_cost_aud_per_mwh_discharged < 0:
        raise ValueError("variable degradation cost must be non-negative")

    prices = np.asarray(price_scenarios, dtype=float)
    if not np.isfinite(prices).all():
        raise ValueError("price scenarios must be finite")
    scenario_count = len(price_scenarios)
    eta = math.sqrt(spec.round_trip_efficiency)
    base_size = 4 * n + 1
    z_index = base_size
    shortfall_start = z_index + 1
    size = base_size + 1 + scenario_count
    expected_price = np.mean(prices, axis=0)
    c = np.zeros(size)
    c[:n] = expected_price * interval_hours
    c[n : 2 * n] = (
        -expected_price + variable_degradation_cost_aud_per_mwh_discharged
    ) * interval_hours
    c[z_index] = -risk_aversion
    c[shortfall_start:] = risk_aversion / ((1 - alpha) * scenario_count)

    lower = np.zeros(size)
    upper = np.full(size, np.inf)
    upper[: 2 * n] = spec.power_mw
    lower[2 * n : 3 * n + 1] = spec.min_soc_fraction * spec.energy_mwh
    upper[2 * n : 3 * n + 1] = spec.max_soc_fraction * spec.energy_mwh
    upper[3 * n + 1 : base_size] = 1
    lower[2 * n] = upper[2 * n] = spec.initial_soc_fraction * spec.energy_mwh
    lower[3 * n] = upper[3 * n] = spec.terminal_soc_fraction * spec.energy_mwh
    lower[z_index] = -np.inf

    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []
    for t in range(n):
        balance = np.zeros(size)
        balance[2 * n + t + 1] = 1
        balance[2 * n + t] = -1
        balance[t] = -eta * interval_hours
        balance[n + t] = interval_hours / eta
        rows.append(balance)
        lb.append(0)
        ub.append(0)
        charge_mode = np.zeros(size)
        charge_mode[t] = 1
        charge_mode[3 * n + 1 + t] = -spec.power_mw
        rows.append(charge_mode)
        lb.append(-np.inf)
        ub.append(0)
        discharge_mode = np.zeros(size)
        discharge_mode[n + t] = 1
        discharge_mode[3 * n + 1 + t] = spec.power_mw
        rows.append(discharge_mode)
        lb.append(-np.inf)
        ub.append(spec.power_mw)
    for scenario_index, scenario_prices in enumerate(prices):
        # z - scenario_margin - u_s <= 0
        tail = np.zeros(size)
        tail[:n] = scenario_prices * interval_hours
        tail[n : 2 * n] = (
            -scenario_prices + variable_degradation_cost_aud_per_mwh_discharged
        ) * interval_hours
        tail[z_index] = 1
        tail[shortfall_start + scenario_index] = -1
        rows.append(tail)
        lb.append(-np.inf)
        ub.append(0)

    integrality = np.zeros(size)
    integrality[3 * n + 1 : base_size] = 1
    solved = milp(
        c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(np.vstack(rows), lb, ub),
    )
    if not solved.success or solved.x is None:
        raise RuntimeError(f"risk-aware dispatch optimisation failed: {solved.message}")
    charge = solved.x[:n]
    discharge = solved.x[n : 2 * n]
    soc = solved.x[2 * n : 3 * n + 1]
    discharged_mwh = float(np.sum(discharge) * interval_hours)
    degradation_proxy = variable_degradation_cost_aud_per_mwh_discharged * discharged_mwh
    scenario_margins = [
        float(np.sum((discharge - charge) * scenario * interval_hours) - degradation_proxy)
        for scenario in prices
    ]
    ordered = sorted(scenario_margins)
    tail_count = max(1, math.ceil((1 - alpha) * scenario_count))
    cvar = float(np.mean(ordered[:tail_count]))
    expected_margin = float(np.mean(scenario_margins))
    gross_expected = expected_margin + degradation_proxy
    throughput = float(np.sum((charge + discharge) * interval_hours))
    spread_value = float(
        np.sum(np.maximum(expected_price, 0) * spec.power_mw * interval_hours)
    )
    dispatch = DispatchResult(
        charge.tolist(),
        discharge.tolist(),
        soc.tolist(),
        gross_expected,
        throughput / (2 * spec.energy_mwh),
        gross_expected / spread_value if spread_value else 0.0,
        (time.perf_counter() - started) * 1000,
        discharged_mwh,
        degradation_proxy,
        expected_margin,
    )
    gap = getattr(solved, "mip_gap", None)
    return RiskAwareDispatchResult(
        dispatch=dispatch,
        scenario_net_margins_aud=scenario_margins,
        expected_net_margin_aud=expected_margin,
        lower_tail_cvar_aud=cvar,
        alpha=alpha,
        risk_aversion=risk_aversion,
        solver_mip_gap=None if gap is None else float(gap),
    )


def threshold_dispatch(
    prices: list[float],
    spec: BatterySpec,
    low: float,
    high: float,
    interval_hours: float = 5 / 60,
    *,
    variable_degradation_cost_aud_per_mwh_discharged: float = 0.0,
) -> DispatchResult:
    """Apply a deterministic threshold baseline while enforcing SoC and terminal state."""

    if not prices:
        raise ValueError("prices must not be empty")
    if low >= high:
        raise ValueError("low threshold must be below high threshold")
    if variable_degradation_cost_aud_per_mwh_discharged < 0:
        raise ValueError("variable degradation cost must be non-negative")
    started = time.perf_counter()
    eta = math.sqrt(spec.round_trip_efficiency)
    soc = spec.initial_soc_fraction * spec.energy_mwh
    minimum = spec.min_soc_fraction * spec.energy_mwh
    maximum = spec.max_soc_fraction * spec.energy_mwh
    charge: list[float] = []
    discharge: list[float] = []
    states = [soc]
    for index, price in enumerate(prices):
        intervals_left = len(prices) - index
        target = spec.terminal_soc_fraction * spec.energy_mwh
        mandatory_charge = max(0.0, (target - soc) / (eta * interval_hours * intervals_left))
        mandatory_discharge = max(0.0, (soc - target) * eta / (interval_hours * intervals_left))
        charging = (
            min(spec.power_mw, (maximum - soc) / (eta * interval_hours))
            if price <= low
            else 0.0
        )
        discharging = (
            min(spec.power_mw, (soc - minimum) * eta / interval_hours)
            if price >= high
            else 0.0
        )
        if intervals_left <= 12:
            if soc < target:
                charging, discharging = max(charging, mandatory_charge), 0.0
            elif soc > target:
                charging, discharging = 0.0, max(discharging, mandatory_discharge)
        soc += eta * charging * interval_hours - discharging * interval_hours / eta
        charge.append(charging)
        discharge.append(discharging)
        states.append(soc)
    margin = float(
        sum(
            (discharging - charging) * price * interval_hours
            for charging, discharging, price in zip(charge, discharge, prices, strict=True)
        )
    )
    throughput = float(
        sum(
            (charging + discharging) * interval_hours
            for charging, discharging in zip(charge, discharge, strict=True)
        )
    )
    discharged_mwh = float(sum(discharge) * interval_hours)
    degradation_proxy = variable_degradation_cost_aud_per_mwh_discharged * discharged_mwh
    return DispatchResult(
        charge,
        discharge,
        states,
        margin,
        throughput / (2 * spec.energy_mwh),
        0.0,
        (time.perf_counter() - started) * 1000,
        discharged_mwh,
        degradation_proxy,
        margin - degradation_proxy,
    )
