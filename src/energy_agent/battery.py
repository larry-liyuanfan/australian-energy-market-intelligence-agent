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


def optimize_dispatch(prices: list[float], spec: BatterySpec, interval_hours: float = 5 / 60) -> DispatchResult:
    """Solve charge/discharge MILP with SoC bounds and a fixed terminal SoC."""
    started = time.perf_counter()
    n = len(prices)
    if n == 0:
        raise ValueError("prices must not be empty")
    eta = math.sqrt(spec.round_trip_efficiency)
    # Variables: charge[n], discharge[n], soc[n+1], binary_mode[n].
    size = 4 * n + 1
    c = np.zeros(size)
    c[:n] = np.asarray(prices) * interval_hours
    c[n : 2 * n] = -np.asarray(prices) * interval_hours
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
    spread_value = float(np.sum(np.maximum(np.asarray(prices), 0) * spec.power_mw * interval_hours))
    return DispatchResult(
        charge.tolist(),
        discharge.tolist(),
        soc.tolist(),
        margin,
        throughput / (2 * spec.energy_mwh),
        margin / spread_value if spread_value else 0.0,
        (time.perf_counter() - started) * 1000,
    )
