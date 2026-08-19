from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Forecast:
    point: list[float]
    lower: list[float]
    upper: list[float]
    method: str


def seasonal_conformal(prices: list[float], horizon: int, alpha: float = 0.1, season: int = 288) -> Forecast:
    """Leakage-safe seasonal baseline with split-conformal absolute residual interval."""
    if len(prices) < max(24, horizon + 2):
        raise ValueError("insufficient history")
    values = np.asarray(prices, dtype=float)
    point = [
        float(values[-season + i]) if len(values) >= season else float(np.median(values[-12:])) for i in range(horizon)
    ]
    if len(values) > season:
        residuals = np.abs(values[season:] - values[:-season])
    else:
        residuals = np.abs(values[1:] - values[:-1])
    q_level = min(1.0, math.ceil((len(residuals) + 1) * (1 - alpha)) / len(residuals))
    q = float(np.quantile(residuals, q_level, method="higher"))
    return Forecast(point, [p - q for p in point], [p + q for p in point], "seasonal_split_conformal")
