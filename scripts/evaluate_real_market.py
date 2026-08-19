from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from energy_agent.battery import DispatchResult, optimize_dispatch
from energy_agent.schemas import BatterySpec


def load(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["intervention"] != "0":
                continue
            grouped[raw["region"]].append(
                {
                    "interval": datetime.strptime(raw["interval"], "%Y/%m/%d %H:%M:%S").replace(tzinfo=UTC),
                    "rrp": float(raw["rrp"]),
                    "demand": float(raw["total_demand_mw"] or "nan"),
                    "available": float(raw["available_generation_mw"] or "nan"),
                    "interchange": float(raw["net_interchange_mw"] or "nan"),
                }
            )
    for rows in grouped.values():
        rows.sort(key=lambda row: row["interval"])
    return grouped


def features(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[datetime]]:
    price = np.asarray([row["rrp"] for row in rows], dtype=float)
    demand = np.asarray([row["demand"] for row in rows], dtype=float)
    available = np.asarray([row["available"] for row in rows], dtype=float)
    interchange = np.asarray([row["interchange"] for row in rows], dtype=float)
    times = [row["interval"] for row in rows]
    output: list[list[float]] = []
    labels: list[float] = []
    output_times: list[datetime] = []
    for i in range(288, len(rows)):
        hour = times[i].hour + times[i].minute / 60
        output.append(
            [
                price[i - 1],
                price[i - 12],
                price[i - 288],
                float(np.mean(price[i - 12 : i])),
                float(np.std(price[i - 12 : i])),
                demand[i - 1],
                available[i - 1],
                interchange[i - 1],
                math.sin(2 * math.pi * hour / 24),
                math.cos(2 * math.pi * hour / 24),
                float(times[i].weekday()),
            ]
        )
        labels.append(price[i])
        output_times.append(times[i])
    matrix = np.nan_to_num(np.asarray(output), nan=0.0)
    return matrix, np.asarray(labels), output_times


def point_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(predicted - actual)),
    }


def interval_metrics(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    return {
        "coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "mean_width": float(np.mean(upper - lower)),
    }


def _realized(schedule: DispatchResult, prices: list[float]) -> float:
    interval_hours = 5 / 60
    return float(
        sum(
            (discharge - charge) * price * interval_hours
            for charge, discharge, price in zip(schedule.charge_mw, schedule.discharge_mw, prices, strict=True)
        )
    )


def threshold_dispatch(prices: list[float], spec: BatterySpec, low: float, high: float) -> DispatchResult:
    eta = math.sqrt(spec.round_trip_efficiency)
    dt = 5 / 60
    soc = spec.initial_soc_fraction * spec.energy_mwh
    minimum = spec.min_soc_fraction * spec.energy_mwh
    maximum = spec.max_soc_fraction * spec.energy_mwh
    charge: list[float] = []
    discharge: list[float] = []
    states = [soc]
    for index, price in enumerate(prices):
        intervals_left = len(prices) - index
        target = spec.terminal_soc_fraction * spec.energy_mwh
        mandatory_charge = max(0.0, (target - soc) / (eta * dt * intervals_left))
        mandatory_discharge = max(0.0, (soc - target) * eta / (dt * intervals_left))
        c = min(spec.power_mw, (maximum - soc) / (eta * dt)) if price <= low else 0.0
        d = min(spec.power_mw, (soc - minimum) * eta / dt) if price >= high else 0.0
        if intervals_left <= 12:
            if soc < target:
                c, d = max(c, mandatory_charge), 0.0
            elif soc > target:
                c, d = 0.0, max(d, mandatory_discharge)
        soc += eta * c * dt - d * dt / eta
        charge.append(c)
        discharge.append(d)
        states.append(soc)
    margin = sum((d - c) * p * dt for c, d, p in zip(charge, discharge, prices, strict=True))
    throughput = sum((c + d) * dt for c, d in zip(charge, discharge, strict=True))
    return DispatchResult(charge, discharge, states, margin, throughput / (2 * spec.energy_mwh), 0.0, 0.0)


def bootstrap_mean(values: list[float], seed: int = 20260820, samples: int = 1000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    data = np.asarray(values)
    means = np.mean(rng.choice(data, size=(samples, len(data)), replace=True), axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def evaluate_region(region: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from lightgbm import LGBMRegressor

    x, y, times = features(rows)
    n = len(y)
    train_end = int(n * 0.70)
    calibration_end = int(n * 0.85)
    x_train, y_train = x[:train_end], y[:train_end]
    x_cal, y_cal = x[train_end:calibration_end], y[train_end:calibration_end]
    x_test, y_test = x[calibration_end:], y[calibration_end:]
    test_times = times[calibration_end:]
    common: dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 20260820,
        "verbosity": -1,
        "n_jobs": 2,
    }
    point_model = LGBMRegressor(objective="regression_l1", **common).fit(x_train, y_train)
    lower_model = LGBMRegressor(objective="quantile", alpha=0.1, **common).fit(x_train, y_train)
    upper_model = LGBMRegressor(objective="quantile", alpha=0.9, **common).fit(x_train, y_train)
    point = np.asarray(point_model.predict(x_test), dtype=float)
    seasonal = x_test[:, 2]
    persistence = x_test[:, 0]
    cal_point = np.asarray(point_model.predict(x_cal), dtype=float)
    absolute_q = float(np.quantile(np.abs(y_cal - cal_point), 0.9, method="higher"))
    fixed_conformal_lower, fixed_conformal_upper = point - absolute_q, point + absolute_q
    q_lower = np.asarray(lower_model.predict(x_test), dtype=float)
    q_upper = np.asarray(upper_model.predict(x_test), dtype=float)
    cal_lower = np.asarray(lower_model.predict(x_cal), dtype=float)
    cal_upper = np.asarray(upper_model.predict(x_cal), dtype=float)
    correction = float(np.quantile(np.maximum(cal_lower - y_cal, y_cal - cal_upper), 0.9, method="higher"))
    fixed_quantile_lower, fixed_quantile_upper = q_lower - max(0, correction), q_upper + max(0, correction)
    # Adaptive conformal uses only calibration residuals and earlier observed test
    # residuals. A 30-day window responds to regime shifts without future leakage.
    residual_window = 30 * 288
    absolute_history = list(np.abs(y_cal - cal_point)[-residual_window:])
    quantile_history = list(np.maximum(cal_lower - y_cal, y_cal - cal_upper)[-residual_window:])
    rolling_lower: list[float] = []
    rolling_upper: list[float] = []
    rolling_q_lower: list[float] = []
    rolling_q_upper: list[float] = []
    for actual, predicted, lower, upper in zip(y_test, point, q_lower, q_upper, strict=True):
        absolute_q = float(np.quantile(absolute_history, 0.9, method="higher"))
        correction = max(0.0, float(np.quantile(quantile_history, 0.9, method="higher")))
        rolling_lower.append(float(predicted - absolute_q))
        rolling_upper.append(float(predicted + absolute_q))
        rolling_q_lower.append(float(lower - correction))
        rolling_q_upper.append(float(upper + correction))
        absolute_history.append(abs(float(actual - predicted)))
        quantile_history.append(max(float(lower - actual), float(actual - upper)))
        absolute_history = absolute_history[-residual_window:]
        quantile_history = quantile_history[-residual_window:]
    # The LightGBM point forecast drives both operational policies. Realized
    # prices are reserved for settlement and the perfect-foresight oracle.
    spec = BatterySpec()
    train_prices = [row["rrp"] for row in rows[: 288 + train_end]]
    low, high = float(np.quantile(train_prices, 0.25)), float(np.quantile(train_prices, 0.75))
    by_day: dict[str, list[float]] = defaultdict(list)
    forecast_by_day: dict[str, list[float]] = defaultdict(list)
    for timestamp, price, predicted in zip(test_times, y_test, point, strict=True):
        by_day[timestamp.date().isoformat()].append(float(price))
        forecast_by_day[timestamp.date().isoformat()].append(float(predicted))
    daily: list[dict[str, float | str]] = []
    for day, actual in sorted(by_day.items()):
        forecast = forecast_by_day[day]
        if len(actual) != 288 or len(forecast) != 288:
            continue
        forecast_schedule = optimize_dispatch(forecast, spec)
        oracle = optimize_dispatch(actual, spec)
        rule_schedule = threshold_dispatch(forecast, spec, low, high)
        daily.append(
            {
                "day": day,
                "forecast_margin": _realized(forecast_schedule, actual),
                "rule_margin": _realized(rule_schedule, actual),
                "oracle_margin": oracle.gross_margin_aud,
                "forecast_cycles": forecast_schedule.equivalent_full_cycles,
            }
        )
    forecast_margins = [float(row["forecast_margin"]) for row in daily]
    rule_margins = [float(row["rule_margin"]) for row in daily]
    oracle_margins = [float(row["oracle_margin"]) for row in daily]
    forecast_total = sum(forecast_margins)
    rule_total = sum(rule_margins)
    oracle_total = sum(oracle_margins)
    annualizer = 365 / len(daily) if daily else 0
    prices_all = np.asarray([row["rrp"] for row in rows])
    median = float(np.median(prices_all))
    mad = float(np.median(np.abs(prices_all - median))) or 1.0
    anomaly_counts = {str(z): int(np.sum(np.abs(0.6745 * (prices_all - median) / mad) >= z)) for z in (4, 5, 6)}
    return {
        "region": region,
        "rows": len(rows),
        "split": {"train": len(y_train), "calibration": len(y_cal), "test": len(y_test)},
        "point": {
            "persistence": point_metrics(y_test, persistence),
            "seasonal": point_metrics(y_test, seasonal),
            "lightgbm": point_metrics(y_test, point),
        },
        "interval": {
            "fixed_conformal_90_ablation": interval_metrics(y_test, fixed_conformal_lower, fixed_conformal_upper),
            "rolling_conformal_90": interval_metrics(y_test, np.asarray(rolling_lower), np.asarray(rolling_upper)),
            "fixed_quantile_conformal_80_ablation": interval_metrics(
                y_test, fixed_quantile_lower, fixed_quantile_upper
            ),
            "rolling_quantile_conformal_80": interval_metrics(
                y_test, np.asarray(rolling_q_lower), np.asarray(rolling_q_upper)
            ),
        },
        "anomaly_stability": {"robust_z_counts": anomaly_counts, "rrp_ge_5000": int(np.sum(prices_all >= 5000))},
        "bess": {
            "test_days": len(daily),
            "gross_spot_margin_aud": forecast_total,
            "aud_per_mw_year": forecast_total * annualizer / spec.power_mw,
            "equivalent_full_cycles": sum(float(row["forecast_cycles"]) for row in daily),
            "rule_margin_aud": rule_total,
            "relative_rule_lift": (forecast_total - rule_total) / abs(rule_total) if rule_total else None,
            "oracle_margin_aud": oracle_total,
            "oracle_regret_aud": oracle_total - forecast_total,
            "daily_forecast_margin_mean_95_interval": bootstrap_mean(forecast_margins),
            "economic_boundary": "historical spot-market gross-margin proxy; excludes CAPEX, degradation, network fees, FCAS and investment returns",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    grouped = load(args.input)
    expected = 365 * 288
    coverage = {region: len(rows) / expected for region, rows in grouped.items()}
    if set(grouped) != {"NSW1", "QLD1", "SA1", "TAS1", "VIC1"} or min(coverage.values()) < 0.98:
        raise SystemExit(f"coverage gate failed: {coverage}")
    results = {region: evaluate_region(region, rows) for region, rows in sorted(grouped.items())}
    metrics = {"scope": "real AEMO NEMWeb rolling test", "coverage": coverage, "regions": results}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "provider_cost_usd": 0.0,
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
