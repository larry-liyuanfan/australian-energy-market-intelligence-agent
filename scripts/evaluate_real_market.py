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

from energy_agent.battery import DispatchResult, optimize_dispatch, threshold_dispatch
from energy_agent.schemas import BatterySpec

DEGRADATION_COSTS_AUD_PER_MWH_DISCHARGED = (0.0, 25.0, 50.0, 100.0)


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


def _discharged_mwh(schedule: DispatchResult) -> float:
    return float(sum(schedule.discharge_mw) * (5 / 60))


def lower_tail_mean(values: list[float], probability: float = 0.05) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    count = max(1, math.ceil(len(ordered) * probability))
    return float(np.mean(ordered[:count]))


def bootstrap_mean(values: list[float], seed: int = 20260820, samples: int = 1000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    data = np.asarray(values)
    means = np.mean(rng.choice(data, size=(samples, len(data)), replace=True), axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def daily_mean_interval(times: list[datetime], values: np.ndarray) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for timestamp, value in zip(times, values, strict=True):
        grouped[timestamp.date().isoformat()].append(float(value))
    return bootstrap_mean([float(np.mean(day_values)) for day_values in grouped.values()])


def evaluate_region(
    region: str,
    rows: list[dict[str, Any]],
    degradation_costs: tuple[float, ...] = DEGRADATION_COSTS_AUD_PER_MWH_DISCHARGED,
) -> dict[str, Any]:
    from lightgbm import LGBMRegressor

    region_started = time.perf_counter()
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
    degradation_sensitivity: dict[str, dict[str, Any]] = {}
    for cost in degradation_costs:
        daily_gross: list[float] = []
        daily_net: list[float] = []
        daily_cycles: list[float] = []
        discharged_mwh = 0.0
        for day, actual in sorted(by_day.items()):
            forecast = forecast_by_day[day]
            if len(actual) != 288 or len(forecast) != 288:
                continue
            schedule = optimize_dispatch(
                forecast,
                spec,
                variable_degradation_cost_aud_per_mwh_discharged=cost,
            )
            gross = _realized(schedule, actual)
            discharged = _discharged_mwh(schedule)
            daily_gross.append(gross)
            daily_net.append(gross - cost * discharged)
            daily_cycles.append(schedule.equivalent_full_cycles)
            discharged_mwh += discharged
        gross_total = sum(daily_gross)
        net_total = sum(daily_net)
        sensitivity_annualizer = 365 / len(daily_net) if daily_net else 0.0
        degradation_sensitivity[str(int(cost))] = {
            "variable_degradation_cost_aud_per_mwh_discharged": cost,
            "gross_spot_margin_aud": gross_total,
            "discharged_mwh": discharged_mwh,
            "variable_degradation_cost_proxy_aud": cost * discharged_mwh,
            "net_operating_margin_proxy_aud": net_total,
            "net_operating_proxy_aud_per_mw_year": net_total
            * sensitivity_annualizer
            / spec.power_mw,
            "equivalent_full_cycles": sum(daily_cycles),
            "positive_day_share": float(np.mean(np.asarray(daily_net) > 0))
            if daily_net
            else 0.0,
            "daily_net_margin_mean_95_interval": bootstrap_mean(daily_net),
            "daily_net_margin_p05": float(np.quantile(daily_net, 0.05)) if daily_net else 0.0,
            "daily_net_margin_cvar05": lower_tail_mean(daily_net),
        }
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
            "persistence": point_metrics(y_test, persistence)
            | {"daily_mae_mean_95_interval": daily_mean_interval(test_times, np.abs(y_test - persistence))},
            "seasonal": point_metrics(y_test, seasonal)
            | {"daily_mae_mean_95_interval": daily_mean_interval(test_times, np.abs(y_test - seasonal))},
            "lightgbm": point_metrics(y_test, point)
            | {"daily_mae_mean_95_interval": daily_mean_interval(test_times, np.abs(y_test - point))},
        },
        "interval": {
            "fixed_conformal_90_ablation": interval_metrics(y_test, fixed_conformal_lower, fixed_conformal_upper),
            "rolling_conformal_90": interval_metrics(y_test, np.asarray(rolling_lower), np.asarray(rolling_upper)),
            "fixed_quantile_conformal_90_ablation": interval_metrics(
                y_test, fixed_quantile_lower, fixed_quantile_upper
            ),
            "rolling_quantile_conformal_90": interval_metrics(
                y_test, np.asarray(rolling_q_lower), np.asarray(rolling_q_upper)
            ),
        },
        "anomaly_stability": {
            "robust_z_counts": anomaly_counts,
            "rrp_ge_5000": int(np.sum(prices_all >= 5000)),
            "z4_to_z5_jaccard": anomaly_counts["5"] / anomaly_counts["4"] if anomaly_counts["4"] else None,
            "z5_to_z6_jaccard": anomaly_counts["6"] / anomaly_counts["5"] if anomaly_counts["5"] else None,
            "z5_daily_event_rate_mean_95_interval": bootstrap_mean(
                [
                    float(np.mean(np.abs(0.6745 * (prices_all[start : start + 288] - median) / mad) >= 5))
                    for start in range(0, len(prices_all), 288)
                ]
            ),
            "label_boundary": "No anomaly ground truth is claimed; fixed-price and robust-z counts are baselines, with threshold and day-level stability only.",
        },
        "bess": {
            "test_days": len(daily),
            "no_storage_margin_aud": 0.0,
            "gross_spot_margin_aud": forecast_total,
            "aud_per_mw_year": forecast_total * annualizer / spec.power_mw,
            "equivalent_full_cycles": sum(float(row["forecast_cycles"]) for row in daily),
            "rule_margin_aud": rule_total,
            "relative_rule_lift": (forecast_total - rule_total) / abs(rule_total) if rule_total else None,
            "oracle_margin_aud": oracle_total,
            "capture_rate": forecast_total / oracle_total if oracle_total else None,
            "oracle_regret_aud": oracle_total - forecast_total,
            "daily_forecast_margin_mean_95_interval": bootstrap_mean(forecast_margins),
            "degradation_sensitivity": degradation_sensitivity,
            "economic_boundary": "historical spot-market gross and variable-degradation sensitivity proxies; the cycling cost is user-supplied rather than asset-specific; excludes CAPEX, fixed O&M, network fees, FCAS and investment returns",
        },
        "calculation_seconds": time.perf_counter() - region_started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--regions",
        default="NSW1,QLD1,SA1,TAS1,VIC1",
        help="comma-separated subset used for pilot or full evaluation",
    )
    parser.add_argument(
        "--degradation-costs",
        default=",".join(str(int(value)) for value in DEGRADATION_COSTS_AUD_PER_MWH_DISCHARGED),
        help="comma-separated non-negative AUD/MWh-discharged sensitivity values",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    grouped = load(args.input)
    expected = 365 * 288
    coverage = {region: len(rows) / expected for region, rows in grouped.items()}
    if set(grouped) != {"NSW1", "QLD1", "SA1", "TAS1", "VIC1"} or min(coverage.values()) < 0.98:
        raise SystemExit(f"coverage gate failed: {coverage}")
    selected_regions = tuple(item.strip() for item in args.regions.split(",") if item.strip())
    unknown_regions = sorted(set(selected_regions) - set(grouped))
    if not selected_regions or unknown_regions:
        raise SystemExit(f"invalid selected regions: {unknown_regions or selected_regions}")
    degradation_costs = tuple(float(item) for item in args.degradation_costs.split(","))
    if not degradation_costs or any(value < 0 for value in degradation_costs):
        raise SystemExit("degradation costs must be a non-empty list of non-negative numbers")
    results = {
        region: evaluate_region(region, grouped[region], degradation_costs)
        for region in selected_regions
    }
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
        "selected_regions": selected_regions,
        "degradation_costs_aud_per_mwh_discharged": degradation_costs,
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
