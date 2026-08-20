from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from energy_agent.battery import (
    DispatchResult,
    optimize_dispatch,
    optimize_dispatch_cvar,
    threshold_dispatch,
)
from energy_agent.evaluation import (
    adaptive_conformal_bounds,
    parse_degradation_costs,
    residual_price_scenarios,
    seasonal_fold_windows,
    select_tail_policy,
)
from energy_agent.schemas import BatterySpec

DEGRADATION_COSTS_AUD_PER_MWH_DISCHARGED = (0.0, 25.0, 50.0, 100.0)
RISK_SCENARIO_COUNT = 10
RISK_ALPHA = 0.8
RISK_AVERSION_CANDIDATES = (0.25, 0.5, 1.0)
RISK_EVALUATION_COST = 50.0


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


def evaluate_seasonal_region(
    region: str,
    rows: list[dict[str, Any]],
    degradation_costs: tuple[float, ...],
) -> dict[str, Any]:
    """Evaluate four leakage-safe seasonal BESS folds on one NEM region."""

    from lightgbm import LGBMRegressor

    started = time.perf_counter()
    x, y, times = features(rows)
    folds = seasonal_fold_windows(times[0], times[-1])
    if {fold.name.split("-")[0] for fold in folds} != {"spring", "summer", "autumn", "winter"}:
        raise ValueError(f"four-season fold gate failed for {region}: {[fold.name for fold in folds]}")
    spec = BatterySpec()
    aggregated: dict[str, dict[str, list[float]]] = {
        str(int(cost)): defaultdict(list) for cost in degradation_costs
    }
    fold_metrics: dict[str, Any] = {}
    common: dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 20260820,
        "verbosity": -1,
        "n_jobs": 2,
    }
    for fold in folds:
        train_indices = [index for index, timestamp in enumerate(times) if timestamp < fold.calibration_start]
        calibration_indices = [
            index
            for index, timestamp in enumerate(times)
            if fold.calibration_start <= timestamp < fold.test_start
        ]
        test_indices = [
            index for index, timestamp in enumerate(times) if fold.test_start <= timestamp < fold.test_end
        ]
        if min(len(train_indices), len(calibration_indices), len(test_indices)) == 0:
            raise ValueError(f"empty seasonal split for {region}/{fold.name}")
        x_train, y_train = x[train_indices], y[train_indices]
        x_cal, y_cal = x[calibration_indices], y[calibration_indices]
        x_test, y_test = x[test_indices], y[test_indices]
        test_times = [times[index] for index in test_indices]
        model = LGBMRegressor(objective="regression_l1", **common).fit(x_train, y_train)
        calibration_predictions = np.asarray(model.predict(x_cal), dtype=float)
        calibration_residuals = y_cal - calibration_predictions
        predictions = np.asarray(model.predict(x_test), dtype=float)
        persistence_predictions = x_test[:, 0]
        conformal_radius = float(
            np.quantile(np.abs(y_cal - calibration_predictions), 0.9, method="higher")
        )
        adaptive = adaptive_conformal_bounds(
            y_test,
            predictions,
            np.abs(y_cal - calibration_predictions),
            target_alpha=0.1,
            gamma=0.005,
            window=30 * 288,
        )
        by_day: dict[str, list[float]] = defaultdict(list)
        forecast_by_day: dict[str, list[float]] = defaultdict(list)
        persistence_by_day: dict[str, list[float]] = defaultdict(list)
        for timestamp, actual, predicted, persistence_prediction in zip(
            test_times, y_test, predictions, persistence_predictions, strict=True
        ):
            by_day[timestamp.date().isoformat()].append(float(actual))
            forecast_by_day[timestamp.date().isoformat()].append(float(predicted))
            persistence_by_day[timestamp.date().isoformat()].append(float(persistence_prediction))
        complete_days = [
            day
            for day in sorted(by_day)
            if len(by_day[day]) == len(forecast_by_day[day]) == len(persistence_by_day[day]) == 288
        ]
        if len(complete_days) < 27:
            raise ValueError(f"seasonal day coverage gate failed for {region}/{fold.name}: {len(complete_days)}")
        low, high = float(np.quantile(y_train, 0.25)), float(np.quantile(y_train, 0.75))
        calibration_by_day: dict[str, list[float]] = defaultdict(list)
        calibration_forecast_by_day: dict[str, list[float]] = defaultdict(list)
        calibration_times = [times[index] for index in calibration_indices]
        for timestamp, actual, predicted in zip(
            calibration_times, y_cal, calibration_predictions, strict=True
        ):
            calibration_by_day[timestamp.date().isoformat()].append(float(actual))
            calibration_forecast_by_day[timestamp.date().isoformat()].append(float(predicted))
        calibration_days = [
            day
            for day in sorted(calibration_by_day)
            if len(calibration_by_day[day]) == len(calibration_forecast_by_day[day]) == 288
        ]
        if len(calibration_days) < 27:
            raise ValueError(
                f"risk-policy calibration day gate failed for {region}/{fold.name}: {len(calibration_days)}"
            )
        scenario_bank_days = calibration_days[: len(calibration_days) // 2]
        validation_days = calibration_days[len(calibration_days) // 2 :]
        scenario_bank_residuals = np.asarray(
            [
                actual - predicted
                for day in scenario_bank_days
                for actual, predicted in zip(
                    calibration_by_day[day], calibration_forecast_by_day[day], strict=True
                )
            ]
        )
        risk_policy_by_cost: dict[str, dict[str, float | str]] = {}
        for cost in degradation_costs:
            if cost != RISK_EVALUATION_COST:
                continue
            point_validation: list[float] = []
            candidate_validation: dict[str, list[float]] = {
                str(value): [] for value in RISK_AVERSION_CANDIDATES
            }
            for day in validation_days:
                actual = calibration_by_day[day]
                forecast = calibration_forecast_by_day[day]
                point_schedule = optimize_dispatch(
                    forecast,
                    spec,
                    variable_degradation_cost_aud_per_mwh_discharged=cost,
                )
                point_validation.append(
                    _realized(point_schedule, actual) - cost * _discharged_mwh(point_schedule)
                )
                validation_scenarios = residual_price_scenarios(
                    forecast,
                    scenario_bank_residuals,
                    scenario_count=RISK_SCENARIO_COUNT,
                    seed_material=f"{region}|{fold.name}|{day}|inner-risk-scenarios-v1",
                )
                for risk_aversion in RISK_AVERSION_CANDIDATES:
                    candidate_schedule = optimize_dispatch_cvar(
                        validation_scenarios,
                        spec,
                        alpha=RISK_ALPHA,
                        risk_aversion=risk_aversion,
                        variable_degradation_cost_aud_per_mwh_discharged=cost,
                    )
                    candidate_validation[str(risk_aversion)].append(
                        _realized(candidate_schedule.dispatch, actual)
                        - cost * _discharged_mwh(candidate_schedule.dispatch)
                    )
            selection = select_tail_policy(point_validation, candidate_validation)
            selection["selected_risk_aversion"] = (
                float(selection["selected_policy"])
                if selection["selected_policy"] != "point"
                else 0.0
            )
            selection["validation_days"] = float(len(validation_days))
            risk_policy_by_cost[str(int(cost))] = selection
        risk_scenarios_by_day = {
            day: residual_price_scenarios(
                forecast_by_day[day],
                calibration_residuals,
                scenario_count=RISK_SCENARIO_COUNT,
                seed_material=f"{region}|{fold.name}|{day}|risk-scenarios-v1",
            )
            for day in complete_days
        }
        cost_metrics: dict[str, Any] = {}
        for cost in degradation_costs:
            key = str(int(cost))
            run_risk_evaluation = cost == RISK_EVALUATION_COST
            daily_gross: list[float] = []
            daily_net: list[float] = []
            daily_rule_net: list[float] = []
            daily_oracle_net: list[float] = []
            daily_persistence_net: list[float] = []
            daily_risk_net: list[float] = []
            daily_cycles: list[float] = []
            daily_risk_cycles: list[float] = []
            daily_discharged: list[float] = []
            risk_mip_gaps: list[float] = []
            for day in complete_days:
                actual = by_day[day]
                forecast = forecast_by_day[day]
                persistence_forecast = persistence_by_day[day]
                schedule = optimize_dispatch(
                    forecast,
                    spec,
                    variable_degradation_cost_aud_per_mwh_discharged=cost,
                )
                oracle = optimize_dispatch(
                    actual,
                    spec,
                    variable_degradation_cost_aud_per_mwh_discharged=cost,
                )
                persistence_schedule = optimize_dispatch(
                    persistence_forecast,
                    spec,
                    variable_degradation_cost_aud_per_mwh_discharged=cost,
                )
                rule = threshold_dispatch(forecast, spec, low, high)
                discharged = _discharged_mwh(schedule)
                gross = _realized(schedule, actual)
                daily_gross.append(gross)
                daily_net.append(gross - cost * discharged)
                daily_rule_net.append(_realized(rule, actual) - cost * _discharged_mwh(rule))
                daily_oracle_net.append(
                    _realized(oracle, actual) - cost * _discharged_mwh(oracle)
                )
                daily_persistence_net.append(
                    _realized(persistence_schedule, actual)
                    - cost * _discharged_mwh(persistence_schedule)
                )
                daily_cycles.append(schedule.equivalent_full_cycles)
                daily_discharged.append(discharged)
                if run_risk_evaluation:
                    risk_selection = risk_policy_by_cost[key]
                    risk_mip_gap: float | None
                    if risk_selection["selected_policy"] == "point":
                        risk_dispatch = schedule
                        risk_mip_gap = 0.0
                    else:
                        risk_schedule = optimize_dispatch_cvar(
                            risk_scenarios_by_day[day],
                            spec,
                            alpha=RISK_ALPHA,
                            risk_aversion=float(risk_selection["selected_risk_aversion"]),
                            variable_degradation_cost_aud_per_mwh_discharged=cost,
                        )
                        risk_dispatch = risk_schedule.dispatch
                        risk_mip_gap = risk_schedule.solver_mip_gap
                    risk_discharged = _discharged_mwh(risk_dispatch)
                    risk_gross = _realized(risk_dispatch, actual)
                    daily_risk_net.append(risk_gross - cost * risk_discharged)
                    daily_risk_cycles.append(risk_dispatch.equivalent_full_cycles)
                    if risk_mip_gap is not None:
                        risk_mip_gaps.append(risk_mip_gap)
            aggregate_series = [
                ("gross", daily_gross),
                ("net", daily_net),
                ("rule_net", daily_rule_net),
                ("oracle_net", daily_oracle_net),
                ("persistence_net", daily_persistence_net),
                ("cycles", daily_cycles),
                ("discharged", daily_discharged),
            ]
            if run_risk_evaluation:
                aggregate_series.extend(
                    [("risk_net", daily_risk_net), ("risk_cycles", daily_risk_cycles)]
                )
            for name, values in aggregate_series:
                aggregated[key][name].extend(values)
            total_net = sum(daily_net)
            total_oracle = sum(daily_oracle_net)
            total_rule = sum(daily_rule_net)
            total_persistence = sum(daily_persistence_net)
            lightgbm_mae = point_metrics(y_test, predictions)["mae"]
            persistence_mae = point_metrics(y_test, persistence_predictions)["mae"]
            mae_winner = "lightgbm" if lightgbm_mae < persistence_mae else "persistence"
            decision_winner = "lightgbm" if total_net > total_persistence else "persistence"
            cost_metrics[key] = {
                "variable_degradation_cost_aud_per_mwh_discharged": cost,
                "test_days": len(daily_net),
                "gross_spot_margin_aud": sum(daily_gross),
                "net_operating_margin_proxy_aud": total_net,
                "net_operating_proxy_aud_per_mw_year": total_net * 365 / len(daily_net),
                "equivalent_full_cycles": sum(daily_cycles),
                "positive_day_share": float(np.mean(np.asarray(daily_net) > 0)),
                "daily_net_margin_mean_95_interval": bootstrap_mean(daily_net),
                "daily_net_margin_cvar05": lower_tail_mean(daily_net),
                "rule_net_margin_aud": total_rule,
                "relative_rule_lift": (total_net - total_rule) / abs(total_rule) if total_rule else None,
                "oracle_net_margin_aud": total_oracle,
                "oracle_capture_rate": total_net / total_oracle if total_oracle else None,
                "oracle_regret_aud": total_oracle - total_net,
                "decision_focused": {
                    "persistence_net_margin_aud": total_persistence,
                    "persistence_oracle_regret_aud": total_oracle - total_persistence,
                    "lightgbm_net_margin_lift_vs_persistence_aud": total_net - total_persistence,
                    "mae_winner": mae_winner,
                    "net_margin_winner": decision_winner,
                    "mae_net_margin_rank_agreement": mae_winner == decision_winner,
                },
            }
            if run_risk_evaluation:
                total_risk = sum(daily_risk_net)
                point_cvar = lower_tail_mean(daily_net)
                risk_cvar = lower_tail_mean(daily_risk_net)
                cost_metrics[key].update(
                    {
                        "risk_aware_net_margin_aud": total_risk,
                        "risk_aware_delta_vs_point_aud": total_risk - total_net,
                        "risk_aware_equivalent_full_cycles": sum(daily_risk_cycles),
                        "risk_aware_positive_day_share": float(
                            np.mean(np.asarray(daily_risk_net) > 0)
                        ),
                        "risk_aware_daily_net_margin_cvar05": risk_cvar,
                        "risk_aware_cvar05_lift_vs_point_aud": risk_cvar - point_cvar,
                        "risk_aware_max_solver_mip_gap": max(risk_mip_gaps, default=0.0),
                        "risk_policy_selection": risk_policy_by_cost[key],
                    }
                )
        fold_metrics[fold.name] = {
            "train_end_exclusive": fold.calibration_start.isoformat(),
            "calibration_window": [fold.calibration_start.isoformat(), fold.test_start.isoformat()],
            "test_window": [fold.test_start.isoformat(), fold.test_end.isoformat()],
            "split_rows": {
                "train": len(train_indices),
                "calibration": len(calibration_indices),
                "test": len(test_indices),
            },
            "complete_test_days": len(complete_days),
            "point": {
                "persistence": point_metrics(y_test, persistence_predictions),
                "lightgbm": point_metrics(y_test, predictions),
            },
            "interval": {
                "fixed_conformal_90_ablation": interval_metrics(
                    y_test,
                    predictions - conformal_radius,
                    predictions + conformal_radius,
                ),
                "adaptive_conformal_90": interval_metrics(
                    y_test,
                    np.asarray(adaptive.lower),
                    np.asarray(adaptive.upper),
                )
                | {
                    "gamma": 0.005,
                    "score_window_intervals": 30 * 288,
                    "alpha_min": min(adaptive.alpha_history),
                    "alpha_max": max(adaptive.alpha_history),
                    "method_boundary": "ACI-inspired online controller with a rolling absolute-residual score window; not a verbatim reproduction of the paper experiments",
                },
            },
            "risk_scenario_design": {
                "scenario_count": RISK_SCENARIO_COUNT,
                "alpha": RISK_ALPHA,
                "risk_aversion_candidates": RISK_AVERSION_CANDIDATES,
                "evaluation_cost_aud_per_mwh_discharged": RISK_EVALUATION_COST,
                "selection": "first half of calibration supplies residual scenarios; second half selects tail policy with a mean guardrail; point dispatch is the fallback",
                "construction": "point forecast plus deterministic complete-day residual paths sampled only from the preceding calibration window",
            },
            "degradation_sensitivity": cost_metrics,
        }

    overall: dict[str, Any] = {}
    for cost in degradation_costs:
        key = str(int(cost))
        aggregate_values = aggregated[key]
        day_count = len(aggregate_values["net"])
        total_net = sum(aggregate_values["net"])
        total_oracle = sum(aggregate_values["oracle_net"])
        total_rule = sum(aggregate_values["rule_net"])
        total_persistence = sum(aggregate_values["persistence_net"])
        overall[key] = {
            "variable_degradation_cost_aud_per_mwh_discharged": cost,
            "test_days": day_count,
            "gross_spot_margin_aud": sum(aggregate_values["gross"]),
            "discharged_mwh": sum(aggregate_values["discharged"]),
            "variable_degradation_cost_proxy_aud": cost * sum(aggregate_values["discharged"]),
            "net_operating_margin_proxy_aud": total_net,
            "net_operating_proxy_aud_per_mw_year": total_net * 365 / day_count,
            "equivalent_full_cycles": sum(aggregate_values["cycles"]),
            "positive_day_share": float(np.mean(np.asarray(aggregate_values["net"]) > 0)),
            "daily_net_margin_mean_95_interval": bootstrap_mean(aggregate_values["net"]),
            "daily_net_margin_p05": float(np.quantile(aggregate_values["net"], 0.05)),
            "daily_net_margin_cvar05": lower_tail_mean(aggregate_values["net"]),
            "rule_net_margin_aud": total_rule,
            "relative_rule_lift": (total_net - total_rule) / abs(total_rule) if total_rule else None,
            "oracle_net_margin_aud": total_oracle,
            "oracle_capture_rate": total_net / total_oracle if total_oracle else None,
            "oracle_regret_aud": total_oracle - total_net,
            "decision_focused": {
                "persistence_net_margin_aud": total_persistence,
                "persistence_oracle_regret_aud": total_oracle - total_persistence,
                "lightgbm_net_margin_lift_vs_persistence_aud": total_net - total_persistence,
            },
        }
        if aggregate_values["risk_net"]:
            total_risk = sum(aggregate_values["risk_net"])
            point_cvar = lower_tail_mean(aggregate_values["net"])
            risk_cvar = lower_tail_mean(aggregate_values["risk_net"])
            overall[key].update(
                {
                    "risk_aware_net_margin_aud": total_risk,
                    "risk_aware_net_operating_proxy_aud_per_mw_year": total_risk
                    * 365
                    / day_count,
                    "risk_aware_delta_vs_point_aud": total_risk - total_net,
                    "risk_aware_equivalent_full_cycles": sum(aggregate_values["risk_cycles"]),
                    "risk_aware_positive_day_share": float(
                        np.mean(np.asarray(aggregate_values["risk_net"]) > 0)
                    ),
                    "risk_aware_daily_net_margin_cvar05": risk_cvar,
                    "risk_aware_cvar05_lift_vs_point_aud": risk_cvar - point_cvar,
                }
            )
    decision_summary: dict[str, Any] = {}
    for cost in degradation_costs:
        key = str(int(cost))
        comparisons = [
            fold["degradation_sensitivity"][key]["decision_focused"]
            for fold in fold_metrics.values()
        ]
        decision_summary[key] = {
            "folds": len(comparisons),
            "lightgbm_mae_wins": sum(item["mae_winner"] == "lightgbm" for item in comparisons),
            "lightgbm_net_margin_wins": sum(
                item["net_margin_winner"] == "lightgbm" for item in comparisons
            ),
            "mae_net_margin_rank_agreements": sum(
                item["mae_net_margin_rank_agreement"] for item in comparisons
            ),
            "evaluation_boundary": "Model selection diagnostic only; models were not trained with SPO+ loss.",
        }
    return {
        "region": region,
        "folds": fold_metrics,
        "overall_degradation_sensitivity": overall,
        "decision_focused_summary": decision_summary,
        "risk_scenario_design": {
            "scenario_count": RISK_SCENARIO_COUNT,
            "alpha": RISK_ALPHA,
            "risk_aversion_candidates": RISK_AVERSION_CANDIDATES,
            "evaluation_cost_aud_per_mwh_discharged": RISK_EVALUATION_COST,
            "information_boundary": "calibration residual paths and forecasts only; realised prices are used solely for settlement",
        },
        "economic_boundary": "112-day four-season historical spot-market operating proxy; user-supplied variable cycling cost; excludes CAPEX, fixed O&M, network fees, FCAS and investment returns",
        "calculation_seconds": time.perf_counter() - started,
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
        help="comma-, colon- or semicolon-separated non-negative AUD/MWh-discharged values",
    )
    parser.add_argument(
        "--seasonal-bess",
        action="store_true",
        help="run four 28-day out-of-time seasonal BESS folds instead of the terminal split",
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
    try:
        degradation_costs = parse_degradation_costs(args.degradation_costs)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    evaluator = evaluate_seasonal_region if args.seasonal_bess else evaluate_region
    results = {region: evaluator(region, grouped[region], degradation_costs) for region in selected_regions}
    scope = (
        "real AEMO NEMWeb four-season out-of-time BESS backtest"
        if args.seasonal_bess
        else "real AEMO NEMWeb rolling test"
    )
    metrics = {"scope": scope, "coverage": coverage, "regions": results}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    git_sha = os.environ.get("ENERGY_GIT_COMMIT") or subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
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
        "seasonal_bess": args.seasonal_bess,
        "risk_aware_dispatch": (
            {
                "scenario_count": RISK_SCENARIO_COUNT,
                "alpha": RISK_ALPHA,
                "risk_aversion_candidates": RISK_AVERSION_CANDIDATES,
                "evaluation_cost_aud_per_mwh_discharged": RISK_EVALUATION_COST,
                "scenario_source": "preceding calibration residual day paths",
            }
            if args.seasonal_bess
            else None
        ),
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
