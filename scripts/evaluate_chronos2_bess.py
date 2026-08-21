from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from energy_agent.battery import DispatchResult, optimize_dispatch
from energy_agent.foundation_forecast import FoundationWindow, load_chronos2, predict_windows
from energy_agent.schemas import BatterySpec

INTERVALS_PER_DAY = 288


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chronos-2 day-ahead AEMO/BESS gate")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--region", default="SA1")
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context-days", type=int, default=14)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--test-end", help="inclusive YYYY-MM-DD; default is last complete day")
    parser.add_argument("--degradation-cost", type=float, default=50.0)
    parser.add_argument("--covariates", action="store_true", help="use past market and known calendar covariates")
    parser.add_argument("--interval-calibration-days", type=int, default=0)
    return parser.parse_args()


def load_region(path: Path, region: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["region"] != region or raw["intervention"] != "0":
                continue
            rows.append(
                {
                    "timestamp": datetime.strptime(raw["interval"], "%Y/%m/%d %H:%M:%S").replace(tzinfo=UTC),
                    "rrp": float(raw["rrp"]),
                    "total_demand_mw": float(raw["total_demand_mw"]),
                    "available_generation_mw": float(raw["available_generation_mw"]),
                    "net_interchange_mw": float(raw["net_interchange_mw"]),
                }
            )
    rows.sort(key=lambda row: row["timestamp"])
    if not rows:
        raise ValueError(f"no non-intervention rows found for {region}")
    return rows


def complete_days(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["timestamp"].date().isoformat()].append(row)
    return {
        day: values
        for day, values in grouped.items()
        if len(values) == INTERVALS_PER_DAY
    }


def known_calendar_covariates(timestamp: datetime) -> dict[str, float]:
    minute = timestamp.hour * 60 + timestamp.minute
    return {
        "minute_sin": float(np.sin(2 * np.pi * minute / 1440)),
        "minute_cos": float(np.cos(2 * np.pi * minute / 1440)),
        "weekday_sin": float(np.sin(2 * np.pi * timestamp.weekday() / 7)),
        "weekday_cos": float(np.cos(2 * np.pi * timestamp.weekday() / 7)),
    }


def build_windows(
    rows: list[dict[str, Any]],
    test_dates: list[str],
    *,
    region: str,
    context_days: int,
    use_covariates: bool = False,
) -> list[FoundationWindow]:
    windows: list[FoundationWindow] = []
    context_length = context_days * INTERVALS_PER_DAY
    for day in test_dates:
        horizon = [row for row in rows if row["timestamp"].date().isoformat() == day]
        if len(horizon) != INTERVALS_PER_DAY:
            raise ValueError(f"incomplete test day: {day}")
        preceding = [row for row in rows if row["timestamp"] < horizon[0]["timestamp"]]
        if len(preceding) < context_length:
            raise ValueError(f"insufficient context for {day}")
        context = preceding[-context_length:]
        context_covariates: dict[str, tuple[float, ...]] = {}
        future_covariates: dict[str, tuple[float, ...]] = {}
        if use_covariates:
            for name in ("total_demand_mw", "available_generation_mw", "net_interchange_mw"):
                context_covariates[name] = tuple(float(row[name]) for row in context)
            context_calendar = [known_calendar_covariates(row["timestamp"]) for row in context]
            future_calendar = [known_calendar_covariates(row["timestamp"]) for row in horizon]
            for name in ("minute_sin", "minute_cos", "weekday_sin", "weekday_cos"):
                context_covariates[name] = tuple(values[name] for values in context_calendar)
                future_covariates[name] = tuple(values[name] for values in future_calendar)
        windows.append(
            FoundationWindow(
                series_id=f"{region}-{day}",
                context_timestamps=tuple(row["timestamp"] for row in context),
                context_target=tuple(row["rrp"] for row in context),
                future_timestamps=tuple(row["timestamp"] for row in horizon),
                context_covariates=context_covariates,
                future_covariates=future_covariates,
            )
        )
    return windows


def day_ahead_features(
    rows: list[dict[str, Any]], *, use_covariates: bool = False
) -> tuple[np.ndarray, np.ndarray, list[datetime]]:
    prices = np.asarray([row["rrp"] for row in rows], dtype=np.float64)
    timestamps = [row["timestamp"] for row in rows]
    covariate_arrays = {
        name: np.asarray([row[name] for row in rows], dtype=np.float64)
        for name in ("total_demand_mw", "available_generation_mw", "net_interchange_mw")
    } if use_covariates else {}
    matrix: list[list[float]] = []
    targets: list[float] = []
    output_times: list[datetime] = []
    for index in range(7 * INTERVALS_PER_DAY, len(rows)):
        timestamp = timestamps[index]
        hour = timestamp.hour + timestamp.minute / 60.0
        previous_day = prices[index - INTERVALS_PER_DAY : index]
        features = [
                prices[index - INTERVALS_PER_DAY],
                prices[index - 2 * INTERVALS_PER_DAY],
                prices[index - 7 * INTERVALS_PER_DAY],
                float(np.mean(previous_day)),
                float(np.std(previous_day)),
                float(np.sin(2 * np.pi * hour / 24.0)),
                float(np.cos(2 * np.pi * hour / 24.0)),
                float(timestamp.weekday()),
            ]
        if use_covariates:
            for values in covariate_arrays.values():
                previous = values[index - INTERVALS_PER_DAY : index]
                features.extend(
                    [
                        float(values[index - INTERVALS_PER_DAY]),
                        float(values[index - 2 * INTERVALS_PER_DAY]),
                        float(values[index - 7 * INTERVALS_PER_DAY]),
                        float(np.mean(previous)),
                        float(np.std(previous)),
                    ]
                )
        matrix.append(features)
        targets.append(float(prices[index]))
        output_times.append(timestamp)
    return np.asarray(matrix), np.asarray(targets), output_times


def realised_net_margin(
    schedule: DispatchResult, actual: list[float], degradation_cost: float
) -> float:
    interval_hours = 5.0 / 60.0
    gross = sum(
        (discharge - charge) * price * interval_hours
        for charge, discharge, price in zip(
            schedule.charge_mw, schedule.discharge_mw, actual, strict=True
        )
    )
    discharged = sum(schedule.discharge_mw) * interval_hours
    return float(gross - degradation_cost * discharged)


def point_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
    }


def paired_moving_block_bootstrap(
    candidate: list[float], baseline: list[float], samples: int = 5000, block_length_days: int = 7
) -> dict[str, Any]:
    difference = np.asarray(candidate) - np.asarray(baseline)
    if len(difference) < block_length_days:
        raise ValueError("daily differences must cover at least one temporal block")
    rng = np.random.default_rng(20260821)
    blocks_per_draw = int(np.ceil(len(difference) / block_length_days))
    starts = rng.integers(0, len(difference), size=(samples, blocks_per_draw))
    offsets = np.arange(block_length_days)
    indices = (starts[:, :, None] + offsets[None, None, :]) % len(difference)
    sampled = difference[indices.reshape(samples, -1)[:, : len(difference)]]
    draws = np.mean(sampled, axis=1)
    return {
        "mean_daily_delta_aud": float(difference.mean()),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "block_length_days": block_length_days,
        "method": "paired circular moving-block bootstrap",
    }


def conformal_interval_adjustment(scores: list[float], coverage: float = 0.8) -> float:
    if not scores or not 0.0 < coverage < 1.0:
        raise ValueError("non-empty scores and coverage between zero and one are required")
    level = min(1.0, np.ceil((len(scores) + 1) * coverage) / len(scores))
    return float(np.quantile(np.asarray(scores, dtype=float), level, method="higher"))


def main() -> None:
    args = parse_args()
    if (
        args.context_days < 7
        or args.test_days < 1
        or args.interval_calibration_days < 0
        or args.degradation_cost < 0
    ):
        raise SystemExit("context/calibration/test days and degradation cost are outside their allowed ranges")
    started = time.perf_counter()
    rows = load_region(args.input, args.region)
    days = sorted(complete_days(rows))
    if args.test_end:
        days = [day for day in days if day <= args.test_end]
    test_dates = days[-args.test_days :]
    if len(test_dates) != args.test_days:
        raise SystemExit("not enough complete test days")
    calibration_dates = (
        days[-(args.test_days + args.interval_calibration_days) : -args.test_days]
        if args.interval_calibration_days
        else []
    )
    if len(calibration_dates) != args.interval_calibration_days:
        raise SystemExit("not enough complete interval-calibration days")
    windows = build_windows(
        rows,
        calibration_dates + test_dates,
        region=args.region,
        context_days=args.context_days,
        use_covariates=args.covariates,
    )
    calibration_windows = windows[: args.interval_calibration_days]
    test_windows = windows[args.interval_calibration_days :]

    pipeline = load_chronos2(args.model_id, device_map=args.device)
    chronos = predict_windows(pipeline, windows, model_id=args.model_id)
    chronos_univariate: dict[str, Any] | None = None
    if args.covariates:
        univariate_windows = build_windows(
            rows,
            calibration_dates + test_dates,
            region=args.region,
            context_days=args.context_days,
            use_covariates=False,
        )
        chronos_univariate = predict_windows(pipeline, univariate_windows, model_id=args.model_id)

    from lightgbm import LGBMRegressor

    x, y, timestamps = day_ahead_features(rows, use_covariates=args.covariates)
    first_test = test_windows[0].future_timestamps[0]
    train_indices = [index for index, timestamp in enumerate(timestamps) if timestamp < first_test]
    test_indices = [index for index, timestamp in enumerate(timestamps) if timestamp.date().isoformat() in test_dates]
    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        random_state=20260821,
        verbosity=-1,
        n_jobs=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2")) // 2),
    ).fit(x[train_indices], y[train_indices])
    lightgbm_all = np.asarray(model.predict(x[test_indices]), dtype=np.float64)
    test_times = [timestamps[index] for index in test_indices]
    lightgbm_by_time = {timestamp: value for timestamp, value in zip(test_times, lightgbm_all, strict=True)}

    actual_all: list[float] = []
    prediction_names = ["persistence", "lightgbm", "chronos2"]
    if chronos_univariate is not None:
        prediction_names.append("chronos2_univariate")
    predictions: dict[str, list[float]] = {name: [] for name in prediction_names}
    interval_lower: list[float] = []
    interval_upper: list[float] = []
    margins: dict[str, list[float]] = {name: [] for name in [*prediction_names, "oracle"]}
    daily_rows: list[dict[str, Any]] = []
    spec = BatterySpec()
    grouped_days = complete_days(rows)
    conformal_adjustment: float | None = None
    if calibration_windows:
        scores: list[float] = []
        for window in calibration_windows:
            day = window.series_id.removeprefix(f"{args.region}-")
            actual = [float(row["rrp"]) for row in grouped_days[day]]
            foundation = chronos[window.series_id]
            scores.extend(
                max(lower - observed, observed - upper)
                for lower, upper, observed in zip(
                    foundation.lower, foundation.upper, actual, strict=True
                )
            )
        conformal_adjustment = conformal_interval_adjustment(scores)
    conformal_lower: list[float] = []
    conformal_upper: list[float] = []
    for window in test_windows:
        day = window.series_id.removeprefix(f"{args.region}-")
        actual = [row["rrp"] for row in grouped_days[day]]
        previous_day = list(window.context_target[-INTERVALS_PER_DAY:])
        lgb = [float(lightgbm_by_time[timestamp]) for timestamp in window.future_timestamps]
        foundation = chronos[window.series_id]
        model_forecasts = {
            "persistence": previous_day,
            "lightgbm": lgb,
            "chronos2": list(foundation.point),
        }
        if chronos_univariate is not None:
            model_forecasts["chronos2_univariate"] = list(
                chronos_univariate[window.series_id].point
            )
        actual_all.extend(actual)
        interval_lower.extend(foundation.lower)
        interval_upper.extend(foundation.upper)
        if conformal_adjustment is not None:
            conformal_lower.extend(value - conformal_adjustment for value in foundation.lower)
            conformal_upper.extend(value + conformal_adjustment for value in foundation.upper)
        row: dict[str, Any] = {"day": day}
        for name, forecast in model_forecasts.items():
            predictions[name].extend(forecast)
            schedule = optimize_dispatch(
                forecast,
                spec,
                variable_degradation_cost_aud_per_mwh_discharged=args.degradation_cost,
            )
            margin = realised_net_margin(schedule, actual, args.degradation_cost)
            margins[name].append(margin)
            row[f"{name}_net_operating_margin_proxy_aud"] = margin
        oracle_schedule = optimize_dispatch(
            actual,
            spec,
            variable_degradation_cost_aud_per_mwh_discharged=args.degradation_cost,
        )
        oracle_margin = realised_net_margin(oracle_schedule, actual, args.degradation_cost)
        margins["oracle"].append(oracle_margin)
        row["oracle_net_operating_margin_proxy_aud"] = oracle_margin
        daily_rows.append(row)

    actual_array = np.asarray(actual_all)
    forecast_metrics = {
        name: point_metrics(actual_array, np.asarray(values)) for name, values in predictions.items()
    }
    interval_lower_array = np.asarray(interval_lower)
    interval_upper_array = np.asarray(interval_upper)
    chronos_interval = {
        "nominal_coverage": 0.8,
        "empirical_coverage": float(
            np.mean((actual_array >= interval_lower_array) & (actual_array <= interval_upper_array))
        ),
        "mean_width_aud_per_mwh": float(np.mean(interval_upper_array - interval_lower_array)),
        "boundary": "raw Chronos-2 q10-q90 interval; not conformalised",
    }
    conformal_interval: dict[str, Any] | None = None
    if conformal_adjustment is not None:
        conformal_lower_array = np.asarray(conformal_lower)
        conformal_upper_array = np.asarray(conformal_upper)
        conformal_interval = {
            "nominal_coverage": 0.8,
            "empirical_coverage": float(
                np.mean(
                    (actual_array >= conformal_lower_array)
                    & (actual_array <= conformal_upper_array)
                )
            ),
            "mean_width_aud_per_mwh": float(
                np.mean(conformal_upper_array - conformal_lower_array)
            ),
            "calibration_days": args.interval_calibration_days,
            "adjustment_aud_per_mwh": conformal_adjustment,
            "boundary": "split conformalised raw q10-q90 interval using only preceding calibration outcomes",
        }
    economics: dict[str, Any] = {}
    for name, values in margins.items():
        economics[name] = {
            "test_net_operating_margin_proxy_aud": float(sum(values)),
            "annualised_net_operating_margin_proxy_aud_per_mw_year": float(sum(values) / len(values) * 365),
            "mean_daily_net_operating_margin_proxy_aud": float(np.mean(values)),
        }
    baseline_names = ["persistence", "lightgbm"]
    if chronos_univariate is not None:
        baseline_names.append("chronos2_univariate")
    baseline_name = max(baseline_names, key=lambda name: sum(margins[name]))
    best_baseline_mae = min(forecast_metrics[name]["mae"] for name in baseline_names)
    economic_delta = float(sum(margins["chronos2"]) - sum(margins[baseline_name]))
    coverage_pass = (
        conformal_interval is None
        or 0.75 <= float(conformal_interval["empirical_coverage"]) <= 0.85
    )
    economic_bootstrap = paired_moving_block_bootstrap(
        margins["chronos2"], margins[baseline_name]
    )
    gate = {
        "best_economic_baseline": baseline_name,
        "chronos2_net_delta_vs_best_baseline_aud": economic_delta,
        "chronos2_mae_ratio_vs_best_baseline": forecast_metrics["chronos2"]["mae"] / best_baseline_mae,
        "interval_coverage_pass": coverage_pass,
        "economic_ci_lower_positive": economic_bootstrap["ci_lower"] > 0,
        "pass": (
            economic_delta > 0
            and forecast_metrics["chronos2"]["mae"] <= 1.1 * best_baseline_mae
            and coverage_pass
            and economic_bootstrap["ci_lower"] > 0
        ),
        "rule": (
            "positive net operating-margin proxy delta, MAE <= 110% of the best baseline, "
            "conformal interval coverage between 75% and 85% when enabled, and positive "
            "paired moving-block economic CI lower bound"
        ),
        "paired_daily_margin_bootstrap": economic_bootstrap,
    }
    metrics = {
        "scope": "real AEMO rolling day-ahead Chronos-2/BESS pilot",
        "region": args.region,
        "test_dates": test_dates,
        "test_intervals": len(actual_all),
        "model_id": args.model_id,
        "context_days": args.context_days,
        "covariates": args.covariates,
        "interval_calibration_dates": calibration_dates,
        "forecast": forecast_metrics,
        "chronos2_interval": chronos_interval,
        "chronos2_conformal_interval": conformal_interval,
        "economics": economics,
        "promotion_gate": gate,
        "economic_boundary": (
            "Historical spot-market net operating-margin proxy after a configurable discharged-MWh cycling-cost proxy; "
            "excludes CAPEX, fixed OPEX, FCAS, network fees and investment return."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output / "daily_decisions.jsonl").open("w", encoding="utf-8") as handle:
        for row in daily_rows:
            handle.write(json.dumps(row) + "\n")
    git_sha = os.environ.get("ENERGY_GIT_COMMIT") or subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "model_id": args.model_id,
        "covariates": args.covariates,
        "context_days": args.context_days,
        "interval_calibration_days": args.interval_calibration_days,
        "test_days": args.test_days,
        "test_end": args.test_end,
        "chronos_forecasting_version": version("chronos-forecasting"),
        "python": sys.version,
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "elapsed_seconds": time.perf_counter() - started,
        "provider_cost_usd": 0.0,
        "data_boundary": "Official AEMO rows only; no interpolation or synthetic market values.",
    }
    try:
        import torch

        manifest["torch"] = torch.__version__
        manifest["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        manifest["cuda_peak_memory_bytes"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    except ImportError:
        pass
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
