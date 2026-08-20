from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from evaluate_real_market import (
    _discharged_mwh,
    _realized,
    features,
    load,
    lower_tail_mean,
    point_metrics,
)

from energy_agent.battery import optimize_dispatch
from energy_agent.evaluation import (
    optimizer_action_weights,
    seasonal_fold_windows,
    select_decision_weighted_model,
)
from energy_agent.schemas import BatterySpec

EVALUATION_COST_AUD_PER_MWH = 50.0
ACTION_EMPHASIS = 4.0


def _complete_day_positions(times: list[datetime], positions: list[int]) -> list[list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for position in positions:
        grouped[times[position].date().isoformat()].append(position)
    return [day for day in grouped.values() if len(day) == 288]


def _training_weights(
    times: list[datetime],
    prices: np.ndarray,
    train_indices: list[int],
    spec: BatterySpec,
) -> tuple[np.ndarray, dict[str, float | int]]:
    local_by_global = {global_index: local_index for local_index, global_index in enumerate(train_indices)}
    weights = np.ones(len(train_indices), dtype=float)
    complete_days = _complete_day_positions(times, train_indices)
    action_intervals = 0
    for positions in complete_days:
        actual = [float(prices[position]) for position in positions]
        oracle = optimize_dispatch(
            actual,
            spec,
            variable_degradation_cost_aud_per_mwh_discharged=EVALUATION_COST_AUD_PER_MWH,
        )
        day_weights = optimizer_action_weights(
            oracle.charge_mw,
            oracle.discharge_mw,
            power_mw=spec.power_mw,
            emphasis=ACTION_EMPHASIS,
        )
        for position, weight in zip(positions, day_weights, strict=True):
            weights[local_by_global[position]] = weight
        action_intervals += int(np.sum(day_weights > 1.0 + 1e-8))
    return weights, {
        "complete_training_days": len(complete_days),
        "action_weighted_intervals": action_intervals,
        "mean_weight": float(np.mean(weights)),
        "max_weight": float(np.max(weights)),
    }


def _daily_net_values(
    times: list[datetime],
    actual: np.ndarray,
    forecast: np.ndarray,
    spec: BatterySpec,
) -> list[float]:
    actual_by_day: dict[str, list[float]] = defaultdict(list)
    forecast_by_day: dict[str, list[float]] = defaultdict(list)
    for timestamp, observed, predicted in zip(times, actual, forecast, strict=True):
        day = timestamp.date().isoformat()
        actual_by_day[day].append(float(observed))
        forecast_by_day[day].append(float(predicted))
    values: list[float] = []
    for day in sorted(actual_by_day):
        observed = actual_by_day[day]
        predicted = forecast_by_day[day]
        if len(observed) != 288 or len(predicted) != 288:
            continue
        schedule = optimize_dispatch(
            predicted,
            spec,
            variable_degradation_cost_aud_per_mwh_discharged=EVALUATION_COST_AUD_PER_MWH,
        )
        values.append(
            _realized(schedule, observed)
            - EVALUATION_COST_AUD_PER_MWH * _discharged_mwh(schedule)
        )
    return values


def _economic_metrics(values: list[float]) -> dict[str, float | int]:
    total = float(sum(values))
    return {
        "days": len(values),
        "net_operating_margin_proxy_aud": total,
        "net_operating_proxy_aud_per_mw_year": total * 365 / len(values),
        "positive_day_share": float(np.mean(np.asarray(values) > 0)),
        "daily_p05_aud": float(np.quantile(values, 0.05)),
        "daily_cvar05_aud": lower_tail_mean(values),
    }


def evaluate_region(region: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from lightgbm import LGBMRegressor

    started = time.perf_counter()
    x, y, times = features(rows)
    folds = seasonal_fold_windows(times[0], times[-1])
    if {fold.name.split("-")[0] for fold in folds} != {"spring", "summer", "autumn", "winter"}:
        raise ValueError(f"four-season fold gate failed for {region}")
    spec = BatterySpec()
    common: dict[str, Any] = {
        "objective": "regression_l1",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 20260820,
        "verbosity": -1,
        "n_jobs": 2,
    }
    fold_metrics: dict[str, Any] = {}
    aggregate: dict[str, list[float]] = {
        "baseline": [],
        "decision_weighted": [],
        "selected": [],
    }
    for fold in folds:
        train_indices = [index for index, timestamp in enumerate(times) if timestamp < fold.calibration_start]
        calibration_indices = [
            index for index, timestamp in enumerate(times) if fold.calibration_start <= timestamp < fold.test_start
        ]
        test_indices = [index for index, timestamp in enumerate(times) if fold.test_start <= timestamp < fold.test_end]
        if min(len(train_indices), len(calibration_indices), len(test_indices)) == 0:
            raise ValueError(f"empty split for {region}/{fold.name}")

        x_train, y_train = x[train_indices], y[train_indices]
        x_cal, y_cal = x[calibration_indices], y[calibration_indices]
        x_test, y_test = x[test_indices], y[test_indices]
        calibration_times = [times[index] for index in calibration_indices]
        test_times = [times[index] for index in test_indices]
        weights, weight_metrics = _training_weights(times, y, train_indices, spec)

        baseline = LGBMRegressor(**common).fit(x_train, y_train)
        weighted = LGBMRegressor(**common).fit(x_train, y_train, sample_weight=weights)
        cal_baseline = np.asarray(baseline.predict(x_cal), dtype=float)
        cal_weighted = np.asarray(weighted.predict(x_cal), dtype=float)
        baseline_validation = _daily_net_values(calibration_times, y_cal, cal_baseline, spec)
        weighted_validation = _daily_net_values(calibration_times, y_cal, cal_weighted, spec)
        if min(len(baseline_validation), len(weighted_validation)) < 27:
            raise ValueError(f"calibration day gate failed for {region}/{fold.name}")
        selection = select_decision_weighted_model(baseline_validation, weighted_validation)

        test_baseline = np.asarray(baseline.predict(x_test), dtype=float)
        test_weighted = np.asarray(weighted.predict(x_test), dtype=float)
        test_selected = test_weighted if selection["selected_model"] == "decision_weighted" else test_baseline
        baseline_values = _daily_net_values(test_times, y_test, test_baseline, spec)
        weighted_values = _daily_net_values(test_times, y_test, test_weighted, spec)
        selected_values = weighted_values if selection["selected_model"] == "decision_weighted" else baseline_values
        if min(len(baseline_values), len(weighted_values), len(selected_values)) < 27:
            raise ValueError(f"test day gate failed for {region}/{fold.name}")
        aggregate["baseline"].extend(baseline_values)
        aggregate["decision_weighted"].extend(weighted_values)
        aggregate["selected"].extend(selected_values)
        fold_metrics[fold.name] = {
            "train_end_exclusive": fold.calibration_start.isoformat(),
            "calibration_window": [fold.calibration_start.isoformat(), fold.test_start.isoformat()],
            "test_window": [fold.test_start.isoformat(), fold.test_end.isoformat()],
            "split_rows": {
                "train": len(train_indices),
                "calibration": len(calibration_indices),
                "test": len(test_indices),
            },
            "training_weight_provenance": weight_metrics,
            "selection": selection,
            "point_mae": {
                "baseline": point_metrics(y_test, test_baseline)["mae"],
                "decision_weighted": point_metrics(y_test, test_weighted)["mae"],
                "selected": point_metrics(y_test, test_selected)["mae"],
            },
            "test_economics": {
                "baseline": _economic_metrics(baseline_values),
                "decision_weighted": _economic_metrics(weighted_values),
                "selected": _economic_metrics(selected_values),
            },
        }
    baseline_total = sum(aggregate["baseline"])
    weighted_total = sum(aggregate["decision_weighted"])
    selected_total = sum(aggregate["selected"])
    return {
        "region": region,
        "folds": fold_metrics,
        "overall": {name: _economic_metrics(values) for name, values in aggregate.items()},
        "test_delta_aud": {
            "decision_weighted_vs_baseline": weighted_total - baseline_total,
            "calibration_selected_vs_baseline": selected_total - baseline_total,
        },
        "selected_model_counts": {
            "baseline": sum(
                fold["selection"]["selected_model"] == "baseline" for fold in fold_metrics.values()
            ),
            "decision_weighted": sum(
                fold["selection"]["selected_model"] == "decision_weighted"
                for fold in fold_metrics.values()
            ),
        },
        "calculation_seconds": time.perf_counter() - started,
    }


def _git_sha() -> str:
    override = os.environ.get("ENERGY_GIT_COMMIT", "").strip()
    if override:
        return override
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regions", default="NSW1,QLD1,SA1,TAS1,VIC1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    grouped = load(args.input)
    requested = [region for region in args.regions.split(",") if region]
    missing = sorted(set(requested) - set(grouped))
    if missing:
        raise SystemExit(f"missing regions: {missing}")
    results = {region: evaluate_region(region, grouped[region]) for region in requested}
    metrics = {
        "evidence_status": "verified-real-if-run-completes",
        "scope": "AEMO NEMWeb four-season out-of-time decision-weighted forecast gate",
        "evaluation_cost_aud_per_mwh_discharged": EVALUATION_COST_AUD_PER_MWH,
        "action_emphasis": ACTION_EMPHASIS,
        "regions": results,
        "boundary": "Training-only perfect-foresight optimiser actions form sample weights; calibration selects the model; unseen test prices are settlement-only. This is an optimiser-informed loss proxy, not SPO+ or an investment-return claim.",
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "input_sha256": input_hash,
        "selected_regions": requested,
        "evaluation_cost_aud_per_mwh_discharged": EVALUATION_COST_AUD_PER_MWH,
        "action_emphasis": ACTION_EMPHASIS,
        "metrics_sha256": hashlib.sha256((args.output / "metrics.json").read_bytes()).hexdigest(),
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
