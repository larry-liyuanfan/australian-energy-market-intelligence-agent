from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

COST_KEY = "50"
REGIONS = {"NSW1", "QLD1", "SA1", "TAS1", "VIC1"}
BOOTSTRAP_SAMPLES = 5_000


def bootstrap_mean_interval(values: list[float], *, seed: int = 20260821) -> list[float]:
    if not values:
        raise ValueError("bootstrap values are empty")
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    samples = rng.choice(array, size=(BOOTSTRAP_SAMPLES, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def summarize(metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen two-year risk/return promotion gate.

    The source evaluator selected each fold's risk aversion using only the
    preceding calibration window. This function never retunes policies; it
    aggregates the already-settled out-of-time fold outcomes.
    """

    regions = metrics.get("regions", {})
    if set(regions) != REGIONS:
        raise ValueError("five-region metrics are required")

    fold_tail_lifts: list[float] = []
    fold_margin_deltas: list[float] = []
    non_point_selections = 0
    positive_tail_regions = 0
    point_annualised: list[float] = []
    risk_annualised: list[float] = []
    region_margin_retentions: list[float] = []
    output_regions: dict[str, Any] = {}

    for region, payload in sorted(regions.items()):
        folds = payload["folds"]
        if len(folds) != 8:
            raise ValueError(f"eight folds are required for {region}")
        selected = 0
        region_tail_lifts: list[float] = []
        region_margin_deltas: list[float] = []
        for fold in folds.values():
            economics = fold["degradation_sensitivity"][COST_KEY]
            point_margin = float(economics["net_operating_margin_proxy_aud"])
            risk_margin = float(economics["risk_aware_net_margin_aud"])
            point_tail = float(economics["daily_net_margin_cvar05"])
            risk_tail = float(economics["risk_aware_daily_net_margin_cvar05"])
            tail_lift = risk_tail - point_tail
            margin_delta = risk_margin - point_margin
            region_tail_lifts.append(tail_lift)
            region_margin_deltas.append(margin_delta)
            fold_tail_lifts.append(tail_lift)
            fold_margin_deltas.append(margin_delta)
            policy = economics["risk_policy_selection"]["selected_policy"]
            selected += int(policy != "point")
        non_point_selections += selected

        overall = payload["overall_degradation_sensitivity"][COST_KEY]
        point_value = float(overall["net_operating_proxy_aud_per_mw_year"])
        risk_value = float(overall["risk_aware_net_operating_proxy_aud_per_mw_year"])
        if point_value <= 0:
            raise ValueError(f"positive point annualised proxy required for {region}")
        retention = risk_value / point_value
        aggregate_tail_lift = float(overall["risk_aware_cvar05_lift_vs_point_aud"])
        point_annualised.append(point_value)
        risk_annualised.append(risk_value)
        region_margin_retentions.append(retention)
        positive_tail_regions += int(aggregate_tail_lift > 0)
        output_regions[region] = {
            "folds": len(folds),
            "non_point_policy_selections": selected,
            "mean_fold_cvar05_lift_aud": float(np.mean(region_tail_lifts)),
            "mean_fold_margin_delta_aud": float(np.mean(region_margin_deltas)),
            "point_net_operating_proxy_aud_per_mw_year": point_value,
            "risk_aware_net_operating_proxy_aud_per_mw_year": risk_value,
            "margin_retention_ratio": retention,
            "point_daily_net_margin_cvar05_aud": float(overall["daily_net_margin_cvar05"]),
            "risk_aware_daily_net_margin_cvar05_aud": float(
                overall["risk_aware_daily_net_margin_cvar05"]
            ),
            "aggregate_cvar05_lift_aud": aggregate_tail_lift,
        }

    fold_tail_interval = bootstrap_mean_interval(fold_tail_lifts)
    fold_margin_interval = bootstrap_mean_interval(fold_margin_deltas, seed=20260822)
    mean_point = float(np.mean(point_annualised))
    mean_risk = float(np.mean(risk_annualised))
    aggregate_retention = mean_risk / mean_point
    minimum_region_retention = min(region_margin_retentions)
    promotion_rules = {
        "fold_cvar05_lift_95pct_bootstrap_lower_must_exceed_aud": 0.0,
        "minimum_positive_aggregate_cvar05_regions": 3,
        "minimum_five_region_mean_margin_retention_ratio": 0.95,
        "minimum_single_region_margin_retention_ratio": 0.90,
    }
    promotion_pass = (
        fold_tail_interval[0] > 0
        and positive_tail_regions >= 3
        and aggregate_retention >= 0.95
        and minimum_region_retention >= 0.90
    )
    return {
        "evidence_status": "verified-real-if-run-completes",
        "gate_kind": "backward-temporal-risk-transport-not-prospective",
        "cost_aud_per_mwh_discharged": 50.0,
        "folds": len(fold_tail_lifts),
        "non_point_policy_selections": non_point_selections,
        "aggregate": {
            "mean_fold_cvar05_lift_aud": float(np.mean(fold_tail_lifts)),
            "mean_fold_cvar05_lift_95pct_paired_bootstrap_aud": fold_tail_interval,
            "mean_fold_margin_delta_aud": float(np.mean(fold_margin_deltas)),
            "mean_fold_margin_delta_95pct_paired_bootstrap_aud": fold_margin_interval,
            "positive_aggregate_cvar05_regions": positive_tail_regions,
            "point_mean_net_operating_proxy_aud_per_mw_year": mean_point,
            "risk_aware_mean_net_operating_proxy_aud_per_mw_year": mean_risk,
            "five_region_mean_margin_retention_ratio": aggregate_retention,
            "minimum_single_region_margin_retention_ratio": minimum_region_retention,
        },
        "regions": output_regions,
        "promotion_rules": promotion_rules,
        "promotion_pass": promotion_pass,
        "boundary": (
            "This gate aggregates already-settled out-of-time folds from a backward historical "
            "transport check. It is not a prospective or untouched trial. Values are historical "
            "spot-market operating proxies and exclude CAPEX, fixed O&M, network charges, FCAS, "
            "taxes, financing and complete degradation. A pass is not ROI or an investment claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.metrics.read_text(encoding="utf-8"))
    summary = summarize(source)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "summary_git_sha": os.environ.get("SUMMARY_GIT_COMMIT", "unversioned-local-run"),
        "evaluation_git_sha": os.environ.get(
            "EVALUATION_GIT_COMMIT", "unversioned-evaluation"
        ),
        "source_metrics_sha256": hashlib.sha256(args.metrics.read_bytes()).hexdigest(),
        "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
