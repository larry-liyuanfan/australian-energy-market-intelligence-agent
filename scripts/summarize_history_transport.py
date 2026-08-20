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


def bootstrap_mean_interval(values: list[float], *, seed: int = 20260821) -> list[float]:
    if not values:
        raise ValueError("bootstrap values are empty")
    rng = np.random.default_rng(seed)
    samples = rng.choice(np.asarray(values), size=(5000, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def summarize(metrics: dict[str, Any]) -> dict[str, Any]:
    regions = metrics.get("regions", {})
    if set(regions) != {"NSW1", "QLD1", "SA1", "TAS1", "VIC1"}:
        raise ValueError("five-region metrics are required")
    fold_deltas: list[float] = []
    output_regions: dict[str, Any] = {}
    wins = 0
    folds = 0
    positive_regions = 0
    annualised_values: list[float] = []
    annualised_rule_values: list[float] = []
    cycle_values: list[float] = []
    capture_values: list[float] = []
    positive_day_values: list[float] = []
    mae_wins = 0
    for region, payload in sorted(regions.items()):
        region_wins = 0
        for fold_name, fold in sorted(payload["folds"].items()):
            economics = fold["degradation_sensitivity"][COST_KEY]
            delta = float(economics["net_operating_margin_proxy_aud"]) - float(
                economics["rule_net_margin_aud"]
            )
            fold_deltas.append(delta)
            region_wins += int(delta > 0)
            folds += 1
        wins += region_wins
        overall = payload["overall_degradation_sensitivity"][COST_KEY]
        annualised = float(overall["net_operating_proxy_aud_per_mw_year"])
        test_days = int(overall["test_days"])
        rule_annualised = float(overall["rule_net_margin_aud"]) / test_days * 365
        decision = payload["decision_focused_summary"][COST_KEY]
        mae_wins += int(decision["lightgbm_mae_wins"])
        annualised_values.append(annualised)
        annualised_rule_values.append(rule_annualised)
        cycle_values.append(float(overall["equivalent_full_cycles"]))
        capture_values.append(float(overall["oracle_capture_rate"]))
        positive_day_values.append(float(overall["positive_day_share"]))
        positive_regions += int(annualised > 0)
        output_regions[region] = {
            "folds": len(payload["folds"]),
            "lightgbm_dispatch_wins": region_wins,
            "lightgbm_mae_wins": int(decision["lightgbm_mae_wins"]),
            "net_operating_proxy_aud_per_mw_year": annualised,
            "rule_net_operating_proxy_aud_per_mw_year": rule_annualised,
            "relative_rule_lift": float(overall["relative_rule_lift"]),
            "equivalent_full_cycles": float(overall["equivalent_full_cycles"]),
            "positive_day_share": float(overall["positive_day_share"]),
            "daily_net_margin_cvar05_aud": float(overall["daily_net_margin_cvar05"]),
            "oracle_capture_rate": float(overall["oracle_capture_rate"]),
            "oracle_regret_aud": float(overall["oracle_regret_aud"]),
        }
    win_share = wins / folds if folds else 0.0
    return {
        "evidence_status": "verified-real-if-run-completes",
        "gate_kind": "backward-temporal-transport-not-prospective",
        "cost_aud_per_mwh_discharged": 50.0,
        "folds": folds,
        "lightgbm_dispatch_wins": wins,
        "lightgbm_dispatch_win_share": win_share,
        "positive_annualised_regions": positive_regions,
        "aggregate": {
            "mean_net_operating_proxy_aud_per_mw_year": float(np.mean(annualised_values)),
            "mean_rule_net_operating_proxy_aud_per_mw_year": float(
                np.mean(annualised_rule_values)
            ),
            "relative_lift_vs_rule": float(
                np.mean(annualised_values) / np.mean(annualised_rule_values) - 1
            ),
            "mean_equivalent_full_cycles": float(np.mean(cycle_values)),
            "mean_oracle_capture_rate": float(np.mean(capture_values)),
            "mean_positive_day_share": float(np.mean(positive_day_values)),
            "lightgbm_mae_wins": mae_wins,
            "mae_win_share": mae_wins / folds if folds else 0.0,
        },
        "fold_delta_mean_aud": float(np.mean(fold_deltas)),
        "fold_delta_mean_95pct_paired_bootstrap": bootstrap_mean_interval(fold_deltas),
        "regions": output_regions,
        "promotion_rules": {
            "minimum_win_share": 0.6,
            "minimum_positive_regions": 4,
        },
        "promotion_pass": win_share >= 0.6 and positive_regions >= 4,
        "boundary": (
            "This is a backward historical transport check of a frozen protocol, not a prospective "
            "or untouched final test. Net margin is a historical spot-market operating proxy and "
            "excludes CAPEX, network charges, FCAS, taxes, financing and complete degradation."
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
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
