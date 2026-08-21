from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REGIONS = ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_moving_block_bootstrap(
    day_deltas: dict[str, list[float]], samples: int = 5_000, block_length_days: int = 7
) -> dict[str, Any]:
    days = sorted(day_deltas)
    if not days or any(len(day_deltas[day]) != len(REGIONS) for day in days):
        raise ValueError("every calendar day must contain all five regional deltas")
    values = np.asarray([np.mean(day_deltas[day]) for day in days], dtype=float)
    if block_length_days < 1 or block_length_days > len(values):
        raise ValueError("block length must be between one and the number of calendar days")
    rng = np.random.default_rng(20260821)
    blocks_per_draw = int(np.ceil(len(values) / block_length_days))
    starts = rng.integers(0, len(values), size=(samples, blocks_per_draw))
    offsets = np.arange(block_length_days)
    indices = (starts[:, :, None] + offsets[None, None, :]) % len(values)
    sampled = values[indices.reshape(samples, -1)[:, : len(values)]]
    draws = np.mean(sampled, axis=1)
    return {
        "units": len(days),
        "block_length_days": block_length_days,
        "method": "paired circular moving-block bootstrap over cross-region mean daily deltas",
        "mean_daily_delta_aud_per_region": float(np.mean(values)),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "boundary": (
            "same-day cross-region dependence and within-block serial dependence are preserved; "
            "28 days still provide limited regime coverage"
        ),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if {record["region"] for record in records} != set(REGIONS):
        raise ValueError("exactly one record per NEM region is required")
    total_intervals = sum(int(record["metrics"]["test_intervals"]) for record in records)
    weighted_chronos_mae = sum(
        float(record["metrics"]["forecast"]["chronos2"]["mae"])
        * int(record["metrics"]["test_intervals"])
        for record in records
    ) / total_intervals
    weighted_best_mae = sum(
        min(
            float(record["metrics"]["forecast"]["persistence"]["mae"]),
            float(record["metrics"]["forecast"]["lightgbm"]["mae"]),
        )
        * int(record["metrics"]["test_intervals"])
        for record in records
    ) / total_intervals
    weighted_coverage = sum(
        float(record["metrics"]["chronos2_interval"]["empirical_coverage"])
        * int(record["metrics"]["test_intervals"])
        for record in records
    ) / total_intervals
    region_rows: list[dict[str, Any]] = []
    day_deltas: dict[str, list[float]] = defaultdict(list)
    total_delta = 0.0
    for record in records:
        metrics = record["metrics"]
        baseline = str(metrics["promotion_gate"]["best_economic_baseline"])
        chronos_total = float(metrics["economics"]["chronos2"]["test_net_operating_margin_proxy_aud"])
        baseline_total = float(metrics["economics"][baseline]["test_net_operating_margin_proxy_aud"])
        delta = chronos_total - baseline_total
        total_delta += delta
        region_rows.append(
            {
                "region": record["region"],
                "best_baseline": baseline,
                "chronos2_mae": metrics["forecast"]["chronos2"]["mae"],
                "best_baseline_mae": min(
                    metrics["forecast"]["persistence"]["mae"],
                    metrics["forecast"]["lightgbm"]["mae"],
                ),
                "coverage": metrics["chronos2_interval"]["empirical_coverage"],
                "net_delta_aud": delta,
            }
        )
        for row in record["daily"]:
            day_deltas[str(row["day"])].append(
                float(row["chronos2_net_operating_margin_proxy_aud"])
                - float(row[f"{baseline}_net_operating_margin_proxy_aud"])
            )
    bootstrap = paired_moving_block_bootstrap(day_deltas)
    positive_regions = sum(float(row["net_delta_aud"]) > 0 for row in region_rows)
    mae_ratio = weighted_chronos_mae / weighted_best_mae
    conditions = {
        "total_net_delta_positive": total_delta > 0,
        "paired_moving_block_bootstrap_ci_lower_positive": bootstrap["ci_lower"] > 0,
        "positive_regions_gte_3": positive_regions >= 3,
        "mae_ratio_lte_1_10": mae_ratio <= 1.10,
        "raw_interval_coverage_between_0_75_and_0_85": 0.75 <= weighted_coverage <= 0.85,
    }
    return {
        "scope": "five-region 28-day real AEMO Chronos-2/BESS transport gate",
        "regions": region_rows,
        "region_days": sum(len(record["daily"]) for record in records),
        "test_intervals": total_intervals,
        "chronos2_weighted_mae": weighted_chronos_mae,
        "best_baseline_weighted_mae": weighted_best_mae,
        "chronos2_mae_ratio": mae_ratio,
        "chronos2_raw_q10_q90_coverage": weighted_coverage,
        "total_net_delta_vs_region_best_baselines_aud": total_delta,
        "positive_regions": positive_regions,
        "paired_calendar_day_moving_block_bootstrap": bootstrap,
        "promotion_conditions": conditions,
        "promotion_pass": all(conditions.values()),
        "boundary": (
            "Historical spot-market net operating-margin proxy at 50 AUD/MWh discharged; "
            "excludes CAPEX, fixed OPEX, FCAS, network fees and investment return."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--expected-input-sha", required=True)
    args = parser.parse_args()
    records: list[dict[str, Any]] = []
    source_hashes: dict[str, dict[str, str]] = {}
    for region in REGIONS:
        root = args.input_root / region
        metrics_path = root / "metrics.json"
        daily_path = root / "daily_decisions.jsonl"
        manifest_path = root / "run_manifest.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        daily = [json.loads(line) for line in daily_path.read_text(encoding="utf-8").splitlines() if line]
        if metrics["region"] != region or manifest["git_sha"] != args.expected_git_sha:
            raise ValueError(f"region or git provenance mismatch for {region}")
        if manifest["input_sha256"] != args.expected_input_sha:
            raise ValueError(f"input provenance mismatch for {region}")
        records.append({"region": region, "metrics": metrics, "daily": daily})
        source_hashes[region] = {
            "metrics_sha256": sha256(metrics_path),
            "daily_sha256": sha256(daily_path),
            "manifest_sha256": sha256(manifest_path),
        }
    summary = summarize_records(records)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": args.expected_git_sha,
        "input_sha256": args.expected_input_sha,
        "source_hashes": source_hashes,
        "metrics_sha256": sha256(metrics_path),
        "python": sys.version,
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
