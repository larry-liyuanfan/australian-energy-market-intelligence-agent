"""Cross-region promotion gate for leakage-safe dispatch ensembles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def summarize_dispatch_ensemble(
    regions: Mapping[str, Mapping[str, Any]],
    *,
    tail_tolerance_fraction: float = 0.1,
    bootstrap_repetitions: int = 5000,
    seed: int = 20260821,
) -> dict[str, Any]:
    """Aggregate fixed regional results and apply a predeclared release gate."""

    if not regions or tail_tolerance_fraction < 0 or bootstrap_repetitions < 100:
        raise ValueError("invalid ensemble gate inputs")
    region_rows: dict[str, dict[str, Any]] = {}
    fold_deltas: list[float] = []
    total_baseline = 0.0
    total_ensemble = 0.0
    total_days = 0
    for region, payload in sorted(regions.items()):
        overall = payload["overall"]
        baseline = overall["baseline"]
        ensemble = overall["ensemble_selected"]
        baseline_total = float(baseline["net_operating_margin_proxy_aud"])
        ensemble_total = float(ensemble["net_operating_margin_proxy_aud"])
        baseline_tail = float(baseline["daily_cvar05_aud"])
        ensemble_tail = float(ensemble["daily_cvar05_aud"])
        tail_floor = baseline_tail - tail_tolerance_fraction * max(abs(baseline_tail), 1.0)
        region_rows[region] = {
            "days": int(baseline["days"]),
            "baseline_net_proxy_aud": baseline_total,
            "ensemble_net_proxy_aud": ensemble_total,
            "delta_aud": ensemble_total - baseline_total,
            "baseline_cvar05_aud": baseline_tail,
            "ensemble_cvar05_aud": ensemble_tail,
            "tail_floor_aud": tail_floor,
            "tail_gate_pass": ensemble_tail >= tail_floor,
            "selected_weight_counts": payload["selected_ensemble_weight_counts"],
        }
        total_baseline += baseline_total
        total_ensemble += ensemble_total
        total_days += int(baseline["days"])
        for fold in payload["folds"].values():
            economics = fold["test_economics"]
            fold_deltas.append(
                float(economics["ensemble_selected"]["net_operating_margin_proxy_aud"])
                - float(economics["baseline"]["net_operating_margin_proxy_aud"])
            )
    values = np.asarray(fold_deltas, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(bootstrap_repetitions, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    positive_regions = sum(row["delta_aud"] > 0 for row in region_rows.values())
    required_positive_regions = len(region_rows) // 2 + 1
    total_delta = total_ensemble - total_baseline
    promotion_pass = (
        positive_regions >= required_positive_regions
        and total_delta > 0
        and all(row["tail_gate_pass"] for row in region_rows.values())
    )
    return {
        "regions": region_rows,
        "aggregate": {
            "region_days": total_days,
            "baseline_net_proxy_aud": total_baseline,
            "ensemble_net_proxy_aud": total_ensemble,
            "delta_aud": total_delta,
            "baseline_aud_per_mw_year": total_baseline * 365 / total_days,
            "ensemble_aud_per_mw_year": total_ensemble * 365 / total_days,
            "relative_lift": total_delta / total_baseline if total_baseline else 0.0,
            "positive_regions": positive_regions,
            "required_positive_regions": required_positive_regions,
            "paired_region_season_delta_mean_aud": float(values.mean()),
            "paired_region_season_delta_mean_95_interval_aud": [
                float(np.quantile(bootstrap_means, 0.025)),
                float(np.quantile(bootstrap_means, 0.975)),
            ],
            "promotion_pass": promotion_pass,
        },
        "gate": {
            "minimum_positive_regions": required_positive_regions,
            "requires_positive_aggregate_delta": True,
            "tail_tolerance_fraction": tail_tolerance_fraction,
            "bootstrap_unit": "region-season total net operating margin proxy",
            "bootstrap_repetitions": bootstrap_repetitions,
            "bootstrap_seed": seed,
        },
    }
