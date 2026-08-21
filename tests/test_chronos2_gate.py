from __future__ import annotations

from typing import Any

from scripts.summarize_chronos2_gate import REGIONS, summarize_records


def records(*, delta: float = 10.0, coverage: float = 0.8, chronos_mae: float = 10.0) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for region in REGIONS:
        daily = [
            {
                "day": f"2026-08-{day:02d}",
                "chronos2_net_operating_margin_proxy_aud": delta,
                "lightgbm_net_operating_margin_proxy_aud": 0.0,
            }
            for day in range(1, 29)
        ]
        output.append(
            {
                "region": region,
                "daily": daily,
                "metrics": {
                    "test_intervals": 28 * 288,
                    "forecast": {
                        "persistence": {"mae": 12.0},
                        "lightgbm": {"mae": 10.0},
                        "chronos2": {"mae": chronos_mae},
                    },
                    "chronos2_interval": {"empirical_coverage": coverage},
                    "economics": {
                        "lightgbm": {"test_net_operating_margin_proxy_aud": 0.0},
                        "chronos2": {"test_net_operating_margin_proxy_aud": 28 * delta},
                    },
                    "promotion_gate": {"best_economic_baseline": "lightgbm"},
                },
            }
        )
    return output


def test_chronos2_gate_passes_only_when_all_predeclared_conditions_pass() -> None:
    summary = summarize_records(records())
    assert summary["region_days"] == 140
    assert summary["positive_regions"] == 5
    assert summary["promotion_pass"]


def test_chronos2_gate_rejects_undercoverage_despite_positive_economics() -> None:
    summary = summarize_records(records(coverage=0.70))
    assert summary["total_net_delta_vs_region_best_baselines_aud"] > 0
    assert not summary["promotion_conditions"]["raw_interval_coverage_between_0_75_and_0_85"]
    assert not summary["promotion_pass"]


def test_chronos2_gate_rejects_mae_regression() -> None:
    summary = summarize_records(records(chronos_mae=11.1))
    assert not summary["promotion_conditions"]["mae_ratio_lte_1_10"]
    assert not summary["promotion_pass"]
