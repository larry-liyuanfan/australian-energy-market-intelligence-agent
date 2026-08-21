from __future__ import annotations

from scripts.summarize_history_transport import summarize


def _metrics(*, wins_per_region: int, annualised: float) -> dict:
    regions = {}
    for region in ("NSW1", "QLD1", "SA1", "TAS1", "VIC1"):
        folds = {}
        for index in range(4):
            wins = index < wins_per_region
            folds[f"fold-{index}"] = {
                "degradation_sensitivity": {
                    "50": {
                        "net_operating_margin_proxy_aud": 20.0 if wins else 5.0,
                        "rule_net_margin_aud": 10.0,
                    }
                }
            }
        regions[region] = {
            "folds": folds,
            "overall_degradation_sensitivity": {
                "50": {
                    "test_days": 112,
                    "net_operating_proxy_aud_per_mw_year": annualised,
                    "rule_net_margin_aud": 20.0,
                    "relative_rule_lift": 0.5,
                    "equivalent_full_cycles": 10.0,
                    "positive_day_share": 0.75,
                    "daily_net_margin_cvar05": -1.0,
                    "oracle_capture_rate": 0.7,
                    "oracle_regret_aud": 100.0,
                }
            },
            "decision_focused_summary": {
                "50": {"lightgbm_mae_wins": 2}
            },
        }
    return {"regions": regions}


def test_history_transport_gate_passes_only_declared_thresholds() -> None:
    passed = summarize(_metrics(wins_per_region=3, annualised=100.0))
    failed_wins = summarize(_metrics(wins_per_region=2, annualised=100.0))
    failed_regions = summarize(_metrics(wins_per_region=3, annualised=-1.0))
    assert passed["promotion_pass"] is True
    assert passed["lightgbm_dispatch_win_share"] == 0.75
    assert passed["aggregate"]["lightgbm_mae_wins"] == 10
    assert passed["aggregate"]["mean_net_operating_proxy_aud_per_mw_year"] == 100.0
    assert failed_wins["promotion_pass"] is False
    assert failed_regions["promotion_pass"] is False
    assert "not a prospective" in passed["boundary"]
