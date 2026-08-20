from __future__ import annotations

from scripts.summarize_risk_transport import summarize


def _metrics(
    *, tail_lift: float = 2.0, risk_annualised: float = 95.0, weak_regions: int = 0
) -> dict:
    regions = {}
    names = ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")
    for region_index, region in enumerate(names):
        region_tail_lift = -1.0 if region_index < weak_regions else tail_lift
        folds = {}
        for fold_index in range(8):
            folds[f"fold-{fold_index}"] = {
                "degradation_sensitivity": {
                    "50": {
                        "net_operating_margin_proxy_aud": 20.0,
                        "risk_aware_net_margin_aud": 19.0,
                        "daily_net_margin_cvar05": -10.0,
                        "risk_aware_daily_net_margin_cvar05": -10.0 + region_tail_lift,
                        "risk_policy_selection": {"selected_policy": "0.5"},
                    }
                }
            }
        regions[region] = {
            "folds": folds,
            "overall_degradation_sensitivity": {
                "50": {
                    "net_operating_proxy_aud_per_mw_year": 100.0,
                    "risk_aware_net_operating_proxy_aud_per_mw_year": risk_annualised,
                    "daily_net_margin_cvar05": -10.0,
                    "risk_aware_daily_net_margin_cvar05": -10.0 + region_tail_lift,
                    "risk_aware_cvar05_lift_vs_point_aud": region_tail_lift,
                }
            },
        }
    return {"regions": regions}


def test_risk_transport_gate_requires_tail_lift_and_margin_guardrails() -> None:
    passed = summarize(_metrics())
    failed_tail = summarize(_metrics(tail_lift=-1.0))
    failed_mean = summarize(_metrics(risk_annualised=89.0))
    failed_regions = summarize(_metrics(weak_regions=3))

    assert passed["promotion_pass"] is True
    assert passed["folds"] == 40
    assert passed["non_point_policy_selections"] == 40
    assert passed["aggregate"]["five_region_mean_margin_retention_ratio"] == 0.95
    assert failed_tail["promotion_pass"] is False
    assert failed_mean["promotion_pass"] is False
    assert failed_regions["promotion_pass"] is False


def test_risk_transport_gate_counts_point_fallback_as_no_tail_win() -> None:
    metrics = _metrics()
    fold = metrics["regions"]["NSW1"]["folds"]["fold-0"]["degradation_sensitivity"]["50"]
    fold["risk_policy_selection"]["selected_policy"] = "point"
    fold["risk_aware_daily_net_margin_cvar05"] = fold["daily_net_margin_cvar05"]
    result = summarize(metrics)
    assert result["non_point_policy_selections"] == 39


def test_risk_transport_protocol_freezes_boundaries() -> None:
    from pathlib import Path

    root = Path(__file__).parents[1]
    document = (root / "docs" / "RISK_TRANSPORT_GATE.md").read_text(encoding="utf-8")
    job = (root / "scripts" / "slurm" / "summarize_risk_transport.sbatch").read_text(
        encoding="utf-8"
    )
    assert "Before inspecting the corrected two-year aggregate risk result" in document
    assert "not a prospective trial" in document
    assert "at least 95%" in document and "at least 90%" in document
    assert '"${SUMMARY_GIT_COMMIT}^{commit}"' in job
    assert '"${EVALUATION_GIT_COMMIT}^{commit}"' in job
    assert "history-transport-e79b2b7-merged/metrics.json" in job
