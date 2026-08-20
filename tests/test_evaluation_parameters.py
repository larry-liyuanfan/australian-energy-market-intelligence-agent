from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from energy_agent.ensemble_gate import summarize_dispatch_ensemble
from energy_agent.evaluation import (
    adaptive_conformal_bounds,
    citation_structure_metrics,
    optimizer_action_weights,
    parse_degradation_costs,
    residual_price_scenarios,
    seasonal_fold_windows,
    select_decision_weighted_model,
    select_dispatch_ensemble_weight,
    select_tail_policy,
)


def test_slurm_evaluation_pins_code_and_manifest_to_one_commit() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "slurm" / "evaluate_real.sbatch"
    ).read_text(encoding="utf-8")

    assert 'git clone --shared --no-checkout "${SOURCE_REPO}" "${CODE_ROOT}"' in script
    assert 'git -C "${CODE_ROOT}" checkout --detach "${ENERGY_GIT_COMMIT}"' in script
    assert 'git -C "${SOURCE_REPO}" archive' not in script
    assert "export ENERGY_GIT_COMMIT" in script
    assert "%A_%a.out" in script

    merge_script = (
        Path(__file__).parents[1] / "scripts" / "slurm" / "merge_real_array.sbatch"
    ).read_text(encoding="utf-8")
    assert 'git clone --shared --no-checkout "${SOURCE_REPO}" "${CODE_ROOT}"' in merge_script
    assert 'git -C "${CODE_ROOT}" checkout --detach "${ENERGY_GIT_COMMIT}"' in merge_script
    assert "export ENERGY_GIT_COMMIT" in merge_script
    assert 'cd "${CODE_ROOT}"' in merge_script

    ensemble_script = (
        Path(__file__).parents[1] / "scripts" / "slurm" / "evaluate_dispatch_ensemble.sbatch"
    ).read_text(encoding="utf-8")
    assert '${ENERGY_GIT_COMMIT:?set exact pushed commit}' in ensemble_script
    assert 'checkout --detach "${ENERGY_GIT_COMMIT}"' in ensemble_script
    assert '[[ "$(git -C "${CODE_ROOT}" rev-parse HEAD)" == "${ENERGY_GIT_COMMIT}" ]]' in ensemble_script
    assert 'REGION="${REGION:-SA1}"' in ensemble_script
    assert "git -C \"${SOURCE_REPO}\" fetch" not in ensemble_script
    assert 'OUTPUT_DIR="${OUTPUT_ROOT}/${REGION}"' in ensemble_script

    merge_ensemble_script = (
        Path(__file__).parents[1] / "scripts" / "slurm" / "merge_dispatch_ensemble.sbatch"
    ).read_text(encoding="utf-8")
    assert '--input-template "${INPUT_TEMPLATE}"' in merge_ensemble_script
    assert '${INPUT_ROOT:?set array result root when INPUT_TEMPLATE is absent}' in merge_ensemble_script


def test_cross_region_ensemble_gate_requires_majority_and_tail_safety() -> None:
    def region(delta: float, tail: float) -> dict[str, object]:
        baseline = {
            "days": 28,
            "net_operating_margin_proxy_aud": 100.0,
            "daily_cvar05_aud": -10.0,
        }
        ensemble = {
            "days": 28,
            "net_operating_margin_proxy_aud": 100.0 + delta,
            "daily_cvar05_aud": tail,
        }
        return {
            "overall": {"baseline": baseline, "ensemble_selected": ensemble},
            "selected_ensemble_weight_counts": {"0.0": 0, "0.25": 1},
            "folds": {
                "fold": {
                    "test_economics": {"baseline": baseline, "ensemble_selected": ensemble}
                }
            },
        }

    passed = summarize_dispatch_ensemble(
        {"A": region(10.0, -10.5), "B": region(5.0, -10.0), "C": region(-1.0, -10.0)},
        bootstrap_repetitions=100,
    )
    assert passed["aggregate"]["promotion_pass"] is True
    failed = summarize_dispatch_ensemble(
        {"A": region(10.0, -12.0), "B": region(5.0, -10.0), "C": region(-1.0, -10.0)},
        bootstrap_repetitions=100,
    )
    assert failed["aggregate"]["promotion_pass"] is False


def test_residual_scenarios_are_deterministic_and_use_complete_days() -> None:
    forecast = [100.0] * 288
    residuals = np.concatenate((np.full(288, -10.0), np.full(288, 20.0)))
    first = residual_price_scenarios(
        forecast,
        residuals,
        scenario_count=4,
        seed_material="SA1|winter|2026-07-01",
    )
    second = residual_price_scenarios(
        forecast,
        residuals,
        scenario_count=4,
        seed_material="SA1|winter|2026-07-01",
    )
    assert first == second
    assert first[0] == forecast
    assert all(set(scenario).issubset({90.0, 120.0}) for scenario in first[1:])


@pytest.mark.parametrize("separator", [",", ":", ";"])
def test_degradation_costs_accept_cli_and_slurm_safe_separators(separator: str) -> None:
    assert parse_degradation_costs(separator.join(("0", "25", "100"))) == (0.0, 25.0, 100.0)


def test_degradation_costs_reject_empty_or_negative_values() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        parse_degradation_costs("")
    with pytest.raises(ValueError, match="non-negative"):
        parse_degradation_costs("0:-1")


def test_seasonal_windows_use_prior_calibration_and_cover_four_seasons() -> None:
    folds = seasonal_fold_windows(
        datetime(2025, 8, 18, tzinfo=UTC),
        datetime(2026, 8, 17, 23, 55, tzinfo=UTC),
    )
    assert [fold.name for fold in folds] == [
        "spring-2025",
        "summer-2026",
        "autumn-2026",
        "winter-2026",
    ]
    assert all(fold.calibration_start < fold.test_start < fold.test_end for fold in folds)
    assert all((fold.test_end - fold.test_start).days == 28 for fold in folds)


def test_adaptive_conformal_emits_before_observing_each_test_label() -> None:
    actual = np.asarray([0.0, 100.0, 100.0])
    predicted = np.zeros(3)
    calibration = np.ones(100)
    result = adaptive_conformal_bounds(actual, predicted, calibration, gamma=0.05, window=100)
    assert result.upper[0] == pytest.approx(1.0)
    assert result.upper[1] == pytest.approx(1.0)
    assert result.alpha_history[2] < result.alpha_history[1]
    assert len(result.lower) == len(actual)


def test_citation_structure_metrics_separate_presence_from_validity() -> None:
    metrics = citation_structure_metrics(
        "Verified outputs:\n- first claim [@ev-1]\n- second claim [@missing]\n- third claim",
        {"ev-1"},
    )
    assert metrics["claim_citation_completeness"] == pytest.approx(2 / 3)
    assert metrics["citation_id_validity"] == pytest.approx(0.5)


def test_tail_policy_selects_improvement_or_falls_back_to_point() -> None:
    selected = select_tail_policy(
        [1.0, 2.0, 10.0, 10.0, 10.0],
        {"0.5": [2.0, 3.0, 9.0, 9.0, 9.0]},
    )
    assert selected["selected_policy"] == "0.5"
    fallback = select_tail_policy(
        [1.0, 2.0, 10.0, 10.0, 10.0],
        {"0.5": [-5.0, -4.0, 20.0, 20.0, 20.0]},
    )
    assert fallback["selected_policy"] == "point"


def test_optimizer_action_weights_are_bounded_and_action_sensitive() -> None:
    weights = optimizer_action_weights(
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 1.0],
        power_mw=1.0,
        emphasis=4.0,
    )
    assert weights.tolist() == pytest.approx([1.0, 3.0, 5.0])


def test_decision_weighted_selector_uses_mean_and_tail_guardrail() -> None:
    selected = select_decision_weighted_model(
        [1.0, 2.0, 10.0, 10.0, 10.0],
        [2.0, 3.0, 10.0, 10.0, 10.0],
    )
    assert selected["selected_model"] == "decision_weighted"
    rejected = select_decision_weighted_model(
        [1.0, 2.0, 10.0, 10.0, 10.0],
        [-10.0, -9.0, 20.0, 20.0, 20.0],
    )
    assert rejected["selected_model"] == "baseline"


def test_dispatch_ensemble_selector_uses_fixed_grid_and_tail_guardrail() -> None:
    selected = select_dispatch_ensemble_weight(
        [1.0, 2.0, 10.0, 10.0, 10.0],
        {
            "0.25": [2.0, 3.0, 10.0, 10.0, 10.0],
            "0.5": [2.0, 3.0, 12.0, 12.0, 12.0],
        },
    )
    assert selected["selected_weight"] == pytest.approx(0.5)
    fallback = select_dispatch_ensemble_weight(
        [1.0, 2.0, 10.0, 10.0, 10.0],
        {"0.25": [-10.0, -9.0, 20.0, 20.0, 20.0]},
    )
    assert fallback["selected_weight"] == pytest.approx(0.0)
