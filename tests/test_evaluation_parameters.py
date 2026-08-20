from datetime import UTC, datetime

import pytest

from energy_agent.evaluation import parse_degradation_costs, seasonal_fold_windows


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
