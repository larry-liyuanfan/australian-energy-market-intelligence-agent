from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scripts.evaluate_chronos2_bess import (
    INTERVALS_PER_DAY,
    build_windows,
    conformal_interval_adjustment,
    day_ahead_features,
    paired_moving_block_bootstrap,
)


def rows(days: int = 8) -> list[dict[str, object]]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        {
            "timestamp": start + timedelta(minutes=5 * index),
            "rrp": float(index % 100),
            "total_demand_mw": 1000.0 + index,
            "available_generation_mw": 1200.0 + index,
            "net_interchange_mw": float(index % 20),
        }
        for index in range(days * INTERVALS_PER_DAY)
    ]


def test_covariate_windows_separate_past_market_and_known_calendar_values() -> None:
    source = rows()
    test_day = source[-1]["timestamp"]
    assert isinstance(test_day, datetime)
    window = build_windows(
        source,
        [test_day.date().isoformat()],
        region="SA1",
        context_days=7,
        use_covariates=True,
    )[0]
    assert set(window.future_covariates) == {
        "minute_sin",
        "minute_cos",
        "weekday_sin",
        "weekday_cos",
    }
    assert "total_demand_mw" in window.context_covariates
    assert "total_demand_mw" not in window.future_covariates
    window.validate()


def test_lightgbm_feature_builder_adds_market_covariates_without_changing_rows() -> None:
    source = rows()
    base_x, base_y, base_times = day_ahead_features(source)
    cov_x, cov_y, cov_times = day_ahead_features(source, use_covariates=True)
    assert cov_x.shape[0] == base_x.shape[0]
    assert cov_x.shape[1] == base_x.shape[1] + 15
    assert cov_y.tolist() == base_y.tolist()
    assert cov_times == base_times


def test_conformal_adjustment_uses_finite_sample_higher_quantile() -> None:
    assert conformal_interval_adjustment([-2.0, -1.0, 0.0, 1.0, 2.0]) == 2.0
    with pytest.raises(ValueError, match="coverage"):
        conformal_interval_adjustment([], coverage=0.8)


def test_paired_moving_block_bootstrap_preserves_positive_constant_delta() -> None:
    result = paired_moving_block_bootstrap([2.0] * 28, [1.0] * 28)
    assert result["ci_lower"] == pytest.approx(1.0)
    assert result["block_length_days"] == 7
