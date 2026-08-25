from __future__ import annotations

from datetime import UTC, datetime, timedelta

from energy_agent.market import fixture_store
from energy_agent.schemas import Region
from energy_agent.snapshots import ForecastSnapshot, ForecastSnapshotStore
from energy_agent.tools import ToolRegistry


def test_exact_offline_snapshot_precedes_seasonal_fallback() -> None:
    start = datetime(2025, 1, 2, tzinfo=UTC)
    end = start + timedelta(days=1)
    snapshot = ForecastSnapshot(
        region=Region.SA1,
        start=start,
        end=end,
        training_cutoff=start,
        created_at=start,
        data_sha256="a" * 64,
        model_sha256="b" * 64,
        model_name="lightgbm_quantile_v1",
        point=[1.0] * 288,
        lower=[0.0] * 288,
        upper=[2.0] * 288,
    )
    registry = ToolRegistry(fixture_store(), forecast_snapshots=ForecastSnapshotStore([snapshot]))
    result = registry.execute(
        "forecast_price_risk",
        {
            "region": "SA1",
            "window": {"start": start.isoformat(), "end": end.isoformat()},
            "horizon_intervals": 12,
        },
    )
    assert result.data["forecast_source"] == "offline_snapshot"
    assert result.data["method"] == "lightgbm_quantile_v1"
    assert result.data["point"] == [1.0] * 12


def test_snapshot_rejects_future_training_cutoff() -> None:
    start = datetime(2025, 1, 2, tzinfo=UTC)
    try:
        ForecastSnapshot(
            region=Region.SA1,
            start=start,
            end=start + timedelta(days=1),
            training_cutoff=start + timedelta(minutes=5),
            created_at=start,
            data_sha256="a" * 64,
            model_sha256="b" * 64,
            model_name="invalid",
            point=[1.0],
            lower=[0.0],
            upper=[2.0],
        )
    except ValueError as exc:
        assert "as-of contract" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("future training cutoff was accepted")
