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
        point=[-100.0] * 144 + [1000.0] * 144,
        lower=[-110.0] * 144 + [990.0] * 144,
        upper=[-90.0] * 144 + [1010.0] * 144,
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
    assert result.data["forecast_snapshot_id"] == snapshot.snapshot_id
    assert result.data["method"] == "lightgbm_quantile_v1"
    assert result.data["point"] == [-100.0] * 12

    dispatch = registry.execute(
        "optimize_battery_dispatch",
        {
            "region": "SA1",
            "window": {"start": start.isoformat(), "end": end.isoformat()},
            "settlement_mode": "historical_replay",
        },
    )
    assert dispatch.data["forecast_source"] == "offline_snapshot"
    assert dispatch.data["forecast_snapshot_id"] == snapshot.snapshot_id
    assert dispatch.data["forecast_data_sha256"] == result.data["data_sha256"]
    assert dispatch.data["forecast_model_sha256"] == result.data["model_sha256"]


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
