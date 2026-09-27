from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from energy_agent.decision_demo import build_demo_bundle, evidence_publication_role
from energy_agent.market import fixture_store
from energy_agent.model_agent import AgentPath, MemoryMode, ModelDrivenAgent
from energy_agent.schemas import Evidence, Region
from energy_agent.snapshots import ForecastSnapshot, ForecastSnapshotStore
from energy_agent.tools import ToolRegistry


def fixture_registry() -> ToolRegistry:
    store = fixture_store()
    common = {
        "url": "https://example.invalid/explicit-test-fixture", "published_at": datetime(2025, 2, 1, tzinfo=UTC),
        "retrieved_at": datetime(2025, 3, 1, tzinfo=UTC), "sha256": "a" * 64,
        "snippet": "Synthetic evidence fixture", "evidence_type": "explanatory",
    }
    store.evidence = [
        Evidence(evidence_id="fixture-text", title="Test text", **common),
        Evidence(evidence_id="fixture-chart", title="Test chart", modality="chart", figure_id="Figure 1",
                 source_cell_preview="Q1 | 25 | 50", **common),
    ]
    return ToolRegistry(store)


def test_recorded_demo_recomputes_settlement_and_labels_retrospective_evidence() -> None:
    registry = fixture_registry()
    question = "Replay SA1 on 2025-01-03 with 1MW/2MWh BESS and chart evidence; degradation cost 0."
    run = ModelDrivenAgent(registry, None).run_turn(
        question, conversation_id="test-demo", path=AgentPath.deterministic, memory_mode=MemoryMode.structured_state,
    )
    bundle = build_demo_bundle(run, registry.store, question)
    assert all(bundle["verification"].values())
    assert bundle["settlement"]["realised_gross_aud"] > 0
    assert bundle["settlement"]["realised_gross_aud"] == bundle["settlement"]["realised_operating_proxy_aud"]
    assert all(c["publication_role"] == "published_later_retrospective_only" for c in bundle["citations"])
    assert len(bundle["series"]["actual"]) == 48
    forecast = next(result for result in run.results if result.tool_name == "forecast_price_risk")
    forecast.data["signal_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="forecast_signal_matches_dispatch"):
        build_demo_bundle(run, registry.store, question)


def test_unknown_report_date_cannot_be_called_available_as_of() -> None:
    evidence = fixture_registry().store.evidence[0].model_copy(update={"published_at": None})
    assert evidence_publication_role(evidence, datetime.now(UTC)) == "publication_unknown_retrospective_only"


def test_short_snapshot_cannot_diverge_forecast_from_dispatch() -> None:
    start = datetime(2025, 1, 2, tzinfo=UTC)
    snapshot = ForecastSnapshot(
        region=Region.SA1, start=start, end=start + timedelta(days=1), training_cutoff=start,
        created_at=start, data_sha256="a" * 64, model_sha256="b" * 64, model_name="short-test-only",
        point=[50.0] * 12, lower=[40.0] * 12, upper=[60.0] * 12,
    )
    store = ForecastSnapshotStore([snapshot])
    assert store.get(Region.SA1, start, start + timedelta(days=1)) is None
