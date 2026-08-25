from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from energy_agent.agent import EnergyAgent
from energy_agent.api import app
from energy_agent.market import fixture_store
from energy_agent.schemas import AgentQueryRequest
from energy_agent.tools import ToolRegistry


def test_event_workflow_is_dependent_and_reaches_diagnosis() -> None:
    agent = EnergyAgent(ToolRegistry(fixture_store()))
    response = agent.run(
        AgentQueryRequest(
            question="Explain what happened during the SA1 price spike on 2025-01-01",
            max_tool_calls=8,
        )
    )
    names = [call.name for call in response.tool_calls if call.status == "ok"]
    assert response.workflow_type == "event_diagnosis"
    assert names.index("detect_price_events") < names.index("diagnose_price_event")
    assert response.market_context["diagnose_price_event"]["event_minus_before"]["rrp"] is not None
    trace = agent.get_trace(response.trace_id)
    assert trace is not None
    assert trace["states"].index("market_context") < trace["states"].index("event_diagnosis")


def test_historical_replay_separates_plan_and_realised_settlement() -> None:
    response = EnergyAgent(ToolRegistry(fixture_store())).run(
        AgentQueryRequest(
            question="Replay a 1 MW / 2 MWh BESS dispatch for SA1 on 2025-01-03",
            max_tool_calls=8,
        )
    )
    assert response.status == "completed"
    assert response.workflow_type == "decision_replay"
    assert response.forecast["forecast_source"] == "seasonal_fallback"
    assert response.forecast["training_cutoff"].startswith("2025-01-03")
    assert response.dispatch["settlement_mode"] == "historical_replay"
    assert response.dispatch["margin_basis"] == "historical_actual_settlement_after_as_of_schedule"
    assert response.dispatch["planned_margin_aud"] is not None
    assert response.dispatch["realized_margin_aud"] is not None
    assert response.dispatch["oracle_regret_aud"] is not None
    assert response.verification["economic_boundary_present"]


def test_all_eight_tools_are_reachable_across_bounded_workflows() -> None:
    agent = EnergyAgent(ToolRegistry(fixture_store()))
    questions = [
        "Get a market snapshot and explain SA1 on 2025-01-01",
        "Compare NSW1 and VIC1 on 2025-01-02",
        "Explain SA1 data coverage on 2025-01-02",
        "Forecast SA1 price risk on 2025-01-02",
        "Explain the SA1 price spike on 2025-01-01",
        "Replay SA1 BESS dispatch on 2025-01-03",
    ]
    observed: set[str] = set()
    for question in questions:
        response = agent.run(AgentQueryRequest(question=question, max_tool_calls=8))
        observed.update(call.name for call in response.tool_calls)
    assert observed == {agent.registry.specs()[index]["name"] for index in range(8)}


def test_historical_replay_rejects_incomplete_actual_window() -> None:
    registry = ToolRegistry(fixture_store())
    start = datetime(2025, 1, 2, tzinfo=UTC)
    result = registry.execute(
        "optimize_battery_dispatch",
        {
            "region": "SA1",
            "window": {"start": start.isoformat(), "end": (start + timedelta(minutes=30)).isoformat()},
            "settlement_mode": "historical_replay",
        },
    )
    assert result.data["realized_margin_aud"] is not None
    assert result.data["margin_basis"] == "historical_actual_settlement_after_as_of_schedule"


def test_demo_is_served_without_external_frontend_runtime() -> None:
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "NEM Decision Replay Agent" in response.text
    assert "cdn" not in response.text.lower()
