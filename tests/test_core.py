import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from energy_agent import api as api_module
from energy_agent.agent import EnergyAgent
from energy_agent.api import app
from energy_agent.battery import optimize_dispatch
from energy_agent.forecast import seasonal_conformal
from energy_agent.market import fixture_store, robust_events
from energy_agent.providers import ModelStudioPlanner
from energy_agent.schemas import TOOL_MODELS, AgentQueryRequest, BatterySpec, ToolResult
from energy_agent.tools import ToolRegistry


class TransientSlowRegistry(ToolRegistry):
    def __init__(self) -> None:
        super().__init__(fixture_store())
        self.slow_once = True

    def execute(self, name: str, arguments: dict[str, object]) -> ToolResult:
        import time

        if self.slow_once:
            self.slow_once = False
            time.sleep(0.08)
        return super().execute(name, arguments)


class UnavailableDependency:
    def ping(self) -> bool:
        raise ConnectionError("test-only failure")


def test_eight_strict_typed_tools() -> None:
    assert len(TOOL_MODELS) == 8
    with pytest.raises(ValidationError):
        TOOL_MODELS["explain_data_coverage"].model_validate({"sql": "drop table market"})


def test_battery_golden_arbitrage_and_constraints() -> None:
    spec = BatterySpec()
    result = optimize_dispatch([-100.0] * 12 + [1000.0] * 12, spec)
    assert result.gross_margin_aud > 0
    assert min(result.soc_mwh) >= 0.2 - 1e-6
    assert max(result.soc_mwh) <= 1.8 + 1e-6
    assert result.soc_mwh[0] == pytest.approx(1.0)
    assert result.soc_mwh[-1] == pytest.approx(1.0)
    assert all(not (c > 1e-7 and d > 1e-7) for c, d in zip(result.charge_mw, result.discharge_mw, strict=True))


def test_forecast_intervals_are_ordered() -> None:
    forecast = seasonal_conformal([float(i % 50) for i in range(400)], 12)
    assert len(forecast.point) == 12
    assert all(lo <= point <= hi for lo, point, hi in zip(forecast.lower, forecast.point, forecast.upper, strict=True))


def test_anomaly_detection() -> None:
    store = fixture_store()
    rows = [row for row in store.rows if row.region.value == "SA1"]
    events = robust_events(rows, 5000, 5)
    assert any(float(event["rrp"]) >= 6000 for event in events)


def test_registry_rejects_unknown_dsl() -> None:
    registry = ToolRegistry(fixture_store())
    with pytest.raises(ValueError):
        registry.validate("raw_sql", {"sql": "select *"})


def test_api_contract_and_trace() -> None:
    with TestClient(app) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        tools = client.get("/api/tools")
        assert tools.status_code == 200
        assert len(tools.json()["tools"]) == 8
        response = client.post(
            "/api/agent/query", json={"question": "Explain SA1 battery risk and price spike 2025-01-01"}
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["trace_id"]
        assert len(payload["answer"]) < 5_000
        trace = client.get(f"/api/agent/traces/{payload['trace_id']}")
        assert trace.status_code == 200
        assert trace.json()["verified_results"]
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert 'energy_agent_queries_total{status="completed"}' in metrics.text
        assert "energy_agent_query_duration_seconds_count" in metrics.text
        assert 'energy_agent_tool_calls_total{tool="search_official_evidence"' in metrics.text
        assert "energy_agent_citations_total" in metrics.text


def test_health_is_degraded_when_configured_dependency_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "redis_url", "redis://configured")
    monkeypatch.setattr(api_module, "redis_client", UnavailableDependency())
    assert api_module.health()["status"] == "degraded"
    assert api_module.health()["redis"] == "unavailable"


def test_agent_recovers_from_transient_timeout_with_bounded_backoff() -> None:
    response = EnergyAgent(TransientSlowRegistry(), timeout_seconds=0.01).run(
        AgentQueryRequest(question="Detect SA1 price events 2025-01-01")
    )
    assert response.status == "completed"
    assert any(call.recovered and call.status == "ok" for call in response.tool_calls)


def test_agent_trace_cache_is_bounded_and_reports_evictions() -> None:
    bounded = EnergyAgent(ToolRegistry(fixture_store()), trace_capacity=2)
    responses = [
        bounded.run(AgentQueryRequest(question=f"Explain SA1 data coverage request {index}")) for index in range(3)
    ]
    assert bounded.get_trace(responses[0].trace_id) is None
    assert bounded.get_trace(responses[-1].trace_id) is not None
    assert bounded.trace_stats() == (2, 2, 1)


def test_model_studio_adapter_accepts_only_registered_typed_calls() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "explain_data_coverage",
                                "arguments": json.dumps({"region": "SA1"}),
                            }
                        }
                    ]
                }
            }
        ]
    }
    provider = ModelStudioPlanner(
        "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        "test-only-not-a-secret",
        transport=lambda _request, _timeout: json.dumps(response).encode(),
    )
    calls = provider.plan("Explain SA1 data coverage", ToolRegistry(fixture_store()), 3)
    assert calls == [("explain_data_coverage", {"region": "SA1"})]
