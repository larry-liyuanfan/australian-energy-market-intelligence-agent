import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from energy_agent.agent import EnergyAgent
from energy_agent.api import app
from energy_agent.battery import optimize_dispatch
from energy_agent.forecast import seasonal_conformal
from energy_agent.market import fixture_store, robust_events
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


def test_agent_recovers_from_transient_timeout_with_bounded_backoff() -> None:
    response = EnergyAgent(TransientSlowRegistry(), timeout_seconds=0.01).run(
        AgentQueryRequest(question="Detect SA1 price events 2025-01-01")
    )
    assert response.status == "completed"
    assert any(call.recovered and call.status == "ok" for call in response.tool_calls)
