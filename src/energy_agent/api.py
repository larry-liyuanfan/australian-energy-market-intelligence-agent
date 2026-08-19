from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from .agent import EnergyAgent
from .market import fixture_store
from .schemas import AgentQueryRequest, AgentQueryResponse
from .tools import ToolRegistry

store = fixture_store()
registry = ToolRegistry(store)
agent = EnergyAgent(registry)
app = FastAPI(title="Australian Energy Market Intelligence Agent", version="0.1.0")


@app.post("/api/agent/query", response_model=AgentQueryResponse)
def query(request: AgentQueryRequest) -> AgentQueryResponse:
    return agent.run(request)


@app.get("/api/agent/traces/{trace_id}")
def trace(trace_id: str) -> dict[str, object]:
    if trace_id not in agent.traces:
        raise HTTPException(404, "trace not found")
    return agent.traces[trace_id]


@app.get("/api/tools")
def tools() -> dict[str, object]:
    return {"tools": registry.specs(), "arbitrary_sql_or_dsl_allowed": False}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return "energy_agent_up 1\n"
