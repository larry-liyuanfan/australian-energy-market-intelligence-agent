from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from .agent import EnergyAgent
from .evidence import HybridEvidenceIndex, load_official_chunks
from .market import fixture_store, load_dispatch_store
from .schemas import AgentQueryRequest, AgentQueryResponse
from .tools import ToolRegistry

data_path = os.getenv("ENERGY_DATA_PATH")
manifest_path = os.getenv("ENERGY_DATA_MANIFEST")
evidence_path = os.getenv("ENERGY_EVIDENCE_PATH")
store = load_dispatch_store(Path(data_path), Path(manifest_path)) if data_path and manifest_path else fixture_store()
evidence_index = HybridEvidenceIndex(load_official_chunks(Path(evidence_path))) if evidence_path else None
registry = ToolRegistry(store, evidence_index)
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
