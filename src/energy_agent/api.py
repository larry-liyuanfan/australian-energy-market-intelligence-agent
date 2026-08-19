from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from .agent import EnergyAgent
from .evidence import HybridEvidenceIndex, load_official_chunks
from .market import fixture_store, load_dispatch_store
from .providers import ModelStudioPlanner
from .schemas import AgentQueryRequest, AgentQueryResponse
from .tools import ToolRegistry

data_path = os.getenv("ENERGY_DATA_PATH")
manifest_path = os.getenv("ENERGY_DATA_MANIFEST")
evidence_path = os.getenv("ENERGY_EVIDENCE_PATH")
store = load_dispatch_store(Path(data_path), Path(manifest_path)) if data_path and manifest_path else fixture_store()
evidence_index = HybridEvidenceIndex(load_official_chunks(Path(evidence_path))) if evidence_path else None
registry = ToolRegistry(store, evidence_index)
planner_provider = ModelStudioPlanner.from_environment()
agent = EnergyAgent(registry, planner_provider=planner_provider)
redis_client = None
elasticsearch_client = None
if redis_url := os.getenv("ENERGY_REDIS_URL"):
    try:
        from redis import Redis

        redis_client = Redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
    except Exception:
        redis_client = None
if elasticsearch_url := os.getenv("ENERGY_ELASTICSEARCH_URL"):
    try:
        from elasticsearch import Elasticsearch

        elasticsearch_client = Elasticsearch(elasticsearch_url)
        elasticsearch_client.info()
    except Exception:
        elasticsearch_client = None
app = FastAPI(title="Australian Energy Market Intelligence Agent", version="0.1.0")


@app.get("/healthz")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "data_version": store.data_version,
        "rows": len(store.rows),
        "evidence_chunks": len(evidence_index.documents) if evidence_index else 0,
        "redis": "connected" if redis_client else "disabled_or_unavailable",
        "elasticsearch": "connected" if elasticsearch_client else "disabled_or_unavailable",
        "model_provider": planner_provider.name if planner_provider else "deterministic",
    }


@app.post("/api/agent/query", response_model=AgentQueryResponse)
def query(request: AgentQueryRequest) -> AgentQueryResponse:
    response = agent.run(request)
    if redis_client:
        redis_client.setex(
            f"energy:trace:{response.trace_id}",
            86400,
            json.dumps(agent.traces[response.trace_id]),
        )
    return response


@app.get("/api/agent/traces/{trace_id}")
def trace(trace_id: str) -> dict[str, object]:
    if trace_id not in agent.traces and redis_client:
        cached = redis_client.get(f"energy:trace:{trace_id}")
        if cached:
            decoded = json.loads(cached)
            if isinstance(decoded, dict):
                return decoded
    if trace_id not in agent.traces:
        raise HTTPException(404, "trace not found")
    return agent.traces[trace_id]


@app.get("/api/tools")
def tools() -> dict[str, object]:
    return {"tools": registry.specs(), "arbitrary_sql_or_dsl_allowed": False}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return "energy_agent_up 1\n"
