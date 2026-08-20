from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from .agent import EnergyAgent
from .evidence import ElasticsearchHybridEvidenceIndex, EvidenceIndex, HybridEvidenceIndex, load_official_chunks
from .market import fixture_store, load_dispatch_store
from .metrics import ServiceMetrics
from .providers import ModelStudioPlanner
from .schemas import AgentQueryRequest, AgentQueryResponse
from .tools import ToolRegistry

logger = logging.getLogger(__name__)

data_path = os.getenv("ENERGY_DATA_PATH")
manifest_path = os.getenv("ENERGY_DATA_MANIFEST")
evidence_path = os.getenv("ENERGY_EVIDENCE_PATH")
store = load_dispatch_store(Path(data_path), Path(manifest_path)) if data_path and manifest_path else fixture_store()
local_evidence_index = HybridEvidenceIndex(load_official_chunks(Path(evidence_path))) if evidence_path else None
evidence_index: EvidenceIndex | None = local_evidence_index
planner_provider = ModelStudioPlanner.from_environment()
redis_client: Any = None
elasticsearch_client: Any = None
if redis_url := os.getenv("ENERGY_REDIS_URL"):
    try:
        from redis import Redis

        redis_client = Redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
    except Exception as exc:
        logger.warning("Redis unavailable; continuing without durable trace cache: %s", type(exc).__name__)
        redis_client = None
if elasticsearch_url := os.getenv("ENERGY_ELASTICSEARCH_URL"):
    try:
        from elasticsearch import Elasticsearch

        elasticsearch_client = Elasticsearch(elasticsearch_url)
        elasticsearch_client.info()
        if local_evidence_index is not None:
            indexed = ElasticsearchHybridEvidenceIndex(local_evidence_index, elasticsearch_client)
            indexed.ensure_index()
            evidence_index = indexed
    except Exception as exc:
        logger.warning("Elasticsearch evidence backend unavailable; using local hybrid: %s", type(exc).__name__)
        elasticsearch_client = None
registry = ToolRegistry(store, evidence_index)
agent = EnergyAgent(registry, planner_provider=planner_provider)
service_metrics = ServiceMetrics()
app = FastAPI(title="Australian Energy Market Intelligence Agent", version="0.1.0")


@app.get("/healthz")
def health() -> dict[str, object]:
    redis_status = "disabled_or_unavailable"
    if redis_client is not None:
        try:
            redis_status = "connected" if redis_client.ping() else "unavailable"
        except Exception:
            redis_status = "unavailable"
    elasticsearch_status = "disabled_or_unavailable"
    if elasticsearch_client is not None:
        try:
            if not elasticsearch_client.ping():
                elasticsearch_status = "unavailable"
            elif isinstance(evidence_index, ElasticsearchHybridEvidenceIndex):
                indexed = int(elasticsearch_client.count(index=evidence_index.alias)["count"])
                elasticsearch_status = "connected_indexed" if indexed == len(evidence_index.documents) else "index_mismatch"
            else:
                elasticsearch_status = "connected"
        except Exception:
            elasticsearch_status = "unavailable"
    dependency_degraded = (redis_url and redis_status != "connected") or (
        elasticsearch_url and elasticsearch_status != "connected_indexed"
    )
    return {
        "status": "degraded" if dependency_degraded else "ok",
        "data_version": store.data_version,
        "rows": len(store.rows),
        "evidence_chunks": len(evidence_index.documents) if evidence_index else 0,
        "redis": redis_status,
        "elasticsearch": elasticsearch_status,
        "evidence_backend": evidence_index.backend if evidence_index else "market_evidence_fallback",
        "evidence_index": evidence_index.index_name if isinstance(evidence_index, ElasticsearchHybridEvidenceIndex) else None,
        "evidence_indexed_documents": (
            evidence_index.indexed_documents if isinstance(evidence_index, ElasticsearchHybridEvidenceIndex) else 0
        ),
        "model_provider": planner_provider.name if planner_provider else "deterministic",
    }


@app.post("/api/agent/query", response_model=AgentQueryResponse)
def query(request: AgentQueryRequest) -> AgentQueryResponse:
    started = time.perf_counter()
    response = agent.run(request)
    service_metrics.observe(response, time.perf_counter() - started)
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
    return service_metrics.render(
        planner_provider.name if planner_provider else "deterministic",
        evidence_index.backend if evidence_index else "market_evidence_fallback",
        len(store.rows),
        len(evidence_index.documents) if evidence_index else 0,
    )
