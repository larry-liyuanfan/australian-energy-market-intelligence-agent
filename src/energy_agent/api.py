from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from .agent import EnergyAgent
from .composite_evidence import CompositeEvidenceIndex
from .evidence import ElasticsearchHybridEvidenceIndex, EvidenceIndex, HybridEvidenceIndex, load_official_chunks
from .market import fixture_store, load_dispatch_store
from .metrics import ServiceMetrics
from .model_agent import AgentPath, ModelAgentQueryRequest, ModelAgentRun, ModelDrivenAgent
from .multimodal import MultimodalEvidenceIndex, load_page_records
from .providers import LlamaCppPlanner, ModelStudioPlanner, OllamaPlanner
from .schemas import AgentQueryRequest, AgentQueryResponse
from .snapshots import ForecastSnapshotStore, load_forecast_snapshots
from .tools import ToolRegistry
from .workbook_evidence import FigureEvidenceIndex, load_figure_evidence_records

logger = logging.getLogger(__name__)

data_path = os.getenv("ENERGY_DATA_PATH")
manifest_path = os.getenv("ENERGY_DATA_MANIFEST")
evidence_path = os.getenv("ENERGY_EVIDENCE_PATH")
figure_evidence_path = os.getenv("ENERGY_FIGURE_EVIDENCE_PATH")
page_evidence_path = os.getenv("ENERGY_PAGE_EVIDENCE_PATH")
forecast_snapshots_path = os.getenv("ENERGY_FORECAST_SNAPSHOTS_PATH")
store = load_dispatch_store(Path(data_path), Path(manifest_path)) if data_path and manifest_path else fixture_store()
local_evidence_index = HybridEvidenceIndex(load_official_chunks(Path(evidence_path))) if evidence_path else None
evidence_index: EvidenceIndex | None = local_evidence_index
figure_index: EvidenceIndex | None = None
page_index: EvidenceIndex | None = None
forecast_snapshots = ForecastSnapshotStore()
if figure_evidence_path and Path(figure_evidence_path).is_file():
    figure_index = FigureEvidenceIndex(load_figure_evidence_records(Path(figure_evidence_path)))
elif figure_evidence_path:
    logger.warning("Figure evidence file is unavailable; text retrieval remains active")
if page_evidence_path and Path(page_evidence_path).is_file():
    page_index = MultimodalEvidenceIndex(load_page_records(Path(page_evidence_path)))
elif page_evidence_path:
    logger.warning("Page evidence file is unavailable; text/figure retrieval remains active")
if forecast_snapshots_path and Path(forecast_snapshots_path).is_file():
    forecast_snapshots = load_forecast_snapshots(
        Path(forecast_snapshots_path),
        expected_data_sha256=store.evidence[0].sha256 if store.evidence else None,
    )
elif forecast_snapshots_path:
    logger.warning("Forecast snapshot file is unavailable; as-of seasonal fallback remains active")
planner_provider = ModelStudioPlanner.from_environment()
turn_planner = planner_provider or LlamaCppPlanner.from_environment() or OllamaPlanner.from_environment()
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
if evidence_index is not None and (figure_index is not None or page_index is not None):
    evidence_index = CompositeEvidenceIndex(evidence_index, figure_index, page_index)
registry = ToolRegistry(store, evidence_index, forecast_snapshots)
agent = EnergyAgent(
    registry,
    planner_provider=planner_provider,
    trace_capacity=int(os.getenv("ENERGY_TRACE_CACHE_SIZE", "128")),
)
model_agent = ModelDrivenAgent(registry, turn_planner)
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
    text_evidence_index: EvidenceIndex | None = (
        evidence_index.text_index if isinstance(evidence_index, CompositeEvidenceIndex) else evidence_index
    )
    if elasticsearch_client is not None:
        try:
            if not elasticsearch_client.ping():
                elasticsearch_status = "unavailable"
            elif isinstance(text_evidence_index, ElasticsearchHybridEvidenceIndex):
                indexed = int(elasticsearch_client.count(index=text_evidence_index.alias)["count"])
                elasticsearch_status = (
                    "connected_indexed" if indexed == len(text_evidence_index.documents) else "index_mismatch"
                )
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
        "evidence_index": (
            text_evidence_index.index_name
            if isinstance(text_evidence_index, ElasticsearchHybridEvidenceIndex)
            else None
        ),
        "evidence_indexed_documents": (
            text_evidence_index.indexed_documents
            if isinstance(text_evidence_index, ElasticsearchHybridEvidenceIndex)
            else 0
        ),
        "text_evidence_chunks": len(text_evidence_index.documents) if text_evidence_index else 0,
        "figure_evidence_records": len(figure_index.documents) if figure_index else 0,
        "page_evidence_records": len(page_index.documents) if page_index else 0,
        "forecast_snapshots": forecast_snapshots.count,
        "model_provider": planner_provider.name if planner_provider else "deterministic",
        "model_agent_provider": turn_planner.name if turn_planner else "deterministic_only",
        "trace_cache": dict(zip(("entries", "capacity", "evictions"), agent.trace_stats(), strict=True)),
    }


@app.post("/api/agent/query", response_model=AgentQueryResponse)
def query(request: AgentQueryRequest) -> AgentQueryResponse:
    started = time.perf_counter()
    response = agent.run(request)
    service_metrics.observe(response, time.perf_counter() - started)
    if redis_client:
        trace_payload = agent.get_trace(response.trace_id)
        if trace_payload is None:
            raise RuntimeError("newly created trace was unexpectedly evicted")
        redis_client.setex(
            f"energy:trace:{response.trace_id}",
            86400,
            json.dumps(trace_payload),
        )
    return response


@app.post("/api/agent/model-query", response_model=ModelAgentRun)
def model_query(request: ModelAgentQueryRequest) -> ModelAgentRun:
    if request.path != AgentPath.deterministic and turn_planner is None:
        raise HTTPException(503, "No real model planner runtime is configured; deterministic path remains available")
    return model_agent.run_turn(
        request.question,
        conversation_id=request.conversation_id,
        path=request.path,
        memory_mode=request.memory_mode,
        seed=request.seed,
        max_tool_calls=request.max_tool_calls,
    )


@app.get("/api/agent/traces/{trace_id}")
def trace(trace_id: str) -> dict[str, object]:
    local_trace = agent.get_trace(trace_id)
    if local_trace is None and redis_client:
        cached = redis_client.get(f"energy:trace:{trace_id}")
        if cached:
            decoded = json.loads(cached)
            if isinstance(decoded, dict):
                return decoded
    if local_trace is None:
        raise HTTPException(404, "trace not found")
    return local_trace


@app.get("/api/tools")
def tools() -> dict[str, object]:
    return {"tools": registry.specs(), "arbitrary_sql_or_dsl_allowed": False}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    trace_entries, trace_capacity, trace_evictions = agent.trace_stats()
    return service_metrics.render(
        planner_provider.name if planner_provider else "deterministic",
        evidence_index.backend if evidence_index else "market_evidence_fallback",
        len(store.rows),
        len(evidence_index.documents) if evidence_index else 0,
        trace_entries,
        trace_capacity,
        trace_evictions,
    )


@app.get("/", include_in_schema=False)
def demo() -> FileResponse:
    return FileResponse(Path(__file__).with_name("demo.html"))
