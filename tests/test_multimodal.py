from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from energy_agent.agent import EnergyAgent
from energy_agent.market import fixture_store
from energy_agent.multimodal import (
    LateInteractionPageIndex,
    MultimodalEvidenceIndex,
    PageRecord,
    classify_page_modality,
    infer_modality_preference,
    maxsim_late_interaction,
)
from energy_agent.schemas import AgentQueryRequest, ToolResult
from energy_agent.tools import ToolRegistry


def test_qwen_slurm_jobs_can_return_results_without_project_writes() -> None:
    root = Path(__file__).parents[1]
    for name in ("evaluate_multimodal_qwen_pilot.sbatch", "evaluate_multimodal_qwen_full.sbatch"):
        script = (root / "scripts" / "slurm" / name).read_text(encoding="utf-8")
        assert 'if [[ "${OUTPUT_DIR}" == "__TASK_SCRATCH__" ]]' in script
        assert 'OUTPUT_DIR="${TASK_SCRATCH}/energy-mm-output-${SLURM_JOB_ID}"' in script
        assert 'if [[ "${EMIT_RESULT_BASE64:-0}" == "1" ]]' in script
        assert 'echo "ENERGY_MM_RESULT_BASE64_BEGIN"' in script
        assert 'echo "ENERGY_MM_RESULT_BASE64_END"' in script


def page(page_id: str, text: str, modality: str, page_number: int) -> PageRecord:
    return PageRecord(
        page_id=page_id,
        source_id="aemo-test",
        title="AEMO Quarterly Energy Dynamics",
        text=text,
        url="https://example.invalid/aemo.pdf",
        published_at="2026-01-01",
        retrieved_at=datetime(2026, 8, 22, tzinfo=UTC).isoformat(),
        source_sha256="a" * 64,
        page_number=page_number,
        modality=modality,  # type: ignore[arg-type]
        asset_id=page_id,
        asset_sha256=("b" if page_number == 1 else "c") * 64,
        width=1200,
        height=1600,
    )


def test_maxsim_late_interaction_preserves_multiple_visual_matches() -> None:
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    complete_page = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    partial_page = np.asarray([[1.0, 0.0]])
    assert maxsim_late_interaction(query, complete_page) == 1.0
    assert maxsim_late_interaction(query, partial_page) == 0.5


def test_multimodal_fusion_routes_chart_intent_and_returns_page_provenance() -> None:
    pages = [
        page("page-text", "South Australia quarterly prices and demand narrative", "text", 1),
        page("page-chart", "Figure 8 South Australia price trend 2025 2026", "chart", 2),
    ]
    query_vector = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    visual = LateInteractionPageIndex(
        {
            "page-text": np.asarray([[1.0, 0.0]]),
            "page-chart": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        },
        query_encoder=lambda _query: query_vector,
        model_id="test-visual-retriever",
        model_revision="immutable-test-revision",
    )
    index = MultimodalEvidenceIndex(pages, visual, dense_dimensions=2)
    hits = index.search_multimodal("Show the chart of South Australia price trend 2025", 2, "auto")
    assert hits[0]["chunk_id"] == "page-chart"
    assert hits[0]["page_number"] == 2
    assert hits[0]["asset_sha256"] == "c" * 64
    assert hits[0]["component_scores"]["visual_score"] == 1.0


def test_tool_and_agent_expose_multimodal_evidence_without_private_asset_paths() -> None:
    pages = [page("page-chart", "Figure: SA1 quarterly price trend", "chart", 2)]
    visual = LateInteractionPageIndex(
        {"page-chart": np.asarray([[1.0, 0.0]])},
        query_encoder=lambda _query: np.asarray([[1.0, 0.0]]),
    )
    registry = ToolRegistry(fixture_store(), MultimodalEvidenceIndex(pages, visual, dense_dimensions=2))
    result = registry.execute(
        "search_official_evidence",
        {
            "query": "Show the SA1 price chart",
            "preferred_modality": "chart",
            "retrieval_mode": "multimodal_fusion",
            "top_k": 1,
        },
    )
    assert result.evidence[0].modality == "chart"
    assert result.evidence[0].source_page == 2
    assert result.evidence[0].asset_id == "page-chart"
    assert "path" not in result.evidence[0].model_dump()

    planned = EnergyAgent(registry)._plan(AgentQueryRequest(question="Show the SA1 price chart"))
    evidence_call = next(arguments for name, arguments in planned if name == "search_official_evidence")
    assert evidence_call["preferred_modality"] == "chart"
    assert evidence_call["retrieval_mode"] == "multimodal_fusion"


def test_page_modality_classification_and_intent_are_deterministic() -> None:
    assert classify_page_modality(image_blocks=1, drawing_objects=12, table_count=0, numeric_ratio=0.1) == "chart"
    assert classify_page_modality(image_blocks=0, drawing_objects=0, table_count=1, numeric_ratio=0.2) == "table"
    assert classify_page_modality(image_blocks=2, drawing_objects=12, table_count=1, numeric_ratio=0.2) == "mixed"
    assert classify_page_modality(image_blocks=0, drawing_objects=0, table_count=0, numeric_ratio=0.0) == "text"
    assert infer_modality_preference("Compare the values in the table") == "table"
    assert infer_modality_preference("Explain the figure") == "chart"


class EmptyVisualRegistry(ToolRegistry):
    def execute(self, name: str, raw_args: dict[str, object]) -> ToolResult:
        if name == "search_official_evidence" and raw_args.get("retrieval_mode") == "multimodal_fusion":
            return ToolResult(tool_name=name)
        return super().execute(name, raw_args)


def test_agent_recovers_empty_visual_route_through_text_without_looping() -> None:
    pages = [page("page-chart", "Figure: SA1 quarterly price trend", "chart", 2)]
    visual = LateInteractionPageIndex(
        {"page-chart": np.asarray([[1.0, 0.0]])},
        query_encoder=lambda _query: np.asarray([[1.0, 0.0]]),
    )
    agent = EnergyAgent(
        EmptyVisualRegistry(fixture_store(), MultimodalEvidenceIndex(pages, visual, dense_dimensions=2))
    )

    response = agent.run(AgentQueryRequest(question="Show the SA1 price chart"))

    recovered = [call for call in response.tool_calls if call.recovered]
    assert response.status == "completed"
    assert len(recovered) == 1
    assert recovered[0].attempt == 2
    assert recovered[0].recovery_strategy == "visual_to_text_fallback"
    assert recovered[0].arguments["retrieval_mode"] == "hybrid_rerank"
    trace = agent.get_trace(response.trace_id)
    assert trace is not None
    assert trace["progress_ledger"]["recovery_attempts"] == 1  # type: ignore[index]
    assert trace["progress_ledger"]["stalled"] is False  # type: ignore[index]


def test_metadata_tie_breakers_cannot_override_both_retrieval_channels() -> None:
    pages = [
        page("joint-first", "Show the target chart evidence for the 2025 period", "mixed", 1),
        page("metadata-match", "2025", "chart", 2),
    ]
    visual = LateInteractionPageIndex(
        {
            "joint-first": np.asarray([[1.0, 0.0]]),
            "metadata-match": np.asarray([[0.8, 0.2]]),
        },
        query_encoder=lambda _query: np.asarray([[1.0, 0.0]]),
    )
    index = MultimodalEvidenceIndex(pages, visual, dense_dimensions=2)

    hits = index.search_multimodal("Show the target chart for 2025", top_k=2, preferred_modality="chart")

    assert hits[0]["chunk_id"] == "joint-first"
    assert hits[1]["component_scores"]["numeric_coverage"] == 1.0
    assert hits[1]["component_scores"]["modality_match"] == 1.0


def test_pdf_transport_benchmark_is_source_disjoint_from_fusion_development() -> None:
    repository = Path(__file__).parents[1]

    def rows(name: str) -> list[dict[str, object]]:
        path = repository / "benchmarks" / name
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    development = rows("multimodal_page_retrieval_q4_2024.jsonl")
    transport = rows("multimodal_page_retrieval_q1_2025_transport.jsonl")
    development_pages = {page for row in development for page in row["expected_page_ids"]}  # type: ignore[union-attr]
    transport_pages = {page for row in transport for page in row["expected_page_ids"]}  # type: ignore[union-attr]

    assert len(transport) == 14
    assert not development_pages & transport_pages
    assert all(str(page).startswith("aemo-qed-q4-2024-") for page in development_pages)
    assert all(str(page).startswith("aemo-qed-q1-2025-") for page in transport_pages)
