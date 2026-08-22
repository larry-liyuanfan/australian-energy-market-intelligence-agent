from __future__ import annotations

from datetime import UTC, datetime

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
from energy_agent.schemas import AgentQueryRequest
from energy_agent.tools import ToolRegistry


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
    assert classify_page_modality(image_blocks=0, drawing_objects=0, table_count=0, numeric_ratio=0.0) == "text"
    assert infer_modality_preference("Compare the values in the table") == "table"
    assert infer_modality_preference("Explain the figure") == "chart"
