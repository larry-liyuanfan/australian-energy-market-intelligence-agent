from __future__ import annotations

from energy_agent.composite_evidence import CompositeEvidenceIndex
from energy_agent.evidence import HybridEvidenceIndex, OfficialChunk
from energy_agent.workbook_evidence import FigureEvidence, FigureEvidenceIndex


def chunk() -> OfficialChunk:
    return OfficialChunk(
        chunk_id="text-1",
        source_id="report",
        title="Quarterly price text",
        text="South Australia average price discussion",
        url="https://aemo.example/report.pdf",
        published_at="2026-04-30",
        retrieved_at="2026-08-22T00:00:00+00:00",
        sha256="a" * 64,
    )


def figure() -> FigureEvidence:
    return FigureEvidence(
        chunk_id="figure-1",
        figure_id="Figure 1",
        figure_number=1,
        source_id="workbook",
        title="South Australia price by quarter",
        subtitle="Regional price trend",
        text="SA1 | Q1 | 120 AUD/MWh",
        url="https://aemo.example/workbook.xlsx",
        published_at="2026-04-30",
        retrieved_at="2026-08-22T00:00:00+00:00",
        sha256="b" * 64,
        image_sha256=("c" * 64,),
        image_count=1,
        row_count=10,
        column_count=3,
        preview_cell_count=30,
    )


def test_composite_routes_chart_and_preserves_source_cell_provenance() -> None:
    index = CompositeEvidenceIndex(
        HybridEvidenceIndex([chunk()], dense_dimensions=2),
        FigureEvidenceIndex([figure()], dense_dimensions=2),
    )
    hits = index.search_multimodal("Show the South Australia price chart", 5, "chart")
    chart = next(hit for hit in hits if hit["modality"] == "chart")
    assert chart["figure_id"] == "Figure 1"
    assert chart["preview_cell_count"] == 30
    assert chart["asset_sha256"] == "c" * 64
    assert all("path" not in key.lower() for hit in hits for key in hit)
