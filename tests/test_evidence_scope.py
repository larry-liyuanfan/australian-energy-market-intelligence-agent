from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest
from test_composite_evidence import chunk, figure

from energy_agent.composite_evidence import CompositeEvidenceIndex
from energy_agent.evidence import HybridEvidenceIndex
from energy_agent.evidence_scope import (
    evidence_excerpt,
    figure_source_preview,
    filter_scoped_hits,
    market_evidence_query,
    official_evidence_url,
    prepare_evidence_hits,
    query_scope,
    scope_support,
)
from energy_agent.workbook_evidence import FigureEvidenceIndex


def test_dated_scope_changes_with_user_day_and_expands_alias() -> None:
    query = market_evidence_query(["SA1", "QLD1"], date(2025, 12, 15), "decision_replay")
    assert "SA1 QLD1 2025-12-15" in query and "South Australia" in query and "battery" in query
    assert query_scope(query).period == "Q4 2025"  # type: ignore[union-attr]
    assert query_scope("SA1 2026-02-05 Q1 2026").period == "Q1 2026"  # type: ignore[union-attr]
    assert query_scope("SA1 2025-12-15 daily event report") is None


def test_wrong_quarter_gas_only_and_other_region_are_rejected_before_fusion() -> None:
    text = replace(chunk(), source_id="qed-q4-2025", text="South Australia NEM spot prices fell.")
    good = replace(figure(), source_id="qed-q4-2025", subtitle="NEM electricity spot prices Q4 2025")
    gas = replace(good, chunk_id="gas", title="Iona gas storage prices", subtitle="Gas market", text="SA Q4 2025 GJ")
    later = replace(good, chunk_id="later", source_id="qed-q1-2026")
    other = replace(good, chunk_id="other", title="Tasmania spot prices", subtitle="Electricity spot prices", text="TAS Q4 2025 AUD/MWh")
    index = CompositeEvidenceIndex(HybridEvidenceIndex([text]), FigureEvidenceIndex([gas, later, other, good]))
    hits = index.search_multimodal("SA1 2025-12-15 Q4 2025 South Australia electricity spot prices", 5, "chart")
    assert {hit["chunk_id"] for hit in hits} == {text.chunk_id, good.chunk_id}
    empty = CompositeEvidenceIndex(HybridEvidenceIndex([replace(text, source_id="qed-q1-2026")]), FigureEvidenceIndex([gas, later]))
    assert empty.search_multimodal("SA1 2025-12-15 Q4 2025 spot prices", 5, "chart") == []


def test_copyright_year_or_publication_date_cannot_prove_report_period() -> None:
    hit = {"title": "NEM electricity spot prices South Australia", "text": "Copyright Q4 2025", "published_at": "2025-12-01"}
    assert filter_scoped_hits([hit], "SA1 2025-12-15 Q4 2025") == []
    scope = query_scope("SA1 2025-12-15 Q4 2025")
    assert scope is not None
    checks = scope_support({**hit, "url": "https://aemo.com.au/qed-q4-2025.pdf", "published_at": "2026-01-29"}, scope)
    assert all(checks.values())  # Report scope; never a claim it was available as of Dec 15.


def test_relevant_passage_after_cover_is_exact_substring_with_offset() -> None:
    content = "Unrelated cover text. " * 100 + "South Australia negative electricity spot prices and batteries. " * 10
    excerpt, start = evidence_excerpt(content, "SA1 2025-12-15 Q4 2025")
    assert start > 500 and "South Australia" in excerpt
    assert excerpt == content[start:start + 500]


@pytest.mark.parametrize(("content", "supported"), [
    ("South Australia spot gas prices increased in the Adelaide STTM market; gas demand rose in GJ.", False),
    ("Domestic spot prices SA GJ STTM", False),
    ("Victoria electricity spot prices in the NEM", False),
    ("NEM-wide electricity prices across the NEM including Victoria", True),
    ("South Australia gas-fired generation sets electricity spot prices in AUD/MWh", True),
])
def test_topic_and_region_false_positive_counterexamples(content: str, supported: bool) -> None:
    hits = [{"title": "Quarterly Energy Dynamics Q4 2025", "text": content}]
    assert bool(filter_scoped_hits(hits, "SA1 2025-12-15 Q4 2025")) is supported


@pytest.mark.parametrize("url", [
    "javascript://aemo.com.au/%0Aalert(1)", "ftp://aemo.com.au/a", "http://aemo.com.au/a",
    "https://user:password@aemo.com.au/a", "https://aemo.com.au.evil.invalid/a",
])
def test_demo_links_reject_unsafe_scheme_credentials_and_suffixes(url: str) -> None:
    assert not official_evidence_url(url)
    assert official_evidence_url("https://www.aemo.com.au/report.pdf")


def test_source_preview_retains_requested_region_row_and_original_header() -> None:
    rows = ["Figure 12", "Wholesale electricity prices", "$/MWh", "Region | Year | Quarter | Energy", "Source: AEMO"]
    rows += ["QLD | 2024 | Q4 | 90"] * 40 + ["SA | 2025 | Q4 | 20", "SA | 2024 | Q4 | 80"]
    preview = figure_source_preview("\n".join(rows), "SA1 2025-12-15 Q4 2025")
    assert "Region | Year | Quarter | Energy" in preview and "SA | 2025 | Q4 | 20" in preview


def test_catalogue_and_only_hidden_context_cannot_pass_returned_passage_gate() -> None:
    hit = {"title": "Q4 2025", "text": "South Australia electricity spot prices " + "Figure 1 Figure 2 Figure 3 Figure 4"}
    assert filter_scoped_hits([hit], "SA1 2025-12-15 Q4 2025") == []
    hit["text"] = "South Australia electricity spot prices " + "x" * 600 + "Victorian gas spot prices " * 20
    prepared = prepare_evidence_hits([hit], "SA1 2025-12-15 Q4 2025", 5)
    assert all(len(item["text"]) <= 500 and all(item["scope_checks"].values()) for item in prepared)
