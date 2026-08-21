import json
from pathlib import Path

from energy_agent.evidence import OfficialChunk
from scripts.evaluate_passage_support import required_terms_present

BENCHMARK = Path("benchmarks/official_passage_support.jsonl")


def test_passage_support_benchmark_is_compact_and_source_balanced() -> None:
    tasks = [json.loads(line) for line in BENCHMARK.read_text(encoding="utf-8").splitlines()]
    assert len(tasks) == 20
    assert len({task["task_id"] for task in tasks}) == 20
    assert {task["source_id"] for task in tasks} == {
        "aemo-qed-q2-2026",
        "aemo-qed-q1-2026",
        "aemo-qed-q4-2025",
        "aemo-qed-q3-2025",
        "aer-significant-q1-2026",
    }
    assert all(task["label"] == "support" for task in tasks)


def test_required_term_check_normalizes_punctuation_without_semantic_claim() -> None:
    chunk = OfficialChunk(
        "x",
        "source",
        "Title",
        "Operational demand averaged 21,700 MW and prices were $74/MWh.",
        "https://example.invalid",
        "2026-01-01",
        "2026-08-21",
        "0" * 64,
    )
    assert required_terms_present(chunk.text, ["21,700", "$74", "operational demand"])
    assert not required_terms_present(chunk.text, ["24,220"])
