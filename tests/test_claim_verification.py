from __future__ import annotations

import json
from pathlib import Path

import pytest

from energy_agent.claim_verification import binary_metrics


def test_binary_metrics_perfect_separation() -> None:
    metrics = binary_metrics([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["support_recall"] == 1.0
    assert metrics["counterfactual_rejection_recall"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["brier_score"] == pytest.approx(0.025)


def test_binary_metrics_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        binary_metrics([], [])
    with pytest.raises(ValueError):
        binary_metrics([2], [0.5])


def test_minicheck_benchmark_is_balanced_and_paired() -> None:
    path = Path("benchmarks/minicheck_claim_support.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 40
    assert sum(int(row["label"]) for row in rows) == 20
    pairs = {str(row["pair_id"]) for row in rows}
    assert len(pairs) == 20
    for pair in pairs:
        selected = [row for row in rows if row["pair_id"] == pair]
        assert sorted(int(row["label"]) for row in selected) == [0, 1]
        assert len({row["evidence_chunk_id"] for row in selected}) == 1
    negative_types = {row["perturbation_type"] for row in rows if int(row["label"]) == 0}
    assert negative_types == {"numeric", "direction", "temporal", "entity", "quantifier_negation"}
