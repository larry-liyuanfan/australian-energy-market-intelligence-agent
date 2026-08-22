from __future__ import annotations

import json
from pathlib import Path

import pytest

from energy_agent.claim_verification import (
    MINICHECK_NEGATIVE_TOKEN_ID,
    MINICHECK_POSITIVE_TOKEN_ID,
    binary_metrics,
    literal_consistency,
    selective_cascade_metrics,
)

DEBERTA_HOLDOUT = Path("benchmarks/minicheck_claim_support_holdout_q2_2025.jsonl")
CASCADE_HOLDOUT = Path("benchmarks/minicheck_claim_support_holdout_q1_2025.jsonl")
FINAL_HOLDOUT = Path("benchmarks/minicheck_claim_support_holdout_q4_2024.jsonl")


def test_binary_metrics_perfect_separation() -> None:
    metrics = binary_metrics([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["support_recall"] == 1.0
    assert metrics["counterfactual_rejection_recall"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["brier_score"] == pytest.approx(0.025)


def test_literal_consistency_catches_numeric_direction_entity_and_binding_changes() -> None:
    document = (
        "Gas-fired generation rose from 1,040 MW in April 2025 to 2,157 MW in June. "
        "Victorian gas demand reached 382 TJ. Batteries provided 61% while gas facilities provided 21%."
    )
    supported = "Gas-fired generation rose from 1,040 MW in April 2025 to 2,157 MW in June."
    assert literal_consistency(document, supported).consistent
    assert literal_consistency(document, supported.replace("rose", "fell")).direction_conflict
    assert literal_consistency(document, supported.replace("2,157", "2,517")).missing_literals == ("2517",)
    assert literal_consistency(document, "Queensland gas demand reached 382 TJ.").missing_entities == ("queensland",)
    region_binding = literal_consistency(
        "Victorian demand rose 6.3%. Queensland demand reached 11,144 MW.",
        "Victoria demand reached 11,144 MW.",
    )
    assert region_binding.missing_bindings == ("victoria:11144",)
    relation_binding = literal_consistency(
        "PV output was 16% above last year while operational demand fell to 21,380 MW.",
        "PV output was 16% below last year while operational demand rose to 21,380 MW.",
    )
    assert relation_binding.direction_conflict
    swapped = literal_consistency(document, "Batteries provided 21% while gas facilities provided 61%.")
    assert swapped.missing_bindings == ("battery:21%", "gas:61%")


def test_selective_cascade_counts_abstention_as_unhandled() -> None:
    metrics = selective_cascade_metrics([1, 0, 1], [0.9, 0.8, 0.4], [True, False, True])
    assert metrics["support_recall"] == 0.5
    assert metrics["counterfactual_rejection_recall"] == 1.0
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["selective_accuracy"] == 1.0
    calibrated = selective_cascade_metrics([1, 0], [0.3, 0.2], [True, True], threshold=0.25)
    assert calibrated["support_recall"] == 1.0


def test_minicheck_checkpoint_label_contract_is_frozen() -> None:
    assert (MINICHECK_NEGATIVE_TOKEN_ID, MINICHECK_POSITIVE_TOKEN_ID) == (3, 209)


def test_deberta_holdout_is_source_disjoint_balanced_and_paired() -> None:
    development = [
        json.loads(line)
        for line in Path("benchmarks/minicheck_claim_support.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    holdout = [json.loads(line) for line in DEBERTA_HOLDOUT.read_text(encoding="utf-8").splitlines() if line]
    assert len(holdout) == 28
    assert sum(int(row["label"]) for row in holdout) == 14
    assert {str(row["source_id"]) for row in holdout} == {"aemo-qed-q2-2025"}
    assert not {str(row["evidence_chunk_id"]) for row in development} & {
        str(row["evidence_chunk_id"]) for row in holdout
    }
    pairs = {str(row["pair_id"]) for row in holdout}
    assert len(pairs) == 14
    for pair in pairs:
        selected = [row for row in holdout if row["pair_id"] == pair]
        assert sorted(int(row["label"]) for row in selected) == [0, 1]
        assert len({row["evidence_chunk_id"] for row in selected}) == 1


def test_cascade_holdout_is_new_source_and_frozen_after_q2_development() -> None:
    q2_development = [json.loads(line) for line in DEBERTA_HOLDOUT.read_text(encoding="utf-8").splitlines() if line]
    q1_holdout = [json.loads(line) for line in CASCADE_HOLDOUT.read_text(encoding="utf-8").splitlines() if line]
    assert len(q1_holdout) == 28
    assert sum(int(row["label"]) for row in q1_holdout) == 14
    assert {str(row["source_id"]) for row in q1_holdout} == {"aemo-qed-q1-2025"}
    assert {str(row["source_id"]) for row in q1_holdout}.isdisjoint(
        {str(row["source_id"]) for row in q2_development}
    )
    pairs = {str(row["pair_id"]) for row in q1_holdout}
    assert len(pairs) == 14
    for pair in pairs:
        selected = [row for row in q1_holdout if row["pair_id"] == pair]
        assert sorted(int(row["label"]) for row in selected) == [0, 1]


def test_final_holdout_is_disjoint_from_both_cascade_development_sources() -> None:
    development_sources = {"aemo-qed-q2-2025", "aemo-qed-q1-2025"}
    rows = [json.loads(line) for line in FINAL_HOLDOUT.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 28
    assert sum(int(row["label"]) for row in rows) == 14
    assert {str(row["source_id"]) for row in rows} == {"aemo-qed-q4-2024"}
    assert {str(row["source_id"]) for row in rows}.isdisjoint(development_sources)
    assert len({str(row["pair_id"]) for row in rows}) == 14


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


def test_minicheck_slurm_cache_has_job_unique_tmp_fallback_and_cleanup() -> None:
    script = Path("scripts/slurm/evaluate_claim_support.sbatch").read_text(encoding="utf-8")
    assert 'TASK_SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"' in script
    assert 'HF_JOB_CACHE="${TASK_SCRATCH}/energy-minicheck-hf-${SLURM_JOB_ID}"' in script
    assert 'export HF_HOME="${HF_JOB_CACHE}"' in script
    assert '"${HF_JOB_CACHE}"' in script.split("trap ", maxsplit=1)[1].split(" EXIT", maxsplit=1)[0]
