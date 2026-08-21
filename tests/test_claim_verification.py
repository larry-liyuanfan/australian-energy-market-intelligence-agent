from __future__ import annotations

import json
from pathlib import Path

import pytest

from energy_agent.claim_verification import (
    MINICHECK_NEGATIVE_TOKEN_ID,
    MINICHECK_POSITIVE_TOKEN_ID,
    binary_metrics,
)

DEBERTA_HOLDOUT = Path("benchmarks/minicheck_claim_support_holdout_q2_2025.jsonl")


def test_binary_metrics_perfect_separation() -> None:
    metrics = binary_metrics([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["support_recall"] == 1.0
    assert metrics["counterfactual_rejection_recall"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["brier_score"] == pytest.approx(0.025)


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
