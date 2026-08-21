from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

DEVELOPMENT = Path("benchmarks/official_passage_support.jsonl")
HOLDOUT = Path("benchmarks/official_passage_support_holdout_q2_2025.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_q2_2025_holdout_is_source_disjoint_from_development_set() -> None:
    development = _read_jsonl(DEVELOPMENT)
    holdout = _read_jsonl(HOLDOUT)

    assert len(holdout) == 14
    assert {str(row["source_id"]) for row in holdout} == {"aemo-qed-q2-2025"}
    assert {str(row["source_id"]) for row in development}.isdisjoint(
        {str(row["source_id"]) for row in holdout}
    )
    assert len({str(row["task_id"]) for row in holdout}) == len(holdout)
    assert all(row["label"] == "support" for row in holdout)


def test_holdout_cli_requires_a_declared_feature_freeze() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_passage_support.py",
            "--evidence",
            "missing.jsonl",
            "--output",
            "missing-output",
            "--evaluation-role",
            "holdout",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "--frozen-feature-sha is required" in result.stderr
