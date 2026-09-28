from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from energy_agent.v2_artifacts import validate_goal_run_directory


def test_goal_run_schema_and_hash_links(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    predictions = tmp_path / "predictions.jsonl"
    metrics.write_text("{}", encoding="utf-8")
    predictions.write_text('{"case_id":"one"}\n', encoding="utf-8")
    manifest = {
        "schema_version": "goal-spec-agent-run-v2",
        "system_label": "qwen3_8b_goal_spec",
        "benchmark_sha256": "a" * 64,
        "gate_sha256": "b" * 64,
        "metrics_sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_goal_run_directory(tmp_path)["rows"] == 1
    predictions.write_text('{"case_id":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="predictions hash mismatch"):
        validate_goal_run_directory(tmp_path)
