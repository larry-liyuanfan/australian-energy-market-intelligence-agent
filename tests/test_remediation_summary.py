from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import summarize_planner_remediation as export


def legacy_compact() -> dict[str, Any]:
    return {
        "schema_version": "planner-remediation-public-v3",
        "manifest": {"scoring_version": "v3"},
        "by_path_memory_and_track": {
            "hybrid|no_memory|real_task": {
                "failure_causes_nonexclusive": {"memory_recall": 3, "tool_path_correct": 3},
                "task_success_rate": 0.5,
                "unsafe_tool_or_dsl_calls": 1,
            },
        },
        "all_tracks_metrics": {"promotion_pass": False, "frozen_score": 0.5},
    }


def test_relabel_preserves_waived_memory_observation_and_all_frozen_scores() -> None:
    original = legacy_compact()
    clarified = export.clarify_diagnostic_labels(original)
    group = clarified["by_path_memory_and_track"]["hybrid|no_memory|real_task"]
    assert "failure_causes_nonexclusive" not in group
    assert group["failed_row_observations_nonexclusive"]["memory_recall"] == 3
    assert group["unsafe_tool_or_dsl_calls"] == 1
    assert clarified["all_tracks_metrics"] == original["all_tracks_metrics"]
    assert clarified["manifest"] == original["manifest"]
    assert "no_memory waives" in clarified["diagnostic_semantics"]
    assert "not causal or exhaustive" in clarified["diagnostic_semantics"]
    assert "failure_causes_nonexclusive" in original["by_path_memory_and_track"]["hybrid|no_memory|real_task"]


@pytest.mark.parametrize("mutation", ["wrong_version", "ambiguous_keys"])
def test_invalid_relabel_is_rejected(mutation: str) -> None:
    summary = legacy_compact()
    if mutation == "wrong_version":
        summary["manifest"]["scoring_version"] = "v1"
    else:
        summary["by_path_memory_and_track"]["hybrid|no_memory|real_task"]["failed_row_observations_nonexclusive"] = {}
    with pytest.raises(ValueError):
        export.clarify_diagnostic_labels(summary)


def test_relabel_cli_is_hash_bound_does_not_rescore_or_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "immutable_compute_summary.json"
    source.write_text(json.dumps(legacy_compact()), encoding="utf-8")
    target = tmp_path / "clarified_summary.json"

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("diagnostic rename must not rescore predictions")

    monkeypatch.setattr(export, "summarize_run", forbidden)
    monkeypatch.setattr(sys, "argv", [
        "export", "--relabel-summary", str(source), "--output", str(target),
    ])
    export.main()
    assert b"\r\n" not in target.read_bytes()
    result = json.loads(target.read_text())
    assert result["source_summary_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["all_tracks_metrics"] == legacy_compact()["all_tracks_metrics"]
    with pytest.raises(FileExistsError):
        export.main()
