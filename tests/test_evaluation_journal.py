from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from energy_agent.model_agent import ModelAgentRun, ModelDrivenAgent
from scripts import evaluate_llm_agent


def configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    turn = {
        "user": "Show SA1 on 2025-01-03 with official evidence.",
        "expected_tools": ["get_market_snapshot", "search_official_evidence"],
        "expected": {"region": "SA1", "date": "2025-01-03"},
    }
    benchmark = tmp_path / "explicit_fixture.jsonl"
    benchmark.write_text(json.dumps({
        "case_id": "journal-fixture", "category": "fixture", "split": "development", "turns": [turn, turn],
    }))
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_llm_agent", "--benchmark", str(benchmark),
        "--gate", "benchmarks/planner_remediation_gate_v3.json", "--output", str(output),
        "--paths", "deterministic", "--memory-modes", "no_memory",
    ])
    return output


def test_completed_journal_matches_final_predictions_and_does_not_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = configure(tmp_path, monkeypatch)
    evaluate_llm_agent.main()
    assert (output / "predictions.partial.jsonl").read_bytes() == (output / "predictions.jsonl").read_bytes()
    assert json.loads((output / "run_manifest.json").read_text())["data_track"] == "explicit_synthetic_fixture"
    with pytest.raises(FileExistsError):
        evaluate_llm_agent.main()


def test_interrupted_journal_keeps_completed_attempt_without_final_success_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = configure(tmp_path, monkeypatch)
    original = ModelDrivenAgent.run_turn
    count = 0

    def interrupted(self: ModelDrivenAgent, *args: Any, **kwargs: Any) -> ModelAgentRun:
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("explicit test-only process interruption")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ModelDrivenAgent, "run_turn", interrupted)
    with pytest.raises(RuntimeError, match="test-only process interruption"):
        evaluate_llm_agent.main()
    assert len((output / "predictions.partial.jsonl").read_text().splitlines()) == 1
    assert not (output / "run_manifest.json").exists()
    assert not (output / "metrics.json").exists()
