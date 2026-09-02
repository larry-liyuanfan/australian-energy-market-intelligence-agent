from __future__ import annotations

from pathlib import Path

from energy_agent.llm_evaluation import aggregate_rows, load_episodes, ordered_subsequence, wilson_interval


def test_holdout_is_separate_and_uses_new_dates() -> None:
    path = Path(__file__).parents[1] / "benchmarks" / "llm_agent_holdout_v1.jsonl"
    episodes = load_episodes(path)
    assert len(episodes) == 12
    text = path.read_text(encoding="utf-8")
    assert "2025-09-15" not in text
    assert "2025-12-15" not in text
    assert "2026-03-15" not in text
    assert all(item["split"] == "holdout" for item in episodes)


def test_tool_path_is_order_sensitive_subsequence() -> None:
    assert ordered_subsequence(["detect", "search"], ["snapshot", "detect", "diagnose", "search"])
    assert not ordered_subsequence(["detect", "search"], ["search", "detect"])


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = wilson_interval(8, 10)
    assert low < 0.8 < high


def test_aggregate_preserves_retries_tokens_latency_and_unsafe_counts() -> None:
    row = {
        "path": "constrained_hybrid",
        "memory_mode": "structured_state",
        "case_id": "case",
        "turn_index": 1,
        "task_success": True,
        "tool_path_correct": True,
        "model_tool_path_correct": False,
        "parameter_accuracy": 1.0,
        "model_parameter_accuracy": 0.8,
        "citation_correct": True,
        "settlement_consistent": True,
        "requires_replan": True,
        "replan_success": True,
        "requires_memory": True,
        "memory_recall": True,
        "state_contaminated": False,
        "steps": 3,
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "end_to_end_latency_ms": 50.0,
        "provider_cost_aud": 0.0,
        "retries": 1,
        "rejected_model_calls": 2,
        "unsafe_tool_or_dsl_calls": 0,
    }
    metrics = aggregate_rows([row])["constrained_hybrid|structured_state"]
    assert metrics["total_retries"] == 1
    assert metrics["average_prompt_tokens"] == 100
    assert metrics["model_parameter_accuracy"] == 0.8
    assert metrics["model_correct_tool_path_rate"] == 0.0
    assert metrics["p95_latency_ms"] == 50.0
    assert metrics["total_rejected_model_calls"] == 2
