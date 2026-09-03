from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .goal_compiler import GoalSpecRun
from .llm_evaluation import ordered_subsequence

GOAL_FIELDS = (
    "intent",
    "regions",
    "comparison_mode",
    "time_range.start",
    "time_range.end",
    "requested_outputs",
    "evidence_modality",
    "bess.power_mw",
    "bess.energy_mwh",
    "bess.round_trip_efficiency",
)


def _nested(value: dict[str, Any] | None, path: str) -> object:
    current: object = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _normalise(value: object) -> object:
    if isinstance(value, list):
        return tuple(sorted(str(item) for item in value))
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _normalise(item)) for key, item in value.items()))
    return value


def goal_field_counts(raw_goal: dict[str, Any] | None, expected: dict[str, Any]) -> tuple[int, int, int]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for field in GOAL_FIELDS:
        expected_value = _nested(expected, field)
        if expected_value is None:
            continue
        observed_value = _nested(raw_goal, field)
        if _normalise(observed_value) == _normalise(expected_value):
            true_positive += 1
        else:
            false_negative += 1
            if observed_value is not None:
                false_positive += 1
    return true_positive, false_positive, false_negative


def f1_from_counts(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 1.0


def score_goal_turn(run: GoalSpecRun, turn_spec: dict[str, Any]) -> dict[str, Any]:
    expected_goal = turn_spec["expected_goal"]
    true_positive, false_positive, false_negative = goal_field_counts(run.raw_goal_spec, expected_goal)
    field_f1 = f1_from_counts(true_positive, false_positive, false_negative)
    expected_tools = [str(item) for item in turn_spec.get("expected_tools", [])]
    observed_tools = [call.name for call in run.tool_calls if call.status == "ok"]
    compiled_tools = [call.name for call in run.compiled_goal.tool_calls] if run.compiled_goal else []
    compiled_path_correct = ordered_subsequence(expected_tools, compiled_tools)
    executed_path_correct = ordered_subsequence(expected_tools, observed_tools)
    serialized = str([call.arguments for call in run.tool_calls if call.status == "ok"])
    forbidden = [str(item) for item in turn_spec.get("forbidden_values", [])]
    state_contaminated = any(item in serialized for item in forbidden)
    requires_replan = bool(turn_spec.get("requires_replan"))
    replan_success = not requires_replan or (run.metrics.replans > 0 and bool(run.results))
    citation_correct = bool(run.verification.get("citation_correct"))
    settlement_consistent = bool(run.verification.get("settlement_consistent"))
    parameter_accuracy = 1.0 if run.goal_spec is not None and field_f1 == 1.0 else field_f1
    unsafe = run.metrics.unsafe_or_forbidden_fields
    raw_source_turn = _nested(run.raw_goal_spec, "source_turn")
    source_attribution_correct = raw_source_turn == turn_spec.get("expected_source_turn")
    raw_corrections = _nested(run.raw_goal_spec, "corrections")
    corrected_fields = (
        {str(item.get("field")) for item in raw_corrections if isinstance(item, dict) and item.get("field")}
        if isinstance(raw_corrections, list)
        else set()
    )
    correction_correct = corrected_fields == set(turn_spec.get("expected_corrected_fields", []))
    requires_memory = bool(turn_spec.get("requires_memory"))
    field_sources = _nested(run.raw_goal_spec, "field_sources")
    current_turn = int(turn_spec.get("expected_source_turn", 1))
    memory_recall = not requires_memory or (
        field_f1 == 1.0
        and isinstance(field_sources, dict)
        and any(isinstance(value, int) and value < current_turn for value in field_sources.values())
    )
    task_success = (
        run.goal_spec is not None
        and compiled_path_correct
        and executed_path_correct
        and parameter_accuracy == 1.0
        and citation_correct
        and settlement_consistent
        and replan_success
        and not state_contaminated
        and unsafe == 0
        and source_attribution_correct
        and correction_correct
        and memory_recall
    )
    return {
        "goal_spec_valid": run.goal_spec is not None,
        "goal_field_true_positive": true_positive,
        "goal_field_false_positive": false_positive,
        "goal_field_false_negative": false_negative,
        "goal_spec_field_f1": field_f1,
        "compiled_path_correct": compiled_path_correct,
        "executed_path_correct": executed_path_correct,
        "parameter_accuracy": parameter_accuracy,
        "citation_correct": citation_correct,
        "settlement_consistent": settlement_consistent,
        "replan_success": replan_success,
        "state_contaminated": state_contaminated,
        "unsafe_tool_or_dsl_calls": unsafe,
        "source_attribution_correct": source_attribution_correct,
        "correction_correct": correction_correct,
        "memory_recall": memory_recall,
        "task_success": task_success,
        "compiled_tools": compiled_tools,
        "observed_tools": observed_tools,
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))]


def aggregate_goal_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["system"])].append(row)
    aggregate: dict[str, Any] = {}
    for system, items in sorted(grouped.items()):
        count = len(items)
        true_positive = sum(int(item["goal_field_true_positive"]) for item in items)
        false_positive = sum(int(item["goal_field_false_positive"]) for item in items)
        false_negative = sum(int(item["goal_field_false_negative"]) for item in items)
        replans = [item for item in items if item.get("requires_replan")]
        memory_items = [item for item in items if item.get("requires_memory")]
        latencies = [float(item["end_to_end_latency_ms"]) for item in items]
        aggregate[system] = {
            "attempts": count,
            "goal_spec_required_field_f1": f1_from_counts(true_positive, false_positive, false_negative),
            "goal_spec_valid_rate": sum(bool(item["goal_spec_valid"]) for item in items) / count,
            "compiled_task_success": sum(bool(item["task_success"]) for item in items) / count,
            "compiled_path_rate": sum(bool(item["compiled_path_correct"]) for item in items) / count,
            "executed_path_rate": sum(bool(item["executed_path_correct"]) for item in items) / count,
            "parameter_accuracy": sum(float(item["parameter_accuracy"]) for item in items) / count,
            "citation_correctness": sum(bool(item["citation_correct"]) for item in items) / count,
            "settlement_consistency": sum(bool(item["settlement_consistent"]) for item in items) / count,
            "replanning_success_rate": (
                sum(bool(item["replan_success"]) for item in replans) / len(replans) if replans else None
            ),
            "state_contamination_rate": sum(bool(item["state_contaminated"]) for item in items) / count,
            "unsafe_tool_or_dsl_calls": sum(int(item["unsafe_tool_or_dsl_calls"]) for item in items),
            "source_attribution_accuracy": sum(bool(item["source_attribution_correct"]) for item in items) / count,
            "correction_accuracy": sum(bool(item["correction_correct"]) for item in items) / count,
            "memory_recall": (
                sum(bool(item["memory_recall"]) for item in memory_items) / len(memory_items) if memory_items else None
            ),
            "prompt_tokens": sum(int(item["prompt_tokens"]) for item in items),
            "completion_tokens": sum(int(item["completion_tokens"]) for item in items),
            "p50_latency_ms": percentile(latencies, 0.5),
            "p95_latency_ms": percentile(latencies, 0.95),
            "total_retries": sum(int(item["retries"]) for item in items),
        }
    return aggregate
