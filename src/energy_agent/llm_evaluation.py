from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .model_agent import MemoryMode, ModelAgentRun


def load_episodes(path: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("split") not in {"development", "holdout"}:
            raise ValueError(f"invalid split at line {line_number}")
        if not isinstance(item.get("turns"), list) or not item["turns"]:
            raise ValueError(f"missing turns at line {line_number}")
        episodes.append(item)
    return episodes


def ordered_subsequence(expected: list[str], observed: list[str]) -> bool:
    cursor = 0
    for name in observed:
        if cursor < len(expected) and name == expected[cursor]:
            cursor += 1
    return cursor == len(expected)


def score_turn(run: ModelAgentRun, turn_spec: dict[str, Any]) -> dict[str, Any]:
    observed = [call.name for call in run.tool_calls if call.status == "ok"]
    expected_tools = [str(name) for name in turn_spec.get("expected_tools", [])]
    tool_path_correct = ordered_subsequence(expected_tools, observed)
    proposed = [call.name for call in run.model_proposed_calls]
    model_tool_path_correct = (
        tool_path_correct if run.path.value == "deterministic" else ordered_subsequence(expected_tools, proposed)
    )
    serialized_calls = json.dumps([call.arguments for call in run.tool_calls if call.status == "ok"], sort_keys=True)
    expected = turn_spec.get("expected", {})

    def accuracy(serialized: str) -> float:
        checks: list[bool] = []
        for key, value in expected.items():
            if key == "date":
                requested = date.fromisoformat(str(value))
                checks.append(str(value) in serialized)
                window_tools = {
                    "compare_region_period",
                    "detect_price_events",
                    "forecast_price_risk",
                    "optimize_battery_dispatch",
                }
                if window_tools.intersection(expected_tools):
                    checks.append((requested + timedelta(days=1)).isoformat() in serialized)
            elif key == "region":
                checks.append(f'"region": "{value}"' in serialized)
            elif key == "regions":
                checks.extend(str(region) in serialized for region in value)
            elif key == "battery_power_mw":
                checks.append(f'"power_mw": {float(value)}' in serialized)
            elif key == "battery_energy_mwh":
                checks.append(f'"energy_mwh": {float(value)}' in serialized)
            elif key == "round_trip_efficiency":
                checks.append(f'"round_trip_efficiency": {float(value)}' in serialized)
        return sum(checks) / len(checks) if checks else 1.0

    parameter_accuracy = accuracy(serialized_calls)
    serialized_model_calls = json.dumps([call.arguments for call in run.model_proposed_calls], sort_keys=True)
    model_parameter_accuracy = (
        parameter_accuracy if run.path.value == "deterministic" else accuracy(serialized_model_calls)
    )
    citation_correct = bool(run.citations) and all(
        item.url.startswith("https://") and len(item.sha256) == 64 for item in run.citations
    )
    settlement_consistent = bool(run.verification.get("settlement_consistent"))
    requires_replan = bool(turn_spec.get("requires_replan", False))
    replan_success = not requires_replan or (run.metrics.replans > 0 and bool(run.results))
    requires_memory = bool(turn_spec.get("requires_memory", False))
    memory_values = json.dumps({key: value.value for key, value in run.resolved_constraints.items()}, sort_keys=True)
    expected_memory_values = [value for key, value in expected.items() if key in {"region", "regions", "date"}]
    flattened: list[str] = []
    for value in expected_memory_values:
        flattened.extend(str(item) for item in value) if isinstance(value, list) else flattened.append(str(value))
    memory_recall = not requires_memory or all(value in memory_values for value in flattened)
    combined_state = serialized_calls + memory_values
    forbidden_values = [str(value) for value in turn_spec.get("forbidden_values", [])]
    state_contaminated = any(value in combined_state for value in forbidden_values)
    unsafe = run.metrics.unsafe_tool_or_dsl_calls
    task_success = (
        tool_path_correct
        and parameter_accuracy == 1.0
        and citation_correct
        and settlement_consistent
        and replan_success
        and (memory_recall or run.memory_mode == MemoryMode.no_memory)
        and not state_contaminated
        and unsafe == 0
    )
    return {
        "task_success": task_success,
        "tool_path_correct": tool_path_correct,
        "model_tool_path_correct": model_tool_path_correct,
        "parameter_accuracy": parameter_accuracy,
        "model_parameter_accuracy": model_parameter_accuracy,
        "citation_correct": citation_correct,
        "settlement_consistent": settlement_consistent,
        "replan_success": replan_success,
        "memory_recall": memory_recall,
        "state_contaminated": state_contaminated,
        "unsafe_tool_or_dsl_calls": unsafe,
        "observed_tools": observed,
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['path']}|{row['memory_mode']}"].append(row)
    output: dict[str, Any] = {}
    for key, items in sorted(grouped.items()):
        n = len(items)
        successes = sum(bool(item["task_success"]) for item in items)
        replan_items = [item for item in items if item["requires_replan"]]
        memory_items = [item for item in items if item["requires_memory"]]
        latencies = [float(item["end_to_end_latency_ms"]) for item in items]
        output[key] = {
            "attempts": n,
            "task_success_rate": successes / n,
            "task_success_wilson_95": wilson_interval(successes, n),
            "correct_tool_path_rate": sum(bool(item["tool_path_correct"]) for item in items) / n,
            "model_correct_tool_path_rate": sum(bool(item.get("model_tool_path_correct", False)) for item in items) / n,
            "parameter_accuracy": sum(float(item["parameter_accuracy"]) for item in items) / n,
            "model_parameter_accuracy": sum(float(item.get("model_parameter_accuracy", 0.0)) for item in items) / n,
            "citation_correctness": sum(bool(item["citation_correct"]) for item in items) / n,
            "settlement_consistency": sum(bool(item["settlement_consistent"]) for item in items) / n,
            "replanning_success_rate": (
                sum(bool(item["replan_success"]) for item in replan_items) / len(replan_items) if replan_items else None
            ),
            "memory_recall": (
                sum(bool(item["memory_recall"]) for item in memory_items) / len(memory_items) if memory_items else None
            ),
            "state_contamination_rate": sum(bool(item["state_contaminated"]) for item in items) / n,
            "average_steps": sum(int(item["steps"]) for item in items) / n,
            "average_prompt_tokens": sum(int(item["prompt_tokens"]) for item in items) / n,
            "average_completion_tokens": sum(int(item["completion_tokens"]) for item in items) / n,
            "p50_latency_ms": percentile(latencies, 0.50),
            "p95_latency_ms": percentile(latencies, 0.95),
            "provider_cost_aud_per_task": sum(float(item["provider_cost_aud"]) for item in items) / n,
            "total_retries": sum(int(item["retries"]) for item in items),
            "total_rejected_model_calls": sum(int(item["rejected_model_calls"]) for item in items),
            "unsafe_tool_or_dsl_calls": sum(int(item["unsafe_tool_or_dsl_calls"]) for item in items),
        }

    stability_groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        key = f"{row['path']}|{row['memory_mode']}|{row['case_id']}|{row['turn_index']}"
        stability_groups[key].append(bool(row["task_success"]))
    output["stability"] = {
        "groups": len(stability_groups),
        "pass_at_1": sum(sum(values) / len(values) for values in stability_groups.values()) / len(stability_groups),
        "pass_all_k": sum(all(values) for values in stability_groups.values()) / len(stability_groups),
        "k_min": min((len(values) for values in stability_groups.values()), default=0),
        "k_max": max((len(values) for values in stability_groups.values()), default=0),
    }
    return output
