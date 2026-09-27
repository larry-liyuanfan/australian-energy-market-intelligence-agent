"""V3 scoring: exact typed parameters and first-proposal attribution.

The historical v1/v2 scorers remain unchanged so published results retain their
original meaning. This scorer is frozen before the new evaluation is executed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from .llm_evaluation import ordered_subsequence, score_turn
from .market import MarketStore
from .model_agent import ModelAgentRun
from .schemas import TOOL_MODELS, Region, ToolCall


def validate_market_windows(episodes: list[dict[str, Any]], store: MarketStore) -> dict[str, int]:
    if "fixture" in store.data_version.lower():
        raise ValueError("real-model evaluation requires a real market store")
    windows: set[tuple[str, str]] = set()
    for episode in episodes:
        for turn in episode["turns"]:
            expected = turn["expected"]
            regions = expected.get("regions", [expected.get("region")])
            for region in regions:
                windows.add((region, expected["date"]))
    for region, day in sorted(windows):
        start = datetime.fromisoformat(day).replace(tzinfo=timezone(timedelta(hours=10)))
        rows = store.select(Region(region), start, start + timedelta(days=1))
        expected_times = [start + timedelta(minutes=5 * i) for i in range(288)]
        if [row.interval for row in rows] != expected_times:
            raise ValueError(f"incomplete five-minute market window: {region}/{day}")
        if len(store.before(Region(region), start, limit=288)) < 288:
            raise ValueError(f"insufficient as-of history: {region}/{day}")
    return {"complete_region_days": len(windows), "intervals_per_day": 288}


def exact_parameter_accuracy(calls: list[ToolCall], expected: dict[str, Any]) -> float:
    checks: list[bool] = []
    day = datetime.fromisoformat(expected["date"]).replace(tzinfo=timezone(timedelta(hours=10)))
    regions = expected.get("regions", [expected.get("region")])
    for call in calls:
        args = TOOL_MODELS[call.name].model_validate(call.arguments).model_dump(mode="json")
        if call.name == "search_official_evidence":
            continue
        if "region" in args:
            checks.append(args["region"] == regions[0])
        if "regions" in args:
            checks.append(set(args["regions"]) == set(regions))
        if "at" in args:
            checks.append(datetime.fromisoformat(args["at"]) == day)
        if "window" in args:
            checks.append(datetime.fromisoformat(args["window"]["start"]) == day)
            checks.append(datetime.fromisoformat(args["window"]["end"]) == day + timedelta(days=1))
        if call.name == "optimize_battery_dispatch":
            checks.append(args.get("settlement_mode") == "historical_replay")
            for source, target in (
                ("battery_power_mw", "power_mw"), ("battery_energy_mwh", "energy_mwh"),
                ("round_trip_efficiency", "round_trip_efficiency"),
            ):
                if source in expected:
                    checks.append(args.get("battery", {}).get(target) == expected[source])
    return sum(checks) / len(checks) if checks else 0.0


def score_remediation_turn(run: ModelAgentRun, turn: dict[str, Any]) -> dict[str, Any]:
    score = score_turn(run, turn)
    successful = [call for call in run.tool_calls if call.status == "ok"]
    initial = run.initial_model_proposed_calls
    expected_tools = turn["expected_tools"]
    score["legacy_all_proposals_path_correct"] = score["model_tool_path_correct"]
    score["parameter_accuracy"] = exact_parameter_accuracy(successful, turn["expected"])
    score["model_parameter_accuracy"] = exact_parameter_accuracy(initial, turn["expected"])
    score["model_tool_path_correct"] = ordered_subsequence(expected_tools, [call.name for call in initial])
    if run.path.value == "deterministic":
        score["model_tool_path_correct"] = score["tool_path_correct"]
        score["model_parameter_accuracy"] = score["parameter_accuracy"]
    required_replan = bool(turn.get("requires_replan"))
    failed_tools = {call.name for call in run.tool_calls if call.status in {"error", "timeout"}}
    recovered_tools = {call.name for call in successful if call.recovered}
    score["replan_success"] = not required_replan or (
        bool(failed_tools) and failed_tools.issubset(recovered_tools)
        and (run.metrics.replans > 0 or (run.path.value == "deterministic" and run.metrics.retries > 0))
    )
    score["missing_initial_tools"] = sorted(set(expected_tools) - {call.name for call in initial})
    score["missing_executed_tools"] = sorted(set(expected_tools) - {call.name for call in successful})
    score["model_requests"] = len(run.planner_attempts)
    score["task_success"] = (
        score["tool_path_correct"] and score["parameter_accuracy"] == 1.0
        and score["citation_correct"] and score["settlement_consistent"] and score["replan_success"]
        and not score["state_contaminated"] and score["unsafe_tool_or_dsl_calls"] == 0
        and (score["memory_recall"] or run.memory_mode.value == "no_memory")
    )
    return score
