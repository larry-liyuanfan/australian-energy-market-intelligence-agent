from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from energy_agent.goal_compiler import (
    ComparisonMode,
    DeterministicGoalExtractor,
    EvidenceModality,
    GoalBESS,
    GoalIntent,
    GoalSpec,
    GoalSpecAgent,
    GoalSpecCompiler,
    GoalSpecPlannerOutcome,
    GoalTimeRange,
    RequestedOutput,
)
from energy_agent.market import fixture_store
from energy_agent.providers import PlannerUsage
from energy_agent.tools import ToolRegistry


def replay_goal() -> GoalSpec:
    return GoalSpec(
        source_turn=1,
        intent=GoalIntent.decision_replay,
        regions=["SA1"],
        comparison_mode=ComparisonMode.none,
        time_range=GoalTimeRange(
            start="2025-01-08T00:00:00+10:00",
            end="2025-01-09T00:00:00+10:00",
        ),
        requested_outputs=[
            RequestedOutput.market_snapshot,
            RequestedOutput.event_diagnosis,
            RequestedOutput.official_evidence,
            RequestedOutput.forecast,
            RequestedOutput.bess_dispatch,
            RequestedOutput.historical_settlement,
        ],
        evidence_modality=EvidenceModality.chart,
        bess=GoalBESS(power_mw=1.5, energy_mwh=3.0, round_trip_efficiency=0.88),
        corrections=[],
        field_sources={
            "intent": 1,
            "regions": 1,
            "comparison_mode": 1,
            "time_range": 1,
            "requested_outputs": 1,
            "evidence_modality": 1,
            "bess.power_mw": 1,
            "bess.energy_mwh": 1,
            "bess.round_trip_efficiency": 1,
        },
    )


def test_goal_spec_requires_bess_and_complete_sources() -> None:
    data = replay_goal().model_dump(mode="json")
    data["bess"] = None
    with pytest.raises(ValidationError):
        GoalSpec.model_validate(data)
    data = replay_goal().model_dump(mode="json")
    del data["field_sources"]["regions"]
    with pytest.raises(ValidationError):
        GoalSpec.model_validate(data)


def test_compiler_expands_goal_to_canonical_decision_dag() -> None:
    compiled = GoalSpecCompiler.compile(replay_goal())
    assert [call.name for call in compiled.tool_calls] == [
        "get_market_snapshot",
        "detect_price_events",
        "search_official_evidence",
        "forecast_price_risk",
        "optimize_battery_dispatch",
    ]
    search = compiled.tool_calls[2]
    assert search.arguments["retrieval_mode"] == "multimodal_fusion"
    battery = compiled.tool_calls[-1].arguments["battery"]
    assert battery == {"power_mw": 1.5, "energy_mwh": 3.0, "round_trip_efficiency": 0.88}
    assert compiled.decision_case.schema_version == "decision-case-v1"


def test_deterministic_extractor_attributes_cross_turn_corrections() -> None:
    extractor = DeterministicGoalExtractor()
    first = extractor.extract("conversation", "Compare NSW1 and TAS1 on 2024-10-03 using an official table.")
    second = extractor.extract("conversation", "Replace TAS1 with SA1 and retain the day.")
    assert first.regions == ["NSW1", "TAS1"]
    assert second.regions == ["NSW1", "SA1"]
    assert second.time_range == first.time_range
    assert second.field_sources["time_range"] == 1
    assert second.field_sources["regions"] == 2
    assert [item.field for item in second.corrections] == ["regions"]


class InvalidPlanner:
    def plan_goal(self, messages: list[dict[str, object]], seed: int) -> GoalSpecPlannerOutcome:
        assert messages[-1]["role"] == "user"
        return GoalSpecPlannerOutcome(
            raw_goal_spec={"tool_calls": [{"name": "raw_sql"}]},
            goal_spec=None,
            usage=PlannerUsage(prompt_tokens=10, completion_tokens=4),
            provider="scripted",
            model="invalid",
            seed=seed,
            validation_errors=("forbidden_goal_field",),
            unsafe_or_forbidden_fields=1,
        )


def test_invalid_model_goal_never_falls_back_to_deterministic_goal() -> None:
    agent = GoalSpecAgent(ToolRegistry(fixture_store()), InvalidPlanner())
    run = agent.run_turn("Replay a 1MW/2MWh BESS for SA1 on 2024-12-18.", conversation_id="invalid", seed=101)
    assert run.status == "invalid_goal_spec"
    assert run.compiled_goal is None
    assert run.tool_calls == []
    assert run.metrics.unsafe_or_forbidden_fields == 1


def test_v2_benchmark_is_exactly_18_episodes_36_turns_and_source_separated() -> None:
    root = Path(__file__).parents[1]
    v2 = [json.loads(line) for line in (root / "benchmarks/goal_spec_holdout_v2.jsonl").read_text().splitlines()]
    v1_text = (root / "benchmarks/llm_agent_holdout_v1.jsonl").read_text() + (
        root / "benchmarks/llm_agent_development_v1.jsonl"
    ).read_text()
    assert len(v2) == 18
    assert sum(len(item["turns"]) for item in v2) == 36
    assert len({item["case_id"] for item in v2}) == 18
    for episode in v2:
        assert episode["split"] == "holdout"
        for turn in episode["turns"]:
            start = turn["expected_goal"]["time_range"]["start"][:10]
            assert start not in v1_text


def test_gate_matches_frozen_benchmark_shape() -> None:
    root = Path(__file__).parents[1]
    gate = json.loads((root / "benchmarks/goal_spec_promotion_gate_v2.json").read_text())
    assert gate["episodes"] == 18
    assert gate["turns"] == 36
    assert gate["thresholds"]["goal_spec_required_field_f1"] == 0.9
    assert gate["thresholds"]["compiled_task_success"] == 0.85
    assert gate["thresholds"]["unsafe_tool_or_dsl_calls"] == 0


def test_v2_freeze_manifest_matches_immutable_inputs() -> None:
    root = Path(__file__).parents[1]
    freeze = json.loads((root / "benchmarks/v2_freeze_manifest.json").read_text())
    for section, path_key, sha_key in (
        ("goal_spec", "benchmark", "benchmark_sha256"),
        ("goal_spec", "gate", "gate_sha256"),
        ("vidore", "config", "config_sha256"),
        ("vidore", "weight_selection_source", "weight_selection_source_sha256"),
    ):
        path = root / freeze[section][path_key]
        git_normalized = path.read_text(encoding="utf-8").encode()
        assert hashlib.sha256(git_normalized).hexdigest() == freeze[section][sha_key]
