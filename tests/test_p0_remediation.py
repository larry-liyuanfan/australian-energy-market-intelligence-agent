from __future__ import annotations

from test_goal_compiler import InvalidPlanner
from test_model_agent import ScriptedPlanner

from energy_agent.goal_compiler import GoalSpecAgent, LlamaCppGoalSpecPlanner
from energy_agent.market import fixture_store
from energy_agent.model_agent import AgentPath, ConversationMemory, MemoryMode, ModelDrivenAgent
from energy_agent.schemas import AgentQueryRequest
from energy_agent.tools import ToolRegistry


def test_replaced_region_cannot_reenter_typed_case_from_original_prose() -> None:
    memory = ConversationMemory()
    memory.begin_turn("c", "Compare SA1 and VIC1 on 2025-01-02.", MemoryMode.structured_state)
    state, _ = memory.begin_turn("c", "Replace VIC1 with QLD1.", MemoryMode.structured_state)
    request = AgentQueryRequest(question="Replace VIC1 with QLD1.")
    case = ModelDrivenAgent._case_from_constraints(request, memory.constraints_for_mode(state, MemoryMode.structured_state))
    assert case.workflow_type == "region_comparison"
    assert case.requested_regions == ["SA1", "QLD1"]
    assert case.window.start.isoformat() == "2025-01-02T00:00:00+10:00"
    assert case.window.end.isoformat() == "2025-01-03T00:00:00+10:00"


def test_comparison_extends_visible_region_but_no_memory_does_not_inherit() -> None:
    for mode, expected in [(MemoryMode.structured_state, ["SA1", "QLD1"]), (MemoryMode.no_memory, ["QLD1"])]:
        memory = ConversationMemory()
        memory.begin_turn("c", "SA1 on 2025-01-02", mode)
        state, _ = memory.begin_turn("c", "Compare it with QLD1.", mode)
        assert memory.constraints_for_mode(state, mode)["regions"].value == expected


def test_hybrid_uses_sourced_battery_constraints_when_model_omits_them() -> None:
    agent = ModelDrivenAgent(ToolRegistry(fixture_store()), ScriptedPlanner([[]]))
    run = agent.run_turn(
        "Replay SA1 on 2025-01-03 with a 2MW/4MWh BESS at 88% efficiency.",
        conversation_id="battery", path=AgentPath.constrained_hybrid, memory_mode=MemoryMode.structured_state,
    )
    dispatch = next(call for call in run.tool_calls if call.name == "optimize_battery_dispatch")
    assert dispatch.arguments["battery"] == {"power_mw": 2.0, "energy_mwh": 4.0, "round_trip_efficiency": 0.88}
    assert run.initial_model_proposed_calls == []
    assert len(run.planner_attempts) == 1 + run.metrics.replans
    assert run.metrics.fallback_calls > 0


def test_goal_decoding_schema_rejects_bess_parent_key() -> None:
    sources = LlamaCppGoalSpecPlanner.decoding_schema()["properties"]["field_sources"]
    assert sources["additionalProperties"] is False
    assert "bess" not in sources["properties"]
    assert "bess.power_mw" in sources["properties"]
    assert "time_range" in sources["required"]


def test_invalid_goal_does_not_erase_attributed_user_context() -> None:
    class CapturingPlanner(InvalidPlanner):
        def __init__(self) -> None:
            self.calls: list[list[dict[str, object]]] = []

        def plan_goal(self, messages: list[dict[str, object]], seed: int):
            self.calls.append(messages)
            return super().plan_goal(messages, seed)

    planner = CapturingPlanner()
    agent = GoalSpecAgent(ToolRegistry(fixture_store()), planner)
    agent.run_turn("Compare SA1 and VIC1 on 2025-01-02.", conversation_id="c")
    agent.run_turn("Replace VIC1 with QLD1.", conversation_id="c")
    assert any("[source_turn=1] Compare SA1" in str(item["content"]) for item in planner.calls[1])
    assert not any("PRIOR_SOURCED_GOAL_SPEC" in str(item["content"]) for item in planner.calls[1])
