from energy_agent.agent import EnergyAgent
from energy_agent.decision_evaluation import bess_golden_checks, decision_tasks, fault_tasks
from energy_agent.market import fixture_store
from energy_agent.schemas import AgentQueryRequest
from energy_agent.tools import ToolRegistry


def test_decision_evaluation_suite_has_declared_category_balance() -> None:
    tasks = decision_tasks()
    assert len(tasks) == 60
    assert {category: sum(task.category == category for task in tasks) for category in {t.category for t in tasks}} == {
        "event_diagnosis": 15,
        "region_comparison": 15,
        "figure_grounding": 15,
        "decision_replay": 15,
    }
    faults = fault_tasks()
    assert len(faults) == 20
    assert all(sum(task.fault == fault for task in faults) == 4 for fault in {task.fault for task in faults})


def test_bess_golden_checks_cover_physics_and_settlement() -> None:
    response = EnergyAgent(ToolRegistry(fixture_store())).run(
        AgentQueryRequest(question="Replay BESS dispatch for SA1 on 2025-01-03", max_tool_calls=8)
    )
    assert all(bess_golden_checks(response).values())
