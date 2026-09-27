from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langgraph", reason="optional graph extra; dedicated graph CI installs it")
from langgraph.checkpoint.sqlite import SqliteSaver

from energy_agent.graph_runtime import build_graph, invoke_graph
from energy_agent.market import fixture_store
from energy_agent.schemas import ToolResult
from energy_agent.tools import ToolRegistry


def registry() -> ToolRegistry:
    store = fixture_store()
    # Explicit synthetic unit fixture, never counted in real-data evidence.
    store.evidence = [e.model_copy(update={"url": "https://nemweb.com.au/fixture-contract"}) for e in store.evidence]
    return ToolRegistry(store)


def test_sqlite_restart_correction_and_thread_isolation(tmp_path: Path) -> None:
    path = str(tmp_path / "private.sqlite")
    with SqliteSaver.from_conn_string(path) as saver:
        graph = build_graph(registry(), saver)
        paused = invoke_graph(graph, "A", question="Show SA1 snapshot for 2025-01-02", pause=True)
        assert paused["__interrupt__"] and paused["calls"] == []
        with pytest.raises(ValueError, match="paused"):
            invoke_graph(graph, "A", question="change VIC1")
    with SqliteSaver.from_conn_string(path) as saver:
        graph = build_graph(registry(), saver)
        completed = invoke_graph(graph, "A", resume="approve")
        assert completed["status"] == "completed"
        corrected = invoke_graph(graph, "A", question="Correction: use VIC1; keep the date.")
        assert corrected["status"] == "completed"
        state = corrected["conversation"]["constraints"]
        assert state["region"]["value"] == "VIC1" and state["region"]["source_turn"] == 2
        assert state["dates"]["value"] == ["2025-01-02"] and state["dates"]["source_turn"] == 1
        other = invoke_graph(graph, "B", question="Show SA1 snapshot for 2025-01-03")
        assert other["conversation"]["constraints"]["region"]["value"] == "SA1"
        assert len(other["conversation"]["user_turns"]) == 1
        assert graph.get_state({"configurable": {"thread_id": "A"}}).values == corrected


@pytest.mark.parametrize("failures,retries,status", [(1, 1, "completed"), (10, 1, "tool_failed")])
def test_transient_retry_is_bounded(tmp_path: Path, failures: int, retries: int, status: str) -> None:
    class FailingRegistry(ToolRegistry):
        attempts = 0

        def execute(self, name: str, arguments: dict[str, object]) -> ToolResult:
            if name == "get_market_snapshot":
                self.attempts += 1
                if self.attempts <= failures:
                    raise TimeoutError("explicit unit fault")
            return super().execute(name, arguments)

    fault = FailingRegistry(registry().store)
    with SqliteSaver.from_conn_string(str(tmp_path / "state.sqlite")) as saver:
        result = invoke_graph(build_graph(fault, saver), "retry", question="Show SA1 snapshot for 2025-01-02")
    assert result["retry_count"] == retries and result["status"] == status
    assert fault.attempts == 2
    if status != "completed":
        assert not result["citations"] and "No complete answer" in result["answer"]


def test_cancel_and_wrong_resume_fail_closed(tmp_path: Path) -> None:
    with SqliteSaver.from_conn_string(str(tmp_path / "state.sqlite")) as saver:
        graph = build_graph(registry(), saver)
        invoke_graph(graph, "A", question="Show SA1 snapshot for 2025-01-02", pause=True)
        with pytest.raises(ValueError, match="no pending"):
            invoke_graph(graph, "B", resume="approve")
        result = invoke_graph(graph, "A", resume="cancel")
        assert result["status"] == "cancelled" and result["calls"] == []


def test_dataset_switch_on_resume_fails(tmp_path: Path) -> None:
    path = str(tmp_path / "state.sqlite")
    with SqliteSaver.from_conn_string(path) as saver:
        invoke_graph(build_graph(registry(), saver), "A", question="Show SA1 snapshot for 2025-01-02", pause=True)
    different = registry()
    different.store.data_version = "different"
    with SqliteSaver.from_conn_string(path) as saver, pytest.raises(ValueError, match="dataset version"):
        invoke_graph(build_graph(different, saver), "A", resume="approve")


def test_result_conflict_not_summarised(tmp_path: Path) -> None:
    class WrongRegion(ToolRegistry):
        def execute(self, name: str, arguments: dict[str, object]) -> ToolResult:
            result = super().execute(name, arguments)
            if name == "get_market_snapshot":
                data: dict[str, Any] = {**result.data, "region": "WRONG"}
                return result.model_copy(update={"data": data})
            return result

    with SqliteSaver.from_conn_string(str(tmp_path / "state.sqlite")) as saver:
        result = invoke_graph(build_graph(WrongRegion(registry().store), saver), "A",
                              question="Show SA1 snapshot for 2025-01-02")
    assert result["status"] == "tool_failed" and not result["citations"]
    assert result["retry_count"] == 0


def test_wrong_tool_cannot_bypass_request_result_validation(tmp_path: Path) -> None:
    class WrongTool(ToolRegistry):
        def execute(self, name: str, arguments: dict[str, object]) -> ToolResult:
            return ToolResult(tool_name="explain_data_coverage", data={"unrelated": True}, evidence=self.store.evidence)

    with SqliteSaver.from_conn_string(str(tmp_path / "state.sqlite")) as saver:
        result = invoke_graph(build_graph(WrongTool(registry().store), saver), "A",
                              question="Show SA1 snapshot for 2025-01-02")
    assert result["status"] == "tool_failed" and result["error"] == "result_tool_identity_mismatch"
    assert not result["citations"] and result["retry_count"] == 0


def test_valid_citation_does_not_authorise_changed_market_value(tmp_path: Path) -> None:
    class WrongPrice(ToolRegistry):
        def execute(self, name: str, arguments: dict[str, object]) -> ToolResult:
            result = super().execute(name, arguments)
            if name == "get_market_snapshot":
                return result.model_copy(update={"data": {**result.data, "rrp": -99999}})
            return result

    with SqliteSaver.from_conn_string(str(tmp_path / "state.sqlite")) as saver:
        result = invoke_graph(build_graph(WrongPrice(registry().store), saver), "A",
                              question="Show SA1 snapshot for 2025-01-02")
    assert result["status"] == "tool_failed" and not result["citations"]


def test_slow_tool_documents_cooperative_not_hard_timeout(tmp_path: Path) -> None:
    class SlowTool(ToolRegistry):
        def execute(self, name: str, arguments: dict[str, object]) -> ToolResult:
            time.sleep(.03)
            return super().execute(name, arguments)

    started = time.perf_counter()
    with SqliteSaver.from_conn_string(str(tmp_path / "state.sqlite")) as saver:
        result = invoke_graph(build_graph(SlowTool(registry().store), saver, timeout_seconds=.001), "A",
                              question="Show SA1 snapshot for 2025-01-02")
    assert result["status"] == "tool_failed" and result["retry_count"] == 1
    assert time.perf_counter() - started >= .06  # existing worker exit is awaited
