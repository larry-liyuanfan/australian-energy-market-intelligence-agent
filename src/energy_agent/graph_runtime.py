"""Opt-in LangGraph market-context adapter; default API/runtime is unchanged.

SQLite state is private and local. This adapter deliberately validates snapshot,
comparison and coverage workflows only; full BESS replay remains in the original
runtime and its separately recorded demonstration. No model is called here.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict
from urllib.parse import urlsplit

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .agent import EnergyAgent
from .market import MarketRow, MarketStore
from .model_agent import ConversationMemory, ConversationState, MemoryMode, ModelDrivenAgent
from .schemas import AgentQueryRequest, Evidence, Region, StrictModel, ToolCall, ToolResult
from .tools import ToolRegistry


class GraphState(TypedDict, total=False):
    question: str
    require_approval: bool
    conversation: dict[str, Any]
    dataset_sha256: str
    plan: list[dict[str, Any]]
    cursor: int
    attempts: int
    retry_count: int
    result: dict[str, Any] | None
    validated_results: list[dict[str, Any]]
    calls: list[dict[str, Any]]
    events: list[str]
    error: str | None
    status: str
    answer: str
    citations: list[dict[str, Any]]


class Approval(StrictModel):
    action: Literal["approve", "cancel"]


def market_answer(results: list[ToolResult]) -> str:
    """Plain-language deterministic answer; never summarise unvalidated output."""
    lines = []
    for result in results:
        data = result.data
        refs = " ".join(f"[@{e.evidence_id}]" for e in result.evidence[:2])
        if result.tool_name == "get_market_snapshot":
            lines.append(f"At {data['interval']}, {data['region']} spot price was AUD {data['rrp']:.2f}/MWh "
                         f"and demand was {data['demand_mw']:.2f} MW. {refs}")
        elif result.tool_name == "compare_region_period":
            values = "; ".join(f"{region}: mean AUD {values['mean_rrp']:.2f}/MWh across {values['count']} intervals"
                               for region, values in data.items())
            lines.append(f"For the requested historical window, {values}. {refs}")
        elif result.tool_name == "explain_data_coverage":
            lines.append(f"This mounted subset covers {data.get('rows')} rows in {data.get('regions')}. {refs}")
        elif result.tool_name == "search_official_evidence":
            lines.append(f"The linked AEMO archive supplies numeric provenance, not a causal explanation. {refs}")
    return "\n".join(lines)


def registry_from_subset(path: Path, expected_sha256: str) -> ToolRegistry:
    """Require a pinned private subset; do not label fixture values as AEMO."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("private market subset hash mismatch")
    payload = json.loads(raw)
    if payload.get("data_track") != "official_aemo_archival_subset":
        raise ValueError("real demo requires official archival subset")
    rows = [MarketRow(
        datetime.strptime(r["interval"], "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone(timedelta(hours=10))),
        Region(r["region"]), float(r["rrp"]), float(r["total_demand_mw"]),
        float(r["available_generation_mw"]), float(r["net_interchange_mw"]), r["intervention"] != "0",
    ) for r in payload["rows"]]
    evidence = [Evidence.model_validate(c) for c in payload["numeric_evidence"]]
    if not evidence or any(e.sha256 != payload["source_sha256"] for e in evidence):
        raise ValueError("subset parent provenance mismatch")
    if len({(row.region, row.interval) for row in rows}) != len(rows):
        raise ValueError("duplicate subset market interval")
    # The subset hash is part of the version, so resuming against different local
    # bytes fails even when the upstream archive digest is unchanged.
    return ToolRegistry(MarketStore(rows, evidence, f"aemo-subset-{expected_sha256}"))


def _restore(payload: dict[str, Any] | None, thread: str) -> tuple[ConversationMemory, ConversationState]:
    memory = ConversationMemory()
    state = memory.state(thread)
    if payload:
        saved = ConversationState.model_validate(payload)
        if saved.conversation_id != thread:
            raise ValueError("checkpoint conversation/thread mismatch")
        state.user_turns = saved.user_turns
        state.constraints = saved.constraints
        state.tool_summaries = saved.tool_summaries
    return memory, state


def build_graph(registry: ToolRegistry, checkpointer: Any, *, timeout_seconds: float = 2) -> Any:
    """LangGraph owns transitions, retry branching and persisted state boundaries."""
    # Existing thread-based timeout detects slow work but waits for the worker to
    # exit. It is NOT a hard wall-clock/cancellation guarantee. This opt-in adapter
    # runs bounded local read-only tools; external/side-effecting tools need a
    # separately cancellable worker boundary before being supported here.
    executor = ModelDrivenAgent(registry, None, timeout_seconds=timeout_seconds)

    def normalise(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
        thread = str(config["configurable"]["thread_id"])
        if state.get("dataset_sha256", registry.store.data_version) != registry.store.data_version:
            raise ValueError("checkpoint dataset version differs from execution registry")
        memory, _ = _restore(state.get("conversation"), thread)
        conversation, _ = memory.begin_turn(thread, state["question"], MemoryMode.structured_state)
        return {"conversation": conversation.model_dump(mode="json"), "dataset_sha256": registry.store.data_version,
                "cursor": 0, "attempts": 0, "retry_count": 0, "result": None, "validated_results": [],
                "calls": [], "events": ["normalise"], "error": None, "status": "planning", "answer": "",
                "citations": []}

    def plan(state: GraphState) -> dict[str, Any]:
        conversation = ConversationState.model_validate(state["conversation"])
        if not {"regions", "dates"}.issubset(conversation.constraints):
            return {"error": "explicit_region_and_date_required", "plan": [], "status": "insufficient_context",
                    "events": [*state["events"], "plan:missing_context"]}
        request = AgentQueryRequest(question=state["question"])
        case = ModelDrivenAgent._case_from_constraints(request, conversation.constraints)
        if case.workflow_type not in {"general_market_query", "region_comparison", "coverage"}:
            return {"error": "workflow_not_validated_in_graph_adapter", "plan": [], "status": "unsupported",
                    "events": [*state["events"], "plan:unsupported"]}
        if case.workflow_type == "region_comparison" and len(case.requested_regions) < 2:
            return {"error": "comparison_requires_two_regions", "plan": [], "status": "insufficient_context",
                    "events": [*state["events"], "plan:missing_context"]}
        calls = EnergyAgent(registry)._plan(request, case=case)
        # Tool definitions/execution remain the existing eight-tool registry.
        planned = [ToolCall(name=name, arguments=registry.validate(name, args).model_dump(mode="json")).model_dump(
            mode="json") for name, args in calls]
        return {"plan": planned, "status": "planned", "events": [*state["events"], "plan:deterministic"]}

    def approval(state: GraphState) -> dict[str, Any]:
        # No side effects before interrupt: this node restarts on resume.
        if state.get("require_approval"):
            response = Approval.model_validate(interrupt({
                "kind": "review_read_only_market_plan", "plan": state["plan"],
                "instruction": "Resume this same thread with action approve or cancel.",
            }))
            if response.action == "cancel":
                return {"status": "cancelled", "events": [*state["events"], "approval:cancelled"]}
        return {"status": "approved", "events": [*state["events"], "approval:approved"]}

    def execute(state: GraphState) -> dict[str, Any]:
        # Dataset check also protects a resumed process, which skips normalise.
        if state["dataset_sha256"] != registry.store.data_version:
            raise ValueError("checkpoint dataset version differs from execution registry")
        call = ToolCall.model_validate(state["plan"][state["cursor"]])
        result, record, error = executor._execute(call.name, call.arguments)
        return {"result": result.model_dump(mode="json") if result else None,
                "calls": [*state["calls"], record.model_dump(mode="json")],
                "attempts": state["attempts"] + 1, "error": None if error == "none" else error,
                "events": [*state["events"], f"execute:{call.name}"]}

    def verify(state: GraphState) -> dict[str, Any]:
        error = state["error"]
        result = ToolResult.model_validate(state["result"]) if state["result"] is not None else None
        if result:
            if result.tool_name != state["plan"][state["cursor"]]["name"]:
                return {"error": "result_tool_identity_mismatch", "status": "tool_failed",
                        "events": [*state["events"], "verify:failed"]}
            expected = {e.evidence_id: (e.url, e.sha256) for e in registry.store.evidence}
            if not result.evidence or any(
                expected.get(e.evidence_id) != (e.url, e.sha256)
                or urlsplit(e.url).scheme != "https"
                or urlsplit(e.url).hostname not in {"www.aemo.com.au", "aemo.com.au", "nemweb.com.au"}
                for e in result.evidence
            ):
                error = "citation_provenance_mismatch"
            args = state["plan"][state["cursor"]]["arguments"]
            if result.tool_name == "get_market_snapshot":
                try:
                    interval = datetime.fromisoformat(str(result.data.get("interval")))
                    source = registry.store.closest(Region(args["region"]), datetime.fromisoformat(args["at"]))
                    matches = (result.data.get("region") == args["region"] and interval.tzinfo is not None
                               and interval == datetime.fromisoformat(args["at"]))
                    matches = matches and source is not None and result.model_dump(mode="json")["data"] == ToolResult(
                        tool_name=result.tool_name, data=source.__dict__).model_dump(mode="json")["data"]
                except (TypeError, ValueError):
                    matches = False
                if not matches:
                    error = "market_scope_or_interval_mismatch"
            if result.tool_name == "compare_region_period":
                if set(result.data) != set(args["regions"]):
                    error = "comparison_scope_mismatch"
                else:
                    for region in args["regions"]:
                        rows = registry.store.select(Region(region), datetime.fromisoformat(args["window"]["start"]),
                                                     datetime.fromisoformat(args["window"]["end"]))
                        observed = result.data[region]
                        if (len(rows) != 288 or observed.get("count") != 288
                                or not math.isclose(float(observed.get("mean_rrp", math.nan)),
                                                    math.fsum(r.rrp for r in rows) / 288, abs_tol=1e-8)
                                or observed.get("max_rrp") != max((r.rrp for r in rows), default=None)):
                            error = "comparison_source_or_window_mismatch"
        valid = result is not None and error is None
        return {"error": error, "status": "validated" if valid else "tool_failed",
                "validated_results": [*state["validated_results"], state["result"]] if valid
                else state["validated_results"],
                "events": [*state["events"], "verify:passed" if valid else "verify:failed"]}

    def advance(state: GraphState) -> dict[str, Any]:
        return {"cursor": state["cursor"] + 1, "attempts": 0, "result": None}

    def retry(state: GraphState) -> dict[str, Any]:
        return {"retry_count": state["retry_count"] + 1,
                "events": [*state["events"], "retry:bounded_once"]}

    def answer(state: GraphState) -> dict[str, Any]:
        success = state["status"] == "validated" and state["cursor"] == len(state["plan"]) - 1
        results = [ToolResult.model_validate(r) for r in state["validated_results"]]
        thread = state["conversation"]["conversation_id"]
        memory, conversation = _restore(state["conversation"], thread)
        if success:
            memory.record_results(conversation, results)
        text = market_answer(results) if success else ""
        citations = {e.evidence_id: e.model_dump(mode="json") for r in results for e in r.evidence} if success else {}
        return {"status": "completed" if success else state["status"],
                "answer": text if success else f"No complete answer: {state.get('error') or state['status']}.",
                "citations": list(citations.values()), "conversation": conversation.model_dump(mode="json"),
                "events": [*state["events"], "answer:deterministic_template"]}

    def after_verification(state: GraphState) -> str:
        if state["error"]:
            return "retry" if state["attempts"] < 2 and state["error"] in {"timeout", "empty_result"} else "answer"
        return "advance" if state["cursor"] + 1 < len(state["plan"]) else "answer"

    builder = StateGraph(GraphState)
    for name, node in (("normalise", normalise), ("plan", plan), ("approval", approval), ("execute", execute),
                       ("verify", verify), ("advance", advance), ("retry", retry), ("answer", answer)):
        builder.add_node(name, node)
    builder.add_edge(START, "normalise")
    builder.add_edge("normalise", "plan")
    builder.add_conditional_edges("plan", lambda s: "answer" if s.get("error") else "approval")
    builder.add_conditional_edges("approval", lambda s: "answer" if s["status"] == "cancelled" else "execute")
    builder.add_edge("execute", "verify")
    builder.add_conditional_edges("verify", after_verification)
    builder.add_edge("advance", "execute")
    builder.add_edge("retry", "execute")
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=checkpointer)


def invoke_graph(graph: Any, thread: str, *, question: str | None = None,
                 resume: Literal["approve", "cancel"] | None = None, pause: bool = False) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", thread):
        raise ValueError("invalid thread ID")
    if (question is None) == (resume is None):
        raise ValueError("provide exactly one of question/resume")
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 40}
    pending = graph.get_state(config).next
    if question is not None and pending:
        raise ValueError("thread is paused; resume or cancel before another user turn")
    if resume is not None and not pending:
        raise ValueError("thread has no pending interrupt")
    value: Any = Command(resume={"action": resume}) if resume else {"question": question, "require_approval": pause}
    return dict(graph.invoke(value, config=config))
