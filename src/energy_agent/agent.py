from __future__ import annotations

import re
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Literal

from .schemas import AgentQueryRequest, AgentQueryResponse, Region, ToolCall, ToolResult
from .tools import ToolRegistry

if TYPE_CHECKING:
    from .providers import ModelStudioPlanner


class EnergyAgent:
    """Bounded deterministic state machine; provider planners may only emit registered typed calls."""

    def __init__(
        self,
        registry: ToolRegistry,
        timeout_seconds: float = 5,
        planner_provider: ModelStudioPlanner | None = None,
        trace_capacity: int = 128,
    ) -> None:
        if trace_capacity < 1:
            raise ValueError("trace_capacity must be positive")
        self.registry = registry
        self.timeout_seconds = timeout_seconds
        self.planner_provider = planner_provider
        self.trace_capacity = trace_capacity
        self._trace_lock = threading.Lock()
        self.traces: OrderedDict[str, dict[str, object]] = OrderedDict()
        self.trace_evictions = 0

    def get_trace(self, trace_id: str) -> dict[str, object] | None:
        with self._trace_lock:
            trace = self.traces.get(trace_id)
            if trace is not None:
                self.traces.move_to_end(trace_id)
            return trace

    def trace_stats(self) -> tuple[int, int, int]:
        with self._trace_lock:
            return len(self.traces), self.trace_capacity, self.trace_evictions

    def _store_trace(self, trace_id: str, trace: dict[str, object]) -> None:
        with self._trace_lock:
            self.traces[trace_id] = trace
            self.traces.move_to_end(trace_id)
            while len(self.traces) > self.trace_capacity:
                self.traces.popitem(last=False)
                self.trace_evictions += 1

    def _plan(self, request: AgentQueryRequest) -> list[tuple[str, dict[str, object]]]:
        if self.planner_provider is not None:
            return self.planner_provider.plan(request.question, self.registry, request.max_tool_calls)
        text = request.question.lower()
        region = next((r for r in Region if r.value.lower() in text or r.value[:-1].lower() in text), Region.NSW1)
        end = datetime.now(UTC)
        start = end - timedelta(days=2)
        dates = re.findall(r"20\d\d-\d\d-\d\d", text)
        if dates:
            nem_time = timezone(timedelta(hours=10))
            start = datetime.fromisoformat(dates[0]).replace(tzinfo=nem_time)
            end = datetime.fromisoformat(dates[-1]).replace(tzinfo=nem_time) + timedelta(days=1)
        window = {"start": start.isoformat(), "end": end.isoformat()}
        calls: list[tuple[str, dict[str, object]]] = []
        if "coverage" in text or "覆盖" in text:
            calls.append(("explain_data_coverage", {"region": region.value}))
        if "battery" in text or "bess" in text or "电池" in text:
            calls.append(("optimize_battery_dispatch", {"region": region.value, "window": window}))
        if "forecast" in text or "risk" in text or "预测" in text or "风险" in text:
            calls.append(("forecast_price_risk", {"region": region.value, "window": window, "horizon_intervals": 12}))
        if "event" in text or "spike" in text or "异常" in text:
            calls.append(("detect_price_events", {"region": region.value, "window": window}))
        chart_terms = ("chart", "figure", "plot", "graph", "trend line", "图表", "图中", "曲线")
        table_terms = ("table", "tabular", "表格", "表中")
        preferred_modality = (
            "chart"
            if any(term in text for term in chart_terms)
            else "table"
            if any(term in text for term in table_terms)
            else "auto"
        )
        retrieval_mode = "multimodal_fusion" if preferred_modality != "auto" else "hybrid_rerank"
        calls.append(
            (
                "search_official_evidence",
                {
                    "query": request.question,
                    "top_k": 5,
                    "preferred_modality": preferred_modality,
                    "retrieval_mode": retrieval_mode,
                },
            )
        )
        return calls[: request.max_tool_calls]

    @staticmethod
    def _summarize(result: ToolResult) -> str:
        data = result.data
        if result.tool_name == "optimize_battery_dispatch":
            return (
                f"{result.tool_name}: gross_margin_aud={data.get('gross_margin_aud')}, "
                f"equivalent_full_cycles={data.get('equivalent_full_cycles')}, "
                f"solver_seconds={data.get('solve_seconds')}, intervals={len(data.get('charge_mw', []))}; "
                f"{data.get('economic_boundary')}"
            )
        if result.tool_name == "forecast_price_risk":
            point = data.get("point", [])
            lower = data.get("lower", [])
            upper = data.get("upper", [])
            return (
                f"{result.tool_name}: horizon={len(point)}, point_range={EnergyAgent._range(point)}, "
                f"interval_range={EnergyAgent._range(lower)}..{EnergyAgent._range(upper)}"
            )
        if result.tool_name == "detect_price_events":
            events = data.get("events", [])
            return f"{result.tool_name}: events={len(events)}, top={events[:3]}"
        return f"{result.tool_name}: {data}"

    @staticmethod
    def _range(values: object) -> str:
        if not isinstance(values, list) or not values:
            return "empty"
        numeric = [float(value) for value in values]
        return f"[{min(numeric):.2f}, {max(numeric):.2f}]"

    @classmethod
    def _cited_summary(cls, result: ToolResult) -> str:
        summary = cls._summarize(result)
        evidence_ids = [evidence.evidence_id for evidence in result.evidence[:2]]
        citations = " ".join(f"[@{evidence_id}]" for evidence_id in evidence_ids)
        return f"- {summary} {citations}".rstrip()

    def run(self, request: AgentQueryRequest) -> AgentQueryResponse:
        trace_id = str(uuid.uuid4())
        records: list[ToolCall] = []
        results = []
        seen: set[str] = set()
        calls: list[tuple[str, dict[str, object]]] = []
        for name, arguments in self._plan(request):
            signature = f"{name}:{arguments}"
            if signature in seen:
                records.append(ToolCall(name=name, arguments=arguments, status="skipped"))
                continue
            seen.add(signature)
            calls.append((name, arguments))
        failed: list[tuple[str, dict[str, object]]] = []
        with ThreadPoolExecutor(max_workers=min(4, len(calls) or 1)) as pool:
            pending = [
                (name, arguments, time.perf_counter(), pool.submit(self.registry.execute, name, arguments))
                for name, arguments in calls
            ]
            for name, arguments, started, future in pending:
                try:
                    result = future.result(timeout=self.timeout_seconds)
                    if result.data or result.evidence:
                        results.append(result)
                        records.append(
                            ToolCall(name=name, arguments=arguments, duration_ms=(time.perf_counter() - started) * 1000)
                        )
                    else:
                        failed.append((name, arguments))
                        records.append(
                            ToolCall(
                                name=name,
                                arguments=arguments,
                                status="error",
                                duration_ms=(time.perf_counter() - started) * 1000,
                            )
                        )
                except TimeoutError:
                    failed.append((name, arguments))
                    records.append(
                        ToolCall(
                            name=name,
                            arguments=arguments,
                            status="timeout",
                            duration_ms=(time.perf_counter() - started) * 1000,
                        )
                    )
                except Exception:
                    failed.append((name, arguments))
                    records.append(
                        ToolCall(
                            name=name,
                            arguments=arguments,
                            status="error",
                            duration_ms=(time.perf_counter() - started) * 1000,
                        )
                    )
        for name, arguments in failed:
            started = time.perf_counter()
            # A transient timeout needs a longer bounded recovery window; retrying
            # with the identical deadline deterministically reproduces the fault.
            retry_timeout = min(30.0, max(0.25, self.timeout_seconds * 10))
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(self.registry.execute, name, arguments).result(timeout=retry_timeout)
                if result.data or result.evidence:
                    results.append(result)
                    records.append(
                        ToolCall(
                            name=name,
                            arguments=arguments,
                            duration_ms=(time.perf_counter() - started) * 1000,
                            recovered=True,
                        )
                    )
                else:
                    records.append(
                        ToolCall(
                            name=name,
                            arguments=arguments,
                            status="error",
                            duration_ms=(time.perf_counter() - started) * 1000,
                            recovered=True,
                        )
                    )
            except TimeoutError:
                records.append(
                    ToolCall(
                        name=name,
                        arguments=arguments,
                        status="timeout",
                        duration_ms=(time.perf_counter() - started) * 1000,
                        recovered=True,
                    )
                )
            except Exception:
                records.append(
                    ToolCall(
                        name=name,
                        arguments=arguments,
                        status="error",
                        duration_ms=(time.perf_counter() - started) * 1000,
                        recovered=True,
                    )
                )
        citations = [ev for result in results for ev in result.evidence]
        citations = list({ev.evidence_id: ev for ev in citations}.values())
        successful_tools = {result.tool_name for result in results}
        required_tools = {name for name, _arguments in calls}
        status: Literal["completed", "insufficient_evidence"] = (
            "completed"
            if results and citations and required_tools.issubset(successful_tools)
            else "insufficient_evidence"
        )
        answer = (
            "Verified tool outputs (citations identify provenance records; they are not a semantic-entailment score):\n"
            + "\n".join(self._cited_summary(result) for result in results)
            if results
            else "No verified result."
        )
        response = AgentQueryResponse(
            trace_id=trace_id,
            status=status,
            answer=answer,
            citations=citations,
            tool_calls=records,
            data_version=self.registry.store.data_version,
        )
        self._store_trace(trace_id, {
            "trace_id": trace_id,
            "states": ["normalize", "plan", "execute", "verify", "synthesize", "done"],
            "verified_results": [result.model_dump(mode="json") for result in results],
            "response": response.model_dump(mode="json"),
        })
        return response
