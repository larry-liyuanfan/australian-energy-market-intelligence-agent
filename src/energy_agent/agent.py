from __future__ import annotations

import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Literal

from .schemas import AgentQueryRequest, AgentQueryResponse, DecisionCase, Region, ToolCall, ToolResult, Window
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
        case = self._build_case(request)
        text = request.question.lower()
        region = case.region
        window = case.window.model_dump(mode="json")
        calls: list[tuple[str, dict[str, object]]] = []
        evidence_only = (
            any(term in text for term in ("chart", "figure", "plot", "graph", "table", "图表", "表格"))
            and not re.search(r"20\d\d-\d\d-\d\d", text)
        )
        if case.workflow_type in {"decision_replay", "event_diagnosis"} or (
            case.workflow_type == "general_market_query" and not evidence_only
        ):
            calls.append(("get_market_snapshot", {"region": region.value, "at": case.window.start.isoformat()}))
        if case.workflow_type == "coverage":
            calls.append(("explain_data_coverage", {"region": region.value}))
        if case.workflow_type == "region_comparison":
            calls.append(
                (
                    "compare_region_period",
                    {"regions": [item.value for item in case.requested_regions], "window": window},
                )
            )
        if case.workflow_type in {"decision_replay", "forecast_risk"}:
            horizon = 288 if case.workflow_type == "decision_replay" else 12
            calls.append(
                ("forecast_price_risk", {"region": region.value, "window": window, "horizon_intervals": horizon})
            )
        if case.workflow_type in {"decision_replay", "event_diagnosis"}:
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
        if case.workflow_type == "decision_replay":
            calls.append(
                (
                    "optimize_battery_dispatch",
                    {
                        "region": region.value,
                        "window": window,
                        "settlement_mode": "historical_replay",
                        "variable_degradation_cost_aud_per_mwh_discharged": 50.0,
                    },
                )
            )
        return calls[: request.max_tool_calls]

    @staticmethod
    def _build_case(request: AgentQueryRequest) -> DecisionCase:
        text = request.question.lower()
        requested_regions = [
            region
            for region in Region
            if region.value.lower() in text or region.value[:-1].lower() in text
        ]
        region = requested_regions[0] if requested_regions else Region.NSW1
        end = datetime.now(UTC)
        start = end - timedelta(days=2)
        dates = re.findall(r"20\d\d-\d\d-\d\d", text)
        if dates:
            nem_time = timezone(timedelta(hours=10))
            start = datetime.fromisoformat(dates[0]).replace(tzinfo=nem_time)
            end = datetime.fromisoformat(dates[-1]).replace(tzinfo=nem_time) + timedelta(days=1)
        battery = any(term in text for term in ("battery", "bess", "dispatch", "电池", "调度"))
        comparison = any(term in text for term in ("compare", "versus", " vs ", "比较", "对比"))
        coverage = "coverage" in text or "覆盖" in text
        forecast = any(term in text for term in ("forecast", "risk", "预测", "风险"))
        event = any(term in text for term in ("event", "spike", "anomaly", "异常", "尖峰", "发生了什么"))
        workflow_type: Literal[
            "decision_replay",
            "event_diagnosis",
            "region_comparison",
            "forecast_risk",
            "coverage",
            "general_market_query",
        ]
        if battery:
            workflow_type = "decision_replay"
        elif comparison:
            workflow_type = "region_comparison"
        elif coverage:
            workflow_type = "coverage"
        elif forecast:
            workflow_type = "forecast_risk"
        elif event:
            workflow_type = "event_diagnosis"
        else:
            workflow_type = "general_market_query"
        if workflow_type == "region_comparison" and len(requested_regions) < 2:
            requested_regions = [region, Region.VIC1 if region != Region.VIC1 else Region.NSW1]
        return DecisionCase(
            workflow_type=workflow_type,
            region=region,
            window=Window(start=start, end=end),
            requested_regions=requested_regions,
            states=["intent", "constraints"],
        )

    @staticmethod
    def _summarize(result: ToolResult) -> str:
        data = result.data
        if result.tool_name == "optimize_battery_dispatch":
            return (
                f"{result.tool_name}: planned_margin_aud={data.get('planned_margin_aud')}, "
                f"realized_margin_aud={data.get('realized_margin_aud')}, "
                f"margin_basis={data.get('margin_basis')}, oracle_regret_aud={data.get('oracle_regret_aud')}, "
                f"equivalent_full_cycles={data.get('equivalent_full_cycles')}, "
                f"solver_ms={data.get('runtime_ms')}, intervals={len(data.get('charge_mw', []))}; "
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

    @staticmethod
    def _recovery_call(
        name: str, arguments: dict[str, object]
    ) -> tuple[dict[str, object], Literal[
        "retry_with_backoff",
        "visual_to_text_fallback",
        "text_to_visual_escalation",
    ]]:
        """Choose a bounded alternate channel for evidence failures.

        Market calculations are retried unchanged after transient failures. Evidence
        search instead changes modality once: visual failures fall back to the cheap
        text index, while empty text retrieval escalates to the visual page index.
        """

        recovered = dict(arguments)
        if name != "search_official_evidence":
            return recovered, "retry_with_backoff"
        if recovered.get("retrieval_mode") == "multimodal_fusion":
            recovered["retrieval_mode"] = "hybrid_rerank"
            recovered["preferred_modality"] = "text"
            return recovered, "visual_to_text_fallback"
        recovered["retrieval_mode"] = "multimodal_fusion"
        recovered["preferred_modality"] = "auto"
        return recovered, "text_to_visual_escalation"

    def run(self, request: AgentQueryRequest) -> AgentQueryResponse:
        trace_id = str(uuid.uuid4())
        case = self._build_case(request)
        records: list[ToolCall] = []
        results: list[ToolResult] = []
        seen: set[str] = set()
        planned_calls: list[tuple[str, dict[str, object]]] = []
        for name, arguments in self._plan(request):
            signature = f"{name}:{arguments}"
            if signature in seen:
                records.append(ToolCall(name=name, arguments=arguments, status="skipped"))
                continue
            seen.add(signature)
            planned_calls.append((name, arguments))

        def execute_stage(calls: list[tuple[str, dict[str, object]]], state: str) -> None:
            if not calls:
                return
            case.states.append(state)
            failed: list[tuple[str, dict[str, object]]] = []
            # SciPy/HiGHS has an upstream Windows access-violation history when
            # native solves run in short-lived Python worker threads.  The
            # production Linux path retains hard timeouts; Windows clean-room
            # runs serialize the one native MILP call on the caller thread.
            windows_native_dispatch = (
                os.name == "nt" and len(calls) == 1 and calls[0][0] == "optimize_battery_dispatch"
            )
            pool: ThreadPoolExecutor | None = None
            pending: list[tuple[str, dict[str, object], float, Future[ToolResult] | None]]
            if windows_native_dispatch:
                pending = [(calls[0][0], calls[0][1], time.perf_counter(), None)]
            else:
                pool = ThreadPoolExecutor(max_workers=min(4, len(calls)))
                pending = [
                    (name, arguments, time.perf_counter(), pool.submit(self.registry.execute, name, arguments))
                    for name, arguments in calls
                ]
            try:
                for name, arguments, started, future in pending:
                    try:
                        if windows_native_dispatch:
                            result = self.registry.execute(name, arguments)
                        else:
                            assert future is not None
                            result = future.result(timeout=self.timeout_seconds)
                        if result.data or result.evidence:
                            results.append(result)
                            records.append(
                                ToolCall(
                                    name=name,
                                    arguments=arguments,
                                    duration_ms=(time.perf_counter() - started) * 1000,
                                )
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
            finally:
                if not windows_native_dispatch:
                    assert pool is not None
                    pool.shutdown(wait=True)
            for name, arguments in failed:
                started = time.perf_counter()
                retry_timeout = min(30.0, max(0.25, self.timeout_seconds * 10))
                recovery_arguments, recovery_strategy = self._recovery_call(name, arguments)
                try:
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        result = pool.submit(self.registry.execute, name, recovery_arguments).result(
                            timeout=retry_timeout
                        )
                    if result.data or result.evidence:
                        results.append(result)
                        records.append(
                            ToolCall(
                                name=name,
                                arguments=recovery_arguments,
                                duration_ms=(time.perf_counter() - started) * 1000,
                                recovered=True,
                                attempt=2,
                                recovery_strategy=recovery_strategy,
                            )
                        )
                    else:
                        records.append(
                            ToolCall(
                                name=name,
                                arguments=recovery_arguments,
                                status="error",
                                duration_ms=(time.perf_counter() - started) * 1000,
                                recovered=True,
                                attempt=2,
                                recovery_strategy=recovery_strategy,
                            )
                        )
                except TimeoutError:
                    records.append(
                        ToolCall(
                            name=name,
                            arguments=recovery_arguments,
                            status="timeout",
                            duration_ms=(time.perf_counter() - started) * 1000,
                            recovered=True,
                            attempt=2,
                            recovery_strategy=recovery_strategy,
                        )
                    )
                except Exception:
                    records.append(
                        ToolCall(
                            name=name,
                            arguments=recovery_arguments,
                            status="error",
                            duration_ms=(time.perf_counter() - started) * 1000,
                            recovered=True,
                            attempt=2,
                            recovery_strategy=recovery_strategy,
                        )
                    )

        context_names = {
            "get_market_snapshot",
            "compare_region_period",
            "detect_price_events",
            "explain_data_coverage",
            "forecast_price_risk",
        }
        execute_stage([call for call in planned_calls if call[0] in context_names], "market_context")

        if len(records) < request.max_tool_calls:
            event_result = next((result for result in results if result.tool_name == "detect_price_events"), None)
            events = event_result.data.get("events", []) if event_result else []
            if events:
                top_event = max(events, key=lambda item: float(item.get("rrp", 0)))
                diagnose_arguments: dict[str, object] = {
                    "region": case.region.value,
                    "interval": str(top_event["interval"]),
                    "context_intervals": 12,
                }
                signature = f"diagnose_price_event:{diagnose_arguments}"
                if signature not in seen:
                    seen.add(signature)
                    planned_calls.append(("diagnose_price_event", diagnose_arguments))
                    execute_stage([("diagnose_price_event", diagnose_arguments)], "event_diagnosis")

        execute_stage(
            [call for call in planned_calls if call[0] == "search_official_evidence"],
            "evidence_retrieval",
        )
        execute_stage(
            [call for call in planned_calls if call[0] == "optimize_battery_dispatch"],
            "dispatch_and_settlement",
        )
        citations = [ev for result in results for ev in result.evidence]
        citations = list({ev.evidence_id: ev for ev in citations}.values())
        successful_tools = {result.tool_name for result in results}
        required_tools = {name for name, _arguments in planned_calls}
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
        by_tool = {result.tool_name: result for result in results}
        case.market_context = {
            name: by_tool[name].data
            for name in (
                "get_market_snapshot",
                "compare_region_period",
                "detect_price_events",
                "diagnose_price_event",
                "explain_data_coverage",
            )
            if name in by_tool
        }
        case.forecast = by_tool.get("forecast_price_risk", ToolResult(tool_name="forecast_price_risk")).data
        case.dispatch = by_tool.get(
            "optimize_battery_dispatch", ToolResult(tool_name="optimize_battery_dispatch")
        ).data
        case.historical_settlement = {
            key: case.dispatch.get(key)
            for key in ("realized_margin_aud", "oracle_regret_aud", "margin_basis")
            if case.dispatch.get(key) is not None
        }
        case.states.extend(["verification", "answer"])
        warnings = list(dict.fromkeys(warning for result in results for warning in result.warnings))
        economic_boundary = case.dispatch.get("economic_boundary")
        if economic_boundary:
            warnings.append(str(economic_boundary))
        verification = {
            "citation_count": len(citations),
            "all_citation_hashes_well_formed": all(len(evidence.sha256) == 64 for evidence in citations),
            "required_tools_satisfied": required_tools.issubset(successful_tools),
            "economic_boundary_present": bool(economic_boundary) if case.dispatch else True,
        }
        if case.historical_settlement:
            decision_summary = (
                f"Historical decision replay completed for {case.region.value}: "
                f"planned margin {case.dispatch.get('planned_margin_aud')}, realised margin "
                f"{case.dispatch.get('realized_margin_aud')} AUD under the declared operating-proxy boundary."
            )
        else:
            decision_summary = f"{case.workflow_type.replace('_', ' ').title()} completed with {len(results)} verified tool outputs."
        response = AgentQueryResponse(
            trace_id=trace_id,
            status=status,
            answer=answer,
            citations=citations,
            tool_calls=records,
            data_version=self.registry.store.data_version,
            workflow_type=case.workflow_type,
            decision_summary=decision_summary,
            market_context=case.market_context,
            forecast=case.forecast,
            dispatch=case.dispatch,
            historical_settlement=case.historical_settlement,
            verification=verification,
            limitations=warnings,
        )
        self._store_trace(
            trace_id,
            {
                "trace_id": trace_id,
                "states": case.states,
                "decision_case": case.model_dump(mode="json"),
                "progress_ledger": {
                    "planned_tools": sorted(required_tools),
                    "successful_tools": sorted(successful_tools),
                    "missing_tools": sorted(required_tools - successful_tools),
                    "unique_call_signatures": len(seen),
                    "skipped_duplicate_calls": sum(record.status == "skipped" for record in records),
                    "recovery_attempts": sum(record.recovered for record in records),
                    "recovery_strategies": [
                        record.recovery_strategy for record in records if record.recovery_strategy != "none"
                    ],
                    "stalled": bool(required_tools - successful_tools),
                },
                "verification": verification,
                "verified_results": [result.model_dump(mode="json") for result in results],
                "response": response.model_dump(mode="json"),
            },
        )
        return response
