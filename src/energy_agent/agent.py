from __future__ import annotations

import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import UTC, datetime, timedelta
from typing import Literal

from .schemas import AgentQueryRequest, AgentQueryResponse, Region, ToolCall
from .tools import ToolRegistry


class EnergyAgent:
    """Bounded deterministic state machine; provider planners may only emit registered typed calls."""

    def __init__(self, registry: ToolRegistry, timeout_seconds: float = 5) -> None:
        self.registry = registry
        self.timeout_seconds = timeout_seconds
        self.traces: dict[str, dict[str, object]] = {}

    def _plan(self, request: AgentQueryRequest) -> list[tuple[str, dict[str, object]]]:
        text = request.question.lower()
        region = next((r for r in Region if r.value.lower() in text or r.value[:-1].lower() in text), Region.NSW1)
        end = datetime.now(UTC)
        start = end - timedelta(days=2)
        dates = re.findall(r"20\d\d-\d\d-\d\d", text)
        if dates:
            start = datetime.fromisoformat(dates[0]).replace(tzinfo=UTC)
            end = datetime.fromisoformat(dates[-1]).replace(tzinfo=UTC) + timedelta(days=1)
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
        calls.append(("search_official_evidence", {"query": request.question, "top_k": 5}))
        return calls[: request.max_tool_calls]

    def run(self, request: AgentQueryRequest) -> AgentQueryResponse:
        trace_id = str(uuid.uuid4())
        records: list[ToolCall] = []
        results = []
        seen: set[str] = set()
        for name, arguments in self._plan(request):
            signature = f"{name}:{arguments}"
            if signature in seen:
                records.append(ToolCall(name=name, arguments=arguments, status="skipped"))
                continue
            seen.add(signature)
            started = time.perf_counter()
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(self.registry.execute, name, arguments).result(timeout=self.timeout_seconds)
                results.append(result)
                records.append(
                    ToolCall(name=name, arguments=arguments, duration_ms=(time.perf_counter() - started) * 1000)
                )
            except TimeoutError:
                records.append(
                    ToolCall(
                        name=name,
                        arguments=arguments,
                        status="timeout",
                        duration_ms=(time.perf_counter() - started) * 1000,
                    )
                )
            except Exception:
                records.append(
                    ToolCall(
                        name=name,
                        arguments=arguments,
                        status="error",
                        duration_ms=(time.perf_counter() - started) * 1000,
                    )
                )
        citations = [ev for result in results for ev in result.evidence]
        citations = list({ev.evidence_id: ev for ev in citations}.values())
        status: Literal["completed", "insufficient_evidence"] = (
            "completed" if results and citations else "insufficient_evidence"
        )
        answer = (
            "Verified tool outputs: " + "; ".join(f"{r.tool_name}={r.data}" for r in results)
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
        self.traces[trace_id] = {
            "trace_id": trace_id,
            "states": ["normalize", "plan", "execute", "verify", "synthesize", "done"],
            "response": response.model_dump(mode="json"),
        }
        return response
