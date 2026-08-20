from __future__ import annotations

import threading
from collections import Counter

from .schemas import AgentQueryResponse


class ServiceMetrics:
    buckets = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.queries: Counter[str] = Counter()
        self.tools: Counter[tuple[str, str, str]] = Counter()
        self.duration_count = 0
        self.duration_sum = 0.0
        self.duration_buckets: Counter[float] = Counter()
        self.citations = 0

    def observe(self, response: AgentQueryResponse, duration_seconds: float) -> None:
        with self._lock:
            self.queries[response.status] += 1
            self.duration_count += 1
            self.duration_sum += duration_seconds
            for bucket in self.buckets:
                if duration_seconds <= bucket:
                    self.duration_buckets[bucket] += 1
            self.citations += len(response.citations)
            for call in response.tool_calls:
                self.tools[(call.name, call.status, str(call.recovered).lower())] += 1

    def render(
        self,
        provider: str,
        evidence_backend: str,
        rows: int,
        chunks: int,
        trace_entries: int,
        trace_capacity: int,
        trace_evictions: int,
    ) -> str:
        with self._lock:
            lines = [
                "# HELP energy_agent_up Process is serving requests.",
                "# TYPE energy_agent_up gauge",
                "energy_agent_up 1",
                "# TYPE energy_agent_queries_total counter",
            ]
            for status, count in sorted(self.queries.items()):
                lines.append(f'energy_agent_queries_total{{status="{status}"}} {count}')
            lines.append("# TYPE energy_agent_query_duration_seconds histogram")
            for bucket in self.buckets:
                lines.append(
                    f'energy_agent_query_duration_seconds_bucket{{le="{bucket:g}"}} {self.duration_buckets[bucket]}'
                )
            lines.extend(
                [
                    f'energy_agent_query_duration_seconds_bucket{{le="+Inf"}} {self.duration_count}',
                    f"energy_agent_query_duration_seconds_sum {self.duration_sum:.9f}",
                    f"energy_agent_query_duration_seconds_count {self.duration_count}",
                    "# TYPE energy_agent_tool_calls_total counter",
                ]
            )
            for (tool, status, recovered), count in sorted(self.tools.items()):
                lines.append(
                    f'energy_agent_tool_calls_total{{tool="{tool}",status="{status}",recovered="{recovered}"}} {count}'
                )
            lines.extend(
                [
                    "# TYPE energy_agent_citations_total counter",
                    f"energy_agent_citations_total {self.citations}",
                    f'energy_agent_provider_info{{provider="{provider}"}} 1',
                    f'energy_agent_evidence_backend_info{{backend="{evidence_backend}"}} 1',
                    f"energy_agent_market_rows {rows}",
                    f"energy_agent_evidence_chunks {chunks}",
                    f"energy_agent_trace_cache_entries {trace_entries}",
                    f"energy_agent_trace_cache_capacity {trace_capacity}",
                    f"energy_agent_trace_evictions_total {trace_evictions}",
                ]
            )
            return "\n".join(lines) + "\n"
