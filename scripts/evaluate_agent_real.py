from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from energy_agent.agent import EnergyAgent
from energy_agent.evaluation import citation_structure_metrics
from energy_agent.evidence import HybridEvidenceIndex, load_official_chunks
from energy_agent.market import MarketStore, load_dispatch_store
from energy_agent.schemas import AgentQueryRequest, ToolResult
from energy_agent.tools import ToolRegistry


@dataclass(frozen=True)
class Task:
    task_id: str
    split: str
    question: str
    expected_tools: list[str]
    fault: str | None = None


class FaultRegistry(ToolRegistry):
    def __init__(self, store: MarketStore, evidence: HybridEvidenceIndex, fault: str | None) -> None:
        super().__init__(store, evidence)
        self.fault = fault
        self.injected = False

    def execute(self, name: str, arguments: dict[str, object]) -> ToolResult:
        if self.fault and not self.injected:
            self.injected = True
            if self.fault == "error":
                raise RuntimeError("injected transient error")
            if self.fault == "timeout":
                time.sleep(0.08)
            if self.fault == "empty":
                return ToolResult(tool_name=name)
        return super().execute(name, arguments)


def tasks() -> list[Task]:
    regions = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]
    dates = ["2025-09-15", "2025-12-15", "2026-03-15", "2026-06-15"]
    output: list[Task] = []
    for region in regions:
        for day in dates:
            suffix = f"{region}-{day}"
            output.extend(
                [
                    Task(
                        f"event-{suffix}",
                        "real_window",
                        f"Detect and explain price events in {region} on {day}",
                        ["detect_price_events", "search_official_evidence"],
                    ),
                    Task(
                        f"forecast-{suffix}",
                        "real_window",
                        f"Forecast price risk for {region} on {day}",
                        ["forecast_price_risk", "search_official_evidence"],
                    ),
                    Task(
                        f"battery-{suffix}",
                        "real_window",
                        f"Optimise BESS dispatch for {region} on {day}",
                        ["optimize_battery_dispatch", "search_official_evidence"],
                    ),
                    Task(
                        f"coverage-{suffix}",
                        "real_window",
                        f"Explain data coverage for {region} on {day}",
                        ["explain_data_coverage", "search_official_evidence"],
                    ),
                ]
            )
    faults = ["error", "empty", "timeout"]
    for index in range(20):
        region = regions[index % len(regions)]
        day = dates[index % len(dates)]
        output.append(
            Task(
                f"fault-{index + 1:03d}",
                "fault_fixture",
                f"Detect price events in {region} on {day}",
                ["detect_price_events", "search_official_evidence"],
                faults[index % len(faults)],
            )
        )
    assert len(output) == 100
    return output


def percentile(values: list[float], quantile: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(quantile * len(values)) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    store = load_dispatch_store(args.data, args.data_manifest)
    evidence = HybridEvidenceIndex(load_official_chunks(args.evidence))
    task_set = tasks()
    rows: list[dict[str, Any]] = []
    for variant in ("direct_no_tools", "single_tool", "state_machine"):
        for task in task_set:
            started = time.perf_counter()
            if variant == "direct_no_tools":
                status = "completed"
                calls = []
                citations = 0
                validity = True
                recovered = False
                citation_structure = {
                    "claim_citation_completeness": 0.0,
                    "citation_id_validity": 0.0,
                }
            else:
                registry = FaultRegistry(store, evidence, task.fault)
                timeout = 0.01 if task.fault == "timeout" else 10.0
                response = EnergyAgent(registry, timeout_seconds=timeout).run(
                    AgentQueryRequest(question=task.question, max_tool_calls=1 if variant == "single_tool" else 6)
                )
                status = response.status
                calls = response.tool_calls
                citations = len(response.citations)
                validity = all(registry.validate(call.name, call.arguments) is not None for call in calls)
                recovered = any(call.recovered and call.status == "ok" for call in calls)
                citation_structure = citation_structure_metrics(
                    response.answer,
                    {citation.evidence_id for citation in response.citations},
                )
            observed = {call.name for call in calls if call.status == "ok"}
            expected_observed = len(set(task.expected_tools) & observed) / len(task.expected_tools)
            attempted = [call for call in calls if call.status != "skipped"]
            successful_attempts = sum(call.status == "ok" for call in attempted)
            success = status == "completed" and all(name in observed for name in task.expected_tools) and citations > 0
            rows.append(
                {
                    "task_id": task.task_id,
                    "split": task.split,
                    "variant": variant,
                    "status": status,
                    "expected_tools": task.expected_tools,
                    "observed_tools": sorted(observed),
                    "schema_valid": validity,
                    "citations": citations,
                    "claim_citation_completeness": citation_structure["claim_citation_completeness"],
                    "citation_id_validity": citation_structure["citation_id_validity"],
                    "recovered": recovered,
                    "logical_tool_success": expected_observed,
                    "attempt_success": successful_attempts / len(attempted) if attempted else 0.0,
                    "task_success": success,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            )
    metrics: dict[str, Any] = {"scope": "80 real-window tasks and 20 separately reported fault fixtures"}
    for variant in ("direct_no_tools", "single_tool", "state_machine"):
        metrics[variant] = {}
        for split in ("real_window", "fault_fixture", "all"):
            selected = [row for row in rows if row["variant"] == variant and (split == "all" or row["split"] == split)]
            latencies = [row["latency_ms"] for row in selected]
            metrics[variant][split] = {
                "tasks": len(selected),
                "task_success": statistics.fmean(row["task_success"] for row in selected),
                "schema_validity": statistics.fmean(row["schema_valid"] for row in selected),
                "citation_completeness": statistics.fmean(row["citations"] > 0 for row in selected),
                "claim_citation_completeness": statistics.fmean(
                    row["claim_citation_completeness"] for row in selected
                ),
                "citation_id_validity": statistics.fmean(row["citation_id_validity"] for row in selected),
                "logical_tool_success": statistics.fmean(row["logical_tool_success"] for row in selected),
                "attempt_success": statistics.fmean(row["attempt_success"] for row in selected),
                "failure_recovery": statistics.fmean(row["recovered"] for row in selected)
                if split == "fault_fixture"
                else None,
                "p50_latency_ms": statistics.median(latencies),
                "p95_latency_ms": percentile(latencies, 0.95),
            }
    tasks_path = args.output / "tasks.jsonl"
    predictions_path = args.output / "predictions.jsonl"
    tasks_path.write_text("".join(json.dumps(asdict(task)) + "\n" for task in task_set), encoding="utf-8")
    predictions_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    manifest = {
        "git_sha": git_sha,
        "python": sys.version,
        "platform": platform.platform(),
        "data_version": store.data_version,
        "task_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "provider": "deterministic",
        "provider_cost_usd": 0.0,
        "task_distribution": dict(Counter(task.split for task in task_set)),
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
