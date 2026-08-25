from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from energy_agent.agent import EnergyAgent
from energy_agent.composite_evidence import CompositeEvidenceIndex
from energy_agent.decision_evaluation import bess_golden_checks, decision_tasks, fault_tasks
from energy_agent.evidence import HybridEvidenceIndex, load_official_chunks
from energy_agent.market import load_dispatch_store
from energy_agent.schemas import AgentQueryRequest, ToolResult
from energy_agent.snapshots import ForecastSnapshotStore, load_forecast_snapshots
from energy_agent.tools import ToolRegistry
from energy_agent.workbook_evidence import FigureEvidenceIndex, load_figure_evidence_records


class FaultRegistry(ToolRegistry):
    def __init__(self, base: ToolRegistry, fault: str) -> None:
        snapshots = ForecastSnapshotStore() if fault == "stale_snapshot" else base.forecast_snapshots
        super().__init__(base.store, base.evidence_index, snapshots)
        self.fault = fault
        self.injected = False

    def execute(self, name: str, arguments: dict[str, object]) -> ToolResult:
        if self.fault == "timeout" and not self.injected:
            self.injected = True
            time.sleep(0.08)
        if self.fault == "empty_figure" and name == "search_official_evidence" and not self.injected:
            self.injected = True
            return ToolResult(tool_name=name)
        result = super().execute(name, arguments)
        if (
            self.fault == "citation_hash"
            and name == "search_official_evidence"
            and result.evidence
            and not self.injected
        ):
            self.injected = True
            evidence = result.evidence[0].model_copy(update={"sha256": "invalid"})
            return result.model_copy(update={"evidence": [evidence, *result.evidence[1:]]})
        return result


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the unified historical decision-replay Agent")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--forecast-snapshots", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    store = load_dispatch_store(args.data, args.data_manifest)
    text_index = HybridEvidenceIndex(load_official_chunks(args.evidence))
    figure_index = FigureEvidenceIndex(load_figure_evidence_records(args.figures))
    evidence_index = CompositeEvidenceIndex(text_index, figure_index)
    snapshots = (
        load_forecast_snapshots(args.forecast_snapshots)
        if args.forecast_snapshots and args.forecast_snapshots.is_file()
        else ForecastSnapshotStore()
    )
    registry = ToolRegistry(store, evidence_index, snapshots)
    real_rows: list[dict[str, Any]] = []
    for task in decision_tasks():
        started = time.perf_counter()
        response = EnergyAgent(registry, timeout_seconds=10).run(
            AgentQueryRequest(question=task.question, max_tool_calls=8)
        )
        observed = {call.name for call in response.tool_calls if call.status == "ok"}
        citation_valid = bool(response.citations) and all(
            citation.url.startswith("https://") and len(citation.sha256) == 64 for citation in response.citations
        )
        figure_grounded = task.category != "figure_grounding" or any(
            citation.modality in {"chart", "table", "mixed", "page_image"}
            and citation.asset_id
            and citation.asset_sha256
            for citation in response.citations
        )
        bess_checks = bess_golden_checks(response) if task.category == "decision_replay" else {}
        passed = (
            response.status == "completed"
            and response.workflow_type == task.expected_workflow
            and set(task.expected_tools).issubset(observed)
            and citation_valid
            and figure_grounded
            and all(bess_checks.values())
            and bool(response.verification.get("required_tools_satisfied"))
        )
        real_rows.append(
            {
                **asdict(task),
                "status": response.status,
                "observed_tools": sorted(observed),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "citation_valid": citation_valid,
                "figure_grounded": bool(figure_grounded),
                "bess_checks": bess_checks,
                "passed": passed,
                "trace_id": response.trace_id,
            }
        )

    fault_rows: list[dict[str, Any]] = []
    for fault_task in fault_tasks():
        fault_registry = FaultRegistry(registry, fault_task.fault)
        timeout = 0.01 if fault_task.fault == "timeout" else 10.0
        response = EnergyAgent(fault_registry, timeout_seconds=timeout).run(
            AgentQueryRequest(question=fault_task.question, max_tool_calls=8)
        )
        if fault_task.fault in {"timeout", "empty_figure"}:
            handled = response.status == "completed" and any(call.recovered for call in response.tool_calls)
        elif fault_task.fault == "incomplete_window":
            handled = response.status == "insufficient_evidence"
        elif fault_task.fault == "stale_snapshot":
            handled = response.status == "completed" and any("seasonal fallback" in item for item in response.limitations)
        else:
            handled = response.verification.get("all_citation_hashes_well_formed") is False
        fault_rows.append(
            {
                **asdict(fault_task),
                "status": response.status,
                "recovery_strategies": [
                    call.recovery_strategy for call in response.tool_calls if call.recovery_strategy != "none"
                ],
                "handled": handled,
            }
        )

    latencies = [float(row["latency_ms"]) for row in real_rows]
    tool_attempts = sum(len(row["observed_tools"]) for row in real_rows)
    expected_attempts = sum(len(row["expected_tools"]) for row in real_rows)
    figure_rows = [row for row in real_rows if row["category"] == "figure_grounding"]
    bess_rows = [row for row in real_rows if row["category"] == "decision_replay"]
    task_success_rate = sum(bool(row["passed"]) for row in real_rows) / len(real_rows)
    tool_execution_success_rate = min(1.0, tool_attempts / expected_attempts)
    citation_validity_rate = sum(bool(row["citation_valid"]) for row in real_rows) / len(real_rows)
    figure_grounding_rate = sum(bool(row["figure_grounded"]) for row in figure_rows) / len(figure_rows)
    bess_golden_rate = sum(all(row["bess_checks"].values()) for row in bess_rows) / len(bess_rows)
    fault_handling_rate = sum(bool(row["handled"]) for row in fault_rows) / len(fault_rows)
    p95_latency_ms = percentile(latencies, 0.95)
    promotion_pass = (
        task_success_rate >= 0.85
        and tool_execution_success_rate >= 0.95
        and citation_validity_rate >= 0.95
        and figure_grounding_rate >= 0.90
        and bess_golden_rate == 1.0
        and fault_handling_rate >= 0.95
        and p95_latency_ms < 2000
    )
    metrics: dict[str, object] = {
        "schema_version": "decision-replay-eval-v1",
        "real_tasks": len(real_rows),
        "fault_tasks": len(fault_rows),
        "task_success_rate": task_success_rate,
        "tool_execution_success_rate": tool_execution_success_rate,
        "citation_validity_rate": citation_validity_rate,
        "figure_grounding_rate": figure_grounding_rate,
        "bess_golden_rate": bess_golden_rate,
        "fault_handling_rate": fault_handling_rate,
        "p95_latency_ms": p95_latency_ms,
        "promotion_pass": promotion_pass,
    }
    predictions_path = args.output / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in [*real_rows, *fault_rows]) + "\n",
        encoding="utf-8",
    )
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    git_sha = os.environ.get("ENERGY_GIT_COMMIT") or subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "python": platform.python_version(),
        "data_manifest_sha256": sha256(args.data_manifest),
        "evidence_sha256": sha256(args.evidence),
        "figure_records_sha256": sha256(args.figures),
        "metrics_sha256": sha256(metrics_path),
        "predictions_sha256": sha256(predictions_path),
        "boundaries": [
            "Historical decision replay, not live trading or automatic bidding.",
            "Figure relevance labels remain author-curated; source-cell provenance is not answer entailment.",
            "Economic values are historical operating proxies and exclude CAPEX, fixed O&M, network fees and FCAS.",
        ],
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if not promotion_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
