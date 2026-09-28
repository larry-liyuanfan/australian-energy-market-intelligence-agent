"""Small real-data CLI/checkpoint demonstration, never a model-quality benchmark."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from energy_agent.graph_runtime import build_graph, registry_from_subset
from energy_agent.model_agent import AgentPath, MemoryMode, ModelDrivenAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New compact public report")
    args = parser.parse_args()
    if args.output.exists() or args.private_output.exists():
        raise ValueError("use new output paths; never overwrite a prior demonstration")
    args.private_output.mkdir(parents=True)
    checkpoint = args.private_output / "checkpoints.sqlite"
    timings: dict[str, float] = {}
    responses: dict[str, Any] = {}
    env = dict(os.environ, LANGSMITH_TRACING="false", LANGCHAIN_TRACING_V2="false")
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")

    def invoke(label: str, thread: str, *options: str) -> dict[str, Any]:
        output = args.private_output / f"{label}.json"
        command = [sys.executable, str(Path(__file__).with_name("run_langgraph_market.py")),
                   "--data", str(args.data), "--data-sha256", args.data_sha256,
                   "--checkpoint", str(checkpoint), "--thread", thread, "--output", str(output), *options]
        started = time.perf_counter()
        subprocess.run(command, check=True, capture_output=True, env=env, timeout=30)
        timings[label] = time.perf_counter() - started
        responses[label] = json.loads(output.read_text())
        return dict(responses[label])

    q1 = "Show the SA1 market snapshot for 2025-12-15 with official evidence."
    q2 = "Correction: use VIC1; keep the date."
    qb = "Compare SA1 and VIC1 market records for 2025-12-15 with official evidence."
    paused1 = invoke("a1_pause", "analyst_A", "--question", q1, "--pause")
    first = invoke("a1_resume", "analyst_A", "--resume", "approve")
    paused2 = invoke("a2_pause", "analyst_A", "--question", q2, "--pause")
    second = invoke("a2_resume", "analyst_A", "--resume", "approve")
    other = invoke("b1_comparison", "analyst_B", "--question", qb)
    incomplete = invoke("c1_missing_date", "analyst_C", "--question", "Show the SA1 market snapshot.")
    registry = registry_from_subset(args.data, args.data_sha256)
    baseline = ModelDrivenAgent(registry, None)
    original1 = baseline.run_turn(q1, conversation_id="baseline_A", path=AgentPath.deterministic,
                                  memory_mode=MemoryMode.structured_state, seed=0)
    original2 = baseline.run_turn(q2, conversation_id="baseline_A", path=AgentPath.deterministic,
                                  memory_mode=MemoryMode.structured_state, seed=0)
    original_b = baseline.run_turn(qb, conversation_id="baseline_B", path=AgentPath.deterministic,
                                   memory_mode=MemoryMode.structured_state, seed=0)
    checks = {
        "first_success": first["status"] == "completed",
        "corrected_success": second["status"] == "completed",
        "comparison_success": other["status"] == "completed",
        "pause_before_tools": paused1["interrupts"] and not paused1["calls"]
        and paused2["interrupts"] and not paused2["calls"],
        "restored_date_source_turn_1": second["conversation"]["constraints"]["dates"] ==
        first["conversation"]["constraints"]["dates"],
        "corrected_region_source_turn_2": second["conversation"]["constraints"]["region"]["value"] == "VIC1"
        and second["conversation"]["constraints"]["region"]["source_turn"] == 2,
        "resume_did_not_duplicate_user_turn": len(first["conversation"]["user_turns"]) == 1
        and len(second["conversation"]["user_turns"]) == 2,
        "no_cross_thread_date_inheritance": incomplete["status"] == "insufficient_context"
        and "dates" not in incomplete["conversation"]["constraints"],
        "original_runtime_equal_first": first["validated_results"] == [r.model_dump(mode="json") for r in original1.results],
        "original_runtime_equal_correction": second["validated_results"] == [r.model_dump(mode="json") for r in original2.results],
        "original_runtime_equal_comparison": other["validated_results"] == [r.model_dump(mode="json") for r in original_b.results],
    }
    with SqliteSaver.from_conn_string(str(checkpoint)) as saver:
        graph = build_graph(registry, saver)
        restored_a = graph.get_state({"configurable": {"thread_id": "analyst_A"}}).values
        restored_b = graph.get_state({"configurable": {"thread_id": "analyst_B"}}).values
        checks["other_thread_did_not_modify_A"] = restored_a["conversation"] == second["conversation"]
        checks["independent_comparison_thread"] = restored_b["conversation"]["constraints"]["regions"]["value"] == ["SA1", "VIC1"]
    # Coerce expressions using list truthiness to explicit JSON booleans.
    checks = {name: bool(value) for name, value in checks.items()}
    public = {
        "schema_version": "energy-langgraph-real-demo-v1", "execution_mode": "live_local_deterministic_graph_no_LLM",
        "data_track": "official_aemo_archival_subset", "data_sha256": args.data_sha256,
        "source_archive_sha256": registry.store.evidence[0].sha256, "market_rows": len(registry.store.rows),
        "checks": checks, "completed_business_turns": sum(r["status"] == "completed" for r in (first, second, other)),
        "process_invocations": len(responses), "cross_process_interrupt_resumes": 2,
        "private_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_backend": "langgraph.checkpoint.sqlite.SqliteSaver",
        "versions": {name: importlib.metadata.version(name) for name in
                     ("langgraph", "langgraph-checkpoint", "langgraph-checkpoint-sqlite", "langchain-core")},
        "invocation_wall_seconds_including_python_startup": timings,
        "model_calls": 0, "model_tokens": 0, "new_gpu_jobs": 0,
        "tool_attempts_completed_turns": sum(len(r["calls"]) for r in (first, second, other)),
        "case_summaries": [{"question": q, "answer": r["answer"], "events": r["events"],
                            "constraints": r["conversation"]["constraints"], "citation_ids": [c["evidence_id"] for c in r["citations"]]}
                           for q, r in ((q1, first), (q2, second), (qb, other))],
        "boundaries": ["Three completed business turns, not a quality benchmark or production SLA.",
                       "Persistent local SQLite, not distributed durability, authentication or exactly-once execution.",
                       "Graph adapter validates market snapshot/comparison/coverage only; full BESS demo uses the original runtime.",
                       "Transient retry was tested with explicit synthetic faults, not a real dependency outage.",
                       "Read-only tools can be repeated after interruption; no trading/bidding side effects."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(public, handle, indent=2, allow_nan=False)
    print(json.dumps({"checks_passed": sum(checks.values()), "checks_total": len(checks),
                      "completed_business_turns": public["completed_business_turns"]}))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
