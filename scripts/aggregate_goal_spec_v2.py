from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate frozen GoalSpec v2 system runs")
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="label=run-directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    expected_benchmark_sha = sha256(args.benchmark)
    expected_gate_sha = sha256(args.gate)
    systems: dict[str, Any] = {}
    manifests: dict[str, Any] = {}
    for value in args.run:
        label, separator, raw_path = value.partition("=")
        if not separator or not label:
            parser.error("--run must use label=run-directory")
        path = Path(raw_path)
        manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest["benchmark_sha256"] != expected_benchmark_sha or manifest["gate_sha256"] != expected_gate_sha:
            raise ValueError(f"{label} was not run on the frozen benchmark and gate")
        if label != manifest["system_label"]:
            raise ValueError(f"system label mismatch for {label}")
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        systems[label] = metrics[label]
        manifests[label] = {
            "git_sha": manifest["git_sha"],
            "model": manifest["model"],
            "seeds": manifest["seeds"],
            "predictions_sha256": manifest["predictions_sha256"],
            "metrics_sha256": manifest["metrics_sha256"],
        }
    missing = set(gate["comparison_systems"]) - systems.keys()
    if missing:
        raise ValueError(f"missing frozen comparison systems: {sorted(missing)}")
    expected_seeds = sorted(int(item) for item in gate["seeds"])
    for label in gate["comparison_systems"]:
        seeds = sorted(int(item) for item in manifests[label]["seeds"])
        expected = [0] if label == "deterministic" else expected_seeds
        if seeds != expected:
            raise ValueError(f"invalid sampling seeds for {label}: {seeds}")
    candidate_name = str(gate["candidate_system"])
    candidate = systems[candidate_name]
    thresholds = gate["thresholds"]
    checks = {
        "goal_spec_required_field_f1": candidate.get("goal_spec_required_field_f1", 0)
        >= thresholds["goal_spec_required_field_f1"],
        "compiled_task_success": candidate.get("compiled_task_success", 0) >= thresholds["compiled_task_success"],
        "citation_correctness": candidate.get("citation_correctness", 0) >= thresholds["citation_correctness"],
        "settlement_consistency": candidate.get("settlement_consistency", 0) >= thresholds["settlement_consistency"],
        "replanning_success_rate": (candidate.get("replanning_success_rate") or 0)
        >= thresholds["replanning_success_rate"],
        "memory_recall": (candidate.get("memory_recall") or 0) >= thresholds["memory_recall"],
        "state_contamination_rate": candidate.get("state_contamination_rate", 1)
        <= thresholds["state_contamination_rate"],
        "unsafe_tool_or_dsl_calls": candidate.get("unsafe_tool_or_dsl_calls", 1)
        == thresholds["unsafe_tool_or_dsl_calls"],
    }
    output = {
        "schema_version": "goal-spec-agent-aggregate-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_sha256": expected_benchmark_sha,
        "gate_sha256": expected_gate_sha,
        "systems": systems,
        "run_provenance": manifests,
        "promotion_candidate": candidate_name,
        "promotion_checks": checks,
        "promotion_pass": all(checks.values()),
        "deployment_decision": (
            "eligible_for_sg_loopback_gate_only" if all(checks.values()) else "retain_deterministic_sg_planner"
        ),
        "boundaries": [
            "Raw model output, deterministic compilation, and execution are distinct measurements.",
            "No model planner is enabled on the public network or Singapore service by this evaluation.",
            "Historical decision replay is not live trading, automatic bidding, or investment advice.",
            "BESS values remain historical operating proxies and are not investment returns.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"promotion_pass": output["promotion_pass"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
