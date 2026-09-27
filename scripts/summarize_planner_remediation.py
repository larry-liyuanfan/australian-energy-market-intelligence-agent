from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from energy_agent.llm_evaluation import aggregate_rows, ordered_subsequence

DIAGNOSTIC_SEMANTICS = (
    "Failed-row observations are co-occurring checks, not causal or exhaustive failure attribution. "
    "In particular, no_memory waives memory recall in task success. "
    "Unsafe proposal errors remain separately reported in unsafe_tool_or_dsl_calls. "
    "Relabelling these diagnostics does not change frozen scores, labels, thresholds or promotion."
)


def clarify_diagnostic_labels(summary: dict[str, Any]) -> dict[str, Any]:
    if (summary.get("schema_version") != "planner-remediation-public-v3"
            or summary.get("manifest", {}).get("scoring_version") != "v3"):
        raise ValueError("only the frozen v3 compact may be relabelled")
    clarified = copy.deepcopy(summary)
    for group in clarified["by_path_memory_and_track"].values():
        old, new = "failure_causes_nonexclusive", "failed_row_observations_nonexclusive"
        if old in group:
            if new in group:
                raise ValueError("ambiguous diagnostic keys")
            group[new] = group.pop(old)
    clarified["diagnostic_revision"] = "cooccurring-observations-r1"
    clarified["diagnostic_semantics"] = DIAGNOSTIC_SEMANTICS
    return clarified


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        track = "fault_injected" if row.get("fault") else "real_task"
        groups[f"{row['path']}|{row['memory_mode']}|{track}"].append(row)
    output = {}
    for key, items in groups.items():
        missing = Counter(tool for row in items for tool in row.get("missing_initial_tools", []))
        guarded_complete = sum(ordered_subsequence(
            row["expected_tools"], [call["name"] for call in row["guarded_calls"]]
        ) for row in items)
        observations: Counter[str] = Counter()
        for row in items:
            if row["task_success"]:
                continue
            for metric in ("tool_path_correct", "citation_correct", "settlement_consistent", "replan_success", "memory_recall"):
                if not row[metric]:
                    observations[metric] += 1
            if row["parameter_accuracy"] < 1:
                observations["executed_parameter_mismatch"] += 1
            if row["state_contaminated"]:
                observations["state_contamination"] += 1
        metrics = next(iter(aggregate_rows(items).values()))
        output[key] = {
            **metrics,
            "initial_model_complete_paths": sum(row["model_tool_path_correct"] for row in items)
            if items[0]["path"] != "deterministic" else None,
            "guarded_complete_paths": guarded_complete,
            "executed_complete_paths": sum(row["tool_path_correct"] for row in items),
            "initial_missing_step_counts": dict(missing) if items[0]["path"] != "deterministic" else {},
            "failed_row_observations_nonexclusive": dict(observations),
            "model_requests": sum(row["model_requests"] for row in items),
            "fallback_calls": sum(row["fallback_calls"] for row in items),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish compact v3 attribution and fault-separated aggregates")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path)
    source.add_argument("--relabel-summary", type=Path, help="Rename diagnostics only; never rescore or infer")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.relabel_summary:
        raw = args.relabel_summary.read_bytes()
        output = clarify_diagnostic_labels(json.loads(raw))
        output["source_summary_sha256"] = hashlib.sha256(raw).hexdigest()
    else:
        output = summarize_run(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps({
        "groups": len(output["by_path_memory_and_track"]),
        "promotion": output["all_tracks_metrics"]["promotion_pass"],
    }))


def summarize_run(run: Path) -> dict[str, Any]:
    predictions = run / "predictions.jsonl"
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    manifest = json.loads((run / "run_manifest.json").read_text())
    if manifest.get("scoring_version") != "v3":
        raise ValueError("v3 aggregate cannot relabel a legacy scorer")
    if manifest["predictions_sha256"] != hashlib.sha256(predictions.read_bytes()).hexdigest():
        raise ValueError("prediction digest differs from run manifest")
    metrics_raw = (run / "metrics.json").read_bytes()
    if manifest["metrics_sha256"] != hashlib.sha256(metrics_raw).hexdigest():
        raise ValueError("metrics digest differs from run manifest")
    metrics = json.loads(metrics_raw)
    return clarify_diagnostic_labels({
        "schema_version": "planner-remediation-public-v3", "manifest": manifest,
        "by_path_memory_and_track": summarize(rows), "all_tracks_metrics": metrics,
        "boundaries": [
            "New authored tasks with disjoint dates/prompts; not an independent human evaluation.",
            "First model proposal, guarded plan and executed path are different quantities.",
            "Legacy v1 metrics used a different parameter/raw-path scorer; no direct lift claim.",
            "Sampling intervals are descriptive; correlated turns and repeated seeds are not independent tasks.",
            "AUD 0 external-provider billing does not mean university compute is free.",
        ],
    })


if __name__ == "__main__":
    main()
