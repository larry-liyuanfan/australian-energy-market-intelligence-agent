"""Read private pilot traces; emit a compact repair report, never a promotion receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(folder: Path, freeze: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    rows = [json.loads(line) for line in (folder / "results/attempts.jsonl").read_text().splitlines()]
    summary = json.loads((folder / "results/summary.json").read_text())
    expected = [("SA1", 1, 2, .9), ("SA1", 1, 3, .9), ("VIC1", 1, 3, .9), ("VIC1", 1, 3, .8)]
    previous = {(r["path"], r["turn"]): r for r in parent["rows"]}
    compact: list[dict[str, Any]] = []
    for row in rows:
        run = row["run"]
        calls = run.get("calls", run.get("tool_calls", []))
        plan = run.get("plan", calls)
        args = next(c["arguments"] for c in plan if c["name"] == "optimize_battery_dispatch")
        spec = args.get("battery", {})
        region, power, energy, efficiency = expected[row["turn"] - 1]
        state = run.get("conversation", {}).get("constraints", run.get("resolved_constraints", {}))
        values = {k: v["value"] for k, v in state.items()}
        expected_sources = {"region": (region, 1 if row["turn"] < 3 else 3),
            "regions": ([region], 1 if row["turn"] < 3 else 3), "dates": (["2025-12-15"], 1),
            "battery_power_mw": (power, 1 if row["turn"] == 1 else 2),
            "battery_energy_mwh": (energy, 1 if row["turn"] == 1 else 2)}
        if row["turn"] == 4:
            expected_sources["round_trip_efficiency"] = (.8, 4)
        state_ok = all(key in state and state[key]["value"] == value
                       and state[key]["source_turn"] == turn and state[key]["source_type"] == "user"
                       for key, (value, turn) in expected_sources.items())
        state_ok = state_ok and values.get("round_trip_efficiency", .9) == efficiency
        dispatch_ok = (args["region"] == region
                       and datetime.fromisoformat(args["window"]["start"]) == datetime.fromisoformat("2025-12-15T00:00:00+10:00")
                       and datetime.fromisoformat(args["window"]["end"]) == datetime.fromisoformat("2025-12-16T00:00:00+10:00")
                       and args.get("settlement_mode") == "historical_replay"
                       and args.get("variable_degradation_cost_aud_per_mwh_discharged") == 50
                       and (spec.get("power_mw", 1), spec.get("energy_mwh", 2), spec.get("round_trip_efficiency", .9))
                       == (power, energy, efficiency))
        results = run["results"]
        results = results if isinstance(results, dict) else {r["tool_name"]: r for r in results}
        data = results["optimize_battery_dispatch"]["data"]
        before = previous[(row["path"], row["turn"])]
        settlement_ok = all(abs(data[k] - before[k]) < 1e-5 for k in ("planned_margin_aud", "realized_margin_aud"))
        counted = next(r for r in summary["rows"] if (r["path"], r["turn"], r["seed"])
                       == (row["path"], row["turn"], row["seed"]))
        requests = row["provider_requests"]
        prompt_tokens = sum(p.get("usage", {}).get("prompt_tokens", 0) for p in requests)
        completion_tokens = sum(p.get("usage", {}).get("completion_tokens", 0) for p in requests)
        exceptions = sum(bool(p.get("error")) for p in requests)
        unknown_usage = sum(p.get("usage_known") is not True for p in requests)
        accounting_ok = (prompt_tokens == counted["known_prompt_tokens"]
                         and completion_tokens == counted["known_completion_tokens"]
                         and exceptions == counted["provider_exceptions"] and unknown_usage == counted["unknown_usage"])
        attempts = run["planner_attempts"]
        graph = row["path"] != "legacy"
        answer = run.get("answer", "")
        source_ok = ("AUD 50/MWh discharged (default assumption)" in answer and "user-supplied" not in answer
                     and "degradation_cost_aud_mwh" not in state) if graph else None
        compact.append({"path": row["path"], "turn": row["turn"], "seed": row["seed"], "status": run["status"],
            "seeds_match_freeze": row["seed"] == 17 and all(p.get("seed") == 17 for p in requests),
            "raw_usage_matches_summary": accounting_ok,
            "state_matches_expected": state_ok, "dispatch_matches_expected": dispatch_ok,
            "independent_checks": row["independent_checks"], "settlement_matches_original": settlement_ok,
            "default_cost_attribution_correct": source_ok,
            "provider_requests": len(row["provider_requests"]),
            "provider_responses": sum(p.get("response_received") is True for p in row["provider_requests"]),
            "provider_exceptions": exceptions, "unknown_usage": unknown_usage,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "wall_seconds": row["wall_seconds"], "actual_calls": len(calls), "reuse": counted["reuse"],
            "executed_owners": dict(Counter(c.get("owner", "legacy_not_tagged") for c in calls)),
            "accepted_canonical_calls": len(attempts[0]["accepted"]) if graph else None,
            "recovery_attempts": sum(a.get("trigger") == "empty_result" for a in attempts),
            "planned_margin_aud": data["planned_margin_aud"], "realized_margin_aud": data["realized_margin_aud"]})
    thread_settings = dict(line.split("=", 1) for line in (folder / "runtime-threads.txt").read_text().splitlines())
    log = (folder / "model-server.log").read_text()
    observed_threads = [int(n) for n in re.findall(r"threadpool init, n_threads = (\d+)", log)]
    help_text = (folder / "runtime-help.txt").read_text()
    checks = {
        "exact_12_path_turns": len(compact) == 12 and len({(r["path"], r["turn"]) for r in compact}) == 12
        and {r["path"] for r in compact} == {"legacy", "full_rerun", "incremental"},
        "all_completed": all(r["status"] == "completed" for r in compact),
        "all_states_and_dispatch": all(r["state_matches_expected"] and r["dispatch_matches_expected"] for r in compact),
        "all_nine_independent_checks": all(len(r["independent_checks"]) == 9 and all(r["independent_checks"].values()) for r in compact),
        "settlements_match_original": all(r["settlement_matches_original"] for r in compact),
        "all_eight_graph_default_attributions": sum(r["default_cost_attribution_correct"] is True for r in compact) == 8,
        "provider_accounting_complete": all(r["provider_requests"] == r["provider_responses"] == 1
            and r["unknown_usage"] == r["provider_exceptions"] == 0
            and r["raw_usage_matches_summary"] and r["seeds_match_freeze"] for r in compact),
        "explicit_requested_thread_limits": thread_settings == {k: "6" for k in
            ("slurm_cpus_per_task", "llama_threads", "llama_threads_batch", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT")},
        "observed_runtime_threads_six": bool(observed_threads) and set(observed_threads) == {6},
        "both_thread_flags_supported": all(re.search(re.escape(flag) + r"\s", help_text) is not None
                                           for flag in ("--threads", "--threads-batch")),
    }
    groups = {}
    for path in ("legacy", "full_rerun", "incremental"):
        selected = [r for r in compact if r["path"] == path]
        groups[path] = {k: sum(r[k] for r in selected) for k in
            ("provider_requests", "provider_responses", "provider_exceptions", "unknown_usage", "prompt_tokens",
             "completion_tokens", "wall_seconds", "recovery_attempts")}
        groups[path].update(actual_calls=[r["actual_calls"] for r in selected], reuse=[r["reuse"] for r in selected],
                            accepted_canonical_calls=[r["accepted_canonical_calls"] for r in selected])
    return {"schema_version": "incremental-repair-confirmation-v1", "freeze": freeze,
            "checks": checks, "confirmation_checks_passed": all(checks.values()), "groups": groups, "rows": compact,
            "thread_settings": thread_settings, "observed_inference_threads": observed_threads,
            "observed_batch_thread_count": None,
            "private_file_hashes": {p.name: sha(p) for p in (folder / "results/attempts.jsonl",
                folder / "results/summary.json", folder / "model-server.log", folder / "runtime-help.txt",
                folder / "runtime-threads.txt")},
            "quality_pass_receipt_created": False, "full_run_authorized": False, "model_gain_demonstrated": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    if sha(args.parent_report) != freeze["parent_public_report_sha256"]:
        raise ValueError("Original report changed")
    report = audit(args.folder, freeze, json.loads(args.parent_report.read_text()))
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"checks": report["checks"], "groups": report["groups"]}, indent=2))


if __name__ == "__main__":
    main()
