"""Read-only frozen-run accounting. No inference, rescoring, confidence interval or promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PATHS = ("legacy", "full_rerun", "incremental")
SEEDS = (17, 29, 43)
CHECKS = {"power_limits", "charge_discharge_exclusive", "soc_limits", "initial_terminal_soc", "energy_balance",
          "planned_cashflow", "realized_cashflow", "signal_binding", "citation_location"}


def number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def read_journal(path: Path) -> tuple[list[dict[str, Any]], bool]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("journal row must be an object")
            rows.append(row)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                return rows, True
            raise ValueError(f"corrupt nonterminal journal line {index + 1}") from None
    return rows, False


def summarize_row(row: dict[str, Any], episode: dict[str, Any], turn: int) -> dict[str, Any]:
    run, score = row.get("run", {}), row.get("score", {})
    status = run.get("status", "exception")
    completed = status == "completed"
    checks = row.get("independent_checks")
    numeric = all(v is True for v in checks.values()) if completed and isinstance(checks, dict) and set(checks) == CHECKS else None
    state_checked = completed and bool(episode["turns"][turn - 1].get("expected"))
    state = score.get("sourced_state_contract") if state_checked else None
    fault = episode.get("fault")
    phase = "natural" if not fault else "target" if turn > 1 else "checkpoint_pause" if fault == "checkpoint_correction" else "setup"
    intervention = row.get("intervention_score") or {}
    calls = run.get("calls", run.get("tool_calls", []))
    tool_trace_available = "calls" in run or "tool_calls" in run
    planner_trace_available = "planner_attempts" in run
    failed_names: set[str] = set()
    retried_successfully = False
    for call in calls:
        if call.get("status") in {"error", "timeout"}:
            failed_names.add(call["name"])
        elif call.get("status") == "ok" and call["name"] in failed_names:
            retried_successfully = True
    provenance = run.get("provenance", [])
    attempts = run.get("planner_attempts", [])
    graph = row["path"] != "legacy"
    boundary = row.get("provider_requests")
    requests = boundary if isinstance(boundary, list) else []
    usage = {"provider_requests": len(requests), "provider_responses": 0, "provider_exceptions": 0,
             "unknown_usage_requests": 0, "known_prompt_tokens": 0, "known_completion_tokens": 0,
             "known_provider_cost_aud": 0.0, "unknown_provider_cost_requests": 0}
    for request in requests:
        values = request.get("usage", {})
        known = request.get("usage_known") is True and all(
            isinstance(values.get(k), int) and not isinstance(values[k], bool) and values[k] >= 0
            for k in ("prompt_tokens", "completion_tokens"))
        usage["provider_responses"] += int(request.get("response_received") is True)
        usage["provider_exceptions"] += int(bool(request.get("error")))
        usage["unknown_usage_requests"] += int(not known)
        for key in ("prompt_tokens", "completion_tokens"):
            if known:
                usage["known_" + key] += values[key]
        cost = values.get("provider_cost_aud")
        if number(cost):
            usage["known_provider_cost_aud"] += cost
        else:
            usage["unknown_provider_cost_requests"] += 1
    unresolved = not isinstance(boundary, list) or (bool(row.get("usage_unknown_on_failure")) and not requests)
    recovery = [a for a in attempts if a.get("trigger") == "empty_result"]
    fault_consistent = row.get("controlled_fault") == (fault if turn > 1 else None)
    contract = score.get("task_contract_success")
    target = intervention.get("target_outcome_verified") if phase == "target" and fault_consistent else None
    result = {"status": status, "recorded_contract_match": contract,
        "tool_trace_available": tool_trace_available, "planner_trace_available": planner_trace_available,
        "completed_numerically_verified": completed and numeric is True,
        "completed_state_and_numerical": completed and numeric is True and state is True,
        "numerical_check": numeric, "state_check": state,
        "contract_matched_noncompletion": not completed and contract is True,
        "target_verified_safe_stop": not completed and target is True,
        "phase": phase, "planned_intervention": phase in {"target", "checkpoint_pause"},
        "recorded_fault_matches_episode": fault_consistent,
        "tool_fault_fired": row.get("fault_fired") is True,
        "checkpoint_pause_observed": bool(run.get("interrupts")) if phase == "checkpoint_pause" else None,
        "intervention_exercised": intervention.get("exercised") if phase == "target" else None,
        "target_outcome_verified": target,
        "recovery_to_verified_completion": target is True and completed and numeric is True and retried_successfully,
        "model_proposals": sum(len(a.get("proposed", a.get("calls", []))) for a in attempts),
        "initial_model_proposals": len(attempts[0].get("proposed", attempts[0].get("calls", []))) if attempts else 0,
        "runtime_accepted_proposals": sum(len(a.get("accepted", [])) for a in attempts) if graph else None,
        "initial_canonical_accepts": len(attempts[0].get("accepted", [])) if graph and attempts else 0 if graph else None,
        "runtime_rejected_proposals": sum(len(a.get("rejected", [])) for a in attempts) if graph else None,
        "provider_rejected_calls": sum(a.get("provider_rejected_calls", a.get("rejected_calls", 0)) for a in attempts),
        "guarded_initial_plan_calls": len(run["guarded_calls"]) if "guarded_calls" in run else None,
        "tool_call_records": len(calls), "executed_tool_attempts": sum(c.get("status") in {"ok", "error", "timeout"} for c in calls),
        "tool_statuses": dict(Counter(c.get("status", "unknown") for c in calls)),
        "model_owned_executions": sum(c.get("owner") == "model_accepted" for c in calls) if graph else None,
        "system_completion_executions": sum(c.get("owner") == "system_completion" for c in calls) if graph else None,
        "reuse": sum(p.get("action") == "reuse" for p in provenance) if graph else None,
        "empty_recovery_choices": dict(Counter(f"{a.get('decision_owner', 'unknown')}:{a.get('decision', 'unknown')}" for a in recovery)),
        "model_rewrite_executions": sum(c.get("owner") == "model_accepted" and any(
            a.get("decision") == "broaden_evidence" and any(c["name"] == item["name"] and c["arguments"] == item["arguments"]
            for item in a.get("accepted", [])) for a in recovery) for c in calls) if graph else None,
        **usage, "request_accounting_unresolved": unresolved,
        "token_totals_are_lower_bounds": unresolved or usage["unknown_usage_requests"] > 0,
        "wall_seconds": row.get("wall_seconds") if number(row.get("wall_seconds")) else None}
    if not tool_trace_available:
        for key in ("tool_call_records", "executed_tool_attempts", "model_owned_executions", "system_completion_executions",
                    "model_rewrite_executions", "recovery_to_verified_completion"):
            result[key] = None
    if not planner_trace_available:
        for key in ("model_proposals", "initial_model_proposals", "runtime_accepted_proposals", "initial_canonical_accepts",
                    "runtime_rejected_proposals", "provider_rejected_calls", "model_rewrite_executions"):
            result[key] = None
    if "provenance" not in run:
        result["reuse"] = None
    return result


def aggregate(slots: list[dict[str, Any]]) -> dict[str, Any]:
    present = [s["observed"] for s in slots if s["observed"] is not None]
    result: dict[str, Any] = {"expected_slots": len(slots), "observed_rows": len(present), "missing_rows": len(slots) - len(present)}
    result["planned_intervention_slots"] = sum(bool(s["episode_fault"]) and
        (s["turn"] > 1 or s["episode_fault"] == "checkpoint_correction") for s in slots)
    for key in ("recorded_contract_match", "completed_numerically_verified", "completed_state_and_numerical",
                "contract_matched_noncompletion", "target_verified_safe_stop", "planned_intervention", "tool_fault_fired",
                "checkpoint_pause_observed", "intervention_exercised", "target_outcome_verified", "recovery_to_verified_completion",
                "numerical_check", "state_check", "recorded_fault_matches_episode"):
        result[key] = {"true": sum(r[key] is True for r in present), "false": sum(r[key] is False for r in present),
                       "unmeasured_including_missing": sum(r[key] is None for r in present) + result["missing_rows"]}
    for key in ("model_proposals", "initial_model_proposals", "runtime_accepted_proposals", "initial_canonical_accepts",
                "runtime_rejected_proposals", "provider_rejected_calls",
                "guarded_initial_plan_calls", "tool_call_records", "executed_tool_attempts", "model_owned_executions", "system_completion_executions", "reuse",
                "model_rewrite_executions", "provider_requests", "provider_responses", "provider_exceptions", "unknown_usage_requests",
                "known_prompt_tokens", "known_completion_tokens", "known_provider_cost_aud", "unknown_provider_cost_requests", "wall_seconds"):
        values = [r[key] for r in present if r[key] is not None]
        result[key] = {"known_sum": sum(values) if values else None, "measured_rows": len(values)}
    result["request_accounting_unresolved_rows"] = sum(r["request_accounting_unresolved"] for r in present)
    result["unavailable_tool_trace_rows"] = sum(not r["tool_trace_available"] for r in present)
    result["unavailable_planner_trace_rows"] = sum(not r["planner_trace_available"] for r in present)
    result["token_totals_are_lower_bounds"] = bool(result["missing_rows"]) or any(r["token_totals_are_lower_bounds"] for r in present)
    for key in ("tool_statuses", "empty_recovery_choices"):
        counts: Counter[str] = Counter()
        for row in present:
            counts.update(row[key])
        result[key] = dict(counts)
    return result


def summarize(episodes: list[dict[str, Any]], rows: list[dict[str, Any]], *,
              paths: tuple[str, ...] = PATHS, seeds: tuple[int, ...] = SEEDS, partial_tail: bool = False) -> dict[str, Any]:
    expected = {(e["episode_id"], t, p, s): (e, t) for e in episodes
                for t in range(1, len(e["turns"]) + 1) for p in paths for s in seeds}
    if len({e["episode_id"] for e in episodes}) != len(episodes):
        raise ValueError("duplicate episode definition")
    observed = {}
    for row in rows:
        key = (row["episode_id"], row["turn"], row["path"], row["seed"])
        if key not in expected or key in observed:
            raise ValueError("unexpected or duplicate slot; do not discard requests")
        episode, turn = expected[key]
        observed[key] = summarize_row(row, episode, turn)
    slots = [{"episode_id": key[0], "turn": key[1], "path": key[2], "seed": key[3],
              "episode_fault": episode.get("fault"), "track": "controlled_fault_or_lifecycle" if episode.get("fault") else "natural_task",
              "observed": observed.get(key)} for key, (episode, _) in expected.items()]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pairs: dict[str, dict[str, Any]] = defaultdict(dict)
    for slot in slots:
        for group in (f"path_track:{slot['path']}:{slot['track']}",
                      f"path_fault:{slot['path']}:{slot['episode_fault'] or 'none'}",
                      f"path_episode:{slot['path']}:{slot['episode_id']}"):
            groups[group].append(slot)
        pairs[f"{slot['episode_id']}:{slot['turn']}:{slot['seed']}"][slot["path"]] = slot["observed"]
    totals = aggregate(slots)
    totals["token_totals_are_lower_bounds"] |= partial_tail
    return {"schema_version": "frozen-full-readonly-summary-v1", "execution": "postprocessing_only_no_inference",
            "episodes": len(episodes), "authored_turns": sum(len(e["turns"]) for e in episodes), "seeds": list(seeds),
            "partial_trailing_record": partial_tail, "unrecorded_tail_request_costs_unknown": bool(
                partial_tail or totals["missing_rows"] or totals["request_accounting_unresolved_rows"]),
            "totals": totals, "groups": {k: aggregate(v) for k, v in groups.items()}, "paired_slots": dict(pairs),
            "boundaries": ["Three model-calling paths; no no-model baseline in this run.",
                "Episode fault defines tracks, including preparation and checkpoint-pause rounds.",
                "Recorded contract match is not task completion or independent semantic correctness.",
                "Unchecked noncompletion state/numerical fields are null, never counted as verified.",
                "Canonical acceptance does not demonstrate model-originated plan changes; legacy owner attribution is unavailable.",
                "Only current row events count; nested history is excluded. Token lower bounds retain missing/failed requests.",
                "Paired seeds/turns are repeated observations of episodes, not independent user tasks; no new interval is estimated.",
                "Fixed path ordering/shared cache prohibit causal speed claims; wall sums are not GPU allocation cost.",
                "This summary does not change frozen scoring, infer new-evidence usefulness, or authorize promotion."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--scenario-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenario_sha = hashlib.sha256(args.scenarios.read_bytes()).hexdigest()
    if scenario_sha != args.scenario_sha256:
        raise ValueError("frozen scenario hash mismatch")
    episodes = [json.loads(line) for line in args.scenarios.read_text(encoding="utf-8").splitlines()]
    rows, partial = read_journal(args.attempts)
    report = summarize(episodes, rows, partial_tail=partial)
    report["input_sha256"] = {"scenarios": scenario_sha, "attempts": hashlib.sha256(args.attempts.read_bytes()).hexdigest()}
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"expected": report["totals"]["expected_slots"], "observed": len(rows), "partial_tail": partial}))


if __name__ == "__main__":
    main()
