"""Four authored real-data CPU pilot turns; no model-quality score is inferred."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from audit_agent_application import audit_schedule
from langgraph.checkpoint.sqlite import SqliteSaver
from run_incremental_replay import inputs, registry

from energy_agent.evidence_scope import market_evidence_query
from energy_agent.incremental_graph import STAGES, build_incremental_graph, invoke_replay
from energy_agent.model_agent import AgentPath, MemoryMode, ModelDrivenAgent
from energy_agent.schemas import Region

QUESTIONS = [
    "Replay SA1 2025-12-15 BESS 1MW/2MWh with official text evidence.",
    "Change BESS to 1MW/3MWh; keep the region and date.",
    "Correction: use VIC1; keep the date and battery.",
    "Change efficiency to 80%; keep the region, date and battery size.",
]


def independent_checks(base: Any, calls: list[dict[str, Any]], results: dict[str, Any]) -> dict[str, bool]:
    from datetime import datetime
    call = next(c for c in calls if c["name"] == STAGES[3])
    args, output, forecast = call["arguments"], results[STAGES[3]]["data"], results[STAGES[2]]["data"]
    prices = [r.rrp for r in base.store.select(Region(args["region"]), datetime.fromisoformat(args["window"]["start"]),
                                             datetime.fromisoformat(args["window"]["end"]))]
    from energy_agent.schemas import BatterySpec
    battery = BatterySpec.model_validate(args.get("battery", {})).model_dump()
    independent = audit_schedule(battery, output, forecast["point"], prices,
                                 args["variable_degradation_cost_aud_per_mwh_discharged"])
    checks: dict[str, bool] = dict(independent["checks"])
    checks["planned_cashflow"] = abs(independent["planned_operating_proxy_aud"]-output["planned_margin_aud"]) < 1e-5
    checks["realized_cashflow"] = abs(independent["realised_operating_proxy_aud"]-output["realized_margin_aud"]) < 1e-5
    checks["signal_binding"] = hashlib.sha256(json.dumps(forecast["point"], separators=(",", ":"), allow_nan=False).encode()).hexdigest() == forecast["signal_sha256"] == output["signal_sha256"]
    # Source lookup is independent of production verify_result; this does not
    # claim causal entailment of a report or evaluate LLM wording.
    sources = {e.evidence_id: (e.url, e.sha256, e.snippet) for e in base.store.evidence}
    sources.update({d.chunk_id: (d.url, d.sha256, d.text) for d in base.evidence_index.documents})
    citations = [e for r in results.values() for e in r["evidence"]]
    checks["citation_location"] = bool(citations) and all(
        e["evidence_id"] in sources and (e["url"], e["sha256"]) == sources[e["evidence_id"]][:2]
        and e["snippet"] in sources[e["evidence_id"]][2] for e in citations)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs(parser)
    parser.add_argument("--private-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.private_run.mkdir(parents=True, exist_ok=False)
    if args.output.exists():
        raise ValueError("output exists")
    base = registry(args)
    records: list[dict[str, Any]] = []
    baseline: list[dict[str, Any]] = []
    old = ModelDrivenAgent(base, None, timeout_seconds=10)
    for index, question in enumerate(QUESTIONS):
        started = time.perf_counter()
        run = old.run_turn(question, conversation_id="pilot", path=AgentPath.deterministic,
                           memory_mode=MemoryMode.structured_state)
        results = {r.tool_name: r.model_dump(mode="json") for r in run.results}
        calls = [c.model_dump(mode="json") for c in run.tool_calls]
        check = independent_checks(base, calls, results)
        settlement = {k: results[STAGES[3]]["data"][k] for k in ("planned_margin_aud", "realized_margin_aud", "margin_basis")}
        baseline.append(settlement)
        records.append({"path": "legacy_deterministic_completion", "turn": index+1, "question": question,
                        "status": run.status, "checks": check, "actual_tool_calls": len(calls), "reuse": 0,
                        "settlement": settlement, "wall_seconds_including_validation": time.perf_counter()-started})
        (args.private_run / f"legacy-{index}.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
    for incremental in (False, True):
        name = "incremental_graph" if incremental else "full_rerun_graph"
        with SqliteSaver.from_conn_string(str(args.private_run / f"{name}.sqlite")) as saver:
            graph = build_incremental_graph(base, saver, incremental=incremental)
            for index, question in enumerate(QUESTIONS):
                started = time.perf_counter()
                result = invoke_replay(graph, "pilot", question=question)
                checks = independent_checks(base, result["plan"], result["results"])
                checks["same_settlement_as_legacy"] = all(
                    abs(result["settlement"][k]-baseline[index][k]) < 1e-5 for k in ("planned_margin_aud", "realized_margin_aud"))
                checks["model_not_called"] = result["planner_attempts"] == []
                records.append({"path": name, "turn": index+1, "question": question, "status": result["status"],
                                "checks": checks, "actual_tool_calls": len(result["calls"]),
                                "reuse": sum(p["action"] == "reuse" for p in result["provenance"]),
                                "settlement": result["settlement"], "provenance": result["provenance"],
                                "invalidations": result["invalidations"],
                                "wall_seconds_including_validation": time.perf_counter()-started})
                (args.private_run / f"{name}-{index}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    from datetime import date
    absent = market_evidence_query(["SA1"], date(2024, 12, 15), "decision_replay")
    # Real index, genuinely absent report period. A larger top_k cannot fabricate
    # evidence: record the negative outcome, not a mocked recovery improvement.
    searches = [base.execute(STAGES[1], {"query": absent, "top_k": 5}),
                base.execute(STAGES[1], {"query": absent.replace(" battery dispatch", ""), "top_k": 10})]
    report = {"schema_version": "incremental-replay-cpu-pilot-v1", "validation_date": "2026-09-29",
              "execution": "real_official_data_deterministic_cpu_no_model", "authored_turns": 4,
              "unique_market_days": 1, "regions": ["SA1", "VIC1"], "market_rows": len(base.store.rows),
              "model_calls": 0, "model_tokens": 0, "qwen_incremental_status": "not_run_pending_resource_release",
              "records": records, "real_missing_period_search_counts": [len(s.evidence) for s in searches],
              "retrieval_dense_enabled": getattr(base.evidence_index, "dense", None) is not None,
              "all_independent_checks_pass": all(all(r["checks"].values()) for r in records),
              "boundaries": ["4 authored turns on one day, not independent held-out model evaluation.",
                             "Timings include repeated full-input hash checks; 576 rows is not a million-row benchmark.",
                             "Tool-call reuse does not imply eliminated internal forecast or oracle computations.",
                             "Fault recovery model counterfactuals are separate explicitly labelled test doubles."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"all_checks": report["all_independent_checks_pass"], "rows": len(records),
                      "missing_period_counts": report["real_missing_period_search_counts"],
                      "calls": {p: [r["actual_tool_calls"] for r in records if r["path"] == p]
                                for p in sorted({r["path"] for r in records})}}))
    if not report["all_independent_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
