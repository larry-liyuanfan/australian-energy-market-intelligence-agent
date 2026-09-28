"""Read-only archival business audit; no model, optimiser or frozen scorer imports.

All outputs are aggregates/short derived facts. Raw predictions, source documents
and five-minute schedules remain private. This is not a new model evaluation.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

NEM = timezone(timedelta(hours=10))
SCOPED = {
    "get_market_snapshot", "compare_region_period", "detect_price_events",
    "forecast_price_risk", "optimize_battery_dispatch", "explain_data_coverage",
}
BATTERY_DEFAULTS = {"power_mw": 1.0, "energy_mwh": 2.0, "round_trip_efficiency": .9,
                    "min_soc_fraction": .1, "max_soc_fraction": .9,
                    "initial_soc_fraction": .5, "terminal_soc_fraction": .5}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def same_instant(value: Any, expected: datetime) -> bool:
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.tzinfo is not None and parsed == expected
    except (TypeError, ValueError):
        return False


def signal_identity(forecast: dict[str, Any], dispatch: dict[str, Any], compact: dict[str, Any]) -> bool:
    sha = hashlib.sha256(json.dumps(forecast["point"], separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return (sha == forecast.get("signal_sha256") == dispatch.get("signal_sha256") == compact.get("signal_sha256")
            and forecast.get("forecast_snapshot_id") == dispatch.get("forecast_snapshot_id")
            == compact.get("forecast_snapshot_id")
            and forecast.get("training_cutoff") == dispatch.get("forecast_training_cutoff")
            == compact.get("training_cutoff"))


def request_errors(calls: list[dict[str, Any]], spec: dict[str, Any]) -> list[str]:
    """Exact latest-turn structured scope, not substring or task-success scoring."""
    expected = spec["expected"]
    start = datetime.fromisoformat(expected["date"]).replace(tzinfo=NEM)
    end = start + timedelta(days=1)
    errors: list[str] = []
    names = {call["name"] for call in calls}
    errors.extend(f"missing:{name}" for name in sorted(set(spec["expected_tools"]) & SCOPED - names))
    for call in calls:
        name, args = call["name"], call["arguments"]
        if name not in SCOPED | {"diagnose_price_event"}:
            continue
        if name == "compare_region_period":
            if sorted(args.get("regions", [])) != sorted(expected.get("regions", [expected.get("region")])):
                errors.append(f"{name}:regions")
        elif args.get("region") not in expected.get("regions", [expected.get("region")]):
            errors.append(f"{name}:region")
        if name == "get_market_snapshot":
            if not same_instant(args.get("at"), start):
                errors.append(f"{name}:date")
        elif name == "diagnose_price_event":
            try:
                valid_interval = start <= datetime.fromisoformat(args.get("interval", "")) < end
            except (ValueError, TypeError):
                valid_interval = False
            if not valid_interval:
                errors.append(f"{name}:date")
        elif name != "explain_data_coverage" and not (
            same_instant(args.get("window", {}).get("start"), start)
            and same_instant(args.get("window", {}).get("end"), end)
        ):
            errors.append(f"{name}:window")
        if name == "optimize_battery_dispatch":
            battery = BATTERY_DEFAULTS | args.get("battery", {})
            for source, field in (("battery_power_mw", "power_mw"), ("battery_energy_mwh", "energy_mwh"),
                                  ("round_trip_efficiency", "round_trip_efficiency")):
                if source in expected and battery.get(field) != expected[source]:
                    errors.append(f"{name}:{field}")
            if args.get("objective", "forecast") != "forecast" or args.get("settlement_mode", "plan") != "historical_replay":
                errors.append(f"{name}:plan_settlement_separation")
    return sorted(set(errors))


def audit_holdout(rows: list[dict[str, Any]], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = {(episode["case_id"], i): turn for episode in episodes
             for i, turn in enumerate(episode["turns"], 1)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples = []
    seen: set[tuple[str, str, int, str, int]] = set()
    for row in rows:
        identity = (row["path"], row["memory_mode"], row["seed"], row["case_id"], row["turn_index"])
        if identity in seen:
            raise ValueError("duplicate archival attempt")
        seen.add(identity)
        spec = tasks[(row["case_id"], row["turn_index"])]
        calls = [{"name": name, "arguments": args} for name, args in
                 zip(row["observed_tools"], row["executed_arguments"], strict=True)]
        initial = request_errors(row["initial_model_proposed_calls"], spec)
        executed = request_errors(calls, spec)
        grouped[f"{row['path']}|{row['memory_mode']}"].append({
            "row": row, "initial": initial, "executed": executed,
        })
        if (row["memory_mode"] == "structured_state" and row["seed"] in {0, 17}
                and (row["case_id"], row["turn_index"]) in {("v3-battery-horizon", 3), ("v3-regional-edit", 2)}):
            examples.append({"case_id": row["case_id"], "turn_index": row["turn_index"],
                             "path": row["path"], "seed": row["seed"], "expected": spec["expected"],
                             "initial_scope_errors": initial if row["path"] != "deterministic" else None,
                             "executed_scope_errors": executed,
                             "executed_scoped_arguments": [call for call in calls if call["name"] in SCOPED]})
    groups = {}
    for name, items in sorted(grouped.items()):
        source = [item["row"] for item in items]
        latency = sorted(row["end_to_end_latency_ms"] / 1000 for row in source)
        track_counts = {}
        for track in ("real_task", "fault_injected"):
            selected = [item for item in items if bool(item["row"].get("fault")) == (track == "fault_injected")]
            track_counts[track] = {"rows": len(selected), "executed_scope_matches":
                                   sum(not item["executed"] for item in selected)}
        groups[name] = {
            "rows": len(items), "distinct_turns": len({(row["case_id"], row["turn_index"]) for row in source}),
            "initial_complete_request_scope_matches": None if name.startswith("deterministic|")
            else sum(not item["initial"] for item in items),
            "executed_complete_request_scope_matches": sum(not item["executed"] for item in items),
            "executed_mismatch_observations": dict(Counter(e for item in items for e in item["executed"])),
            "by_track": track_counts,
            "latency_p50_seconds": latency[math.ceil(len(latency) * .5) - 1],
            "latency_p95_seconds": latency[math.ceil(len(latency) * .95) - 1],
            "mean_tool_attempts": sum(row["steps"] for row in source) / len(source),
            "model_requests": sum(row["model_requests"] for row in source),
            "replans": sum(row["replans"] for row in source),
            "tool_retries": sum(row["retries"] for row in source),
            "prompt_tokens": sum(row["prompt_tokens"] for row in source),
            "completion_tokens": sum(row["completion_tokens"] for row in source),
        }
    return {
        "rows": len(rows), "distinct_turns": len({(r["case_id"], r["turn_index"]) for r in rows}),
        "retained_payload_counts": {key: sum(key in row for row in rows)
                                    for key in ("answer", "results", "citations", "resolved_constraints")},
        "definition": "Independent latest-turn structured request scope, including missing scoped tools. Not frozen task success or answer accuracy. Coverage tool accepts region only; report-query prose is not scope-scored.",
        "answer_level_denominator": 0,
        "unavailable": "No final answer text, citation objects, source-turn state or full schedules retained in these 504 rows; cannot retrospectively verify 18-turn answer semantics, citation grounding or cash flow.",
        "by_path_memory": groups, "examples": examples,
    }


def audit_schedule(battery: dict[str, Any], dispatch: dict[str, Any], point: list[float],
                   actual: list[float], cost_rate: float) -> dict[str, Any]:
    c, d, soc = (dispatch[key] for key in ("charge_mw", "discharge_mw", "soc_mwh"))
    n, dt, tol = len(actual), 1 / 12, 1e-5
    if not n or not (len(c) == len(d) == len(point) == n and len(soc) == n + 1):
        raise ValueError("unaligned archived calculation arrays")
    if not all(math.isfinite(v) for seq in (c, d, soc, point, actual) for v in seq):
        raise ValueError("non-finite archived calculation")
    eta, capacity = math.sqrt(battery["round_trip_efficiency"]), battery["energy_mwh"]
    residual = max(abs(soc[i + 1] - soc[i] - (eta * c[i] - d[i] / eta) * dt) for i in range(n))
    checks = {
        "power_limits": all(-tol <= v <= battery["power_mw"] + tol for v in c + d),
        "charge_discharge_exclusive": all(min(x, y) <= tol for x, y in zip(c, d, strict=True)),
        "soc_limits": all(capacity * battery["min_soc_fraction"] - tol <= v <=
                          capacity * battery["max_soc_fraction"] + tol for v in soc),
        "initial_terminal_soc": abs(soc[0] - capacity * battery["initial_soc_fraction"]) < tol
        and abs(soc[-1] - capacity * battery["terminal_soc_fraction"]) < tol,
        "energy_balance": residual < tol,
    }
    planned = math.fsum((y - x) * price * dt for x, y, price in zip(c, d, point, strict=True))
    realised = math.fsum((y - x) * price * dt for x, y, price in zip(c, d, actual, strict=True))
    discharged = math.fsum(d) * dt
    sensitivity = discharged * cost_rate
    return {"intervals": n, "checks": checks, "max_energy_residual_mwh": residual,
            "planned_gross_aud": planned, "realised_gross_aud": realised,
            "sensitivity_cost_aud": sensitivity,
            "planned_operating_proxy_aud": planned - sensitivity,
            "realised_operating_proxy_aud": realised - sensitivity,
            "discharged_mwh": discharged}


def citation_checks(citation: dict[str, Any], source: dict[str, Any]) -> dict[str, bool]:
    url = urlsplit(citation["url"])
    checks = {"official_url_and_record": url.scheme == "https"
              and url.hostname in {"www.aemo.com.au", "aemo.com.au", "nemweb.com.au"}
              and citation["url"] == source["url"],
              "source_digest_matches": citation["sha256"] == source["sha256"]}
    if citation["modality"] == "text":
        excerpt = citation.get("short_text_excerpt")
        if citation.get("display_role") != "source_link_only":
            start, end = citation.get("excerpt_source_start"), citation.get("excerpt_source_end")
            checks["excerpt_locates_in_source_chunk"] = (
                isinstance(excerpt, str) and bool(excerpt) and isinstance(start, int) and isinstance(end, int)
                and 0 <= start < end <= len(source["text"]) and source["text"][start:end] == excerpt)
    else:
        preview = citation.get("source_cell_preview") or ""
        checks["figure_id_matches"] = citation["figure_id"] == source["figure_id"]
        checks["asset_digest_matches_manifest"] = citation["asset_sha256"] in source["image_sha256"]
        # Router may reorder rows and truncate its final row. Each displayed row
        # must still be an exact substring of the archived extracted source text.
        checks["preview_rows_locate_in_source"] = bool(preview) and all(
            line in source["text"] for line in preview.splitlines() if line
        )
    return checks


def audit_p1(full: dict[str, Any], compact: dict[str, Any], market: list[dict[str, str]],
             sources: list[dict[str, Any]]) -> dict[str, Any]:
    results = {r["tool_name"]: r["data"] for r in full["results"]}
    call = next(c for c in full["tool_calls"] if c["name"] == "optimize_battery_dispatch")
    args, forecast, dispatch = call["arguments"], results["forecast_price_risk"], results["optimize_battery_dispatch"]
    actual = [float(row["rrp"]) for row in market]
    battery = BATTERY_DEFAULTS | args.get("battery", {})
    calculation = audit_schedule(battery, dispatch, forecast["point"], actual,
                                 args["variable_degradation_cost_aud_per_mwh_discharged"])
    start = datetime.fromisoformat(args["window"]["start"])
    times = [datetime.strptime(row["interval"], "%Y/%m/%d %H:%M:%S").replace(tzinfo=NEM) for row in market]
    checks = calculation["checks"]
    checks["complete_unique_day"] = times == [start + timedelta(minutes=5 * i) for i in range(288)]
    spec = {"expected": {"region": "SA1", "date": "2025-12-15", "battery_power_mw": 1,
                         "battery_energy_mwh": 2, "round_trip_efficiency": .9},
            "expected_tools": ["optimize_battery_dispatch"]}
    checks["question_constraints_match_executed_plan"] = not request_errors([call], spec)
    checks["compact_question_scope_matches"] = (compact["region"] == "SA1" and compact["day"] == "2025-12-15"
                                               and compact["battery"] == battery)
    checks["trace_identity"] = full["trace_id"] == compact["trace_id"]
    checks["forecast_as_of"] = datetime.fromisoformat(forecast["training_cutoff"]) <= start
    checks["forecast_signal_snapshot_and_cutoff_identity"] = signal_identity(forecast, dispatch, compact["forecast"])
    checks["schedule_hash"] = hashlib.sha256(json.dumps(
        [dispatch[k] for k in ("charge_mw", "discharge_mw", "soc_mwh")],
        separators=(",", ":")).encode()).hexdigest() == compact["schedule_sha256"]
    for field in ("planned_gross_aud", "realised_gross_aud", "sensitivity_cost_aud",
                  "planned_operating_proxy_aud", "realised_operating_proxy_aud"):
        checks[f"compact_{field}"] = abs(calculation[field] - compact["settlement"][field]) < 1e-5
    checks["tool_plan_cashflow"] = abs(calculation["planned_operating_proxy_aud"] - dispatch["planned_margin_aud"]) < 1e-5
    checks["tool_realised_cashflow"] = abs(calculation["realised_operating_proxy_aud"] - dispatch["realized_margin_aud"]) < 1e-5
    checks["market_summary"] = (
        abs(math.fsum(actual) / len(actual) - compact["market_context"]["mean_rrp_aud_mwh"]) < 1e-5
        and min(actual) == compact["market_context"]["minimum_rrp_aud_mwh"]
        and sum(value < 0 for value in actual) == compact["market_context"]["negative_price_intervals"])
    index = {source["chunk_id"]: source for source in sources}
    citations = [{"evidence_id": c["evidence_id"], "display_role": c["display_role"],
                  "checks": citation_checks(c, index[c["evidence_id"]])} for c in compact["citations"]]
    checks["all_citations_locate_in_archived_source_records"] = bool(citations) and all(
        all(c["checks"].values()) for c in citations)
    return {
        "case_denominator": 1, "archived_natural_language_answer_count": 0,
        "checks": checks, "checks_passed": sum(checks.values()), "checks_total": len(checks),
        "calculation": {k: v for k, v in calculation.items() if k != "checks"},
        "battery_default_boundary": "Missing optional fields in retained calls are resolved with the existing BatterySpec defaults; this is effective executed input, not evidence the model explicitly proposed those values.",
        "citations": citations, "citation_count": len(citations),
        "citation_denominators": {
            "source_records": len(citations),
            "visible_text_excerpts": sum("excerpt_locates_in_source_chunk" in c["checks"] for c in citations),
            "figure_previews": sum("figure_id_matches" in c["checks"] for c in citations),
            "source_link_only": sum(c["display_role"] == "source_link_only" for c in citations),
        },
        "limitations": ["Source-record location is not semantic entailment or day-specific causal support.",
                        "PDF page numbers and original spreadsheet coordinates are absent; do not invent them.",
                        "No optimiser rerun; this validates the recorded schedule, not optimality or oracle regret."],
        "sourced_constraints": full["resolved_constraints"],
        "memory_boundary": "One-turn sourced state. Efficiency 90% matches the question/executed plan but has no separately retained source-turn constraint; multi-turn attribution is not proven by this case.",
        "model_accepted_stages": sum(w["origin"] == "model proposal accepted" for w in compact["workflow"]),
        "runtime_owned_stages": sum(w["origin"] != "model proposal accepted" for w in compact["workflow"]),
        "metrics": full["metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("predictions", "benchmark", "frozen_manifest", "p1_full_run", "p1_compact", "p1_manifest",
                 "market", "evidence", "figures", "workbook", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--graph-input-output", type=Path, help="Private two-region/day real-data subset; never publish")
    args = parser.parse_args()
    frozen, manifest, compact = (read_json(path) for path in (args.frozen_manifest, args.p1_manifest, args.p1_compact))
    hashes = {name: digest(getattr(args, name)) for name in vars(args) if name not in {"output", "graph_input_output"}}
    expected = {"predictions": frozen["sha256"]["private_predictions"], "benchmark": frozen["sha256"]["benchmark"],
                "market": manifest["input_sha256"]["data"], "evidence": manifest["input_sha256"]["text_evidence"],
                "figures": manifest["input_sha256"]["figure_records"],
                "p1_compact": manifest["artifacts_sha256"]["docs/demos/model-replay-20260927/recorded_run.json"]}
    mismatches = [name for name, sha in expected.items() if hashes[name] != sha]
    if mismatches:
        raise ValueError(f"archived input hash mismatch for labels: {mismatches}")
    with gzip.open(args.market, "rt", encoding="utf-8", newline="") as handle:
        subset = [row for row in csv.DictReader(handle) if row["region"] in {"SA1", "VIC1"}
                  and row["interval"].startswith(compact["day"].replace("-", "/"))]
    market = [row for row in subset if row["region"] == compact["region"]]
    market.sort(key=lambda row: row["interval"])
    figures = read_rows(args.figures)
    if any(f["sha256"] != hashes["workbook"] for f in figures):
        raise ValueError("workbook content differs from archived figure source hashes")
    p1 = audit_p1(read_json(args.p1_full_run), compact, market, [*read_rows(args.evidence), *figures])
    output: dict[str, Any] = {"schema_version": "agent-application-archival-audit-v1", "source_sha256": hashes,
              "holdout": audit_holdout(read_rows(args.predictions), read_rows(args.benchmark)), "p1": p1,
              "boundary": "Post-hoc independent archival business checks; no model calls, score rewriting, new labels, optimisation or promotion."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
    print(json.dumps({"p1_checks_passed": p1["checks_passed"], "p1_checks_total": p1["checks_total"],
                      "holdout_rows": output["holdout"]["rows"], "distinct_turns": output["holdout"]["distinct_turns"]}))
    if not all(p1["checks"].values()):
        raise SystemExit(1)
    if args.graph_input_output:
        private_subset = {"schema_version": "energy-private-market-subset-v1",
                          "data_track": "official_aemo_archival_subset", "source_sha256": hashes["market"],
                          "day": compact["day"], "regions": ["SA1", "VIC1"], "rows": subset,
                          "numeric_evidence": [c for c in read_json(args.p1_full_run)["citations"]
                                               if c["evidence_type"] == "numeric"]}
        with args.graph_input_output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(private_subset, handle, allow_nan=False)


if __name__ == "__main__":
    main()
