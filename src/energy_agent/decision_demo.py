"""Compact, auditable recorded-run export for the interview demonstrator."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from typing import Any

from .evidence_scope import EvidenceScope, official_evidence_url, scope_support
from .market import MarketStore
from .model_agent import ModelAgentRun
from .schemas import TOOL_MODELS, Evidence, OptimizeBatteryDispatchInput


def evidence_publication_role(evidence: Evidence, decision_as_of: datetime) -> str:
    if evidence.published_at is None:
        return "publication_unknown_retrospective_only"
    if evidence.published_at.tzinfo is None:
        return "publication_time_or_timezone_uncertain_retrospective_only"
    return "available_by_decision_time" if evidence.published_at <= decision_as_of else "published_later_retrospective_only"


def build_demo_bundle(
    run: ModelAgentRun, store: MarketStore, question: str, *, allow_fixture: bool = False,
) -> dict[str, Any]:
    if "fixture" in store.data_version.lower() and not allow_fixture:
        raise ValueError("fixture cannot be exported as a real recorded demonstration")
    results = {result.tool_name: result for result in run.results}
    forecast = results["forecast_price_risk"].data
    dispatch = results["optimize_battery_dispatch"].data
    dispatch_call = next(call for call in run.tool_calls if call.name == "optimize_battery_dispatch" and call.status == "ok")
    request = OptimizeBatteryDispatchInput.model_validate(dispatch_call.arguments)
    rows = store.select(request.region, request.window.start, request.window.end)
    n = len(rows)
    charge = dispatch["charge_mw"]
    discharge = dispatch["discharge_mw"]
    soc = dispatch["soc_mwh"]
    point = forecast["point"]
    if n != 288 or any(len(series) != n for series in (charge, discharge, point)) or len(soc) != n + 1:
        raise ValueError("demonstrator requires one complete day and aligned forecast/schedule/settlement")
    dt = 5 / 60
    cost_rate = request.variable_degradation_cost_aud_per_mwh_discharged
    sensitivity_cost = math.fsum(discharge) * dt * cost_rate
    planned_gross = math.fsum((d - c) * p * dt for c, d, p in zip(charge, discharge, point, strict=True))
    realised_gross = math.fsum((d - c) * row.rrp * dt for c, d, row in zip(charge, discharge, rows, strict=True))
    planned = planned_gross - sensitivity_cost
    realised = realised_gross - sensitivity_cost
    battery = request.battery
    eta = math.sqrt(battery.round_trip_efficiency)
    tolerance = 1e-5
    physical = (
        abs(soc[0] - battery.initial_soc_fraction * battery.energy_mwh) < tolerance
        and abs(soc[-1] - battery.terminal_soc_fraction * battery.energy_mwh) < tolerance
        and all(battery.min_soc_fraction * battery.energy_mwh - tolerance <= value
                <= battery.max_soc_fraction * battery.energy_mwh + tolerance for value in soc)
        and all(-tolerance <= value <= battery.power_mw + tolerance for value in charge + discharge)
        and all(c * d < tolerance for c, d in zip(charge, discharge, strict=True))
        and all(abs(soc[i + 1] - soc[i] - dt * (eta * charge[i] - discharge[i] / eta)) < tolerance for i in range(n))
    )
    cutoff = datetime.fromisoformat(forecast["training_cutoff"])
    checks = {
        "complete_day": True,
        "as_of_training_cutoff": cutoff <= request.window.start,
        "forecast_signal_matches_dispatch": bool(forecast.get("signal_sha256"))
        and forecast.get("signal_sha256") == dispatch.get("signal_sha256"),
        "same_snapshot": forecast.get("forecast_snapshot_id") == dispatch.get("forecast_snapshot_id"),
        "battery_constraints": physical,
        "planned_margin_independently_recomputed": abs(planned - dispatch["planned_margin_aud"]) < tolerance,
        "realised_margin_independently_recomputed": abs(realised - dispatch["realized_margin_aud"]) < tolerance,
        "forecast_objective_only": request.objective == "forecast",
    }
    if not all(checks.values()):
        raise ValueError(f"recorded-demo calculation gate failed: {[name for name, passed in checks.items() if not passed]}")
    citations: list[dict[str, Any]] = []
    quoted_sources: set[str] = set()
    scope = EvidenceScope((request.region.value,), request.window.start.year, (request.window.start.month - 1) // 3 + 1)
    for evidence in run.citations:
        if evidence.evidence_type != "explanatory":
            continue
        excerpt = " ".join(evidence.snippet.split()[:18]) if (
            evidence.modality == "text" and evidence.url not in quoted_sources
        ) else None
        if excerpt:
            quoted_sources.add(evidence.url)
        support = scope_support({
            "title": evidence.title, "url": evidence.url, "text": evidence.snippet,
            "source_cell_preview": evidence.source_cell_preview or "",
        }, scope)
        citations.append({
            "evidence_id": evidence.evidence_id, "title": evidence.title, "url": evidence.url,
            "published_at": evidence.published_at.isoformat() if evidence.published_at else None,
            "publication_role": evidence_publication_role(evidence, request.window.start),
            "modality": evidence.modality, "page": evidence.source_page, "figure_id": evidence.figure_id,
            "source_cell_preview": evidence.source_cell_preview,
            "short_text_excerpt": excerpt,
            "sha256": evidence.sha256, "asset_sha256": evidence.asset_sha256,
            "source_text_start": evidence.source_text_start,
            "scope_checks": support,
            "context_scope": "quarterly_electricity_context" if all(support.values()) else "unsupported_context",
        })
    checks["official_text_and_figure_present"] = (
        any(item["modality"] == "text" for item in citations)
        and any(item["figure_id"] and item["source_cell_preview"] for item in citations)
    )
    if not checks["official_text_and_figure_present"]:
        raise ValueError("recorded-demo evidence gate requires both official text and workbook source cells")
    checks["all_citations_topic_region_and_report_period_supported"] = all(
        all(item["scope_checks"].values()) for item in citations
    )
    checks["official_citation_origins"] = all(official_evidence_url(item["url"]) for item in citations)
    if not all(checks.values()):
        raise ValueError("recorded-demo evidence scope gate failed: topic/region/report period or official origin mismatch")

    def signature(name: str, args: dict[str, Any]) -> str:
        return name + TOOL_MODELS[name].model_validate(args).model_dump_json()

    initial = {signature(call.name, call.arguments) for call in run.initial_model_proposed_calls}
    workflow = []
    for call in run.tool_calls:
        origin = (
            "model proposal accepted" if signature(call.name, call.arguments) in initial
            else "deterministic event dependency" if call.name == "diagnose_price_event"
            else "bounded recovery" if call.attempt > 1 else "deterministic guard / completion"
        )
        workflow.append({"tool": call.name, "status": call.status, "origin": origin, "duration_ms": call.duration_ms})
    sample = list(range(0, n, 6))
    peak = max(rows, key=lambda row: row.rrp)
    events = results.get("detect_price_events")
    event_records = events.data.get("events", []) if events else []
    diagnosis = results.get("diagnose_price_event")
    return {
        "schema_version": "energy-interview-recorded-run-v1",
        "recorded_at": datetime.now(UTC).isoformat(), "question": question,
        "mode": "recorded real-model evaluation" if run.metrics.provider != "deterministic" else "recorded deterministic baseline",
        "region": request.region.value, "day": request.window.start.date().isoformat(),
        "trace_id": run.trace_id, "status": run.status, "workflow": workflow,
        "market_context": {
            "mean_rrp_aud_mwh": math.fsum(row.rrp for row in rows) / n,
            "minimum_rrp_aud_mwh": min(row.rrp for row in rows),
            "peak_rrp_aud_mwh": peak.rrp, "peak_interval": peak.interval.isoformat(),
            "negative_price_intervals": sum(row.rrp < 0 for row in rows),
            "detected_event_count": len(event_records),
            "diagnosis": diagnosis.model_dump(mode="json")["data"] if diagnosis else None,
            "interpretation_boundary": "Market changes are associations. Quarterly report context does not establish the cause of this day's event.",
        },
        "model": {"provider": run.metrics.provider, "name": run.metrics.model,
                  "initial_proposals": [call.model_dump(mode="json") for call in run.initial_model_proposed_calls],
                  "metrics": run.metrics.model_dump(mode="json")},
        "forecast": {key: value for key, value in forecast.items() if key not in {"point", "lower", "upper"}},
        "battery": battery.model_dump(mode="json"),
        "settlement": {
            "planned_gross_aud": planned_gross, "realised_gross_aud": realised_gross,
            "sensitivity_cost_aud": sensitivity_cost, "sensitivity_aud_per_mwh_discharged": cost_rate,
            "planned_operating_proxy_aud": planned, "realised_operating_proxy_aud": realised,
            "oracle_regret_aud": dispatch["oracle_regret_aud"],
            "equivalent_full_cycles": dispatch["equivalent_full_cycles"],
            "boundary": dispatch["economic_boundary"],
        },
        "series": {"label": "Every sixth five-minute point (30-minute display sampling); all 288 used in checks",
                   "nem_time": [rows[i].interval.isoformat() for i in sample],
                   "actual": [rows[i].rrp for i in sample], "forecast": [point[i] for i in sample],
                   "lower": [forecast["lower"][i] for i in sample], "upper": [forecast["upper"][i] for i in sample],
                   "soc": [soc[i] for i in [*sample, n]]},
        "citations": citations, "verification": checks,
        "data_version": store.data_version,
        "data_sha256": store.evidence[0].sha256 if store.evidence else None,
        "schedule_sha256": hashlib.sha256(json.dumps([charge, discharge, soc], separators=(",", ":")).encode()).hexdigest(),
        "limitations": [
            "Retrospective reports explain history; they do not enter the forecast or optimisation signal.",
            "The forecast was reconstructed from earlier market data, not proven to have been published on that historical day.",
            "Displayed curves are downsampled; calculations use every five-minute interval.",
            "Historical operating proxy, not investment return, live trading, automatic bidding or a public SLA.",
            "Workbook previews preserve values but do not claim original spreadsheet cell coordinates.",
            "Evidence scope checks reject obvious mismatches; quarterly context is not day-specific causal evidence or an entailment score.",
        ],
    }
