from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from .schemas import AgentQueryResponse

FaultKind = Literal["timeout", "empty_figure", "incomplete_window", "stale_snapshot", "citation_hash"]


@dataclass(frozen=True)
class DecisionTask:
    task_id: str
    category: Literal["event_diagnosis", "region_comparison", "figure_grounding", "decision_replay"]
    question: str
    expected_workflow: str
    expected_tools: tuple[str, ...]


@dataclass(frozen=True)
class FaultTask:
    task_id: str
    fault: FaultKind
    question: str


def decision_tasks() -> list[DecisionTask]:
    regions = ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")
    days = ("2025-09-15", "2025-12-15", "2026-03-15")
    tasks: list[DecisionTask] = []
    for region_index, region in enumerate(regions):
        partner = regions[(region_index + 1) % len(regions)]
        for day in days:
            suffix = f"{region}-{day}"
            tasks.extend(
                [
                    DecisionTask(
                        f"event-{suffix}",
                        "event_diagnosis",
                        f"Detect and explain price events in {region} on {day}",
                        "event_diagnosis",
                        ("get_market_snapshot", "detect_price_events", "search_official_evidence"),
                    ),
                    DecisionTask(
                        f"compare-{suffix}",
                        "region_comparison",
                        f"Compare {region} and {partner} on {day}",
                        "region_comparison",
                        ("compare_region_period", "search_official_evidence"),
                    ),
                    DecisionTask(
                        f"figure-{suffix}",
                        "figure_grounding",
                        f"Show the official chart for the {region} price trend around {day}",
                        "general_market_query",
                        ("get_market_snapshot", "search_official_evidence"),
                    ),
                    DecisionTask(
                        f"bess-{suffix}",
                        "decision_replay",
                        f"Replay a 1 MW / 2 MWh BESS dispatch for {region} on {day}",
                        "decision_replay",
                        (
                            "get_market_snapshot",
                            "forecast_price_risk",
                            "detect_price_events",
                            "search_official_evidence",
                            "optimize_battery_dispatch",
                        ),
                    ),
                ]
            )
    if len(tasks) != 60:
        raise AssertionError("decision suite must contain 60 real-window tasks")
    return tasks


def fault_tasks() -> list[FaultTask]:
    faults: tuple[FaultKind, ...] = (
        "timeout",
        "empty_figure",
        "incomplete_window",
        "stale_snapshot",
        "citation_hash",
    )
    regions = ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")
    tasks: list[FaultTask] = []
    for index in range(20):
        fault = faults[index % len(faults)]
        region = regions[index % len(regions)]
        day = "2035-01-01" if fault == "incomplete_window" else "2025-12-15"
        if fault == "empty_figure":
            question = f"Show the official {region} price chart around {day}"
        elif fault == "stale_snapshot":
            question = f"Replay BESS dispatch for {region} on {day}"
        else:
            question = f"Detect and explain price events in {region} on {day}"
        tasks.append(FaultTask(f"fault-{index + 1:03d}", fault, question))
    return tasks


def bess_golden_checks(response: AgentQueryResponse) -> dict[str, bool]:
    dispatch = response.dispatch
    charge = [float(value) for value in dispatch.get("charge_mw", [])]
    discharge = [float(value) for value in dispatch.get("discharge_mw", [])]
    soc = [float(value) for value in dispatch.get("soc_mwh", [])]
    aligned = bool(charge) and len(charge) == len(discharge) and len(soc) == len(charge) + 1
    no_simultaneous = aligned and all(min(c, d) <= 1e-7 for c, d in zip(charge, discharge, strict=True))
    bounds = bool(soc) and min(soc) >= 0.2 - 1e-7 and max(soc) <= 1.8 + 1e-7
    terminal = bool(soc) and math.isclose(soc[0], 1.0, abs_tol=1e-6) and math.isclose(soc[-1], 1.0, abs_tol=1e-6)
    settlement = (
        dispatch.get("settlement_mode") == "historical_replay"
        and dispatch.get("margin_basis") == "historical_actual_settlement_after_as_of_schedule"
        and dispatch.get("realized_margin_aud") is not None
    )
    return {
        "aligned": aligned,
        "no_simultaneous_charge_discharge": no_simultaneous,
        "soc_bounds": bounds,
        "terminal_soc": terminal,
        "settlement_basis": settlement,
    }


def question_day(question: str) -> str | None:
    match = re.search(r"20\d\d-\d\d-\d\d", question)
    return match.group(0) if match else None
