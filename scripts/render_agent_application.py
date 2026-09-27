"""Render a cited deterministic business answer from the verified P1 recording."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def answer_blocks(compact: dict[str, Any]) -> list[dict[str, Any]]:
    market, battery, settlement = (compact[key] for key in ("market_context", "battery", "settlement"))
    numeric = "aemo-dispatch-12m"
    return [
        {"kind": "market_observation", "evidence_ids": [numeric], "text": (
            f"{compact['region']} on {compact['day']} recorded {market['negative_price_intervals']}/288 negative-price "
            f"intervals. The minimum was AUD {market['minimum_rrp_aud_mwh']:.2f}/MWh; mean price was "
            f"AUD {market['mean_rrp_aud_mwh']:.2f}/MWh. These are observations, not a causal explanation.")},
        {"kind": "retrospective_context", "evidence_ids": ["aemo-qed-q4-2025-figure-012"], "text": (
            "AEMO's Q4 2025 workbook Figure 12 supplies quarterly regional price context. It was published after "
            "this market day and does not establish why the individual event happened. The retrieved short "
            "text excerpt is insufficient for a day-specific causal answer.")},
        {"kind": "forecast_plan_and_historical_settlement", "evidence_ids": [numeric, "calculation:schedule"], "text": (
            f"For {battery['power_mw']:g} MW / {battery['energy_mwh']:g} MWh at "
            f"{battery['round_trip_efficiency'] * 100:g}% round-trip efficiency, the as-of seasonal forecast "
            f"produced a fixed constrained schedule. Planned gross margin was AUD {settlement['planned_gross_aud']:.2f}; "
            f"settling that schedule on actual historical prices gave AUD {settlement['realised_gross_aud']:.2f}. "
            f"The separately assumed AUD {settlement['sensitivity_aud_per_mwh_discharged']:g}/MWh discharged cost "
            f"subtracts AUD {settlement['sensitivity_cost_aud']:.2f}, not an observed asset cost.")},
        {"kind": "decision_boundary", "evidence_ids": ["calculation:schedule"], "text": (
            "The schedule respects power limits, 10–90% SoC and 50% initial/terminal SoC. Historical operating "
            "proxies exclude CAPEX, fixed O&M, network fees, FCAS and investment return. This is analysis, not "
            "live trading, automated bidding or a recommendation to invest.")},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compact, audit, graph = [json.loads(p.read_text(encoding="utf-8")) for p in (args.compact, args.audit, args.graph)]
    if not all(audit["p1"]["checks"].values()) or not all(graph["checks"].values()):
        raise ValueError("refusing to present failed checks as a completed application")
    payload = {"recording": compact, "audit": audit, "graph": graph, "answer_blocks": answer_blocks(compact),
               "answer_origin": "New deterministic presentation of archived verified outputs; not an archived or live LLM-written answer."}
    embedded = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
    template = Path(__file__).parents[1] / "src/energy_agent/application_demo.html"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(template.read_text(encoding="utf-8").replace("__APPLICATION_DATA__", embedded))


if __name__ == "__main__":
    main()
