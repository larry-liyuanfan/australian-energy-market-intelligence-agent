from __future__ import annotations

import statistics
from datetime import timedelta
from typing import Any

from .battery import optimize_dispatch
from .evidence import HybridEvidenceIndex
from .forecast import seasonal_conformal
from .market import MarketStore, robust_events
from .schemas import TOOL_MODELS, Evidence, StrictModel, ToolResult


class ToolRegistry:
    def __init__(self, store: MarketStore, evidence_index: HybridEvidenceIndex | None = None) -> None:
        self.store = store
        self.evidence_index = evidence_index

    def validate(self, name: str, arguments: dict[str, object]) -> StrictModel:
        if name not in TOOL_MODELS:
            raise ValueError("unknown tool")
        return TOOL_MODELS[name].model_validate(arguments)

    def specs(self) -> list[dict[str, object]]:
        return [{"name": name, "parameters": model.model_json_schema()} for name, model in TOOL_MODELS.items()]

    def execute(self, name: str, arguments: dict[str, object]) -> ToolResult:
        args: Any = self.validate(name, arguments)
        if name == "explain_data_coverage":
            return ToolResult(tool_name=name, data=self.store.coverage(), evidence=self.store.evidence)
        if name == "get_market_snapshot":
            row = self.store.closest(args.region, args.at)
            return ToolResult(tool_name=name, data={} if row is None else row.__dict__, evidence=self.store.evidence)
        if name == "compare_region_period":
            comparisons: dict[str, dict[str, float | int | None]] = {}
            for region in args.regions:
                rows = self.store.select(region, args.window.start, args.window.end)
                comparisons[region.value] = {
                    "count": len(rows),
                    "mean_rrp": statistics.fmean(row.rrp for row in rows) if rows else None,
                    "max_rrp": max((row.rrp for row in rows), default=None),
                }
            return ToolResult(tool_name=name, data=comparisons, evidence=self.store.evidence)
        if name == "detect_price_events":
            rows = self.store.select(args.region, args.window.start, args.window.end)
            return ToolResult(
                tool_name=name,
                data={"events": robust_events(rows, args.threshold_aud_mwh, args.robust_z_threshold)},
                evidence=self.store.evidence,
            )
        if name == "diagnose_price_event":
            half = timedelta(minutes=5 * args.context_intervals)
            rows = self.store.select(args.region, args.interval - half, args.interval + half)
            target = self.store.closest(args.region, args.interval)
            data = {
                "target": target.__dict__ if target else None,
                "context_count": len(rows),
                "diagnostic_only": "Associations do not establish causal attribution; use official evidence.",
            }
            return ToolResult(tool_name=name, data=data, evidence=self.store.evidence)
        if name == "search_official_evidence":
            if self.evidence_index is not None:
                hits = self.evidence_index.search(args.query, args.top_k)
                evidence = [
                    Evidence(
                        evidence_id=hit["chunk_id"],
                        title=hit["title"],
                        url=hit["url"],
                        published_at=hit["published_at"],
                        retrieved_at=hit["retrieved_at"],
                        sha256=hit["sha256"],
                        snippet=hit["text"][:500],
                        evidence_type="explanatory",
                        score=hit["score"],
                    )
                    for hit in hits
                ]
                return ToolResult(
                    tool_name=name,
                    data={"retrieval": "BM25+dense+RRF+rerank", "hits": len(evidence)},
                    evidence=evidence,
                )
            query = args.query.lower()
            ranked = sorted(
                self.store.evidence, key=lambda ev: (query in (ev.title + ev.snippet).lower(), ev.score), reverse=True
            )[: args.top_k]
            return ToolResult(
                tool_name=name,
                data={"retrieval": "BM25+dense adapters use RRF; local preflight uses deterministic lexical fallback"},
                evidence=ranked,
            )
        if name == "forecast_price_risk":
            rows = self.store.select(args.region, args.window.start, args.window.end)
            forecast = seasonal_conformal([row.rrp for row in rows], args.horizon_intervals, args.alpha)
            return ToolResult(
                tool_name=name,
                data=forecast.__dict__,
                evidence=self.store.evidence,
                warnings=["Risk interval is a historical conformal estimate, not a guarantee."],
            )
        if name == "optimize_battery_dispatch":
            rows = self.store.select(args.region, args.window.start, args.window.end)
            dispatch = optimize_dispatch([row.rrp for row in rows], args.battery)
            data = dispatch.__dict__ | {
                "economic_boundary": "historical spot-market gross-margin proxy; excludes CAPEX, degradation, network fees, FCAS and investment returns"
            }
            return ToolResult(tool_name=name, data=data, evidence=self.store.evidence)
        raise AssertionError(name)
