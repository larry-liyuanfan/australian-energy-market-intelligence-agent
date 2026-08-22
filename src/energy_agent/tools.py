from __future__ import annotations

import statistics
from datetime import timedelta
from typing import Any

from .battery import optimize_dispatch, threshold_dispatch
from .evidence import EvidenceIndex
from .forecast import seasonal_conformal
from .market import MarketStore, robust_events
from .schemas import TOOL_MODELS, Evidence, StrictModel, ToolResult


class ToolRegistry:
    def __init__(self, store: MarketStore, evidence_index: EvidenceIndex | None = None) -> None:
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
                multimodal_search = getattr(self.evidence_index, "search_multimodal", None)
                if args.retrieval_mode == "multimodal_fusion" and callable(multimodal_search):
                    hits = multimodal_search(args.query, args.top_k, args.preferred_modality)
                else:
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
                        modality=hit.get("modality", "text"),
                        source_page=hit.get("page_number"),
                        asset_id=hit.get("asset_id"),
                        asset_sha256=hit.get("asset_sha256"),
                        retrieval_scores={
                            key: float(value)
                            for key, value in hit.get("component_scores", {}).items()
                        },
                    )
                    for hit in hits
                ]
                return ToolResult(
                    tool_name=name,
                    data={
                        "retrieval": self.evidence_index.backend,
                        "hits": len(evidence),
                        "modalities": sorted({item.modality for item in evidence}),
                        "preferred_modality": args.preferred_modality,
                    },
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
            interval_count = int((args.window.end - args.window.start).total_seconds() // 300)
            if interval_count > 288:
                raise ValueError("dispatch window must not exceed 288 five-minute intervals")
            warnings: list[str] = []
            if args.objective == "perfect_foresight":
                rows = self.store.select(args.region, args.window.start, args.window.end)
                if len(rows) != interval_count:
                    raise ValueError("perfect-foresight window has incomplete market coverage")
                signal = [row.rrp for row in rows]
                dispatch = optimize_dispatch(
                    signal,
                    args.battery,
                    variable_degradation_cost_aud_per_mwh_discharged=(
                        args.variable_degradation_cost_aud_per_mwh_discharged
                    ),
                )
                warnings.append("Perfect foresight is an historical oracle, not a deployable policy.")
            else:
                history = self.store.before(args.region, args.window.start, limit=30 * 288)
                forecast = seasonal_conformal([row.rrp for row in history], interval_count)
                signal = forecast.point
                if args.objective == "forecast":
                    dispatch = optimize_dispatch(
                        signal,
                        args.battery,
                        variable_degradation_cost_aud_per_mwh_discharged=(
                            args.variable_degradation_cost_aud_per_mwh_discharged
                        ),
                    )
                else:
                    history_prices = [row.rrp for row in history]
                    dispatch = threshold_dispatch(
                        signal,
                        args.battery,
                        statistics.quantiles(history_prices, n=4)[0],
                        statistics.quantiles(history_prices, n=4)[2],
                        variable_degradation_cost_aud_per_mwh_discharged=(
                            args.variable_degradation_cost_aud_per_mwh_discharged
                        ),
                    )
                warnings.append(
                    "Schedule uses only observations before the requested window; reported margin is on the forecast signal, not realized settlement."
                )
            data = dispatch.__dict__ | {
                "objective": args.objective,
                "signal_intervals": len(signal),
                "economic_boundary": "historical spot-market operating proxy; variable degradation is a user-supplied sensitivity, not an asset-specific estimate; excludes CAPEX, fixed O&M, network fees, FCAS and investment returns"
            }
            return ToolResult(
                tool_name=name,
                data=data,
                evidence=self.store.evidence,
                warnings=warnings,
            )
        raise AssertionError(name)
