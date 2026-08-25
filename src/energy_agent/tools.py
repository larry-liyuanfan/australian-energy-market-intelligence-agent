from __future__ import annotations

import statistics
from datetime import timedelta
from typing import Any

from .battery import optimize_dispatch, threshold_dispatch
from .evidence import EvidenceIndex
from .forecast import seasonal_conformal
from .market import MarketStore, robust_events
from .schemas import TOOL_MODELS, Evidence, StrictModel, ToolResult
from .snapshots import ForecastSnapshotStore


class ToolRegistry:
    def __init__(
        self,
        store: MarketStore,
        evidence_index: EvidenceIndex | None = None,
        forecast_snapshots: ForecastSnapshotStore | None = None,
    ) -> None:
        self.store = store
        self.evidence_index = evidence_index
        self.forecast_snapshots = forecast_snapshots or ForecastSnapshotStore()

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
            if row is not None and abs((row.interval - args.at).total_seconds()) > 300:
                row = None
            if row is None:
                return ToolResult(
                    tool_name=name,
                    data={},
                    warnings=["No market interval exists within five minutes of the requested timestamp."],
                )
            return ToolResult(tool_name=name, data=row.__dict__, evidence=self.store.evidence)
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
            before = [row for row in rows if row.interval < args.interval]
            after = [row for row in rows if row.interval > args.interval]

            def mean(values: list[float | None]) -> float | None:
                observed = [value for value in values if value is not None]
                return statistics.fmean(observed) if observed else None

            before_summary = {
                "rrp": mean([row.rrp for row in before]),
                "demand_mw": mean([row.demand_mw for row in before]),
                "available_mw": mean([row.available_mw for row in before]),
                "net_interchange_mw": mean([row.net_interchange_mw for row in before]),
            }
            after_summary = {
                "rrp": mean([row.rrp for row in after]),
                "demand_mw": mean([row.demand_mw for row in after]),
                "available_mw": mean([row.available_mw for row in after]),
                "net_interchange_mw": mean([row.net_interchange_mw for row in after]),
            }
            event_summary = (
                {
                    "rrp": target.rrp,
                    "demand_mw": target.demand_mw,
                    "available_mw": target.available_mw,
                    "net_interchange_mw": target.net_interchange_mw,
                    "intervention": target.intervention,
                }
                if target
                else {}
            )
            def delta(key: str) -> float | None:
                event_value = event_summary.get(key)
                before_value = before_summary.get(key)
                if not isinstance(event_value, (int, float)) or not isinstance(before_value, (int, float)):
                    return None
                return float(event_value) - float(before_value)

            deltas = {
                key: delta(key)
                for key in ("rrp", "demand_mw", "available_mw", "net_interchange_mw")
            }
            data = {
                "target": target.__dict__ if target else None,
                "context_count": len(rows),
                "before_mean": before_summary,
                "event": event_summary,
                "after_mean": after_summary,
                "event_minus_before": deltas,
                "intervention_observed_in_context": any(row.intervention for row in rows),
                "diagnostic_only": "Observed associations do not establish causal attribution; official evidence is required.",
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
                if not evidence:
                    return ToolResult(
                        tool_name=name,
                        warnings=["Evidence route returned no results; bounded alternate-modality recovery is required."],
                    )
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
            snapshot = self.forecast_snapshots.get(args.region, args.window.start, args.window.end)
            forecast_data: dict[str, Any]
            used_snapshot = snapshot is not None and len(snapshot.point) >= args.horizon_intervals
            if used_snapshot and snapshot is not None:
                forecast_data = {
                    "point": snapshot.point[: args.horizon_intervals],
                    "lower": snapshot.lower[: args.horizon_intervals],
                    "upper": snapshot.upper[: args.horizon_intervals],
                    "method": snapshot.model_name,
                    "forecast_source": "offline_snapshot",
                    "forecast_snapshot_id": snapshot.snapshot_id,
                    "training_cutoff": snapshot.training_cutoff.isoformat(),
                    "data_sha256": snapshot.data_sha256,
                    "model_sha256": snapshot.model_sha256,
                }
            else:
                history = self.store.before(args.region, args.window.start, limit=30 * 288)
                forecast = seasonal_conformal(
                    [row.rrp for row in history], args.horizon_intervals, args.alpha
                )
                forecast_data = forecast.__dict__ | {
                    "forecast_source": "seasonal_fallback",
                    "training_cutoff": args.window.start.isoformat(),
                    "history_intervals": len(history),
                }
            return ToolResult(
                tool_name=name,
                data=forecast_data,
                evidence=self.store.evidence,
                warnings=[
                    "Risk interval is an as-of historical estimate, not a guarantee.",
                    *([] if used_snapshot else ["No published forecast snapshot matched; used the seasonal fallback."]),
                ],
            )
        if name == "optimize_battery_dispatch":
            interval_count = int((args.window.end - args.window.start).total_seconds() // 300)
            if interval_count > 288:
                raise ValueError("dispatch window must not exceed 288 five-minute intervals")
            warnings: list[str] = []
            forecast_source = "historical_oracle"
            forecast_metadata: dict[str, Any] = {}
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
                snapshot = self.forecast_snapshots.get(args.region, args.window.start, args.window.end)
                if snapshot is not None and len(snapshot.point) >= interval_count:
                    signal = snapshot.point[:interval_count]
                    forecast_source = "offline_snapshot"
                    forecast_metadata = {
                        "forecast_snapshot_id": snapshot.snapshot_id,
                        "forecast_data_sha256": snapshot.data_sha256,
                        "forecast_model_sha256": snapshot.model_sha256,
                        "forecast_training_cutoff": snapshot.training_cutoff.isoformat(),
                    }
                else:
                    history = self.store.before(args.region, args.window.start, limit=30 * 288)
                    forecast = seasonal_conformal([row.rrp for row in history], interval_count)
                    signal = forecast.point
                    forecast_source = "seasonal_fallback"
                    forecast_metadata = {
                        "forecast_training_cutoff": args.window.start.isoformat(),
                        "forecast_history_intervals": len(history),
                    }
                if args.objective == "forecast":
                    dispatch = optimize_dispatch(
                        signal,
                        args.battery,
                        variable_degradation_cost_aud_per_mwh_discharged=(
                            args.variable_degradation_cost_aud_per_mwh_discharged
                        ),
                    )
                else:
                    history = self.store.before(args.region, args.window.start, limit=30 * 288)
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
                if forecast_source == "seasonal_fallback":
                    warnings.append("No published forecast snapshot matched; dispatch used the seasonal fallback.")
            planned_margin = dispatch.net_operating_margin_proxy_aud
            realized_margin: float | None = None
            oracle_regret: float | None = None
            actual_rows = self.store.select(args.region, args.window.start, args.window.end)
            if args.settlement_mode == "historical_replay":
                if len(actual_rows) != interval_count:
                    raise ValueError("historical replay requires complete realised market coverage")
                actual_prices = [row.rrp for row in actual_rows]
                interval_hours = 5 / 60
                realized_gross = sum(
                    (discharge - charge) * price * interval_hours
                    for charge, discharge, price in zip(
                        dispatch.charge_mw, dispatch.discharge_mw, actual_prices, strict=True
                    )
                )
                realized_margin = realized_gross - dispatch.variable_degradation_cost_proxy_aud
                oracle = optimize_dispatch(
                    actual_prices,
                    args.battery,
                    variable_degradation_cost_aud_per_mwh_discharged=(
                        args.variable_degradation_cost_aud_per_mwh_discharged
                    ),
                )
                oracle_regret = oracle.net_operating_margin_proxy_aud - realized_margin
                warnings.append(
                    "Realised prices are used only after the historical schedule is fixed; this is replay, not live trading."
                )
            data = dispatch.__dict__ | {
                "objective": args.objective,
                "settlement_mode": args.settlement_mode,
                "signal_intervals": len(signal),
                "planned_margin_aud": planned_margin,
                "realized_margin_aud": realized_margin,
                "oracle_regret_aud": oracle_regret,
                "margin_basis": (
                    "historical_actual_settlement_after_as_of_schedule"
                    if realized_margin is not None
                    else "forecast_signal_only"
                ),
                "forecast_source": (
                    "historical_oracle" if args.objective == "perfect_foresight" else forecast_source
                ),
                **({} if args.objective == "perfect_foresight" else forecast_metadata),
                "economic_boundary": "historical spot-market operating proxy; variable degradation is a user-supplied sensitivity, not an asset-specific estimate; excludes CAPEX, fixed O&M, network fees, FCAS and investment returns"
            }
            return ToolResult(
                tool_name=name,
                data=data,
                evidence=self.store.evidence,
                warnings=warnings,
            )
        raise AssertionError(name)
