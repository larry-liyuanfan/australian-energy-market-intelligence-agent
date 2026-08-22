from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Region(StrEnum):
    NSW1 = "NSW1"
    QLD1 = "QLD1"
    SA1 = "SA1"
    TAS1 = "TAS1"
    VIC1 = "VIC1"


class Window(StrictModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def ordered(self) -> Window:
        if self.start >= self.end:
            raise ValueError("start must precede end")
        return self


class MarketFilter(StrictModel):
    region: Region
    window: Window


class GetMarketSnapshotInput(StrictModel):
    region: Region
    at: datetime


class CompareRegionPeriodInput(StrictModel):
    regions: list[Region] = Field(min_length=2, max_length=5)
    window: Window


class DetectPriceEventsInput(MarketFilter):
    threshold_aud_mwh: float = Field(default=5000, ge=-1000, le=20000)
    robust_z_threshold: float = Field(default=5, ge=1, le=20)


class DiagnosePriceEventInput(StrictModel):
    region: Region
    interval: datetime
    context_intervals: int = Field(default=12, ge=1, le=288)


class SearchOfficialEvidenceInput(StrictModel):
    query: str = Field(min_length=3, max_length=500)
    published_after: datetime | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    preferred_modality: Literal["auto", "text", "visual", "chart", "table"] = "auto"
    retrieval_mode: Literal["hybrid_rerank", "multimodal_fusion"] = "hybrid_rerank"


class ForecastPriceRiskInput(MarketFilter):
    horizon_intervals: int = Field(default=12, ge=1, le=288)
    alpha: float = Field(default=0.1, gt=0, lt=0.5)


class BatterySpec(StrictModel):
    power_mw: float = Field(default=1.0, gt=0, le=1000)
    energy_mwh: float = Field(default=2.0, gt=0, le=10000)
    round_trip_efficiency: float = Field(default=0.9, gt=0, le=1)
    min_soc_fraction: float = Field(default=0.1, ge=0, lt=1)
    max_soc_fraction: float = Field(default=0.9, gt=0, le=1)
    initial_soc_fraction: float = Field(default=0.5, ge=0, le=1)
    terminal_soc_fraction: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def feasible(self) -> BatterySpec:
        if not self.min_soc_fraction <= self.initial_soc_fraction <= self.max_soc_fraction:
            raise ValueError("initial SoC outside bounds")
        if not self.min_soc_fraction <= self.terminal_soc_fraction <= self.max_soc_fraction:
            raise ValueError("terminal SoC outside bounds")
        return self


class OptimizeBatteryDispatchInput(MarketFilter):
    battery: BatterySpec = Field(default_factory=BatterySpec)
    objective: Literal["forecast", "perfect_foresight", "threshold_rule"] = "forecast"
    variable_degradation_cost_aud_per_mwh_discharged: float = Field(default=0.0, ge=0, le=10_000)


class ExplainDataCoverageInput(StrictModel):
    region: Region | None = None


TOOL_MODELS: dict[str, type[StrictModel]] = {
    "get_market_snapshot": GetMarketSnapshotInput,
    "compare_region_period": CompareRegionPeriodInput,
    "detect_price_events": DetectPriceEventsInput,
    "diagnose_price_event": DiagnosePriceEventInput,
    "search_official_evidence": SearchOfficialEvidenceInput,
    "forecast_price_risk": ForecastPriceRiskInput,
    "optimize_battery_dispatch": OptimizeBatteryDispatchInput,
    "explain_data_coverage": ExplainDataCoverageInput,
}


class Evidence(StrictModel):
    evidence_id: str
    title: str
    url: str
    published_at: datetime | None = None
    retrieved_at: datetime
    sha256: str
    snippet: str
    evidence_type: Literal["numeric", "explanatory"]
    score: float = 0
    modality: Literal["text", "page_image", "chart", "table", "mixed"] = "text"
    source_page: int | None = Field(default=None, ge=1)
    asset_id: str | None = None
    asset_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    retrieval_scores: dict[str, float] = Field(default_factory=dict)


class ToolResult(StrictModel):
    tool_name: str
    data: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ToolCall(StrictModel):
    name: str
    arguments: dict[str, Any]
    status: Literal["ok", "error", "timeout", "skipped"] = "ok"
    duration_ms: float = 0
    recovered: bool = False


class AgentQueryRequest(StrictModel):
    question: str = Field(min_length=3, max_length=1000)
    max_tool_calls: int = Field(default=6, ge=1, le=8)


class AgentQueryResponse(StrictModel):
    trace_id: str
    status: Literal["completed", "insufficient_evidence", "clarification_required", "failed"]
    answer: str
    citations: list[Evidence] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    data_version: str
