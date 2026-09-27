"""Conservative, inspectable scope checks for dated NEM evidence retrieval.

These checks reject obvious topic/period mismatches. They do not prove causal
explanations or sentence entailment, and never supply a forecast input.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urlparse

REGION_NAMES = {
    "SA1": "South Australia", "VIC1": "Victoria", "QLD1": "Queensland",
    "NSW1": "New South Wales", "TAS1": "Tasmania",
}
REGION_SHORT = {"SA1": "SA", "VIC1": "VIC", "QLD1": "QLD", "NSW1": "NSW", "TAS1": "TAS"}
PERIOD = re.compile(r"\bq([1-4])[\s_-]*(20\d{2})\b", re.IGNORECASE)


@dataclass(frozen=True)
class EvidenceScope:
    regions: tuple[str, ...]
    year: int
    quarter: int

    @property
    def period(self) -> str:
        return f"Q{self.quarter} {self.year}"


def market_evidence_query(regions: list[str], day: date, workflow: str) -> str:
    aliases = " ".join(REGION_NAMES[region] for region in regions)
    topic = "electricity spot prices demand generation"
    if workflow == "decision_replay":
        topic += " battery dispatch"
    return f"{' '.join(regions)} {day.isoformat()} {aliases} NEM {topic} Q{(day.month - 1) // 3 + 1} {day.year}"


def query_scope(query: str) -> EvidenceScope | None:
    match = re.search(r"\b20\d{2}-\d{2}-\d{2}\b", query)
    regions = tuple(region for region in REGION_NAMES if re.search(rf"\b{region}\b", query, re.IGNORECASE))
    if not match or not regions:
        return None
    try:
        day = date.fromisoformat(match.group())
    except ValueError:
        return None
    return EvidenceScope(regions, day.year, (day.month - 1) // 3 + 1)


def scope_support(hit: dict[str, Any], scope: EvidenceScope) -> dict[str, bool]:
    identity = " ".join(str(hit.get(key, "")) for key in ("source_id", "url", "title"))
    periods = {(int(year), int(quarter)) for quarter, year in PERIOD.findall(identity)}
    title = str(hit.get("title", "")).lower()
    content = " ".join(str(hit.get(key, "")) for key in ("title", "subtitle", "text", "source_cell_preview")).lower()
    # Quarter metadata selects the report, not its publication date. Unknown
    # and different report periods are not silently treated as compatible.
    period_ok = periods == {(scope.year, scope.quarter)}
    electricity = bool(re.search(r"\b(nem|electricity|mwh|rrp|spot|batter\w*|generation|solar|pv)\b", content))
    prices = bool(re.search(r"\b(price\w*|pricing|rrp)\b", content))
    gas_only = bool(re.search(r"\b(iona|lng|sttm|dwgm|gas|gj)\b", title)) and not bool(
        re.search(r"\b(electricity|generation|nem|battery)\b", title)
    )
    requested_region = any(
        re.search(rf"\b{re.escape(alias.lower())}\b", content)
        for region in scope.regions for alias in (region, REGION_SHORT[region], REGION_NAMES[region])
    )
    region_ok = requested_region or bool(re.search(r"\b(nem|national electricity market)\b", content))
    return {"report_period_matches": period_ok, "electricity_price_topic": electricity and prices and not gas_only,
            "region_or_nem_context": bool(region_ok)}


def filter_scoped_hits(hits: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    scope = query_scope(query)
    if scope is None:
        return hits
    return [{**hit, "scope_checks": checks, "context_scope": "quarterly_electricity_context"}
            for hit in hits if all((checks := scope_support(hit, scope)).values())]


def evidence_excerpt(text: str, query: str, limit: int = 500) -> tuple[str, int]:
    """Select an exact contiguous passage; preserve its chunk-relative offset."""
    scope = query_scope(query)
    if scope is None or len(text) <= limit:
        return text[:limit], 0
    terms = "|".join(["negative", "spot", "electricity", "battery", "batteries", "prices",
                      *(re.escape(REGION_NAMES[region]) for region in scope.regions)])
    anchors = [m.start() for m in re.finditer(rf"\b({terms})\b", text, re.IGNORECASE)]
    starts = {0, *anchors}

    def score(start: int) -> tuple[int, int]:
        passage = text[start:start + limit].lower()
        regional = sum(REGION_NAMES[region].lower() in passage for region in scope.regions)
        topical = len(set(re.findall(r"\b(negative|spot|electricity|battery|batteries|prices|pricing)\b", passage)))
        return 3 * regional + topical, -start

    start = max(starts, key=score)
    return text[start:start + limit], start


def official_evidence_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == domain or host.endswith("." + domain) for domain in ("aemo.com.au", "aer.gov.au"))
