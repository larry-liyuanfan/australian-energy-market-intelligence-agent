from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from energy_agent.market import MarketStore, fixture_store
from energy_agent.remediation import exact_parameter_accuracy, validate_market_windows
from energy_agent.schemas import ToolCall


def test_coverage_preflight_rejects_outside_data_before_inference() -> None:
    fixture = fixture_store()
    store = MarketStore(fixture.rows, fixture.evidence, "unit-test-labelled-as-loaded-store")
    episode = {"turns": [{"expected": {"region": "SA1", "date": "2025-01-03"}}]}
    assert validate_market_windows([episode], store)["complete_region_days"] == 1
    episode["turns"][0]["expected"]["date"] = "2024-12-31"
    with pytest.raises(ValueError, match="incomplete five-minute"):
        validate_market_windows([episode], store)


def test_coverage_preflight_rejects_duplicate_interval_and_fixture_label() -> None:
    fixture = fixture_store()
    episodes = [{"turns": [{"expected": {"region": "SA1", "date": "2025-01-03"}}]}]
    with pytest.raises(ValueError, match="real market store"):
        validate_market_windows(episodes, fixture)
    rows = list(fixture.rows)
    index = next(i for i, row in enumerate(rows) if row.region == "SA1" and row.interval.day == 3)
    rows.insert(index, rows[index])
    with pytest.raises(ValueError, match="incomplete five-minute"):
        validate_market_windows(episodes, MarketStore(rows, data_version="unit-test"))


def test_exact_score_cannot_hide_extra_region_or_wrong_half_open_boundary() -> None:
    calls = [ToolCall(name="compare_region_period", arguments={
        "regions": ["SA1", "QLD1", "VIC1"],
        "window": {"start": "2025-01-03T00:00:00+10:00", "end": "2025-01-03T23:59:59+10:00"},
    })]
    assert exact_parameter_accuracy(calls, {"regions": ["SA1", "QLD1"], "date": "2025-01-03"}) == 1 / 3


def test_v3_tasks_and_dates_are_disjoint_from_legacy_and_development() -> None:
    root = Path(__file__).parents[1] / "benchmarks"
    holdout = (root / "planner_remediation_holdout_v3.jsonl").read_text()
    prior_files = [path for path in root.glob("*.jsonl") if path.name != "planner_remediation_holdout_v3.jsonl"]
    prior = "\n".join(path.read_text(encoding="utf-8") for path in prior_files)
    dates = set(re.findall(r"20\d\d-\d\d-\d\d", holdout))
    assert dates.isdisjoint(re.findall(r"20\d\d-\d\d-\d\d", prior))
    episodes = [json.loads(line) for line in holdout.splitlines()]
    assert len(episodes) == 10
    assert sum(len(episode["turns"]) for episode in episodes) == 18
