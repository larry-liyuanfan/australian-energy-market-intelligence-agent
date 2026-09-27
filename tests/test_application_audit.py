from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime

import pytest

from energy_agent.schemas import BatterySpec
from scripts.audit_agent_application import (
    BATTERY_DEFAULTS,
    audit_schedule,
    citation_checks,
    request_errors,
    same_instant,
    signal_identity,
)


def test_latest_turn_battery_correction_is_not_a_substring_check() -> None:
    spec = {"expected": {"region": "NSW1", "date": "2026-02-07", "battery_power_mw": 1.5,
                         "battery_energy_mwh": 3, "round_trip_efficiency": .87},
            "expected_tools": ["optimize_battery_dispatch"]}
    call = {"name": "optimize_battery_dispatch", "arguments": {
        "region": "NSW1", "window": {"start": "2026-02-07T00:00:00+10:00", "end": "2026-02-08T00:00:00+10:00"},
        "battery": {"power_mw": 1.5, "energy_mwh": 3, "round_trip_efficiency": .87},
        "objective": "forecast", "settlement_mode": "historical_replay"}}
    assert request_errors([call], spec) == []
    stale = copy.deepcopy(call)
    stale["arguments"]["battery"]["round_trip_efficiency"] = .9  # type: ignore[index]
    assert "optimize_battery_dispatch:round_trip_efficiency" in request_errors([stale], spec)
    assert request_errors([], spec) == ["missing:optimize_battery_dispatch"]


def test_independent_cashflow_and_energy_equation() -> None:
    battery = {"power_mw": 1, "energy_mwh": 2, "round_trip_efficiency": 1,
               "min_soc_fraction": .1, "max_soc_fraction": .9, "initial_soc_fraction": .5,
               "terminal_soc_fraction": .5}
    dispatch = {"charge_mw": [1, 0], "discharge_mw": [0, 1], "soc_mwh": [1, 1 + 1 / 12, 1]}
    checked = audit_schedule(battery, dispatch, [-12, 24], [-24, 48], 12)
    assert all(checked["checks"].values())
    assert checked["planned_gross_aud"] == 3
    assert checked["realised_gross_aud"] == 6
    assert checked["realised_operating_proxy_aud"] == 5
    dispatch["soc_mwh"][1] += .01
    assert not audit_schedule(battery, dispatch, [-12, 24], [-24, 48], 12)["checks"]["energy_balance"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_calculation_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        audit_schedule({}, {"charge_mw": [bad], "discharge_mw": [0], "soc_mwh": [1, 1]}, [0], [0], 0)


def test_citation_checks_exact_source_location_not_just_hash_length() -> None:
    source = {"url": "https://www.aemo.com.au/report.pdf", "sha256": "a" * 64, "text": "price context"}
    cite = {"url": source["url"], "sha256": source["sha256"], "modality": "text",
            "short_text_excerpt": "price", "excerpt_source_start": 0, "excerpt_source_end": 5}
    assert all(citation_checks(cite, source).values())
    cite["short_text_excerpt"] = "cause"
    assert not citation_checks(cite, source)["excerpt_locates_in_source_chunk"]
    cite["sha256"] = "b" * 64
    assert not citation_checks(cite, source)["source_digest_matches"]


@pytest.mark.parametrize("value,expected", [
    ("2025-12-15T01:00:00+11:00", True), ("2025-12-15T00:00:00+10:00", True),
    ("2025-12-15T00:00:00", False), ("2025-12-16T00:00:00+10:00", False), ("bad", False),
])
def test_time_comparison_is_aware_and_representation_independent(value: str, expected: bool) -> None:
    assert same_instant(value, datetime.fromisoformat("2025-12-15T00:00:00+10:00")) is expected


def test_private_consistency_cannot_override_public_signal() -> None:
    point = [1.0, 2.0]
    sha = hashlib.sha256(json.dumps(point, separators=(",", ":")).encode()).hexdigest()
    forecast = {"point": point, "signal_sha256": sha, "forecast_snapshot_id": "snapshot", "training_cutoff": "cutoff"}
    dispatch = {"signal_sha256": sha, "forecast_snapshot_id": "snapshot", "forecast_training_cutoff": "cutoff"}
    compact = {"signal_sha256": "x" * 64, "forecast_snapshot_id": "snapshot", "training_cutoff": "cutoff"}
    assert not signal_identity(forecast, dispatch, compact)
    compact["signal_sha256"] = sha
    assert signal_identity(forecast, dispatch, compact)


def test_source_link_is_not_counted_as_verified_excerpt() -> None:
    source = {"url": "https://www.aemo.com.au/report.pdf", "sha256": "a" * 64, "text": "price context"}
    cite = {"url": source["url"], "sha256": source["sha256"], "modality": "text",
            "display_role": "source_link_only", "short_text_excerpt": None}
    assert "excerpt_locates_in_source_chunk" not in citation_checks(cite, source)
    cite["display_role"] = "visible_evidence"
    assert not citation_checks(cite, source)["excerpt_locates_in_source_chunk"]


def test_independent_declared_defaults_match_original_tool_contract() -> None:
    assert BATTERY_DEFAULTS == BatterySpec().model_dump()
