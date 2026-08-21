from __future__ import annotations

from scripts.validate_dispatch_coverage import time_axis_issues


def _row(interval: str, region: str = "SA1") -> dict[str, str]:
    return {"interval": interval, "region": region, "intervention": "0"}


def test_time_axis_gate_detects_duplicate_and_gap() -> None:
    issues = time_axis_issues(
        [
            _row("2026/03/10 00:00:00"),
            _row("2026/03/10 00:00:00"),
            _row("2026/03/10 00:10:00"),
        ]
    )
    assert issues["duplicate_key_count"] == 1
    assert issues["gap_count"] == 1


def test_time_axis_gate_accepts_contiguous_intervals() -> None:
    issues = time_axis_issues(
        [
            _row("2026/03/10 00:00:00"),
            _row("2026/03/10 00:05:00"),
            _row("2026/03/10 00:10:00"),
        ]
    )
    assert issues["duplicate_key_count"] == 0
    assert issues["gap_count"] == 0
