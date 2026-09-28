"""Synthetic accounting fixtures only; no market, model or frozen evaluation execution."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from scripts.summarize_incremental_full import CHECKS, read_journal, summarize


def episode(fault: str | None = None) -> dict[str, Any]:
    return {"episode_id": "synthetic-accounting", "fault": fault,
            "turns": [{"expected": {"region": "SA1"}}, {"expected": {"region": "VIC1"}}]}


def row(path: str = "incremental", turn: int = 1, status: str = "completed") -> dict[str, Any]:
    return {"episode_id": "synthetic-accounting", "path": path, "seed": 17, "turn": turn,
            "execution": "synthetic_postprocessing_fixture", "controlled_fault": None, "wall_seconds": 2.0,
            "score": {"task_contract_success": True, "sourced_state_contract": True},
            "independent_checks": dict.fromkeys(CHECKS, True),
            "provider_requests": [{"usage_known": True, "response_received": True,
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "provider_cost_aud": 0.0}}],
            "run": {"status": status, "calls": [{"name": "get_market_snapshot", "status": "ok", "owner": "system_completion"}],
                "provenance": [{"action": "execute"}], "planner_attempts": [{"proposed": [{"name": "get_market_snapshot"}],
                    "accepted": [], "rejected": ["get_market_snapshot"]}]}}


def run(rows: list[dict[str, Any]], fault: str | None = None, partial: bool = False) -> dict[str, Any]:
    return summarize([episode(fault)], rows, paths=("legacy", "incremental"), seeds=(17,), partial_tail=partial)


def test_history_never_doubles_cost_or_tools() -> None:
    record = row()
    record["run"]["history"] = [copy.deepcopy(record["run"])] * 3
    report = run([record])
    assert report["totals"]["executed_tool_attempts"]["known_sum"] == 1
    assert report["totals"]["known_prompt_tokens"]["known_sum"] == 10
    assert report["totals"]["missing_rows"] == 3
    assert report["totals"]["token_totals_are_lower_bounds"]


def test_fault_preparation_and_paused_checkpoint_are_not_natural_completion() -> None:
    record = row(status="validated")
    record["run"]["interrupts"] = [{"stage": "forecast"}]
    report = run([record], "checkpoint_correction")
    assert not any("natural_task" in k for k in report["groups"])
    result = report["paired_slots"]["synthetic-accounting:1:17"]["incremental"]
    assert result["phase"] == "checkpoint_pause" and result["checkpoint_pause_observed"]
    assert result["recorded_contract_match"] and not result["completed_numerically_verified"]
    assert result["numerical_check"] is None and result["state_check"] is None
    assert report["totals"]["planned_intervention_slots"] == 4
    setup = run([row()], "empty_evidence")["paired_slots"]["synthetic-accounting:1:17"]["incremental"]
    assert setup["phase"] == "setup" and not setup["planned_intervention"]


def test_legacy_ownership_is_unavailable_not_zero_or_inferred() -> None:
    result = run([row("legacy")])["paired_slots"]["synthetic-accounting:1:17"]["legacy"]
    assert result["runtime_accepted_proposals"] is None and result["system_completion_executions"] is None
    assert result["reuse"] is None


def test_provider_failure_survives_successful_fallback() -> None:
    record = row()
    record["provider_requests"] = [{"error": "ProviderUnavailable", "usage_known": False}]
    result = run([record])["paired_slots"]["synthetic-accounting:1:17"]["incremental"]
    assert result["completed_numerically_verified"]
    assert result["provider_exceptions"] == result["unknown_usage_requests"] == 1
    assert result["known_prompt_tokens"] == 0 and result["token_totals_are_lower_bounds"]


def test_missing_boundary_differs_from_no_request_and_unchecked_numeric() -> None:
    absent, empty = row(), row()
    del absent["provider_requests"]
    empty["provider_requests"] = []
    absent["independent_checks"] = None
    a = run([absent])["paired_slots"]["synthetic-accounting:1:17"]["incremental"]
    b = run([empty])["paired_slots"]["synthetic-accounting:1:17"]["incremental"]
    assert a["request_accounting_unresolved"] and a["numerical_check"] is None
    assert not b["request_accounting_unresolved"]


def test_duplicate_unknown_slots_fail_and_missing_slots_keep_denominator() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        run([row(), row()])
    bad = row()
    bad["seed"] = 99
    with pytest.raises(ValueError, match="unexpected"):
        run([bad])
    report = run([])
    assert report["totals"]["expected_slots"] == report["totals"]["missing_rows"] == 4
    assert len(report["paired_slots"]) == 2


def test_safe_stop_contract_is_not_recovery_completion() -> None:
    record = row(turn=2, status="clarification_required")
    record.update(controlled_fault="empty_evidence", fault_fired=True,
                  intervention_score={"exercised": True, "target_outcome_verified": True})
    record["run"]["planner_attempts"] = [{"trigger": "empty_result", "decision": "clarify", "decision_owner": "model"}]
    result = run([record], "empty_evidence")["paired_slots"]["synthetic-accounting:2:17"]["incremental"]
    assert result["target_verified_safe_stop"] and result["contract_matched_noncompletion"]
    assert not result["recovery_to_verified_completion"] and result["numerical_check"] is None
    assert result["empty_recovery_choices"] == {"model:clarify": 1}


def test_partial_tail_is_visible_and_interior_corruption_fails(tmp_path: Path) -> None:
    journal = tmp_path / "synthetic.jsonl"
    journal.write_text('{}\n{"partial":', encoding="utf-8")
    records, partial = read_journal(journal)
    assert records == [{}] and partial
    assert run([], partial=True)["unrecorded_tail_request_costs_unknown"]
    journal.write_text('not json\n{}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="nonterminal"):
        read_journal(journal)


def test_graph_retry_uses_ordered_events_not_unset_attempt_flag() -> None:
    record = row(turn=2)
    record.update(controlled_fault="timeout_once", fault_fired=True,
                  intervention_score={"exercised": True, "target_outcome_verified": True})
    record["run"]["calls"] = [{"name": "search_official_evidence", "status": status, "attempt": 1, "recovered": False}
                              for status in ("timeout", "ok")]
    result = run([record], "timeout_once")["paired_slots"]["synthetic-accounting:2:17"]["incremental"]
    assert result["recovery_to_verified_completion"]
    record["run"]["calls"].reverse()
    result = run([record], "timeout_once")["paired_slots"]["synthetic-accounting:2:17"]["incremental"]
    assert not result["recovery_to_verified_completion"]


def test_missing_run_preserves_known_requests_but_not_zero_tool_claims() -> None:
    record = row()
    del record["run"]
    report = run([record])
    result = report["paired_slots"]["synthetic-accounting:1:17"]["incremental"]
    assert result["provider_requests"] == 1 and result["known_prompt_tokens"] == 10
    for key in ("tool_call_records", "executed_tool_attempts", "model_proposals", "initial_model_proposals",
                "system_completion_executions", "model_owned_executions", "runtime_accepted_proposals", "reuse"):
        assert result[key] is None
    assert report["totals"]["unavailable_tool_trace_rows"] == 1


def test_clean_newline_does_not_prove_tail_costs_complete() -> None:
    report = run([row()], partial=False)
    assert not report["partial_trailing_record"] and report["unrecorded_tail_request_costs_unknown"]
    complete = [row(path, turn) for path in ("legacy", "incremental") for turn in (1, 2)]
    assert not run(complete)["unrecorded_tail_request_costs_unknown"]
    del complete[0]["provider_requests"]
    assert run(complete)["unrecorded_tail_request_costs_unknown"]
