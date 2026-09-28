from __future__ import annotations

import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from energy_agent.incremental_evaluation import normalize_usage, score_intervention, usage_accounting
from scripts.check_local_model_owner import check_owner, socket_inodes


def test_flat_nested_and_unknown_usage() -> None:
    flat = {"prompt_tokens": 50, "completion_tokens": 7, "latency_ms": 12.5}
    assert normalize_usage(flat)["usage_known"]
    assert normalize_usage({"usage": flat, "usage_known": True})["usage"] == normalize_usage(flat)["usage"]
    assert not normalize_usage({"usage": flat, "usage_known": False})["usage_known"]
    legacy = usage_accounting({"run": {"planner_attempts": [flat]}})
    assert legacy["known_prompt_tokens"] == 50 and legacy["unknown_usage"] == 0
    missing = usage_accounting({"usage_unknown_on_failure": True})
    assert missing["unknown_usage"] == 1 and missing["unresolved_request_accounting"]
    observed = usage_accounting({"provider_requests": [{"error": "TimeoutError", "usage_known": False}],
                                "usage_unknown_on_failure": True})
    assert observed["model_attempts"] == 1 and observed["provider_exceptions"] == 1 and observed["unknown_usage"] == 1


def test_unrelated_failure_cannot_pass_target_intervention() -> None:
    row: dict[str, Any] = {"controlled_fault": "settlement_conflict", "fault_fired": False,
                           "run": {"status": "failed", "error": "unrelated"}}
    result = score_intervention(row)
    assert result and not result["exercised"] and not result["target_outcome_verified"]
    row["fault_fired"] = True
    result = score_intervention(row)
    assert result and not result["target_outcome_verified"]
    row["run"].update(error="verification:settlement_mismatch", calls=[{
        "name": "optimize_battery_dispatch", "error_category": "verification:settlement_mismatch"}])
    result = score_intervention(row)
    assert result and result["target_outcome_verified"]


def test_version_and_thread_lifecycle_do_not_require_tool_fault_fired() -> None:
    result = score_intervention({"controlled_fault": "snapshot_content_changed", "fault_fired": False,
        "intervention": {"version_changed": True}, "run": {"status": "completed", "calls": [{"name": "forecast_price_risk"}]}})
    assert result and result["target_outcome_verified"]
    result = score_intervention({"controlled_fault": "thread_isolation", "fault_fired": False,
        "intervention": {"new_thread_selected": True}, "run": {"status": "insufficient_evidence", "resolved_constraints": {}}})
    assert result and result["target_outcome_verified"]
    result = score_intervention({"controlled_fault": "planner_unsafe", "intervention": {"prompt_appended": True},
                                "run": {"status": "completed"}, "provider_requests": []})
    assert result and not result["exercised"]
    result = score_intervention({"controlled_fault": "planner_unsafe", "intervention": {"prompt_appended": True},
        "run": {"status": "completed"}, "provider_requests": [{"error": "ProviderUnavailable"}]})
    assert result and not result["exercised"] and not result["target_outcome_verified"]


def test_loopback_ignores_proxy_and_rejects_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    from urllib.request import Request

    from energy_agent.loopback import loopback_transport
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("no_proxy", "")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(302 if self.path == "/redirect" else 200)
            self.send_header("Location", "https://not-an-endpoint.invalid")
            self.end_headers()
            self.wfile.write(b"local-test-double")

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            assert loopback_transport(Request(url), 1) == b"local-test-double"
            with pytest.raises(ValueError, match="redirect"):
                loopback_transport(Request(url+"/redirect"), 1)
            with pytest.raises(ValueError, match="loopback"):
                loopback_transport(Request("https://not-an-endpoint.invalid"), 1)
        finally:
            server.shutdown()
            worker.join(timeout=2)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux proc ownership contract")
def test_listener_ownership_and_model_identity() -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"test-double-not-a-model"}]}')

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            port = server.server_address[1]
            check_owner(os.getpid(), port, "test-double-not-a-model")
            with pytest.raises(ValueError, match="identity"):
                check_owner(os.getpid(), port, "wrong-model")
            with pytest.raises((ValueError, PermissionError)):
                check_owner(os.getppid(), port, "test-double-not-a-model")
        finally:
            server.shutdown()
            worker.join(timeout=2)


def test_disappearing_fd_does_not_skip_listener_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def readlink(path: Path) -> str:
        seen.append(path.name)
        if path.name == "closed":
            raise FileNotFoundError("descriptor closed")
        if path.name == "denied":
            raise PermissionError("proc unavailable")
        return "socket:[123]"

    monkeypatch.setattr(os, "readlink", readlink)
    monkeypatch.setattr(Path, "iterdir", lambda self: iter([Path("closed"), Path("listener")]))
    assert socket_inodes(1) == {"123"} and seen == ["closed", "listener"]
    monkeypatch.setattr(Path, "iterdir", lambda self: iter([Path("closed")]))
    assert socket_inodes(1) == set()
    monkeypatch.setattr(Path, "iterdir", lambda self: iter([Path("denied")]))
    with pytest.raises(PermissionError):
        socket_inodes(1)
