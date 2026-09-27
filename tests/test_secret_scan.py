from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts import secret_scan


@pytest.mark.parametrize("rg_status,expected", [(0, 1), (1, 0), (2, 2), (127, 2), (-9, 2)])
def test_rg_exit_status_is_fail_closed_without_logging_values(
    rg_status: int, expected: int, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    sensitive_fixture = "sk-" + "explicitfixture" * 3

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        assert command[:4] == ["rg", "--no-config", "--files-with-matches", "--hidden"]
        assert "-q" not in command and "--quiet" not in command
        assert "!.git/**" in command and "!.venv/**" in command
        assert kwargs == {"capture_output": True, "check": False, "timeout": 120}
        raw_output = sensitive_fixture.encode("ascii") + b"\xff\xfe"
        return subprocess.CompletedProcess(command, rg_status, raw_output, raw_output)

    monkeypatch.setattr(secret_scan.subprocess, "run", fake_run)
    assert secret_scan.scan(Path(".")) == expected
    output = capsys.readouterr()
    assert sensitive_fixture not in output.out + output.err
    assert ("completed: no matches" in output.out) is (rg_status == 1)


@pytest.mark.parametrize("failure", ["missing", "permission", "timeout"])
def test_execution_failure_cannot_pass_or_leak_output(
    failure: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    sensitive_fixture = "ghp_" + "explicitfixture" * 3

    def cannot_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if failure == "missing":
            raise FileNotFoundError(sensitive_fixture)
        if failure == "permission":
            raise PermissionError(sensitive_fixture)
        raise subprocess.TimeoutExpired("rg", 120, output=sensitive_fixture, stderr=sensitive_fixture)

    monkeypatch.setattr(secret_scan.subprocess, "run", cannot_run)
    assert secret_scan.scan(Path(".")) == 2
    output = capsys.readouterr()
    assert "ERROR" in output.out
    assert sensitive_fixture not in output.out + output.err


@pytest.mark.parametrize("contains_match", [False, True])
def test_real_rg_scans_hidden_fixture_without_echoing_content(
    contains_match: bool, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    # Required in CI: missing tooling must not silently skip these tests.
    assert shutil.which("rg"), "ripgrep is required; install it before running scanner tests"
    sensitive_fixture = "sk-" + "explicitfixture" * 3
    content = sensitive_fixture if contains_match else "Explicit clean fixture, not a market result."
    (tmp_path / ".hidden-fixture").write_text(content, encoding="utf-8")
    assert secret_scan.scan(tmp_path) == (1 if contains_match else 0)
    output = capsys.readouterr()
    assert sensitive_fixture not in output.out + output.err
    assert ("BLOCKED" in output.out) is contains_match


def test_ci_installs_scanner_before_required_tests_and_calls_fail_closed_wrapper() -> None:
    workflow = (Path(__file__).parents[1] / ".github/workflows/quality.yml").read_text()
    assert workflow.index("sudo apt-get install --yes ripgrep") < workflow.index("Unit and contract tests")
    assert "command -v rg" in workflow and "rg --version" in workflow
    assert "run: python scripts/secret_scan.py" in workflow
    assert "if rg " not in workflow
