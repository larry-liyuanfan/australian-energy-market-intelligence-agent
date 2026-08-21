from __future__ import annotations

from pathlib import Path


def test_history_gate_job_pins_summary_and_evaluation_commits() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "slurm" / "summarize_history_transport.sbatch"
    ).read_text(encoding="utf-8")
    assert '"${SUMMARY_GIT_COMMIT}^{commit}"' in script
    assert '"${EVALUATION_GIT_COMMIT}^{commit}"' in script
    assert 'checkout --detach "${SUMMARY_GIT_COMMIT}"' in script
    assert "pip install 'numpy>=2,<3'" in script
