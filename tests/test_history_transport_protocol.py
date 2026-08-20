from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_history_gate_is_explicitly_not_a_new_untouched_test() -> None:
    text = (ROOT / "docs" / "HISTORICAL_TRANSPORT_GATE.md").read_text(encoding="utf-8")
    assert "backward temporal transport check" in text
    assert "not a prospective trial" in text
    assert "at least 60%" in text
    assert "at least four of five regions" in text


def test_history_jobs_use_short_sapphire_pilot_before_full_year() -> None:
    pilot = (ROOT / "scripts" / "slurm" / "ingest_history_pilot.sbatch").read_text(
        encoding="utf-8"
    )
    full = (ROOT / "scripts" / "slurm" / "ingest_history_year.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --partition=sapphire" in pilot and "#SBATCH --time=00:10:00" in pilot
    assert "--start 2024-08-18 --end 2024-08-24" in pilot
    assert "#SBATCH --partition=sapphire" in full and "#SBATCH --time=00:20:00" in full
    assert "--start 2024-08-18 --end 2025-08-17" in full
    assert "ENERGY_GIT_COMMIT" in pilot and "ENERGY_GIT_COMMIT" in full
