from __future__ import annotations

from pathlib import Path


def test_v2_slurm_jobs_are_isolated_and_exact_commit() -> None:
    root = Path(__file__).parents[1]
    paths = [
        root / "scripts/slurm/visual_goal_v2_preflight.sbatch",
        root / "scripts/slurm/goal_spec_v2_pilot.sbatch",
        root / "scripts/slurm/goal_spec_v2_full.sbatch",
        root / "scripts/slurm/vidore_v2_pilot.sbatch",
        root / "scripts/slurm/vidore_v2_full.sbatch",
    ]
    for path in paths:
        text = path.read_text()
        assert "#SBATCH --account=punim2936" in text
        assert "/energy-visual-goal-compiler-v2/" in text
        assert "ENERGY_GIT_COMMIT" in text
        assert "checkout --detach" in text
        assert "llm-agent-eval-20260903/private" not in text
        assert "risk-" not in text
        assert "chronos" not in text.lower()


def test_full_jobs_never_emit_private_predictions_to_github_paths() -> None:
    root = Path(__file__).parents[1]
    for name in ("goal_spec_v2_full.sbatch", "vidore_v2_full.sbatch"):
        text = (root / "scripts/slurm" / name).read_text()
        assert "artifacts/public" not in text
        assert "private/" in text
