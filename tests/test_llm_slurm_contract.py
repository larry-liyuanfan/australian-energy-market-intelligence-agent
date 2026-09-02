from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_llm_jobs_use_isolated_energy_paths_and_never_trip_paths() -> None:
    scripts = [
        ROOT / "scripts" / "slurm" / "llm_agent_runtime_preflight.sbatch",
        ROOT / "scripts" / "slurm" / "llm_agent_gpu_pilot.sbatch",
        ROOT / "scripts" / "slurm" / "llm_agent_holdout.sbatch",
    ]
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        assert "/energy-agent/llm-agent-eval-20260903" in text
        assert "Trip_Project" not in text
        assert "climate" not in text.lower()
        assert "flare" not in text.lower()


def test_holdout_job_binds_real_aemo_inputs_and_real_model_server() -> None:
    text = (ROOT / "scripts" / "slurm" / "llm_agent_holdout.sbatch").read_text(encoding="utf-8")
    assert "dispatch_features_repaired_v2.csv.gz" in text
    assert "evidence_documents.jsonl" in text
    assert "forecast-snapshots-c66e415.jsonl" in text
    assert "llama-server" in text
    assert "Qwen3-8B-Q4_K_M.gguf" in text
    assert "MODEL_SHA256=" in text
    assert "--host 127.0.0.1" in text
    assert "llm_agent_holdout_v1.jsonl" in text


def test_runtime_progression_is_short_preflight_then_gpu_pilot_then_holdout() -> None:
    preflight = (ROOT / "scripts" / "slurm" / "llm_agent_runtime_preflight.sbatch").read_text(encoding="utf-8")
    pilot = (ROOT / "scripts" / "slurm" / "llm_agent_gpu_pilot.sbatch").read_text(encoding="utf-8")
    holdout = (ROOT / "scripts" / "slurm" / "llm_agent_holdout.sbatch").read_text(encoding="utf-8")
    assert "--time=00:15:00" in preflight and "--partition=cascade" in preflight
    assert "module load GCCcore/11.3.0" in preflight
    assert "--time=00:35:00" in pilot and "--partition=gpu-a100-mig" in pilot
    assert "--time=01:20:00" in holdout and "--partition=gpu-a100-mig" in holdout
    assert "--gres=gpu:1g.20gb:1" in pilot
    assert "--tmp=10000" in pilot and "SLURM_TMPDIR" in pilot
    assert "--tmp=10000" in holdout and "SLURM_TMPDIR" in holdout
    assert "libpython3.11.so.1.0" in pilot and "LD_LIBRARY_PATH" in pilot
    assert "libpython3.11.so.1.0" in holdout and "LD_LIBRARY_PATH" in holdout
    assert "llama.cpp-$LLAMA_COMMIT" in pilot
