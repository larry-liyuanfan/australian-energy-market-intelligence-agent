from __future__ import annotations

import json
from pathlib import Path

from scripts.publish_llm_agent_summary import build_summary


def test_public_summary_keeps_hashes_and_removes_private_paths(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    manifest_path = tmp_path / "manifest.json"
    hashes_path = tmp_path / "hashes.txt"
    usage_path = tmp_path / "usage.txt"
    metrics_path.write_text(
        json.dumps(
            {
                "constrained_hybrid|structured_state": {"task_success_rate": 0.9},
                "memory_required_subset": {"hybrid_rows": 3},
                "stability": {"pass_at_1": 0.9},
                "promotion_pass": False,
                "promotion_checks": {"correct_tool_path_rate": False},
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": "2026-09-03T00:00:00+00:00",
                "git_sha": "abc123",
                "provider": "llama_cpp_local",
                "model": "Qwen3-8B-Q4_K_M.gguf",
                "model_runtime_is_real": True,
                "environment": {"SLURM_JOB_ID": "123"},
                "benchmark_sha256": "a" * 64,
                "gate_sha256": "b" * 64,
                "data_manifest_sha256": "c" * 64,
                "evidence_sha256": "d" * 64,
                "forecast_snapshots_sha256": "e" * 64,
                "boundaries": ["historical only"],
            }
        ),
        encoding="utf-8",
    )
    hashes_path.write_text(
        f"{'f' * 64}  /private/bin/llama-server\n{'1' * 64}  /scratch/Qwen3-8B-Q4_K_M.gguf\n",
        encoding="utf-8",
    )
    usage_path.write_text(
        "JobID|State|Elapsed|MaxRSS|AllocTRES\n123|COMPLETED|01:00:00|6G|cpu=6,mem=20G\n",
        encoding="utf-8",
    )

    summary = build_summary(metrics_path, manifest_path, hashes_path, usage_path)

    serialized = json.dumps(summary)
    assert summary["runtime"]["real_model_runtime"] is True
    assert summary["runtime"]["sha256"]["model_gguf"] == "1" * 64
    assert summary["resource_usage"]["state"] == "COMPLETED"
    assert "/private/" not in serialized
    assert "/scratch/" not in serialized
