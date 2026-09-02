from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _runtime_hashes(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            continue
        name = Path(parts[1].strip()).name
        label = "model_gguf" if name.endswith(".gguf") else "llama_server"
        hashes[label] = parts[0]
    return hashes


def _slurm_usage(path: Path, job_id: str) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="|"))
    exact = next((row for row in rows if row.get("JobID") == job_id), None)
    if exact is None:
        raise ValueError(f"Slurm usage does not contain exact job {job_id}")
    return {
        "state": exact.get("State", ""),
        "elapsed": exact.get("Elapsed", ""),
        "max_rss": exact.get("MaxRSS", ""),
        "allocated_tres": exact.get("AllocTRES", ""),
    }


def build_summary(
    metrics_path: Path,
    manifest_path: Path,
    runtime_hashes_path: Path,
    slurm_usage_path: Path,
) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job_id = str(manifest.get("environment", {}).get("SLURM_JOB_ID", ""))
    if not job_id:
        raise ValueError("run manifest is missing SLURM_JOB_ID")
    group_metrics = {key: value for key, value in metrics.items() if "|" in key}
    return {
        "schema_version": "llm-agent-public-summary-v1",
        "created_at": manifest["created_at"],
        "evaluation_split": "frozen_holdout",
        "git_sha": manifest["git_sha"],
        "runtime": {
            "provider": manifest["provider"],
            "model": manifest["model"],
            "real_model_runtime": manifest["model_runtime_is_real"],
            "job_id": job_id,
            "sha256": _runtime_hashes(runtime_hashes_path),
        },
        "input_sha256": {
            key: manifest.get(key)
            for key in (
                "benchmark_sha256",
                "gate_sha256",
                "data_manifest_sha256",
                "evidence_sha256",
                "forecast_snapshots_sha256",
            )
        },
        "metrics": group_metrics,
        "memory_required_subset": metrics["memory_required_subset"],
        "stability": metrics["stability"],
        "promotion": {
            "passed": metrics["promotion_pass"],
            "checks": metrics["promotion_checks"],
        },
        "resource_usage": _slurm_usage(slurm_usage_path, job_id),
        "cost_boundary": {
            "external_provider_cost_aud": 0.0,
            "note": "Local research-compute allocation was consumed; it is reported as Slurm resources, not converted to a commercial AUD price.",
        },
        "boundaries": manifest["boundaries"],
        "privacy": "Per-turn prompts, predictions, private paths, and source rows remain outside GitHub.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact public summary from a private LLM Agent run")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-hashes", type=Path, required=True)
    parser.add_argument("--slurm-usage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.metrics, args.manifest, args.runtime_hashes, args.slurm_usage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
