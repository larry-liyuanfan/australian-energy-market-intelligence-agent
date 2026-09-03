from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_goal_run_directory(path: Path) -> dict[str, Any]:
    manifest_path = path / "run_manifest.json"
    metrics_path = path / "metrics.json"
    predictions_path = path / "predictions.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "goal-spec-agent-run-v2":
        raise ValueError("invalid GoalSpec run schema_version")
    if sha256(metrics_path) != manifest.get("metrics_sha256"):
        raise ValueError("GoalSpec metrics hash mismatch")
    if sha256(predictions_path) != manifest.get("predictions_sha256"):
        raise ValueError("GoalSpec predictions hash mismatch")
    if not manifest.get("benchmark_sha256") or not manifest.get("gate_sha256"):
        raise ValueError("GoalSpec run is not bound to benchmark and gate hashes")
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("GoalSpec run has no prediction rows")
    return {"rows": len(rows), "system_label": manifest["system_label"], "manifest": manifest}


def validate_goal_aggregate(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != "goal-spec-agent-aggregate-v2":
        raise ValueError("invalid GoalSpec aggregate schema_version")
    required = {
        "deterministic",
        "qwen3_8b_direct_tool",
        "qwen3_8b_goal_spec",
        "qwen3_14b_goal_spec",
    }
    if set(artifact.get("systems", {})) != required:
        raise ValueError("GoalSpec aggregate does not contain the four frozen systems")
    checks = artifact.get("promotion_checks", {})
    if bool(artifact.get("promotion_pass")) != all(bool(value) for value in checks.values()):
        raise ValueError("GoalSpec promotion decision is inconsistent with checks")
    return artifact


def validate_vidore_run_directory(path: Path) -> dict[str, Any]:
    manifest_path = path / "run_manifest.json"
    metrics_path = path / "metrics.json"
    predictions_path = path / "predictions.jsonl"
    dataset_manifest_path = path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "vidore-retrieval-run-manifest-v2":
        raise ValueError("invalid ViDoRe run schema_version")
    for artifact_path, key in (
        (metrics_path, "metrics_sha256"),
        (predictions_path, "predictions_sha256"),
        (dataset_manifest_path, "dataset_manifest_sha256"),
    ):
        if sha256(artifact_path) != manifest.get(key):
            raise ValueError(f"ViDoRe {key} mismatch")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("schema_version") != "vidore-retrieval-evaluation-v2":
        raise ValueError("invalid ViDoRe metrics schema_version")
    if set(metrics.get("tasks", {})) != {"docvqa", "infovqa", "tatdqa"}:
        raise ValueError("ViDoRe run does not contain the three frozen tasks")
    for candidate, decision in metrics.get("promotion", {}).items():
        expected = len(decision.get("significant_tasks", [])) >= int(decision.get("required_tasks", 2))
        if bool(decision.get("promotion_pass")) != expected:
            raise ValueError(f"inconsistent ViDoRe promotion decision for {candidate}")
    return {"manifest": manifest, "metrics": metrics}
