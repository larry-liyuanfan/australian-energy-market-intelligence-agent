"""Reusable evaluation parameter parsing and sharded-run quality gates."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


def parse_degradation_costs(value: str) -> tuple[float, ...]:
    """Parse comma-separated CLI values or Slurm-safe colon/semicolon values."""

    costs = tuple(float(item) for item in re.split(r"[,;:]", value) if item.strip())
    if not costs or any(item < 0 for item in costs):
        raise ValueError("degradation costs must be a non-empty list of non-negative numbers")
    return costs


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return cast(dict[str, Any], value)


def merge_evaluation_runs(
    run_dirs: list[Path], expected_regions: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge single-region artifacts only when provenance fields agree."""

    if not run_dirs:
        raise ValueError("at least one run directory is required")
    manifests: list[dict[str, Any]] = []
    merged_regions: dict[str, Any] = {}
    coverage: dict[str, float] | None = None
    scope: str | None = None
    for run_dir in run_dirs:
        metrics = _read_json(run_dir / "metrics.json")
        manifest = _read_json(run_dir / "run_manifest.json")
        selected = tuple(manifest.get("selected_regions", ()))
        if len(selected) != 1 or set(metrics.get("regions", {})) != set(selected):
            raise ValueError(f"{run_dir} is not a single-region shard: {selected}")
        region = selected[0]
        if region in merged_regions:
            raise ValueError(f"duplicate region shard: {region}")
        merged_regions[region] = metrics["regions"][region]
        manifests.append(manifest)
        coverage = metrics["coverage"] if coverage is None else coverage
        scope = metrics["scope"] if scope is None else scope
        if metrics["coverage"] != coverage or metrics["scope"] != scope:
            raise ValueError(f"inconsistent metric metadata in {run_dir}")

    actual_regions = tuple(sorted(merged_regions))
    if set(actual_regions) != set(expected_regions):
        raise ValueError(f"region gate failed: expected {sorted(expected_regions)}, got {list(actual_regions)}")
    for field in ("git_sha", "input_sha256", "degradation_costs_aud_per_mwh_discharged"):
        values = {json.dumps(manifest.get(field), sort_keys=True) for manifest in manifests}
        if len(values) != 1:
            raise ValueError(f"inconsistent {field}: {sorted(values)}")

    metrics = {"scope": scope, "coverage": coverage, "regions": merged_regions}
    component_manifest_sha256 = {
        region: hashlib.sha256((run_dir / "run_manifest.json").read_bytes()).hexdigest()
        for region, run_dir in zip(
            (manifest["selected_regions"][0] for manifest in manifests), run_dirs, strict=True
        )
    }
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": manifests[0]["git_sha"],
        "input_sha256": manifests[0]["input_sha256"],
        "regions": sorted(merged_regions),
        "degradation_costs_aud_per_mwh_discharged": manifests[0][
            "degradation_costs_aud_per_mwh_discharged"
        ],
        "component_manifest_sha256": component_manifest_sha256,
        "component_elapsed_seconds_sum": round(sum(float(item["elapsed_seconds"]) for item in manifests), 3),
        "component_elapsed_seconds_max": round(max(float(item["elapsed_seconds"]) for item in manifests), 3),
        "provider_cost_usd": 0.0,
        "merge_gate": "five unique regions; identical git SHA, input SHA256, degradation costs, scope and coverage",
    }
    return metrics, manifest
