from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

EXPECTED_REGIONS = ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return cast(dict[str, Any], value)


def merge_runs(run_dirs: list[Path], expected_regions: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
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
        for region, run_dir in zip((manifest["selected_regions"][0] for manifest in manifests), run_dirs, strict=True)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regions", default=",".join(EXPECTED_REGIONS))
    args = parser.parse_args()
    expected = tuple(item.strip() for item in args.regions.split(",") if item.strip())
    run_dirs = [args.input_root / region for region in expected]
    metrics, manifest = merge_runs(run_dirs, expected)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    manifest["metrics_sha256"] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
