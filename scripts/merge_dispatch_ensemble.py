from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from energy_agent.ensemble_gate import summarize_dispatch_ensemble


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--input-template")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-git-sha", required=True)
    parser.add_argument("--merge-git-sha", required=True)
    args = parser.parse_args()
    if (args.input_root is None) == (args.input_template is None):
        parser.error("set exactly one of --input-root or --input-template")
    regions = ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")
    region_payloads: dict[str, dict[str, Any]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for region in regions:
        source = (
            Path(args.input_template.format(region=region))
            if args.input_template is not None
            else args.input_root / region
        )
        metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
        manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["git_sha"] != args.evaluation_git_sha
            or manifest["selected_regions"] != [region]
        ):
            raise SystemExit(f"provenance gate failed for {region}")
        expected_hash = hashlib.sha256((source / "metrics.json").read_bytes()).hexdigest()
        if manifest["metrics_sha256"] != expected_hash:
            raise SystemExit(f"metrics hash gate failed for {region}")
        region_payloads[region] = metrics["regions"][region]
        manifests[region] = manifest
    input_hashes = {manifest["input_sha256"] for manifest in manifests.values()}
    if len(input_hashes) != 1:
        raise SystemExit("input hash mismatch across regions")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = summarize_dispatch_ensemble(region_payloads)
    summary.update(
        {
            "evidence_status": "verified-real-out-of-time-if-run-completes",
            "scope": "AEMO five-region four-season calibration-selected dispatch ensemble gate",
            "boundary": "Historical spot-market net operating proxy for a 1 MW/2 MWh BESS at a user-specified cycling-cost sensitivity. Excludes CAPEX, fixed O&M, network charges, FCAS and investment return.",
        }
    )
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    merged_manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_git_sha": args.evaluation_git_sha,
        "merge_git_sha": args.merge_git_sha,
        "input_sha256": next(iter(input_hashes)),
        "regions": list(regions),
        "source_manifests": manifests,
        "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(merged_manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "aggregate": summary["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
