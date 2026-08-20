from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from energy_agent.merge import merge_evaluation_runs

EXPECTED_REGIONS = ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regions", default=",".join(EXPECTED_REGIONS))
    args = parser.parse_args()
    expected = tuple(item.strip() for item in args.regions.split(",") if item.strip())
    run_dirs = [args.input_root / region for region in expected]
    metrics, manifest = merge_evaluation_runs(run_dirs, expected)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    manifest["metrics_sha256"] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    manifest["merge_git_sha"] = os.environ.get("ENERGY_GIT_COMMIT", "unversioned-local-run")
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
