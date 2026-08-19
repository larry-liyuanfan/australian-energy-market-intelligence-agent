from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile an ingest manifest after an official gap repair.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if digest != manifest["data_sha256"]:
        raise SystemExit("data hash does not match final manifest")
    if coverage["status"] != "complete" or coverage["incomplete_region_days"]:
        raise SystemExit("coverage validation is not complete")
    expected_rows = int(manifest["requested_days"]) * 288 * len(manifest["regions"])
    if (
        int(coverage["observed_days"]) != int(manifest["requested_days"])
        or int(manifest["rows"]) != expected_rows
        or int(coverage["total_rows"]) != expected_rows
    ):
        raise SystemExit("row count does not match requested day-region grid")
    manifest["complete_1440_row_days"] = int(manifest["requested_days"])
    manifest["manifest_reconciled_from_coverage"] = args.coverage.name
    temporary = args.manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.manifest)
    print(json.dumps({"status": "complete", "days": manifest["complete_1440_row_days"], "data_sha256": digest}))


if __name__ == "__main__":
    main()
