from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

REGIONS = {"NSW1", "QLD1", "SA1", "TAS1", "VIC1"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    counts: Counter[tuple[str, str]] = Counter()
    intervention_rows = 0
    total_rows = 0
    with gzip.open(args.data, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            total_rows += 1
            if row["intervention"] == "0":
                counts[(row["source_day"], row["region"])] += 1
            else:
                intervention_rows += 1
    original = json.loads(args.manifest.read_text(encoding="utf-8"))
    days = sorted({day for day, _ in counts})
    incomplete = [
        {"day": day, "region": region, "intervals": counts[(day, region)]}
        for day in days
        for region in sorted(REGIONS)
        if counts[(day, region)] != 288
    ]
    validation = {
        "status": "complete"
        if len(days) == original["requested_days"] and not incomplete and original["failed_days"] == 0
        else "incomplete",
        "requested_days": original["requested_days"],
        "observed_days": len(days),
        "standard_rows": sum(counts.values()),
        "intervention_rows": intervention_rows,
        "total_rows": total_rows,
        "incomplete_region_days": incomplete,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
    }
    output = args.manifest.parent / "coverage_validation.json"
    output.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    if validation["status"] != "complete":
        raise SystemExit("coverage validation failed")


if __name__ == "__main__":
    main()
