from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

REGIONS = {"NSW1", "QLD1", "SA1", "TAS1", "VIC1"}


def time_axis_issues(rows: list[dict[str, str]]) -> dict[str, object]:
    keys = Counter(
        (row["interval"], row["region"], row["intervention"])
        for row in rows
        if row["intervention"] == "0"
    )
    duplicates = [
        {"interval": key[0], "region": key[1], "count": count}
        for key, count in sorted(keys.items())
        if count > 1
    ]
    gaps = []
    for region in sorted(REGIONS):
        timestamps = sorted(
            {
                datetime.strptime(row["interval"], "%Y/%m/%d %H:%M:%S").replace(tzinfo=UTC)
                for row in rows
                if row["region"] == region and row["intervention"] == "0"
            }
        )
        for previous, current in itertools.pairwise(timestamps):
            if current - previous != timedelta(minutes=5):
                gaps.append(
                    {
                        "region": region,
                        "previous": previous.isoformat(),
                        "current": current.isoformat(),
                        "delta_seconds": int((current - previous).total_seconds()),
                    }
                )
    return {
        "duplicate_key_count": len(duplicates),
        "duplicate_keys": duplicates[:100],
        "gap_count": len(gaps),
        "gaps": gaps[:100],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    counts: Counter[tuple[str, str]] = Counter()
    intervention_rows = 0
    total_rows = 0
    all_rows: list[dict[str, str]] = []
    with gzip.open(args.data, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            all_rows.append(row)
            total_rows += 1
            if row["intervention"] == "0":
                counts[(row["source_day"], row["region"])] += 1
            else:
                intervention_rows += 1
    original = json.loads(args.manifest.read_text(encoding="utf-8"))
    time_axis = time_axis_issues(all_rows)
    days = sorted({day for day, _ in counts})
    incomplete = [
        {"day": day, "region": region, "intervals": counts[(day, region)]}
        for day in days
        for region in sorted(REGIONS)
        if counts[(day, region)] != 288
    ]
    validation = {
        "status": "complete"
        if len(days) == original["requested_days"]
        and not incomplete
        and original["failed_days"] == 0
        and time_axis["duplicate_key_count"] == 0
        and time_axis["gap_count"] == 0
        else "incomplete",
        "requested_days": original["requested_days"],
        "observed_days": len(days),
        "standard_rows": sum(counts.values()),
        "intervention_rows": intervention_rows,
        "total_rows": total_rows,
        "incomplete_region_days": incomplete,
        "time_axis": time_axis,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
    }
    output = args.output or args.manifest.parent / "coverage_validation.json"
    output.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    if validation["status"] != "complete":
        raise SystemExit("coverage validation failed")


if __name__ == "__main__":
    main()
