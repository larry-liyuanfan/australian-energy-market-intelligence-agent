from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

FIELDS = (
    "interval",
    "region",
    "rrp",
    "total_demand_mw",
    "available_generation_mw",
    "net_interchange_mw",
    "intervention",
    "source_day",
)
REGIONS = {"NSW1", "QLD1", "SA1", "TAS1", "VIC1"}


def load_unique(paths: list[Path]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    sources: list[dict[str, Any]] = []
    for path in paths:
        source_rows = 0
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"schema mismatch: {path}")
            for row in reader:
                key = (row["interval"], row["region"], row["intervention"])
                if key in seen:
                    raise ValueError(f"duplicate dispatch key: {key}")
                seen.add(key)
                rows.append(row)
                source_rows += 1
        sources.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "rows": source_rows,
            }
        )
    rows.sort(key=lambda row: (row["interval"], row["region"], row["intervention"]))
    return rows, sources


def run(paths: list[Path], start: date, end: date, output: Path) -> dict[str, Any]:
    if end < start or len(paths) < 2:
        raise ValueError("ordered date range and at least two inputs are required")
    rows, sources = load_unique(paths)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "dispatch_features.csv.gz"
    with gzip.open(data_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(
        (row["source_day"], row["region"])
        for row in rows
        if row["intervention"] == "0"
    )
    requested_days = (end - start).days + 1
    incomplete = [
        {"day": day.isoformat(), "region": region, "intervals": counts[(day.isoformat(), region)]}
        for offset in range(requested_days)
        for day in [date.fromordinal(start.toordinal() + offset)]
        for region in sorted(REGIONS)
        if counts[(day.isoformat(), region)] != 288
    ]
    unexpected_days = sorted(
        {day for day, _ in counts if day < start.isoformat() or day > end.isoformat()}
    )
    manifest = {
        "status": "complete" if not incomplete and not unexpected_days else "incomplete",
        "created_at": datetime.now(UTC).isoformat(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "requested_days": requested_days,
        "standard_rows": sum(counts.values()),
        "total_rows": len(rows),
        "regions": sorted(REGIONS),
        "incomplete_region_days": incomplete,
        "unexpected_days": unexpected_days,
        "sources": sources,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "data_bytes": data_path.stat().st_size,
        "boundary": "Official AEMO periods are concatenated without interpolation or synthetic fill.",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if manifest["status"] != "complete":
        raise SystemExit("merged period coverage gate failed")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.start, args.end, args.output)


if __name__ == "__main__":
    main()
