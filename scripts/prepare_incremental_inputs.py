"""CPU Slurm-only streaming extraction of private November inputs; no model inference."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from energy_agent.forecast import seasonal_conformal


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta, template = [json.loads(p.read_text()) for p in (args.manifest, args.template)]
    parent = sha(args.market)
    if parent != meta["data_sha256"] or parent != template["source_sha256"]:
        raise ValueError("input parent mismatch")
    regions = ["SA1", "VIC1", "QLD1", "TAS1"]
    retained: dict[str, list[dict[str, str]]] = {r: [] for r in regions}
    with gzip.open(args.market, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["region"] in retained and "2025/10/19" <= row["interval"] < "2025/11/20":
                retained[row["region"]].append(row)
    selected: list[dict[str, str]] = []
    snapshots: list[dict[str, Any]] = []
    tz = timezone(timedelta(hours=10))
    model_sha = sha(Path(__file__).parents[1] / "src/energy_agent/forecast.py")
    for region, rows in retained.items():
        rows.sort(key=lambda row: row["interval"])
        for day in (18, 19):
            start = datetime(2025, 11, day, tzinfo=tz)
            end = start+timedelta(days=1)
            lo, hi = start.strftime("%Y/%m/%d %H:%M:%S"), end.strftime("%Y/%m/%d %H:%M:%S")
            actual = [row for row in rows if lo <= row["interval"] < hi]
            if len(actual) != 288 or any(row["interval"] != (start+timedelta(minutes=5*i)).strftime("%Y/%m/%d %H:%M:%S") for i, row in enumerate(actual)):
                raise ValueError("non-unique/incomplete November window")
            history = [float(row["rrp"]) for row in rows if row["interval"] < lo][-30*288:]
            if len(history) != 30*288:
                raise ValueError("incomplete as-of history")
            forecast = seasonal_conformal(history, 288)
            selected.extend(actual)
            snapshots.append({"region": region, "start": start.isoformat(), "end": end.isoformat(),
                "training_cutoff": start.isoformat(), "created_at": datetime.now(UTC).isoformat(),
                "data_sha256": parent, "model_sha256": model_sha, "model_name": forecast.method,
                "point": forecast.point, "lower": forecast.lower, "upper": forecast.upper})
    args.output.mkdir(parents=True, exist_ok=False)
    subset = {"schema_version": "energy-private-market-subset-v1", "data_track": "official_aemo_archival_subset",
              "source_sha256": parent, "rows": selected, "numeric_evidence": template["numeric_evidence"],
              "regions": regions, "dates": ["2025-11-18", "2025-11-19"]}
    data, forecasts = args.output/"november-market.json", args.output/"november-snapshots.jsonl"
    data.write_text(json.dumps(subset, separators=(",", ":")), encoding="utf-8")
    forecasts.write_text("\n".join(json.dumps(row) for row in snapshots)+"\n", encoding="utf-8")
    manifest = {"rows": len(selected), "snapshots": len(snapshots), "history_per_snapshot": 30*288,
                "parent_sha256": parent, "data_sha256": sha(data), "snapshots_sha256": sha(forecasts),
                "forecast_code_sha256": model_sha, "model_inference": False, "private_sources_not_for_git": True}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
