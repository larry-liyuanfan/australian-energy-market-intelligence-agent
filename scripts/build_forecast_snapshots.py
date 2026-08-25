from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from energy_agent.forecast import seasonal_conformal
from energy_agent.market import load_dispatch_store
from energy_agent.schemas import Region


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish immutable as-of forecast snapshots for decision replay")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--days", required=True, help="Comma-separated NEM calendar days")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    store = load_dispatch_store(args.data, args.data_manifest)
    manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    data_sha256 = str(manifest["data_sha256"])
    model_contract = {
        "model_name": "seasonal_split_conformal_snapshot_v1",
        "history_intervals": 30 * 288,
        "horizon_intervals": 288,
        "alpha": 0.1,
        "season": 288,
    }
    model_sha256 = hashlib.sha256(
        json.dumps(model_contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    nem_time = timezone(timedelta(hours=10))
    rows: list[dict[str, object]] = []
    for raw_day in (item.strip() for item in args.days.split(",") if item.strip()):
        start = datetime.fromisoformat(raw_day).replace(tzinfo=nem_time)
        end = start + timedelta(days=1)
        for region in Region:
            history = store.before(region, start, limit=30 * 288)
            forecast = seasonal_conformal([row.rrp for row in history], 288, 0.1)
            rows.append(
                {
                    "region": region.value,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "training_cutoff": start.isoformat(),
                    "created_at": datetime.now(UTC).isoformat(),
                    "data_sha256": data_sha256,
                    "model_sha256": model_sha256,
                    "model_name": forecast.method,
                    "point": forecast.point,
                    "lower": forecast.lower,
                    "upper": forecast.upper,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"snapshots": len(rows), "model_sha256": model_sha256, "data_sha256": data_sha256}))


if __name__ == "__main__":
    main()
