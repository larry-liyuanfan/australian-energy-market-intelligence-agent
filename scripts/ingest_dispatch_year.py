from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import time
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REGIONS = {"NSW1", "QLD1", "SA1", "TAS1", "VIC1"}
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


def days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def download(day: date, retries: int = 3) -> tuple[bytes, dict[str, Any]]:
    stamp = day.strftime("%Y%m%d")
    url = f"https://nemweb.com.au/Reports/Archive/DispatchIS_Reports/PUBLIC_DISPATCHIS_{stamp}.zip"
    last_error = ""
    for attempt in range(1, retries + 1):
        started = time.perf_counter()
        try:
            request = Request(url, headers={"User-Agent": "energy-agent-research/0.1"})
            with urlopen(request, timeout=90) as response:  # fixed official origin
                payload = response.read()
            return payload, {
                "day": day.isoformat(),
                "url": url,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "download_seconds": round(time.perf_counter() - started, 3),
                "attempt": attempt,
            }
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"download failed for {day}: {last_error}")


def _value(record: list[str], header: dict[str, int], field: str, default: str = "") -> str:
    index = header.get(field)
    return record[index] if index is not None and index < len(record) else default


def parse(payload: bytes, source_day: date) -> list[dict[str, str]]:
    combined: dict[tuple[str, str, str], dict[str, str]] = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        for outer_name in outer.namelist():
            outer_bytes = outer.read(outer_name)
            if not outer_name.lower().endswith(".zip"):
                continue
            with zipfile.ZipFile(io.BytesIO(outer_bytes)) as inner:
                for inner_name in inner.namelist():
                    headers: dict[str, dict[str, int]] = {}
                    text = inner.read(inner_name).decode("utf-8-sig", errors="replace")
                    for record in csv.reader(io.StringIO(text)):
                        if len(record) < 4 or record[1] != "DISPATCH":
                            continue
                        table = record[2]
                        if record[0] == "I" and table in {"PRICE", "REGIONSUM"}:
                            headers[table] = {name: index for index, name in enumerate(record)}
                            continue
                        if record[0] != "D" or table not in headers:
                            continue
                        header = headers[table]
                        region = _value(record, header, "REGIONID")
                        if region not in REGIONS:
                            continue
                        interval = _value(record, header, "SETTLEMENTDATE")
                        intervention = _value(record, header, "INTERVENTION", "0")
                        key = (interval, region, intervention)
                        row = combined.setdefault(
                            key,
                            {
                                "interval": interval,
                                "region": region,
                                "rrp": "",
                                "total_demand_mw": "",
                                "available_generation_mw": "",
                                "net_interchange_mw": "",
                                "intervention": intervention,
                                "source_day": source_day.isoformat(),
                            },
                        )
                        if table == "PRICE":
                            row["rrp"] = _value(record, header, "RRP")
                        else:
                            row["total_demand_mw"] = _value(record, header, "TOTALDEMAND")
                            row["available_generation_mw"] = _value(record, header, "AVAILABLEGENERATION")
                            row["net_interchange_mw"] = _value(record, header, "NETINTERCHANGE")
    return [row for row in combined.values() if row["rrp"]]


def run(start: date, end: date, output: Path, max_failures: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "dispatch_features.csv.gz"
    provenance_path = output / "source_provenance.jsonl"
    errors_path = output / "errors.jsonl"
    started = time.perf_counter()
    row_count = 0
    failures: list[dict[str, str]] = []
    provenance: list[dict[str, Any]] = []
    with gzip.open(data_path, "wt", newline="", encoding="utf-8") as data_file:
        writer = csv.DictWriter(data_file, fieldnames=FIELDS)
        writer.writeheader()
        for day in days(start, end):
            try:
                payload, source = download(day)
                parsed = parse(payload, day)
                writer.writerows(parsed)
                source["parsed_rows"] = len(parsed)
                source["expected_rows"] = 1440
                source["complete"] = len(parsed) == 1440
                provenance.append(source)
                row_count += len(parsed)
                print(json.dumps({"day": day.isoformat(), "rows": len(parsed), "total_rows": row_count}), flush=True)
            except Exception as exc:  # fail is recorded and bounded at the day boundary
                failure = {"day": day.isoformat(), "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                print(json.dumps(failure), flush=True)
                if len(failures) > max_failures:
                    break
    provenance_path.write_text("".join(json.dumps(row) + "\n" for row in provenance), encoding="utf-8")
    errors_path.write_text("".join(json.dumps(row) + "\n" for row in failures), encoding="utf-8")
    requested_days = len(days(start, end))
    manifest = {
        "status": "complete" if not failures and len(provenance) == requested_days else "incomplete",
        "created_at": datetime.now(UTC).isoformat(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "requested_days": requested_days,
        "completed_days": len(provenance),
        "complete_1440_row_days": sum(bool(row["complete"]) for row in provenance),
        "failed_days": len(failures),
        "rows": row_count,
        "regions": sorted(REGIONS),
        "fields": list(FIELDS),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        "data_bytes": data_path.stat().st_size,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "source_terms_boundary": "AEMO source terms apply; raw daily archives were streamed and not retained.",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if manifest["status"] != "complete":
        raise SystemExit(f"coverage gate failed: {manifest}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-failures", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.start, args.end, args.output, args.max_failures), indent=2))


if __name__ == "__main__":
    main()
