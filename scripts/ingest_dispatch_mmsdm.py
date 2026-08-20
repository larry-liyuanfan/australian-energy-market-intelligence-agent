from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import time
import zipfile
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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


def months(start: date, end: date) -> list[tuple[int, int]]:
    if end < start:
        raise ValueError("end precedes start")
    current = date(start.year, start.month, 1)
    output = []
    while current <= end:
        output.append((current.year, current.month))
        current = date(current.year + (current.month == 12), current.month % 12 + 1, 1)
    return output


def source_url(year: int, month: int, table: str) -> str:
    stamp = f"{year}{month:02d}010000"
    base = (
        f"https://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/{year}/"
        f"MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/DATA"
    )
    return f"{base}/PUBLIC_ARCHIVE%23{table}%23FILE01%23{stamp}.zip"


def download(url: str, retries: int = 3) -> tuple[bytes, dict[str, Any]]:
    last_error = ""
    for attempt in range(1, retries + 1):
        started = time.perf_counter()
        try:
            request = Request(url, headers={"User-Agent": "energy-agent-research/0.1"})
            with urlopen(request, timeout=120) as response:  # fixed official origin
                payload = response.read()
            return payload, {
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
    raise RuntimeError(f"download failed: {last_error}")


def parse_monthly(
    payload: bytes,
    table: str,
    start: date,
    end: date,
) -> dict[tuple[str, str], dict[str, str]]:
    output: dict[tuple[str, str], dict[str, str]] = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            header: dict[str, int] = {}
            rows = csv.reader(io.StringIO(archive.read(name).decode("utf-8-sig", errors="replace")))
            for record in rows:
                if len(record) < 4 or record[1:3] != ["DISPATCH", table]:
                    continue
                if record[0] == "I":
                    header = {field: index for index, field in enumerate(record)}
                    continue
                if record[0] != "D" or not header:
                    continue
                interval = record[header["SETTLEMENTDATE"]]
                interval_day = date.fromisoformat(interval[:10].replace("/", "-"))
                region = record[header["REGIONID"]]
                intervention = record[header["INTERVENTION"]]
                if not start <= interval_day <= end or region not in REGIONS or intervention != "0":
                    continue
                row = output.setdefault(
                    (interval, region),
                    {
                        "interval": interval,
                        "region": region,
                        "rrp": "",
                        "total_demand_mw": "",
                        "available_generation_mw": "",
                        "net_interchange_mw": "",
                        "intervention": "0",
                        "source_day": interval_day.isoformat(),
                    },
                )
                if table == "PRICE":
                    row["rrp"] = record[header["RRP"]]
                else:
                    row["total_demand_mw"] = record[header["TOTALDEMAND"]]
                    row["available_generation_mw"] = record[header["AVAILABLEGENERATION"]]
                    row["net_interchange_mw"] = record[header["NETINTERCHANGE"]]
    return output


def run(start: date, end: date, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    combined: dict[tuple[str, str], dict[str, str]] = {}
    provenance: list[dict[str, Any]] = []
    for year, month in months(start, end):
        monthly: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
        for archive_table, table in (
            ("DISPATCHPRICE", "PRICE"),
            ("DISPATCHREGIONSUM", "REGIONSUM"),
        ):
            url = source_url(year, month, archive_table)
            payload, source = download(url)
            monthly[table] = parse_monthly(payload, table, start, end)
            source.update(
                {
                    "year": year,
                    "month": month,
                    "archive_table": archive_table,
                    "table": table,
                    "selected_rows": len(monthly[table]),
                }
            )
            provenance.append(source)
        for key, row in monthly["PRICE"].items():
            combined[key] = row
        for key, values in monthly["REGIONSUM"].items():
            combined.setdefault(key, values).update(
                {
                    field: values[field]
                    for field in (
                        "total_demand_mw",
                        "available_generation_mw",
                        "net_interchange_mw",
                    )
                }
            )
    data_path = output / "dispatch_features.csv.gz"
    with gzip.open(data_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(combined.values(), key=lambda row: (row["interval"], row["region"])))
    provenance_path = output / "source_provenance.jsonl"
    provenance_path.write_text("".join(json.dumps(row) + "\n" for row in provenance), encoding="utf-8")
    counts = Counter((row["source_day"], row["region"]) for row in combined.values())
    requested_days = (end - start).days + 1
    incomplete = [
        {"day": day.isoformat(), "region": region, "intervals": counts[(day.isoformat(), region)]}
        for offset in range(requested_days)
        for day in [date.fromordinal(start.toordinal() + offset)]
        for region in sorted(REGIONS)
        if counts[(day.isoformat(), region)] != 288
    ]
    complete_rows = [
        row
        for row in combined.values()
        if row["rrp"]
        and row["total_demand_mw"]
        and row["available_generation_mw"]
        and row["net_interchange_mw"]
    ]
    manifest = {
        "status": "complete" if not incomplete and len(complete_rows) == len(combined) else "incomplete",
        "created_at": datetime.now(UTC).isoformat(),
        "source_kind": "official AEMO monthly MMSDM archive",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "requested_days": requested_days,
        "rows": len(combined),
        "standard_rows": len(combined),
        "regions": sorted(REGIONS),
        "fields": list(FIELDS),
        "incomplete_region_days": incomplete,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        "data_bytes": data_path.stat().st_size,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "source_terms_boundary": "AEMO source terms apply; monthly archives were streamed and not retained.",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if manifest["status"] != "complete":
        raise SystemExit("monthly MMSDM coverage gate failed")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.start, args.end, args.output)


if __name__ == "__main__":
    main()
