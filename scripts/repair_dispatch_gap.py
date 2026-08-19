from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
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
BASE = "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2026/MMSDM_2026_03/MMSDM_Historical_Data_SQLLoader/DATA"
SOURCES = {
    "PRICE": f"{BASE}/PUBLIC_ARCHIVE%23DISPATCHPRICE%23FILE01%23202603010000.zip",
    "REGIONSUM": f"{BASE}/PUBLIC_ARCHIVE%23DISPATCHREGIONSUM%23FILE01%23202603010000.zip",
}


def download(url: str) -> tuple[bytes, dict[str, object]]:
    request = Request(url, headers={"User-Agent": "energy-agent-research/0.1"})
    with urlopen(request, timeout=120) as response:  # fixed official origin
        payload = response.read()
    return payload, {
        "url": url,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def parse_monthly(payload: bytes, table: str, day: str) -> dict[tuple[str, str], dict[str, str]]:
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
                region = record[header["REGIONID"]]
                intervention = record[header["INTERVENTION"]]
                if not interval.startswith(day.replace("-", "/")) or region not in REGIONS or intervention != "0":
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
                        "source_day": day,
                    },
                )
                if table == "PRICE":
                    row["rrp"] = record[header["RRP"]]
                else:
                    row["total_demand_mw"] = record[header["TOTALDEMAND"]]
                    row["available_generation_mw"] = record[header["AVAILABLEGENERATION"]]
                    row["net_interchange_mw"] = record[header["NETINTERCHANGE"]]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--day", default="2026-03-10")
    args = parser.parse_args()
    tables: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    provenance = []
    for table, url in SOURCES.items():
        payload, source = download(url)
        tables[table] = parse_monthly(payload, table, args.day)
        source["table"] = table
        source["selected_rows"] = len(tables[table])
        provenance.append(source)
    replacement = tables["PRICE"]
    for key, values in tables["REGIONSUM"].items():
        replacement.setdefault(key, values).update(
            {field: values[field] for field in ("total_demand_mw", "available_generation_mw", "net_interchange_mw")}
        )
    if len(replacement) != 1440 or any(not row["rrp"] for row in replacement.values()):
        raise SystemExit(f"monthly repair coverage failed: {len(replacement)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    retained = 0
    with (
        gzip.open(args.input, "rt", encoding="utf-8", newline="") as source,
        gzip.open(args.output, "wt", encoding="utf-8", newline="") as target,
    ):
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        for row in reader:
            if row["source_day"] == args.day:
                continue
            writer.writerow(row)
            retained += 1
        writer.writerows(sorted(replacement.values(), key=lambda row: (row["interval"], row["region"])))
    manifest = {
        "status": "complete",
        "repair_day": args.day,
        "retained_rows": retained,
        "replacement_rows": len(replacement),
        "total_rows": retained + len(replacement),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "sources": provenance,
        "boundary": "Missing official daily intervals repaired only from official AEMO monthly MMSDM archive; no interpolation or synthetic fill.",
    }
    repair_path = args.output.parent / "repair_manifest.json"
    repair_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    base_manifest = json.loads((args.input.parent / "run_manifest.json").read_text(encoding="utf-8"))
    final_manifest = {
        **base_manifest,
        "status": "complete",
        "rows": retained + len(replacement),
        "standard_rows": retained + len(replacement),
        "intervention_rows": 0,
        "data_bytes": args.output.stat().st_size,
        "data_sha256": manifest["output_sha256"],
        "original_data_sha256": manifest["input_sha256"],
        "repair_manifest_sha256": hashlib.sha256(repair_path.read_bytes()).hexdigest(),
        "repair_day": args.day,
        "repair_source": "official AEMO monthly MMSDM archive",
    }
    (args.output.parent / "final_manifest.json").write_text(json.dumps(final_manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
