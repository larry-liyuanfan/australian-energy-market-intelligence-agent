from __future__ import annotations

import csv
import hashlib
import io
import json
import statistics
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen

from .schemas import Evidence, Region


@dataclass(frozen=True)
class MarketRow:
    interval: datetime
    region: Region
    rrp: float
    demand_mw: float
    available_mw: float | None = None
    net_interchange_mw: float | None = None
    intervention: bool = False


class MarketStore:
    def __init__(
        self, rows: list[MarketRow], evidence: list[Evidence] | None = None, data_version: str = "fixture-v1"
    ) -> None:
        self.rows = sorted(rows, key=lambda row: row.interval)
        self.evidence = evidence or []
        self.data_version = data_version

    def select(self, region: Region, start: datetime, end: datetime) -> list[MarketRow]:
        return [row for row in self.rows if row.region == region and start <= row.interval <= end]

    def closest(self, region: Region, at: datetime) -> MarketRow | None:
        rows = [row for row in self.rows if row.region == region]
        return min(rows, key=lambda row: abs((row.interval - at).total_seconds())) if rows else None

    def coverage(self) -> dict[str, object]:
        if not self.rows:
            return {"rows": 0, "regions": []}
        return {
            "rows": len(self.rows),
            "regions": sorted({row.region.value for row in self.rows}),
            "start": self.rows[0].interval.isoformat(),
            "end": self.rows[-1].interval.isoformat(),
            "data_version": self.data_version,
        }


def fixture_store() -> MarketStore:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows: list[MarketRow] = []
    for r_idx, region in enumerate(Region):
        for i in range(576):
            price = 55 + 25 * __import__("math").sin(2 * __import__("math").pi * i / 288) + r_idx * 4
            if i in {140, 141} and region == Region.SA1:
                price = 6000 + 500 * (i - 140)
            rows.append(
                MarketRow(
                    start + timedelta(minutes=5 * i),
                    region,
                    price,
                    5000 + r_idx * 800,
                    6200 + r_idx * 900,
                    20 - r_idx * 10,
                )
            )
    body = b"Synthetic fixture for contract and fault tests only."
    evidence = [
        Evidence(
            evidence_id="fixture-boundary",
            title="Synthetic fixture boundary",
            url="https://example.invalid/fixture",
            retrieved_at=datetime.now(UTC),
            sha256=hashlib.sha256(body).hexdigest(),
            snippet=body.decode(),
            evidence_type="explanatory",
        )
    ]
    return MarketStore(rows, evidence, "synthetic-fixture-v1")


def download_dispatch_day(day: str, output: Path) -> dict[str, object]:
    """Download an official AEMO NEMWeb daily DispatchIS archive without embedding it in git."""
    url = f"https://nemweb.com.au/Reports/Archive/DispatchIS_Reports/PUBLIC_DISPATCHIS_{day}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response:
        payload = response.read()
    output.write_bytes(payload)
    return {
        "url": url,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def parse_dispatch_archive(path: Path) -> list[MarketRow]:
    rows: list[MarketRow] = []
    with zipfile.ZipFile(path) as outer:
        for name in outer.namelist():
            data = outer.read(name)
            files = [(name, data)]
            if name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(data)) as inner:
                    files = [(inner_name, inner.read(inner_name)) for inner_name in inner.namelist()]
            for _, raw in files:
                text = raw.decode("utf-8-sig", errors="replace")
                price_header: dict[str, int] = {}
                for record in csv.reader(io.StringIO(text)):
                    if len(record) >= 5 and record[0] == "I" and record[1:3] == ["DISPATCH", "PRICE"]:
                        price_header = {name: index for index, name in enumerate(record)}
                        continue
                    if not price_header or len(record) < 10 or record[0] != "D" or record[1:3] != ["DISPATCH", "PRICE"]:
                        continue
                    try:
                        interval = datetime.strptime(
                            record[price_header["SETTLEMENTDATE"]], "%Y/%m/%d %H:%M:%S"
                        ).replace(tzinfo=timezone(timedelta(hours=10)))
                        region = Region(record[price_header["REGIONID"]])
                        rrp = float(record[price_header["RRP"]])
                        intervention = record[price_header["INTERVENTION"]] != "0"
                        rows.append(MarketRow(interval, region, rrp, 0.0, intervention=intervention))
                    except (KeyError, ValueError, IndexError):
                        continue
    return rows


def robust_events(rows: list[MarketRow], threshold: float, z_threshold: float) -> list[dict[str, object]]:
    if not rows:
        return []
    values = [row.rrp for row in rows]
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values) or 1.0
    return [
        {"interval": row.interval.isoformat(), "rrp": row.rrp, "robust_z": 0.6745 * (row.rrp - median) / mad}
        for row in rows
        if row.rrp >= threshold or abs(0.6745 * (row.rrp - median) / mad) >= z_threshold
    ]


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
