from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

from scripts.merge_dispatch_periods import FIELDS, load_unique


def _write(path: Path, interval: str, region: str = "SA1") -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "interval": interval,
                "region": region,
                "rrp": "100",
                "total_demand_mw": "1000",
                "available_generation_mw": "1200",
                "net_interchange_mw": "0",
                "intervention": "0",
                "source_day": interval[:10].replace("/", "-"),
            }
        )


def test_merge_sorts_adjacent_periods_and_records_sources(tmp_path: Path) -> None:
    later, earlier = tmp_path / "later.csv.gz", tmp_path / "earlier.csv.gz"
    _write(later, "2025/08/18 00:05:00")
    _write(earlier, "2024/08/18 00:05:00")
    rows, sources = load_unique([later, earlier])
    assert [row["interval"] for row in rows] == [
        "2024/08/18 00:05:00",
        "2025/08/18 00:05:00",
    ]
    assert len(sources) == 2 and all(source["sha256"] for source in sources)


def test_merge_rejects_duplicate_dispatch_keys(tmp_path: Path) -> None:
    first, second = tmp_path / "first.csv.gz", tmp_path / "second.csv.gz"
    _write(first, "2024/08/18 00:05:00")
    _write(second, "2024/08/18 00:05:00")
    with pytest.raises(ValueError, match="duplicate dispatch key"):
        load_unique([first, second])
