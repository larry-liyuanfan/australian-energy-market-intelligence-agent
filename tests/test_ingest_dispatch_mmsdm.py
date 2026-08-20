from __future__ import annotations

import csv
import io
import zipfile
from datetime import date

from scripts.ingest_dispatch_mmsdm import months, parse_monthly, source_url


def _archive(table: str, records: list[list[str]]) -> bytes:
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "I",
            "DISPATCH",
            table,
            "1",
            "SETTLEMENTDATE",
            "REGIONID",
            "INTERVENTION",
            "RRP",
            "TOTALDEMAND",
            "AVAILABLEGENERATION",
            "NETINTERCHANGE",
        ]
    )
    writer.writerows(records)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("PUBLIC_DATA.CSV", stream.getvalue())
    return output.getvalue()


def test_month_sequence_and_official_archive_url() -> None:
    assert months(date(2024, 12, 1), date(2025, 2, 28)) == [
        (2024, 12),
        (2025, 1),
        (2025, 2),
    ]
    url = source_url(2024, 8, "DISPATCHPRICE")
    assert "MMSDM_2024_08" in url
    assert "PUBLIC_ARCHIVE%23DISPATCHPRICE%23FILE01%23202408010000.zip" in url


def test_monthly_parser_filters_dates_regions_and_intervention() -> None:
    payload = _archive(
        "PRICE",
        [
            [
                "D",
                "DISPATCH",
                "PRICE",
                "1",
                "2024/08/18 00:05:00",
                "SA1",
                "0",
                "100",
                "",
                "",
                "",
            ],
            [
                "D",
                "DISPATCH",
                "PRICE",
                "1",
                "2024/08/18 00:05:00",
                "SA1",
                "1",
                "999",
                "",
                "",
                "",
            ],
            [
                "D",
                "DISPATCH",
                "PRICE",
                "1",
                "2024/08/17 23:55:00",
                "SA1",
                "0",
                "50",
                "",
                "",
                "",
            ],
            [
                "D",
                "DISPATCH",
                "PRICE",
                "1",
                "2024/08/18 00:05:00",
                "WA1",
                "0",
                "80",
                "",
                "",
                "",
            ],
        ],
    )
    result = parse_monthly(
        payload,
        "PRICE",
        date(2024, 8, 18),
        date(2024, 8, 24),
    )
    assert list(result) == [("2024/08/18 00:05:00", "SA1")]
    assert next(iter(result.values()))["rrp"] == "100"
