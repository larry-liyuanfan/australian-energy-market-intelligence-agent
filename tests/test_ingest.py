from datetime import date
from pathlib import Path

from energy_agent.market import parse_dispatch_archive


def test_real_preflight_archive_has_complete_price_grid() -> None:
    path = Path("data/raw/PUBLIC_DISPATCHIS_20260817.zip")
    if not path.exists():
        return
    rows = parse_dispatch_archive(path)
    assert len(rows) == 1440
    assert {row.region.value for row in rows} == {"NSW1", "QLD1", "SA1", "TAS1", "VIC1"}
    assert {row.interval.date() for row in rows} <= {date(2026, 8, 17), date(2026, 8, 18)}
