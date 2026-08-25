from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .schemas import Region


@dataclass(frozen=True)
class ForecastSnapshot:
    """Immutable offline forecast published for one region/window."""

    region: Region
    start: datetime
    end: datetime
    training_cutoff: datetime
    created_at: datetime
    data_sha256: str
    model_sha256: str
    model_name: str
    point: list[float]
    lower: list[float]
    upper: list[float]

    def __post_init__(self) -> None:
        if not self.start < self.end or self.training_cutoff > self.start:
            raise ValueError("snapshot timestamps violate the as-of contract")
        if not self.point or not (len(self.point) == len(self.lower) == len(self.upper)):
            raise ValueError("snapshot forecast arrays must be aligned and non-empty")
        for digest in (self.data_sha256, self.model_sha256):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("snapshot hashes must be lowercase SHA-256")

    @property
    def snapshot_id(self) -> str:
        return (
            f"{self.region.value}:{self.start.isoformat()}:{self.end.isoformat()}:"
            f"{self.model_sha256[:12]}"
        )


class ForecastSnapshotStore:
    def __init__(self, snapshots: list[ForecastSnapshot] | None = None) -> None:
        self._snapshots = {
            (snapshot.region, snapshot.start.isoformat(), snapshot.end.isoformat()): snapshot
            for snapshot in snapshots or []
        }

    def get(self, region: Region, start: datetime, end: datetime) -> ForecastSnapshot | None:
        return self._snapshots.get((region, start.isoformat(), end.isoformat()))


def load_forecast_snapshots(path: Path) -> ForecastSnapshotStore:
    snapshots: list[ForecastSnapshot] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        snapshots.append(
            ForecastSnapshot(
                region=Region(row["region"]),
                start=datetime.fromisoformat(row["start"]),
                end=datetime.fromisoformat(row["end"]),
                training_cutoff=datetime.fromisoformat(row["training_cutoff"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                data_sha256=str(row["data_sha256"]),
                model_sha256=str(row["model_sha256"]),
                model_name=str(row["model_name"]),
                point=[float(value) for value in row["point"]],
                lower=[float(value) for value in row["lower"]],
                upper=[float(value) for value in row["upper"]],
            )
        )
    return ForecastSnapshotStore(snapshots)
