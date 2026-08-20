from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.merge_real_market_array import merge_runs


def _write_shard(root: Path, region: str, *, git_sha: str = "a" * 40) -> Path:
    shard = root / region
    shard.mkdir()
    (shard / "metrics.json").write_text(
        json.dumps(
            {
                "scope": "real AEMO NEMWeb rolling test",
                "coverage": {name: 1.0 for name in ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")},
                "regions": {region: {"region": region, "bess": {"test_days": 53}}},
            }
        ),
        encoding="utf-8",
    )
    (shard / "run_manifest.json").write_text(
        json.dumps(
            {
                "git_sha": git_sha,
                "input_sha256": "b" * 64,
                "selected_regions": [region],
                "degradation_costs_aud_per_mwh_discharged": [0.0, 100.0],
                "elapsed_seconds": 2.5,
            }
        ),
        encoding="utf-8",
    )
    return shard


def test_merge_requires_and_combines_all_region_shards(tmp_path: Path) -> None:
    regions = ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")
    shards = [_write_shard(tmp_path, region) for region in regions]
    metrics, manifest = merge_runs(shards, regions)
    assert set(metrics["regions"]) == set(regions)
    assert manifest["component_elapsed_seconds_sum"] == 12.5
    assert manifest["component_elapsed_seconds_max"] == 2.5
    assert set(manifest["component_manifest_sha256"]) == set(regions)


def test_merge_fails_closed_on_inconsistent_git_sha(tmp_path: Path) -> None:
    regions = ("NSW1", "QLD1")
    shards = [_write_shard(tmp_path, "NSW1"), _write_shard(tmp_path, "QLD1", git_sha="c" * 40)]
    with pytest.raises(ValueError, match="inconsistent git_sha"):
        merge_runs(shards, regions)
