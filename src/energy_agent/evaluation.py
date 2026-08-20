"""Reusable evaluation parameter parsing and sharded-run quality gates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np


@dataclass(frozen=True)
class SeasonalFold:
    name: str
    calibration_start: datetime
    test_start: datetime
    test_end: datetime


@dataclass(frozen=True)
class AdaptiveIntervalResult:
    """Sequential conformal bounds and the pre-observation miscoverage levels."""

    lower: list[float]
    upper: list[float]
    alpha_history: list[float]


def adaptive_conformal_bounds(
    actual: np.ndarray,
    predicted: np.ndarray,
    calibration_scores: np.ndarray,
    *,
    target_alpha: float = 0.1,
    gamma: float = 0.005,
    window: int = 30 * 288,
) -> AdaptiveIntervalResult:
    """Build leakage-safe online intervals with an adaptive miscoverage controller.

    This is a compact method-inspired implementation of Adaptive Conformal
    Inference: the quantile level is adjusted after each observed miss. Test
    labels are never used before their interval is emitted. A bounded score
    window lets the conformal radius respond to market-regime changes.
    """

    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    scores = [float(value) for value in np.asarray(calibration_scores, dtype=float)]
    if actual_values.shape != predicted_values.shape:
        raise ValueError("actual and predicted must have identical shapes")
    if not scores:
        raise ValueError("calibration_scores must not be empty")
    if not 0 < target_alpha < 1:
        raise ValueError("target_alpha must be between zero and one")
    if gamma <= 0 or window < 2:
        raise ValueError("gamma and window must be positive")

    adaptive_alpha = target_alpha
    lower: list[float] = []
    upper: list[float] = []
    alpha_history: list[float] = []
    for observed, point in zip(actual_values, predicted_values, strict=True):
        history = scores[-window:]
        minimum_alpha = 1 / (len(history) + 1)
        effective_alpha = min(1 - minimum_alpha, max(minimum_alpha, adaptive_alpha))
        quantile_level = min(
            1.0,
            math.ceil((len(history) + 1) * (1 - effective_alpha)) / len(history),
        )
        radius = float(np.quantile(history, quantile_level, method="higher"))
        lo, hi = float(point - radius), float(point + radius)
        lower.append(lo)
        upper.append(hi)
        alpha_history.append(effective_alpha)
        error = float(not (lo <= observed <= hi))
        adaptive_alpha += gamma * (target_alpha - error)
        scores.append(abs(float(observed - point)))
    return AdaptiveIntervalResult(lower, upper, alpha_history)


def citation_structure_metrics(answer: str, valid_evidence_ids: set[str]) -> dict[str, float | int]:
    """Measure claim-level citation presence and ID validity without claiming entailment."""

    claim_lines = [line.strip() for line in answer.splitlines() if line.strip().startswith("-")]
    cited_ids = [re.findall(r"\[@([^\[\]]+)\]", line) for line in claim_lines]
    cited_claims = sum(bool(ids) for ids in cited_ids)
    flat_ids = [evidence_id for ids in cited_ids for evidence_id in ids]
    valid_ids = sum(evidence_id in valid_evidence_ids for evidence_id in flat_ids)
    return {
        "claims": len(claim_lines),
        "cited_claims": cited_claims,
        "claim_citation_completeness": cited_claims / len(claim_lines) if claim_lines else 0.0,
        "citation_ids": len(flat_ids),
        "valid_citation_ids": valid_ids,
        "citation_id_validity": valid_ids / len(flat_ids) if flat_ids else 0.0,
    }


def seasonal_fold_windows(start: datetime, end: datetime) -> tuple[SeasonalFold, ...]:
    """Return complete 28-day spring/summer/autumn/winter test windows.

    Each fold reserves the immediately preceding 28 days for calibration and
    requires at least 30 days of earlier training history.
    """

    seasons = ((2, "summer"), (5, "autumn"), (7, "winter"), (11, "spring"))
    folds: list[SeasonalFold] = []
    for year in range(start.year, end.year + 1):
        for month, season in seasons:
            test_start = datetime(year, month, 1, tzinfo=UTC)
            test_end = test_start + timedelta(days=28)
            calibration_start = test_start - timedelta(days=28)
            if calibration_start - start < timedelta(days=30) or test_end > end:
                continue
            folds.append(SeasonalFold(f"{season}-{year}", calibration_start, test_start, test_end))
    return tuple(sorted(folds, key=lambda item: item.test_start))


def parse_degradation_costs(value: str) -> tuple[float, ...]:
    """Parse comma-separated CLI values or Slurm-safe colon/semicolon values."""

    costs = tuple(float(item) for item in re.split(r"[,;:]", value) if item.strip())
    if not costs or any(item < 0 for item in costs):
        raise ValueError("degradation costs must be a non-empty list of non-negative numbers")
    return costs


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return cast(dict[str, Any], value)


def merge_evaluation_runs(
    run_dirs: list[Path], expected_regions: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge single-region artifacts only when provenance fields agree."""

    if not run_dirs:
        raise ValueError("at least one run directory is required")
    manifests: list[dict[str, Any]] = []
    merged_regions: dict[str, Any] = {}
    coverage: dict[str, float] | None = None
    scope: str | None = None
    for run_dir in run_dirs:
        metrics = _read_json(run_dir / "metrics.json")
        manifest = _read_json(run_dir / "run_manifest.json")
        selected = tuple(manifest.get("selected_regions", ()))
        if len(selected) != 1 or set(metrics.get("regions", {})) != set(selected):
            raise ValueError(f"{run_dir} is not a single-region shard: {selected}")
        region = selected[0]
        if region in merged_regions:
            raise ValueError(f"duplicate region shard: {region}")
        merged_regions[region] = metrics["regions"][region]
        manifests.append(manifest)
        coverage = metrics["coverage"] if coverage is None else coverage
        scope = metrics["scope"] if scope is None else scope
        if metrics["coverage"] != coverage or metrics["scope"] != scope:
            raise ValueError(f"inconsistent metric metadata in {run_dir}")

    actual_regions = tuple(sorted(merged_regions))
    if set(actual_regions) != set(expected_regions):
        raise ValueError(f"region gate failed: expected {sorted(expected_regions)}, got {list(actual_regions)}")
    for field in (
        "git_sha",
        "input_sha256",
        "degradation_costs_aud_per_mwh_discharged",
        "risk_aware_dispatch",
    ):
        values = {json.dumps(manifest.get(field), sort_keys=True) for manifest in manifests}
        if len(values) != 1:
            raise ValueError(f"inconsistent {field}: {sorted(values)}")

    metrics = {"scope": scope, "coverage": coverage, "regions": merged_regions}
    component_manifest_sha256 = {
        region: hashlib.sha256((run_dir / "run_manifest.json").read_bytes()).hexdigest()
        for region, run_dir in zip(
            (manifest["selected_regions"][0] for manifest in manifests), run_dirs, strict=True
        )
    }
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": manifests[0]["git_sha"],
        "input_sha256": manifests[0]["input_sha256"],
        "regions": sorted(merged_regions),
        "degradation_costs_aud_per_mwh_discharged": manifests[0][
            "degradation_costs_aud_per_mwh_discharged"
        ],
        "risk_aware_dispatch": manifests[0].get("risk_aware_dispatch"),
        "component_manifest_sha256": component_manifest_sha256,
        "component_elapsed_seconds_sum": round(sum(float(item["elapsed_seconds"]) for item in manifests), 3),
        "component_elapsed_seconds_max": round(max(float(item["elapsed_seconds"]) for item in manifests), 3),
        "provider_cost_usd": 0.0,
        "merge_gate": "five unique regions; identical git SHA, input SHA256, degradation costs, scope and coverage",
    }
    return metrics, manifest
