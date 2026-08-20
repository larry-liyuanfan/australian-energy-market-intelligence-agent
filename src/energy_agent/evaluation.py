"""Reusable evaluation parameter parsing and sharded-run quality gates."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

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


def residual_price_scenarios(
    forecast: list[float],
    calibration_residuals: np.ndarray,
    *,
    scenario_count: int,
    seed_material: str,
) -> list[list[float]]:
    """Sample complete calibration-day residual paths without test labels."""

    if len(forecast) != 288:
        raise ValueError("risk scenarios require one complete five-minute day")
    residuals = np.asarray(calibration_residuals, dtype=float)
    complete_days = len(residuals) // 288
    if complete_days < 2 or scenario_count < 2:
        raise ValueError("risk scenarios require at least two calibration days and scenarios")
    blocks = residuals[-complete_days * 288 :].reshape(complete_days, 288)
    seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    selected = rng.choice(complete_days, size=scenario_count - 1, replace=True)
    point = np.asarray(forecast, dtype=float)
    return [point.tolist(), *[(point + blocks[index]).tolist() for index in selected]]


def select_tail_policy(
    point_values: list[float],
    candidate_values: dict[str, list[float]],
    *,
    tail_probability: float = 0.2,
    mean_tolerance_fraction: float = 0.1,
) -> dict[str, float | str]:
    """Select a tail policy on pre-test validation values or fall back to point dispatch."""

    if not point_values or not 0 < tail_probability <= 1 or mean_tolerance_fraction < 0:
        raise ValueError("invalid tail-policy selection inputs")
    if any(len(values) != len(point_values) for values in candidate_values.values()):
        raise ValueError("candidate validation values must align with the point baseline")

    def tail_mean(values: list[float]) -> float:
        count = max(1, math.ceil(len(values) * tail_probability))
        return float(np.mean(sorted(values)[:count]))

    point_mean = float(np.mean(point_values))
    point_tail = tail_mean(point_values)
    mean_floor = point_mean - mean_tolerance_fraction * max(abs(point_mean), 1.0)
    eligible: list[tuple[float, float, str]] = []
    for name, values in candidate_values.items():
        candidate_mean = float(np.mean(values))
        candidate_tail = tail_mean(values)
        if candidate_mean >= mean_floor and candidate_tail > point_tail:
            eligible.append((candidate_tail, candidate_mean, name))
    if not eligible:
        return {
            "selected_policy": "point",
            "validation_mean_aud": point_mean,
            "validation_tail_mean_aud": point_tail,
            "point_validation_mean_aud": point_mean,
            "point_validation_tail_mean_aud": point_tail,
        }
    candidate_tail, candidate_mean, selected = max(eligible)
    return {
        "selected_policy": selected,
        "validation_mean_aud": candidate_mean,
        "validation_tail_mean_aud": candidate_tail,
        "point_validation_mean_aud": point_mean,
        "point_validation_tail_mean_aud": point_tail,
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
