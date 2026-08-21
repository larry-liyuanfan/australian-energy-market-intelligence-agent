"""Typed, lazy-loaded adapter for zero-shot time-series foundation models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class FoundationWindow:
    series_id: str
    context_timestamps: tuple[datetime, ...]
    context_target: tuple[float, ...]
    future_timestamps: tuple[datetime, ...]

    def validate(self) -> None:
        if not self.series_id:
            raise ValueError("series_id must not be empty")
        if len(self.context_timestamps) != len(self.context_target) or not self.context_target:
            raise ValueError("context timestamps and targets must be non-empty and aligned")
        if not self.future_timestamps:
            raise ValueError("future timestamps must not be empty")
        if tuple(sorted(self.context_timestamps)) != self.context_timestamps:
            raise ValueError("context timestamps must be sorted")
        if tuple(sorted(self.future_timestamps)) != self.future_timestamps:
            raise ValueError("future timestamps must be sorted")
        if self.context_timestamps[-1] >= self.future_timestamps[0]:
            raise ValueError("context must end before the forecast horizon")


@dataclass(frozen=True)
class FoundationForecast:
    series_id: str
    timestamps: tuple[datetime, ...]
    point: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    model_id: str


def _model_timestamp(value: datetime) -> datetime:
    """Convert aware timestamps to UTC-naive values required by Chronos DataFrames."""

    return value.astimezone(UTC).replace(tzinfo=None) if value.tzinfo is not None else value


def predict_windows(
    pipeline: Any,
    windows: Sequence[FoundationWindow],
    *,
    model_id: str,
    lower_quantile: float = 0.1,
    upper_quantile: float = 0.9,
) -> dict[str, FoundationForecast]:
    """Run a Chronos-2-compatible pipeline without importing it at module import time."""

    import pandas as pd

    if not windows:
        raise ValueError("at least one forecast window is required")
    if not 0.0 < lower_quantile < 0.5 < upper_quantile < 1.0:
        raise ValueError("quantiles must straddle the median")
    lengths = {len(window.future_timestamps) for window in windows}
    if len(lengths) != 1:
        raise ValueError("all windows in one batch must have the same horizon")
    seen: set[str] = set()
    context_rows: list[dict[str, Any]] = []
    future_rows: list[dict[str, Any]] = []
    for window in windows:
        window.validate()
        if window.series_id in seen:
            raise ValueError(f"duplicate series_id: {window.series_id}")
        seen.add(window.series_id)
        context_rows.extend(
            {
                "id": window.series_id,
                "timestamp": _model_timestamp(timestamp),
                "target": float(target),
            }
            for timestamp, target in zip(
                window.context_timestamps, window.context_target, strict=True
            )
        )
        future_rows.extend(
            {"id": window.series_id, "timestamp": _model_timestamp(timestamp)}
            for timestamp in window.future_timestamps
        )
    context_frame = pd.DataFrame(context_rows)
    future_frame = pd.DataFrame(future_rows)
    context_frame["timestamp"] = pd.to_datetime(context_frame["timestamp"]).astype("datetime64[ns]")
    future_frame["timestamp"] = pd.to_datetime(future_frame["timestamp"]).astype("datetime64[ns]")
    prediction = pipeline.predict_df(
        context_frame,
        future_df=future_frame,
        prediction_length=lengths.pop(),
        quantile_levels=[lower_quantile, 0.5, upper_quantile],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
        # Each row-group is a different rolling forecast origin. Chronos-2's
        # cross-learning would let later origins share information with earlier
        # origins in the same batch, so it is explicitly disabled here.
        cross_learning=False,
    )
    required = {"id", "timestamp", "predictions", str(lower_quantile), str(upper_quantile)}
    missing = required - set(prediction.columns)
    if missing:
        raise ValueError(f"foundation model output is missing columns: {sorted(missing)}")
    output: dict[str, FoundationForecast] = {}
    expected = {window.series_id: window for window in windows}
    for series_id, group in prediction.groupby("id", sort=False):
        identifier = str(series_id)
        if identifier not in expected:
            raise ValueError(f"unexpected prediction series_id: {identifier}")
        ordered = group.sort_values("timestamp")
        model_timestamps = tuple(pd.Timestamp(value).to_pydatetime() for value in ordered["timestamp"])
        expected_model_timestamps = tuple(
            _model_timestamp(value) for value in expected[identifier].future_timestamps
        )
        if model_timestamps != expected_model_timestamps:
            raise ValueError(f"forecast timestamps do not align for {identifier}")
        output[identifier] = FoundationForecast(
            series_id=identifier,
            timestamps=expected[identifier].future_timestamps,
            point=tuple(float(value) for value in ordered["predictions"]),
            lower=tuple(float(value) for value in ordered[str(lower_quantile)]),
            upper=tuple(float(value) for value in ordered[str(upper_quantile)]),
            model_id=model_id,
        )
    if set(output) != set(expected):
        raise ValueError(f"missing forecast series: {sorted(set(expected) - set(output))}")
    return output


def load_chronos2(model_id: str = "amazon/chronos-2", *, device_map: str = "cuda") -> Any:
    """Load the official Chronos-2 pipeline lazily so core/API installs stay lightweight."""

    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(model_id, device_map=device_map)
