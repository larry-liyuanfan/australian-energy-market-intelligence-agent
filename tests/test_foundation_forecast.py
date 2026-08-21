from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from energy_agent.foundation_forecast import FoundationWindow, predict_windows


class FakeChronos2:
    def predict_df(self, context_df: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        future = kwargs["future_df"]
        assert isinstance(future, pd.DataFrame)
        assert kwargs["prediction_length"] == 2
        assert kwargs["cross_learning"] is False
        assert context_df["timestamp"].dtype == "datetime64[ns]"
        assert future["timestamp"].dtype == "datetime64[ns]"
        output = future.copy()
        output["predictions"] = [10.0, 20.0]
        output["0.1"] = [5.0, 15.0]
        output["0.5"] = [10.0, 20.0]
        output["0.9"] = [15.0, 25.0]
        return output


def window() -> FoundationWindow:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return FoundationWindow(
        series_id="SA1-2026-01-02",
        context_timestamps=(start, start + timedelta(minutes=5)),
        context_target=(1.0, 2.0),
        future_timestamps=(start + timedelta(minutes=10), start + timedelta(minutes=15)),
    )


def test_predict_windows_validates_and_aligns_pipeline_output() -> None:
    result = predict_windows(FakeChronos2(), [window()], model_id="amazon/chronos-2")
    assert result[window().series_id].point == (10.0, 20.0)
    assert result[window().series_id].lower == (5.0, 15.0)
    assert result[window().series_id].timestamps == window().future_timestamps


def test_window_rejects_future_leakage() -> None:
    item = window()
    leaking = FoundationWindow(
        series_id=item.series_id,
        context_timestamps=(item.future_timestamps[0],),
        context_target=(1.0,),
        future_timestamps=item.future_timestamps,
    )
    with pytest.raises(ValueError, match="context must end"):
        leaking.validate()


def test_predict_windows_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate series_id"):
        predict_windows(FakeChronos2(), [window(), window()], model_id="amazon/chronos-2")
