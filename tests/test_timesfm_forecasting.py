from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from dohasecuritiesstockai.timesfm_forecasting.cli import _lookback_start
from dohasecuritiesstockai.timesfm_forecasting.market_data import (
    MarketCandle,
    fetch_dse_candles,
    normalize_resolution,
)
from dohasecuritiesstockai.timesfm_forecasting.pipeline import _build_result, _future_timestamps
from dohasecuritiesstockai.timesfm_forecasting.repository import PredictionRepository


class FakeDSEClient:
    def __init__(self, payload):
        self.payload = payload
        self.request = None

    def get(self, service, path, *, params=None):
        self.request = (service, path, params)
        return self.payload


class LinearForecastBackend:
    name = "TimesFM"
    version = "test"
    checkpoint = "local/test"
    parameters = 200_000_000
    device = "cpu"
    gpu_name = None
    max_context = 0
    max_horizon = 0

    def prepare(self, context_points: int, requested_horizon: int) -> None:
        self.max_context = context_points
        self.max_horizon = requested_horizon

    def forecast_open_loop(self, context: np.ndarray, horizon: int):
        point = context[-1] + np.arange(1, horizon + 1, dtype=np.float32)
        quantiles = np.empty((horizon, 10), dtype=np.float32)
        quantiles[:, 0] = point
        for index in range(1, 10):
            quantiles[:, index] = point + (index - 5) * 0.1
        return point, quantiles, 1


def _linear_candles(count: int) -> list[MarketCandle]:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        MarketCandle(
            time=start + timedelta(days=index),
            open=float(index + 1),
            high=float(index + 1.5),
            low=float(index + 0.5),
            close=float(index + 1),
            volume=float(1000 + index),
        )
        for index in range(count)
    ]


def test_resolution_aliases_keep_tradingview_month_distinct_from_minute():
    assert normalize_resolution("1min").server_value == "1"
    assert normalize_resolution("1M").key == "1mo"
    assert normalize_resolution("week").server_value == "1W"
    assert normalize_resolution("12M").key == "1y"
    assert normalize_resolution("daily").timesfm_frequency == 0
    assert normalize_resolution("weekly").timesfm_frequency == 1
    assert normalize_resolution("yearly").timesfm_frequency == 2


def test_daily_future_timestamps_follow_dse_trading_week():
    thursday = datetime(2025, 1, 2, tzinfo=timezone.utc)

    future = _future_timestamps(thursday, normalize_resolution("1d"), 3)

    assert [point.date().isoformat() for point in future] == [
        "2025-01-05",
        "2025-01-06",
        "2025-01-07",
    ]


def test_lookback_is_separate_from_candle_resolution():
    assert _lookback_start(datetime(2026, 8, 10).date(), "1y") == datetime(2025, 8, 10).date()
    assert _lookback_start(datetime(2024, 3, 31).date(), "1mo") == datetime(2024, 2, 29).date()
    assert _lookback_start(datetime(2026, 8, 10).date(), "max") is None


def test_fetch_dse_candles_normalizes_envelope_and_ohlcv_arrays():
    first = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
    client = FakeDSEClient(
        {
            "data": {
                "t": [first, first + 86400],
                "o": [10, 11],
                "h": [12, 13],
                "l": [9, 10],
                "c": [11, 12],
                "v": [100, 120],
            }
        }
    )

    resolution, candles = fetch_dse_candles(
        "GP'PB",
        "1D",
        start=datetime(2025, 1, 1).date(),
        end=datetime(2025, 1, 2).date(),
        client=client,
    )

    assert resolution.key == "1d"
    assert [candle.close for candle in candles] == [11.0, 12.0]
    assert client.request[0:2] == ("analytics", "/candlesticks/limited")
    assert client.request[2]["symbol"] == "GP"
    assert client.request[2]["resolution"] == "1D"


def test_pipeline_scores_only_the_hidden_half_and_builds_future_points():
    candles = _linear_candles(80)
    result = _build_result(
        "GP",
        normalize_resolution("1d"),
        candles,
        split_ratio=0.5,
        future_steps=5,
        backend=LinearForecastBackend(),
    )

    assert result.data.context_points == 40
    assert result.data.holdout_points == 40
    assert len(result.history) == 80
    assert len(result.backtest) == 40
    assert len(result.future) == 5
    assert result.metrics.accuracy_score == 100
    assert result.metrics.mae == 0
    assert result.backtest[0].actual == 41
    assert result.backtest[0].predicted == 41
    assert result.future[0].predicted == 81


def test_pipeline_rejects_series_too_short_for_a_50_50_timesfm_context():
    with pytest.raises(ValueError, match="at least 64 bars"):
        _build_result(
            "GP",
            normalize_resolution("1y"),
            _linear_candles(15),
            split_ratio=0.5,
            future_steps=2,
            backend=LinearForecastBackend(),
        )


def test_prediction_repository_round_trip(tmp_path):
    result = _build_result(
        "GP",
        normalize_resolution("1w"),
        _linear_candles(64),
        split_ratio=0.5,
        future_steps=3,
        backend=LinearForecastBackend(),
    )
    repository = PredictionRepository(tmp_path)
    path = repository.save(result)

    assert path.is_file()
    assert repository.get(result.run_id) == result
    assert repository.latest("GP", "1w") == result
    assert repository.get("../escape") is None
