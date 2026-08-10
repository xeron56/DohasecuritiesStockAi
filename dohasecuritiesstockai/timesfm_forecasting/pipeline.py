"""End-to-end historical split, TimesFM backtest, and future forecast pipeline."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Protocol
from uuid import uuid4

import numpy as np

from dohasecuritiesstockai.dataflows.config import get_config
from dohasecuritiesstockai.dataflows.dse import DEFAULT_DSE_GATEWAY_URL, normalize_dse_symbol

from .forecaster import TimesFMBackend
from .market_data import MarketCandle, Resolution, fetch_dse_candles
from .metrics import calculate_accuracy_metrics
from .schema import (
    BacktestPoint,
    CandlePoint,
    DataMetadata,
    ForecastPoint,
    LiveFeedMetadata,
    ModelMetadata,
    TimesFMPredictionResult,
)


class ForecastBackend(Protocol):
    name: str
    version: str
    checkpoint: str
    parameters: int
    max_context: int
    max_horizon: int
    device: str
    gpu_name: str | None

    def prepare(self, context_points: int, requested_horizon: int) -> None: ...

    def forecast_open_loop(
        self, context: np.ndarray, horizon: int
    ) -> tuple[np.ndarray, np.ndarray, int]: ...


def _future_timestamps(
    last_time: datetime,
    resolution: Resolution,
    count: int,
) -> list[datetime]:
    if resolution.key == "1d":
        # DSE's regular trading week is Sunday through Thursday. Daily
        # forecasts therefore skip Friday (4) and Saturday (5).
        timestamps: list[datetime] = []
        candidate = last_time
        while len(timestamps) < count:
            candidate += timedelta(days=1)
            if candidate.weekday() not in {4, 5}:
                timestamps.append(candidate)
        return timestamps
    return [
        last_time + timedelta(seconds=resolution.seconds * step) for step in range(1, count + 1)
    ]


def _build_result(
    symbol: str,
    resolution: Resolution,
    candles: list[MarketCandle],
    split_ratio: float,
    future_steps: int,
    backend: ForecastBackend,
) -> TimesFMPredictionResult:
    if not 0.1 <= split_ratio <= 0.9:
        raise ValueError("split_ratio must be between 0.1 and 0.9.")
    if future_steps < 1:
        raise ValueError("future_steps must be at least 1.")
    if len(candles) < 64:
        hint = (
            "If you meant one year of daily history, use `--resolution 1d --lookback 1y`."
            if resolution.key == "1y"
            else "Use a finer resolution or a longer lookback."
        )
        raise ValueError(
            f"TimesFM backtesting requires at least 64 bars so the context has 32 or more; "
            f"the server returned {len(candles)} {resolution.label} bars. {hint}"
        )

    split_index = int(len(candles) * split_ratio)
    if split_index < 32 or split_index >= len(candles):
        raise ValueError("The selected split does not leave a valid context and holdout.")

    values = np.asarray([candle.close for candle in candles], dtype=np.float32)
    context = values[:split_index]
    actual = values[split_index:]
    requested_horizon = min(1024, max(len(actual), future_steps))
    backend.prepare(len(values) + future_steps, requested_horizon)

    backtest_point, backtest_quantiles, backtest_chunks = backend.forecast_open_loop(
        context, len(actual)
    )
    future_point, future_quantiles, future_chunks = backend.forecast_open_loop(values, future_steps)

    q10 = backtest_quantiles[:, 1]
    q90 = backtest_quantiles[:, 9]
    metrics = calculate_accuracy_metrics(
        actual,
        backtest_point,
        q10,
        q90,
        last_context_value=float(context[-1]),
    )

    history = [
        CandlePoint(
            time=candle.time,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            segment="context" if index < split_index else "holdout",
        )
        for index, candle in enumerate(candles)
    ]
    backtest = []
    for index, (candle, predicted) in enumerate(
        zip(candles[split_index:], backtest_point, strict=True)
    ):
        actual_value = float(candle.close)
        predicted_value = float(predicted)
        absolute_error = abs(predicted_value - actual_value)
        backtest.append(
            BacktestPoint(
                time=candle.time,
                actual=actual_value,
                predicted=predicted_value,
                q10=float(backtest_quantiles[index, 1]),
                q50=float(backtest_quantiles[index, 5]),
                q90=float(backtest_quantiles[index, 9]),
                error=predicted_value - actual_value,
                absolute_error=absolute_error,
                absolute_percentage_error=(
                    absolute_error / abs(actual_value) * 100 if actual_value else None
                ),
            )
        )

    future = [
        ForecastPoint(
            time=forecast_time,
            predicted=float(future_point[index]),
            q10=float(future_quantiles[index, 1]),
            q50=float(future_quantiles[index, 5]),
            q90=float(future_quantiles[index, 9]),
        )
        for index, forecast_time in enumerate(
            _future_timestamps(candles[-1].time, resolution, future_steps)
        )
    ]

    canonical = normalize_dse_symbol(symbol)
    config = get_config()
    gateway = str(config.get("dse_gateway_url") or DEFAULT_DSE_GATEWAY_URL).rstrip("/")
    generated_at = datetime.now(timezone.utc)
    run_id = f"{canonical.lower()}-{resolution.key}-{generated_at:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    return TimesFMPredictionResult(
        run_id=run_id,
        generated_at=generated_at,
        symbol=canonical,
        model=ModelMetadata(
            name=backend.name,
            version=backend.version,
            checkpoint=backend.checkpoint,
            parameters=backend.parameters,
            device=backend.device,
            gpu_name=backend.gpu_name,
            max_context=backend.max_context,
            max_horizon=backend.max_horizon,
            recursive_chunks=backtest_chunks + future_chunks,
        ),
        data=DataMetadata(
            vendor="Doha Securities DSE gateway",
            endpoint="/market-analytics-service/v1/candlesticks/limited",
            symbol=canonical,
            requested_resolution=resolution.key,
            server_resolution=resolution.server_value,
            resolution_label=resolution.label,
            first_timestamp=candles[0].time,
            last_timestamp=candles[-1].time,
            total_points=len(candles),
            context_points=split_index,
            holdout_points=len(candles) - split_index,
            split_ratio=split_ratio,
        ),
        metrics=metrics,
        history=history,
        backtest=backtest,
        future=future,
        live_feed=LiveFeedMetadata(
            enabled=True,
            url=f"{gateway}/notificationservice/ws-notification",
            topic="/topic/stock-updates",
            stock_code=f"{canonical}'PB",
            note=(
                "The Angular UI filters live STOMP updates for this symbol and compares "
                "the latest traded price with the nearest future forecast point."
            ),
        ),
    )


def run_stock_prediction(
    symbol: str,
    resolution: str = "1d",
    *,
    split_ratio: float = 0.5,
    future_steps: int = 12,
    start: date | None = None,
    end: date | None = None,
    limit: int = 100_000,
    backend: ForecastBackend | None = None,
) -> TimesFMPredictionResult:
    """Fetch DSE history, backtest the held-out half, and forecast future bars."""

    resolved, candles = fetch_dse_candles(
        symbol,
        resolution,
        start=start,
        end=end,
        limit=limit,
    )
    return _build_result(
        symbol,
        resolved,
        candles,
        split_ratio,
        future_steps,
        backend or TimesFMBackend(frequency=resolved.timesfm_frequency),
    )
