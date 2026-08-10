"""Stable prediction-result contract shared by the CLI, API, and Angular UI."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandlePoint(PredictionModel):
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    segment: Literal["context", "holdout"]


class ForecastPoint(PredictionModel):
    time: datetime
    predicted: float
    q10: float
    q50: float
    q90: float


class BacktestPoint(ForecastPoint):
    actual: float
    error: float
    absolute_error: float
    absolute_percentage_error: float | None


class AccuracyMetrics(PredictionModel):
    accuracy_score: float = Field(ge=0, le=100)
    accuracy_definition: str
    mae: float
    rmse: float
    mape_percent: float | None
    smape_percent: float
    r_squared: float | None
    directional_accuracy_percent: float | None
    interval_80_coverage_percent: float
    naive_mae: float
    skill_vs_naive_percent: float | None


class ModelMetadata(PredictionModel):
    name: str
    version: str
    checkpoint: str
    parameters: int
    backend: Literal["torch"] = "torch"
    device: str
    gpu_name: str | None = None
    max_context: int
    max_horizon: int
    recursive_chunks: int


class DataMetadata(PredictionModel):
    vendor: str
    endpoint: str
    symbol: str
    requested_resolution: str
    server_resolution: str
    resolution_label: str
    first_timestamp: datetime
    last_timestamp: datetime
    total_points: int
    context_points: int
    holdout_points: int
    split_ratio: float
    target: Literal["close"] = "close"


class LiveFeedMetadata(PredictionModel):
    enabled: bool
    transport: Literal["sockjs_stomp"] = "sockjs_stomp"
    url: str
    topic: str
    stock_code: str
    note: str


class TimesFMPredictionResult(PredictionModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    generated_at: datetime
    symbol: str
    currency: str = "BDT"
    model: ModelMetadata
    data: DataMetadata
    metrics: AccuracyMetrics
    history: list[CandlePoint]
    backtest: list[BacktestPoint]
    future: list[ForecastPoint]
    live_feed: LiveFeedMetadata
    disclaimer: str = (
        "TimesFM is a statistical forecasting model, not investment advice. "
        "Backtest results do not guarantee future performance."
    )
