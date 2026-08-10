"""Differential Graph Transformer forecasting for Doha Securities DSE data.

This module replaces the original Colab-only experiment.  It uses the existing
``dohasecuritiesstockai`` authenticated candle loader, makes a chronological
train/validation/holdout split, trains a Differential Graph Transformer (DGT),
compares recursive holdout predictions with real closing prices, and produces
future forecasts.  Results use the project's prediction JSON contract, so the
existing API and Angular prediction dashboard can display them.

Example::

    dohasecuritiesstockai-dgt-predict GP \
        --resolution 1d --lookback 2y --peers DSEX,BRACBANK,SQURPHARMA \
        --future-steps 12 --epochs 80

DSE authentication is read by the existing data layer.  Set ``DSE_ACCESS_TOKEN``
or ``DSE_EMAIL_OR_PHONE`` and ``DSE_PASSWORD`` before running the command.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import re
from calendar import monthrange
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dohasecuritiesstockai.dataflows.config import get_config
from dohasecuritiesstockai.dataflows.dse import (
    DEFAULT_DSE_GATEWAY_URL,
    DSEClient,
    normalize_dse_symbol,
)
from dohasecuritiesstockai.default_config import DEFAULT_CONFIG
from dohasecuritiesstockai.timesfm_forecasting.market_data import (
    MarketCandle,
    Resolution,
    fetch_dse_candles,
)
from dohasecuritiesstockai.timesfm_forecasting.metrics import calculate_accuracy_metrics
from dohasecuritiesstockai.timesfm_forecasting.repository import PredictionRepository
from dohasecuritiesstockai.timesfm_forecasting.schema import (
    BacktestPoint,
    CandlePoint,
    DataMetadata,
    ForecastPoint,
    LiveFeedMetadata,
    ModelMetadata,
    TimesFMPredictionResult,
)


@dataclass(frozen=True)
class DGTConfig:
    """Training and architecture settings for one forecast run."""

    window: int = 32
    split_ratio: float = 0.8
    validation_ratio: float = 0.15
    future_steps: int = 12
    epochs: int = 80
    patience: int = 12
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_size: int = 32
    num_heads: int = 2
    num_layers: int = 2
    dropout: float = 0.1
    graph_top_k: int = 5
    use_spatial: bool = True
    refit_for_future: bool = True
    seed: int = 42

    def validate(self) -> None:
        if self.window < 4:
            raise ValueError("window must be at least 4 bars.")
        if not 0.5 <= self.split_ratio <= 0.95:
            raise ValueError("split_ratio must be between 0.5 and 0.95.")
        if not 0.05 <= self.validation_ratio <= 0.4:
            raise ValueError("validation_ratio must be between 0.05 and 0.4.")
        if self.future_steps < 1:
            raise ValueError("future_steps must be at least 1.")
        if min(self.epochs, self.patience, self.batch_size) < 1:
            raise ValueError("epochs, patience, and batch_size must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if self.hidden_size < 1 or self.num_heads < 1:
            raise ValueError("hidden_size and num_heads must be positive.")
        if self.hidden_size % (2 * self.num_heads) != 0:
            raise ValueError("hidden_size must be divisible by two times num_heads.")
        if self.num_layers < 1 or self.graph_top_k < 1:
            raise ValueError("num_layers and graph_top_k must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be at least zero and less than one.")


@dataclass(frozen=True)
class MarketGraphData:
    """Aligned closing prices for a target security and its graph peers."""

    symbol: str
    symbols: tuple[str, ...]
    resolution: Resolution
    target_candles: tuple[MarketCandle, ...]
    closes: np.ndarray  # [time, nodes]


@dataclass(frozen=True)
class TrainingSummary:
    best_epoch: int
    epochs_ran: int
    best_validation_mse: float
    device: str
    parameter_count: int
    train_samples: int
    validation_samples: int
    future_refit_epochs: int
    graph_symbols: tuple[str, ...]


@dataclass(frozen=True)
class ForecastRun:
    result: TimesFMPredictionResult
    summary: TrainingSummary
    model_state: dict[str, torch.Tensor]
    model_config: dict[str, object]


class PriceScaler:
    """Per-security z-score scaler fitted only on the training period."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, values: np.ndarray) -> PriceScaler:
        mean = np.mean(values, axis=0, dtype=np.float64).astype(np.float32)
        scale = np.std(values, axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float32) - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) * self.scale + self.mean).astype(np.float32)


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = values * torch.rsqrt(
            values.float().square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        return normalized.type_as(values) * self.weight


def _lambda_initial_value(depth: int) -> float:
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadDifferentialAttention(nn.Module):
    """Differential attention optionally gated by a dense graph prior."""

    def __init__(self, hidden_size: int, num_heads: int, depth: int) -> None:
        super().__init__()
        if hidden_size % (2 * num_heads) != 0:
            raise ValueError("hidden_size must be divisible by 2 * num_heads.")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // (2 * num_heads)
        self.scaling = self.head_size**-0.5
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lambda_initial = _lambda_initial_value(depth)
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_size) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_size) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_size) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_size) * 0.1)
        self.sub_layer_norm = RMSNorm(2 * self.head_size, epsilon=1e-5)

    def forward(
        self,
        values: torch.Tensor,
        *,
        graph_prior: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = values.shape
        query = self.query(values).view(
            batch_size, sequence_length, 2 * self.num_heads, self.head_size
        )
        key = self.key(values).view(batch_size, sequence_length, 2 * self.num_heads, self.head_size)
        projected_value = self.value(values).view(
            batch_size, sequence_length, self.num_heads, 2 * self.head_size
        )
        query = query.transpose(1, 2) * self.scaling
        key = key.transpose(1, 2)
        projected_value = projected_value.transpose(1, 2)

        attention_logits = torch.matmul(query, key.transpose(-1, -2))
        if causal:
            mask = torch.triu(
                torch.ones(
                    sequence_length,
                    sequence_length,
                    dtype=torch.bool,
                    device=values.device,
                ),
                diagonal=1,
            )
            attention_logits = attention_logits.masked_fill(mask, float("-inf"))
        attention = torch.softmax(attention_logits.float(), dim=-1).type_as(values)
        attention = attention.view(
            batch_size,
            self.num_heads,
            2,
            sequence_length,
            sequence_length,
        )

        lambda_one = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lambda_two = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        differential_lambda = lambda_one - lambda_two + self.lambda_initial
        differential = attention[:, :, 0] - differential_lambda * attention[:, :, 1]

        if graph_prior is not None:
            prior = graph_prior.to(device=values.device, dtype=values.dtype)
            if prior.ndim != 2 or prior.shape != (sequence_length, sequence_length):
                raise ValueError("graph_prior shape must match the attended node dimension.")
            differential = differential * prior.view(1, 1, sequence_length, sequence_length)
            normalizer = differential.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
            differential = differential / normalizer

        attended = torch.matmul(differential, projected_value)
        attended = self.sub_layer_norm(attended) * (1 - self.lambda_initial)
        attended = attended.transpose(1, 2).reshape(batch_size, sequence_length, self.hidden_size)
        return self.output(attended)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = MultiheadDifferentialAttention(hidden_size, num_heads, depth)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.feed_forward_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        *,
        graph_prior: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        attended = self.attention(values, graph_prior=graph_prior, causal=causal)
        values = self.attention_norm(values + self.dropout(attended))
        return self.feed_forward_norm(values + self.dropout(self.feed_forward(values)))


class DifferentialGraphTransformer(nn.Module):
    """Temporal differential attention followed by inter-security attention."""

    def __init__(
        self,
        node_count: int,
        window: int,
        *,
        hidden_size: int = 32,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_spatial: bool = True,
    ) -> None:
        super().__init__()
        self.node_count = node_count
        self.window = window
        self.use_spatial = use_spatial and node_count > 1
        self.input_projection = nn.Linear(1, hidden_size)
        self.time_embedding = nn.Embedding(window, hidden_size)
        self.node_embedding = nn.Embedding(node_count, hidden_size)
        self.temporal_blocks = nn.ModuleList(
            [AttentionBlock(hidden_size, num_heads, layer, dropout) for layer in range(num_layers)]
        )
        self.spatial_blocks = nn.ModuleList(
            [AttentionBlock(hidden_size, num_heads, layer, dropout) for layer in range(num_layers)]
            if self.use_spatial
            else []
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, values: torch.Tensor, graph_prior: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4:
            raise ValueError("DGT input must have shape [batch, nodes, time, features].")
        batch_size, node_count, window, feature_count = values.shape
        if (node_count, window, feature_count) != (self.node_count, self.window, 1):
            raise ValueError(
                f"Expected [batch, {self.node_count}, {self.window}, 1], got {tuple(values.shape)}."
            )

        encoded = self.input_projection(values)
        times = torch.arange(window, device=values.device)
        nodes = torch.arange(node_count, device=values.device)
        encoded = encoded + self.time_embedding(times).view(1, 1, window, -1)
        encoded = encoded + self.node_embedding(nodes).view(1, node_count, 1, -1)

        for layer, temporal_block in enumerate(self.temporal_blocks):
            temporal = encoded.reshape(batch_size * node_count, window, -1)
            temporal = temporal_block(temporal, causal=True)
            encoded = temporal.reshape(batch_size, node_count, window, -1)

            if self.use_spatial:
                spatial = encoded.permute(0, 2, 1, 3).reshape(batch_size * window, node_count, -1)
                spatial = self.spatial_blocks[layer](spatial, graph_prior=graph_prior)
                encoded = spatial.reshape(batch_size, window, node_count, -1).permute(0, 2, 1, 3)

        # Learning a change around the last observed close gives the model a
        # strong persistence anchor while still allowing nonlinear movement.
        predicted_change = self.regression_head(encoded[:, :, -1]).squeeze(-1)
        return values[:, :, -1, 0] + predicted_change


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str = "auto") -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if device.type == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is not available.")
    return device


def _canonical_symbols(target: str, peers: Sequence[str]) -> tuple[str, ...]:
    canonical_target = normalize_dse_symbol(target)
    ordered = [canonical_target]
    for peer in peers:
        if not str(peer).strip():
            continue
        canonical = normalize_dse_symbol(peer)
        if canonical not in ordered:
            ordered.append(canonical)
    return tuple(ordered)


def load_dse_market_graph(
    symbol: str,
    peers: Sequence[str] = ("DSEX",),
    resolution: str = "1d",
    *,
    start: date | None = None,
    end: date | None = None,
    limit: int = 100_000,
    client: DSEClient | None = None,
    fetcher: Callable[..., tuple[Resolution, list[MarketCandle]]] = fetch_dse_candles,
) -> MarketGraphData:
    """Fetch and causally align target and peer closing prices.

    Peer values are forward-filled onto target candle timestamps.  Back-filling
    is deliberately not used because it would leak later prices into earlier
    model inputs.  Leading timestamps without all requested peers are dropped.
    """

    symbols = _canonical_symbols(symbol, peers)
    candles_by_symbol: dict[str, list[MarketCandle]] = {}
    resolved: Resolution | None = None
    shared_client = client or DSEClient()
    for current_symbol in symbols:
        current_resolution, candles = fetcher(
            current_symbol,
            resolution,
            start=start,
            end=end,
            limit=limit,
            client=shared_client,
        )
        if resolved is None:
            resolved = current_resolution
        elif current_resolution.key != resolved.key:
            raise ValueError("DSE returned inconsistent resolutions across graph symbols.")
        candles_by_symbol[current_symbol] = candles

    assert resolved is not None
    target_candles = candles_by_symbol[symbols[0]]
    target_times = [candle.time for candle in target_candles]
    target_positions = {timestamp: index for index, timestamp in enumerate(target_times)}
    matrix = np.full((len(target_times), len(symbols)), np.nan, dtype=np.float64)

    for node_index, current_symbol in enumerate(symbols):
        observed = {candle.time: candle.close for candle in candles_by_symbol[current_symbol]}
        last_value = math.nan
        for row_index, timestamp in enumerate(target_times):
            if timestamp in observed:
                last_value = float(observed[timestamp])
            matrix[row_index, node_index] = last_value

    valid_rows = np.isfinite(matrix).all(axis=1) & (matrix > 0).all(axis=1)
    aligned_times = [
        timestamp for timestamp, valid in zip(target_times, valid_rows, strict=True) if valid
    ]
    aligned_candles = tuple(
        target_candles[target_positions[timestamp]] for timestamp in aligned_times
    )
    closes = matrix[valid_rows].astype(np.float32)
    if not aligned_candles:
        raise ValueError("The target and peer histories have no causally aligned candles.")
    return MarketGraphData(
        symbol=symbols[0],
        symbols=symbols,
        resolution=resolved,
        target_candles=aligned_candles,
        closes=closes,
    )


def build_graph_prior(training_closes: np.ndarray, top_k: int = 5) -> np.ndarray:
    """Build a sparse, non-negative Pearson graph from training returns only."""

    prices = np.asarray(training_closes, dtype=np.float64)
    if prices.ndim != 2 or prices.shape[0] < 3:
        raise ValueError("At least three time rows are required to build a market graph.")
    node_count = prices.shape[1]
    if node_count == 1:
        return np.ones((1, 1), dtype=np.float32)

    returns = np.diff(np.log(np.clip(prices, 1e-8, None)), axis=0)
    correlation = np.corrcoef(returns, rowvar=False)
    correlation = np.nan_to_num(np.abs(correlation), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)

    keep_count = min(max(1, top_k), node_count)
    sparse = np.zeros_like(correlation)
    for row in range(node_count):
        selected = np.argpartition(correlation[row], -keep_count)[-keep_count:]
        sparse[row, selected] = correlation[row, selected]
    # Make the graph undirected and always retain self loops.
    sparse = np.maximum(sparse, sparse.T)
    np.fill_diagonal(sparse, 1.0)
    row_sum = sparse.sum(axis=1, keepdims=True)
    return (sparse / np.maximum(row_sum, 1e-8)).astype(np.float32)


def _make_supervised_samples(
    scaled_closes: np.ndarray,
    window: int,
    target_indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    features = np.stack(
        [scaled_closes[index - window : index].T[..., np.newaxis] for index in target_indices]
    ).astype(np.float32)
    targets = np.stack([scaled_closes[index] for index in target_indices]).astype(np.float32)
    return torch.from_numpy(features), torch.from_numpy(targets)


def _new_model(
    node_count: int, config: DGTConfig, device: torch.device
) -> DifferentialGraphTransformer:
    return DifferentialGraphTransformer(
        node_count=node_count,
        window=config.window,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        use_spatial=config.use_spatial,
    ).to(device)


def _split_positions(point_count: int, config: DGTConfig) -> tuple[int, int]:
    split_index = int(point_count * config.split_ratio)
    context_samples = split_index - config.window
    minimum_train_samples = max(4, min(config.batch_size, 16))
    if context_samples < minimum_train_samples + 1 or split_index >= point_count:
        required = math.ceil((config.window + minimum_train_samples + 1) / config.split_ratio)
        raise ValueError(
            f"DGT needs at least {required} aligned bars for window={config.window} and "
            f"split={config.split_ratio:.0%}; received {point_count}."
        )
    validation_samples = max(1, int(round(context_samples * config.validation_ratio)))
    validation_samples = min(validation_samples, context_samples - minimum_train_samples)
    validation_start = split_index - validation_samples
    return split_index, validation_start


def _train_model(
    scaled_closes: np.ndarray,
    graph_prior: np.ndarray,
    config: DGTConfig,
    validation_start: int,
    split_index: int,
    device: torch.device,
    *,
    progress: Callable[[str], None] | None = print,
) -> tuple[DifferentialGraphTransformer, TrainingSummary, np.ndarray]:
    train_indices = list(range(config.window, validation_start))
    validation_indices = list(range(validation_start, split_index))
    train_x, train_y = _make_supervised_samples(scaled_closes, config.window, train_indices)
    validation_x, validation_y = _make_supervised_samples(
        scaled_closes, config.window, validation_indices
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=min(config.batch_size, len(train_x)),
        shuffle=True,
        generator=generator,
    )

    model = _new_model(scaled_closes.shape[1], config, device)
    adjacency = torch.from_numpy(graph_prior).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    best_epoch = 0
    stale_epochs = 0
    epochs_ran = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        training_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(batch_x, adjacency)
            loss = F.mse_loss(predicted, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            training_loss += loss.item() * len(batch_x)

        model.eval()
        with torch.no_grad():
            validation_prediction = model(validation_x.to(device), adjacency)
            validation_loss = F.mse_loss(validation_prediction, validation_y.to(device)).item()
        epochs_ran = epoch
        mean_training_loss = training_loss / len(train_x)
        if progress and (epoch == 1 or epoch % 10 == 0):
            progress(
                f"epoch {epoch:03d} | train MSE {mean_training_loss:.6f} | "
                f"validation MSE {validation_loss:.6f}"
            )

        if validation_loss < best_validation - 1e-7:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_prediction = model(validation_x.to(device), adjacency).cpu().numpy()
    summary = TrainingSummary(
        best_epoch=best_epoch,
        epochs_ran=epochs_ran,
        best_validation_mse=float(best_validation),
        device=str(device),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        train_samples=len(train_indices),
        validation_samples=len(validation_indices),
        future_refit_epochs=0,
        graph_symbols=(),  # populated by the pipeline once symbols are known
    )
    return model, summary, validation_prediction


def _refit_model_for_future(
    scaled_closes: np.ndarray,
    graph_prior: np.ndarray,
    config: DGTConfig,
    epochs: int,
    device: torch.device,
    *,
    progress: Callable[[str], None] | None = print,
) -> DifferentialGraphTransformer:
    """Fit a fresh deployment model on every real bar after backtest scoring."""

    _set_seed(config.seed)
    target_indices = list(range(config.window, len(scaled_closes)))
    features, targets = _make_supervised_samples(scaled_closes, config.window, target_indices)
    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=min(config.batch_size, len(features)),
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    model = _new_model(scaled_closes.shape[1], config, device)
    adjacency = torch.from_numpy(graph_prior).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    refit_epochs = max(1, epochs)
    for epoch in range(1, refit_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(batch_x, adjacency), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
        if progress and (epoch == 1 or epoch % 10 == 0 or epoch == refit_epochs):
            progress(
                f"future refit epoch {epoch:03d}/{refit_epochs:03d} | "
                f"train MSE {total_loss / len(features):.6f}"
            )
    model.eval()
    return model


def _recursive_forecast(
    model: DifferentialGraphTransformer,
    scaled_context: np.ndarray,
    graph_prior: np.ndarray,
    window: int,
    horizon: int,
    device: torch.device,
) -> np.ndarray:
    if len(scaled_context) < window:
        raise ValueError("Forecast context is shorter than the model window.")
    history = [row.copy() for row in np.asarray(scaled_context, dtype=np.float32)]
    predictions: list[np.ndarray] = []
    adjacency = torch.from_numpy(graph_prior).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            model_input = np.asarray(history[-window:], dtype=np.float32).T[
                np.newaxis, ..., np.newaxis
            ]
            predicted = model(torch.from_numpy(model_input).to(device), adjacency)[0]
            # Prevent an unstable recursive path from escaping far beyond the
            # training scale while still allowing substantial new movement.
            next_row = predicted.clamp(-8.0, 8.0).cpu().numpy().astype(np.float32)
            history.append(next_row)
            predictions.append(next_row)
    return np.stack(predictions)


def _future_timestamps(last_time: datetime, resolution: Resolution, count: int) -> list[datetime]:
    if resolution.key == "1d":
        timestamps: list[datetime] = []
        candidate = last_time
        while len(timestamps) < count:
            candidate += timedelta(days=1)
            # DSE's regular week is Sunday through Thursday.
            if candidate.weekday() not in {4, 5}:
                timestamps.append(candidate)
        return timestamps
    return [
        last_time + timedelta(seconds=resolution.seconds * step) for step in range(1, count + 1)
    ]


def _prediction_intervals(
    point: np.ndarray,
    validation_residuals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    residuals = np.asarray(validation_residuals, dtype=np.float64)
    radius = float(np.quantile(np.abs(residuals), 0.8)) if residuals.size else 0.0
    horizon_scale = np.sqrt(np.arange(1, len(point) + 1, dtype=np.float64))
    widths = radius * horizon_scale
    lower = np.maximum(0.0, np.asarray(point, dtype=np.float64) - widths)
    upper = np.asarray(point, dtype=np.float64) + widths
    return lower.astype(np.float32), upper.astype(np.float32)


def run_dgt_prediction(
    symbol: str,
    resolution: str = "1d",
    *,
    peers: Sequence[str] = ("DSEX",),
    start: date | None = None,
    end: date | None = None,
    limit: int = 100_000,
    config: DGTConfig | None = None,
    device: str = "auto",
    market_data: MarketGraphData | None = None,
    progress: Callable[[str], None] | None = print,
) -> ForecastRun:
    """Fetch DSE data, train DGT, backtest it, and forecast future bars."""

    settings = config or DGTConfig()
    settings.validate()
    _set_seed(settings.seed)
    graph_data = market_data or load_dse_market_graph(
        symbol,
        peers,
        resolution,
        start=start,
        end=end,
        limit=limit,
    )
    if graph_data.closes.ndim != 2:
        raise ValueError("Market close data must have shape [time, graph nodes].")
    if graph_data.closes.shape != (
        len(graph_data.target_candles),
        len(graph_data.symbols),
    ):
        raise ValueError("Candle, symbol, and close-matrix dimensions are inconsistent.")
    if not np.isfinite(graph_data.closes).all() or np.any(graph_data.closes <= 0):
        raise ValueError("Market close data must contain only finite positive prices.")
    split_index, validation_start = _split_positions(len(graph_data.closes), settings)

    # Scaling and graph estimation end before validation begins; neither sees
    # validation or held-out prices.
    scaler = PriceScaler.fit(graph_data.closes[:validation_start])
    scaled_closes = scaler.transform(graph_data.closes)
    graph_prior = build_graph_prior(
        graph_data.closes[:validation_start], top_k=settings.graph_top_k
    )
    resolved_device = _resolve_device(device)
    if progress:
        progress(
            f"training DGT on {resolved_device} with {len(graph_data.closes)} aligned bars, "
            f"{len(graph_data.symbols)} graph nodes, and a {settings.window}-bar window"
        )
    model, raw_summary, validation_prediction_scaled = _train_model(
        scaled_closes,
        graph_prior,
        settings,
        validation_start,
        split_index,
        resolved_device,
        progress=progress,
    )
    validation_prediction = scaler.inverse_transform(validation_prediction_scaled)[:, 0]
    validation_actual = graph_data.closes[validation_start:split_index, 0]
    validation_residuals = validation_actual - validation_prediction

    backtest_scaled = _recursive_forecast(
        model,
        scaled_closes[:split_index],
        graph_prior,
        settings.window,
        len(graph_data.closes) - split_index,
        resolved_device,
    )
    backtest_prediction = scaler.inverse_transform(backtest_scaled)[:, 0]
    actual = graph_data.closes[split_index:, 0]
    backtest_q10, backtest_q90 = _prediction_intervals(backtest_prediction, validation_residuals)
    metrics = calculate_accuracy_metrics(
        actual,
        backtest_prediction,
        backtest_q10,
        backtest_q90,
        last_context_value=float(graph_data.closes[split_index - 1, 0]),
    )

    deployment_model = model
    deployment_scaler = scaler
    deployment_graph = graph_prior
    deployment_scaled_closes = scaled_closes
    refit_epochs = 0
    if settings.refit_for_future:
        if progress:
            progress("backtest scoring complete; refitting a deployment model on all real bars")
        deployment_scaler = PriceScaler.fit(graph_data.closes)
        deployment_scaled_closes = deployment_scaler.transform(graph_data.closes)
        deployment_graph = build_graph_prior(graph_data.closes, top_k=settings.graph_top_k)
        refit_epochs = max(1, raw_summary.best_epoch)
        deployment_model = _refit_model_for_future(
            deployment_scaled_closes,
            deployment_graph,
            settings,
            refit_epochs,
            resolved_device,
            progress=progress,
        )

    future_scaled = _recursive_forecast(
        deployment_model,
        deployment_scaled_closes,
        deployment_graph,
        settings.window,
        settings.future_steps,
        resolved_device,
    )
    future_prediction = deployment_scaler.inverse_transform(future_scaled)[:, 0]
    future_q10, future_q90 = _prediction_intervals(future_prediction, actual - backtest_prediction)
    summary = TrainingSummary(
        **{
            **asdict(raw_summary),
            "future_refit_epochs": refit_epochs,
            "graph_symbols": graph_data.symbols,
        }
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
        for index, candle in enumerate(graph_data.target_candles)
    ]
    backtest = []
    for offset, candle in enumerate(graph_data.target_candles[split_index:]):
        predicted = float(backtest_prediction[offset])
        actual_value = float(candle.close)
        absolute_error = abs(predicted - actual_value)
        backtest.append(
            BacktestPoint(
                time=candle.time,
                actual=actual_value,
                predicted=predicted,
                q10=float(backtest_q10[offset]),
                q50=predicted,
                q90=float(backtest_q90[offset]),
                error=predicted - actual_value,
                absolute_error=absolute_error,
                absolute_percentage_error=(
                    absolute_error / abs(actual_value) * 100 if actual_value else None
                ),
            )
        )
    future = [
        ForecastPoint(
            time=forecast_time,
            predicted=float(future_prediction[index]),
            q10=float(future_q10[index]),
            q50=float(future_prediction[index]),
            q90=float(future_q90[index]),
        )
        for index, forecast_time in enumerate(
            _future_timestamps(
                graph_data.target_candles[-1].time,
                graph_data.resolution,
                settings.future_steps,
            )
        )
    ]

    generated_at = datetime.now(timezone.utc)
    run_id = (
        f"dgt-{graph_data.symbol.lower()}-{graph_data.resolution.key}-"
        f"{generated_at:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    runtime_config = get_config()
    gateway = str(runtime_config.get("dse_gateway_url") or DEFAULT_DSE_GATEWAY_URL).rstrip("/")
    result = TimesFMPredictionResult(
        run_id=run_id,
        generated_at=generated_at,
        symbol=graph_data.symbol,
        model=ModelMetadata(
            name="Differential Graph Transformer",
            version="DGT-DSE-1.0",
            checkpoint=f"dgt_forecasts/{run_id}/model.pt",
            parameters=summary.parameter_count,
            device=str(resolved_device),
            gpu_name=(
                torch.cuda.get_device_name(resolved_device)
                if resolved_device.type == "cuda"
                else None
            ),
            max_context=settings.window,
            max_horizon=max(len(backtest), len(future)),
            recursive_chunks=2,
        ),
        data=DataMetadata(
            vendor="Doha Securities DSE gateway",
            endpoint="/market-analytics-service/v1/candlesticks/limited",
            symbol=graph_data.symbol,
            requested_resolution=graph_data.resolution.key,
            server_resolution=graph_data.resolution.server_value,
            resolution_label=graph_data.resolution.label,
            first_timestamp=graph_data.target_candles[0].time,
            last_timestamp=graph_data.target_candles[-1].time,
            total_points=len(graph_data.target_candles),
            context_points=split_index,
            holdout_points=len(graph_data.target_candles) - split_index,
            split_ratio=settings.split_ratio,
        ),
        metrics=metrics,
        history=history,
        backtest=backtest,
        future=future,
        live_feed=LiveFeedMetadata(
            enabled=True,
            url=f"{gateway}/notificationservice/ws-notification",
            topic="/topic/stock-updates",
            stock_code=f"{graph_data.symbol}'PB",
            note=(
                "The UI compares live DSE updates for this symbol with the nearest "
                "Differential Graph Transformer future forecast."
            ),
        ),
        disclaimer=(
            "The Differential Graph Transformer is an experimental statistical model, "
            "not investment advice. Backtest results do not guarantee future performance."
        ),
    )
    model_state = {
        name: value.detach().cpu() for name, value in deployment_model.state_dict().items()
    }
    model_config = {
        "dgt": asdict(settings),
        "symbols": list(graph_data.symbols),
        "resolution": graph_data.resolution.key,
        "scaler_mean": deployment_scaler.mean.tolist(),
        "scaler_scale": deployment_scaler.scale.tolist(),
        "graph_prior": deployment_graph.tolist(),
    }
    return ForecastRun(result, summary, model_state, model_config)


def save_forecast_run(
    run: ForecastRun,
    output_dir: str | Path,
    *,
    make_plot: bool = True,
) -> dict[str, Path]:
    """Save the checkpoint, real-vs-predicted CSV, future CSV, and chart."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    result = run.result
    paths = {
        "json": directory / "prediction.json",
        "checkpoint": directory / "model.pt",
        "comparison": directory / "backtest_comparison.csv",
        "future": directory / "future_forecast.csv",
    }
    paths["json"].write_text(result.model_dump_json(indent=2), encoding="utf-8")
    torch.save(
        {
            "state_dict": run.model_state,
            "model_config": run.model_config,
            "training_summary": asdict(run.summary),
            "run_id": result.run_id,
        },
        paths["checkpoint"],
    )

    with paths["comparison"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time",
                "actual",
                "predicted",
                "q10",
                "q90",
                "error",
                "absolute_error",
                "absolute_percentage_error",
            ]
        )
        for point in result.backtest:
            writer.writerow(
                [
                    point.time.isoformat(),
                    point.actual,
                    point.predicted,
                    point.q10,
                    point.q90,
                    point.error,
                    point.absolute_error,
                    point.absolute_percentage_error,
                ]
            )

    with paths["future"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", "predicted", "q10", "q50", "q90"])
        for point in result.future:
            writer.writerow(
                [point.time.isoformat(), point.predicted, point.q10, point.q50, point.q90]
            )

    if make_plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib is required for plots; install the dgt extra or use --no-plot."
            ) from exc
        plot_path = directory / "actual_vs_predicted.png"
        context = result.history[: result.data.context_points]
        figure, axis = plt.subplots(figsize=(13, 7))
        axis.plot(
            [point.time for point in context],
            [point.close for point in context],
            color="#64748b",
            linewidth=1.2,
            label="Training context",
        )
        axis.plot(
            [point.time for point in result.backtest],
            [point.actual for point in result.backtest],
            color="#16a34a",
            linewidth=1.8,
            label="Real holdout",
        )
        axis.plot(
            [point.time for point in result.backtest],
            [point.predicted for point in result.backtest],
            color="#2563eb",
            linewidth=1.8,
            label="DGT holdout prediction",
        )
        axis.fill_between(
            [point.time for point in result.backtest],
            [point.q10 for point in result.backtest],
            [point.q90 for point in result.backtest],
            color="#2563eb",
            alpha=0.12,
        )
        axis.plot(
            [point.time for point in result.future],
            [point.predicted for point in result.future],
            color="#f97316",
            linestyle="--",
            linewidth=2,
            marker="o",
            markersize=3,
            label="DGT future forecast",
        )
        axis.fill_between(
            [point.time for point in result.future],
            [point.q10 for point in result.future],
            [point.q90 for point in result.future],
            color="#f97316",
            alpha=0.12,
        )
        axis.axvline(result.backtest[0].time, color="#0f172a", linestyle=":", alpha=0.6)
        axis.axvline(result.future[0].time, color="#f97316", linestyle=":", alpha=0.6)
        axis.set_title(f"{result.symbol}: real prices, DGT backtest, and future forecast")
        axis.set_xlabel("Date")
        axis.set_ylabel(f"Closing price ({result.currency})")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(plot_path, dpi=160)
        plt.close(figure)
        paths["plot"] = plot_path
    return paths


_LOOKBACK_RE = re.compile(r"^(?P<count>[1-9]\d*)(?P<unit>d|w|mo|y)$")


def _lookback_start(end_date: date, lookback: str) -> date | None:
    value = str(lookback or "").strip().lower()
    if value in {"all", "max"}:
        return None
    match = _LOOKBACK_RE.fullmatch(value)
    if match is None:
        raise ValueError("lookback must look like 90d, 52w, 24mo, 2y, or max.")
    count = int(match.group("count"))
    unit = match.group("unit")
    if unit == "d":
        return end_date - timedelta(days=count)
    if unit == "w":
        return end_date - timedelta(weeks=count)
    if unit == "y":
        try:
            return end_date.replace(year=end_date.year - count)
        except ValueError:
            return end_date.replace(year=end_date.year - count, day=28)
    month_index = end_date.year * 12 + end_date.month - 1 - count
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    day = min(end_date.day, monthrange(year, month)[1])
    return end_date.replace(year=year, month=month, day=day)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Differential Graph Transformer on Doha Securities DSE candles, "
            "compare it with held-out real data, and forecast future bars."
        )
    )
    parser.add_argument("symbol", help="DSE symbol, for example GP or BRACBANK.")
    parser.add_argument(
        "--peers",
        default="DSEX",
        help="Comma-separated related DSE symbols used as graph nodes (default: DSEX).",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        default="1d",
        help="1min, 15min, 30min, 1h, 1d, 1w, 1mo, or 1y.",
    )
    parser.add_argument("--start", help="First historical date (YYYY-MM-DD).")
    parser.add_argument("--end", help="Final historical date (YYYY-MM-DD).")
    parser.add_argument(
        "--lookback",
        help="History ending at --end/today: 90d, 52w, 24mo, 2y, or max (default: 2y).",
    )
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--split", type=float, default=0.8, dest="split_ratio")
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--window", type=int, default=32, help="Past bars per model input.")
    parser.add_argument("--future-steps", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--graph-top-k", type=int, default=5)
    parser.add_argument("--no-spatial", action="store_true")
    parser.add_argument(
        "--no-refit",
        action="store_true",
        help="Do not refit the deployment model on all real bars before future prediction.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--open-ui", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def _print_result(run: ForecastRun, paths: dict[str, Path], repository_path: Path) -> None:
    result = run.result
    metrics = result.metrics
    print("\nHeld-out real data vs DGT prediction")
    print(f"  Accuracy (100 - sMAPE): {metrics.accuracy_score:.2f}%")
    print(f"  MAE:                     {metrics.mae:.4f} BDT")
    print(f"  RMSE:                    {metrics.rmse:.4f} BDT")
    print(f"  Directional accuracy:    {(metrics.directional_accuracy_percent or 0.0):.2f}%")
    print(f"  Skill vs naive:          {(metrics.skill_vs_naive_percent or 0.0):.2f}%")
    print(
        f"  Split:                    {result.data.context_points} context / "
        f"{result.data.holdout_points} real holdout bars"
    )
    print(
        f"  Training:                 best epoch {run.summary.best_epoch}, "
        f"{run.summary.parameter_count:,} parameters on {run.summary.device}"
    )
    print(f"  Graph:                    {', '.join(run.summary.graph_symbols)}")
    print("\nFuture forecast")
    print("  time                       predicted       q10       q90")
    for point in result.future:
        print(
            f"  {point.time.isoformat():25s} {point.predicted:9.4f} "
            f"{point.q10:9.4f} {point.q90:9.4f}"
        )
    print(f"\nSaved dashboard/API JSON: {repository_path}")
    for name, path in paths.items():
        print(f"Saved {name:10s}: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    if args.start and args.lookback:
        raise ValueError("Use either --start or --lookback, not both.")
    end_date = date.fromisoformat(args.end) if args.end else date.today()
    start_date = (
        date.fromisoformat(args.start)
        if args.start
        else _lookback_start(end_date, args.lookback or "2y")
    )
    peers = tuple(value.strip() for value in args.peers.split(",") if value.strip())
    config = DGTConfig(
        window=args.window,
        split_ratio=args.split_ratio,
        validation_ratio=args.validation_split,
        future_steps=args.future_steps,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        num_heads=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
        graph_top_k=args.graph_top_k,
        use_spatial=not args.no_spatial,
        refit_for_future=not args.no_refit,
        seed=args.seed,
    )
    run = run_dgt_prediction(
        args.symbol,
        args.resolution,
        peers=peers,
        start=start_date,
        end=end_date,
        limit=args.limit,
        config=config,
        device=args.device,
    )

    repository = PredictionRepository(DEFAULT_CONFIG["results_dir"])
    repository_path = repository.save(run.result)
    if args.output_json:
        repository.save(run.result, args.output_json)
    output_dir = args.output_dir or (
        Path(DEFAULT_CONFIG["results_dir"]) / "dgt_forecasts" / run.result.run_id
    )
    paths = save_forecast_run(run, output_dir, make_plot=not args.no_plot)
    _print_result(run, paths, repository_path)
    if args.open_ui:
        from dohasecuritiesstockai.dashboard import launch_prediction_dashboard

        launch_prediction_dashboard(run.result.run_id, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
