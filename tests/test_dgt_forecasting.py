from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import dohasecuritiesstockai.dgt_forecasting as dgt  # noqa: E402
from dohasecuritiesstockai.timesfm_forecasting.market_data import (  # noqa: E402
    MarketCandle,
    normalize_resolution,
)


def _candles(values: list[float], *, start_day: int = 0) -> list[MarketCandle]:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [
        MarketCandle(
            time=start + timedelta(days=start_day + index),
            open=value - 0.5,
            high=value + 1,
            low=value - 1,
            close=value,
            volume=1000 + index,
        )
        for index, value in enumerate(values)
    ]


def test_market_graph_alignment_forward_fills_without_back_filling():
    payloads = {
        "GP": _candles([10, 11, 12, 13, 14]),
        "DSEX": _candles([100, 102, 104], start_day=2),
    }

    def fake_fetcher(symbol, resolution, **kwargs):
        del kwargs
        return normalize_resolution(resolution), payloads[symbol]

    graph = dgt.load_dse_market_graph(
        "GP",
        ("DSEX",),
        "1d",
        fetcher=fake_fetcher,
    )

    # The first two target dates are dropped, not back-filled from DSEX day 3.
    assert [candle.close for candle in graph.target_candles] == [12, 13, 14]
    np.testing.assert_array_equal(graph.closes[:, 1], [100, 102, 104])


def test_graph_prior_uses_top_correlations_and_is_row_normalized():
    prices = np.array(
        [
            [10, 20, 30],
            [11, 22, 29],
            [12, 24, 31],
            [13, 26, 28],
            [14, 28, 32],
        ],
        dtype=np.float32,
    )

    prior = dgt.build_graph_prior(prices, top_k=2)

    assert prior.shape == (3, 3)
    assert np.all(prior >= 0)
    np.testing.assert_allclose(prior.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(np.diag(prior) > 0)


def test_dgt_forward_returns_one_prediction_per_graph_node():
    model = dgt.DifferentialGraphTransformer(
        node_count=3,
        window=8,
        hidden_size=8,
        num_heads=1,
        num_layers=1,
        dropout=0,
    )
    features = torch.randn(4, 3, 8, 1)
    prior = torch.eye(3)

    prediction = model(features, prior)

    assert prediction.shape == (4, 3)
    assert torch.isfinite(prediction).all()


def test_end_to_end_run_builds_real_comparison_and_future_forecast(tmp_path):
    point_count = 55
    step = np.arange(point_count, dtype=np.float32)
    target = 100 + 0.2 * step + np.sin(step / 5)
    benchmark = 6000 + 2 * step + 10 * np.sin(step / 6)
    market = dgt.MarketGraphData(
        symbol="GP",
        symbols=("GP", "DSEX"),
        resolution=normalize_resolution("1d"),
        target_candles=tuple(_candles(target.tolist())),
        closes=np.column_stack((target, benchmark)).astype(np.float32),
    )
    config = dgt.DGTConfig(
        window=6,
        split_ratio=0.8,
        validation_ratio=0.2,
        future_steps=3,
        epochs=1,
        patience=1,
        batch_size=16,
        hidden_size=8,
        num_heads=1,
        num_layers=1,
        dropout=0,
    )

    run = dgt.run_dgt_prediction(
        "GP", config=config, device="cpu", market_data=market, progress=None
    )

    assert len(run.result.history) == point_count
    assert len(run.result.backtest) == 11
    assert len(run.result.future) == 3
    assert run.result.backtest[0].actual == market.target_candles[44].close
    assert run.result.model.name == "Differential Graph Transformer"
    assert run.summary.graph_symbols == ("GP", "DSEX")
    assert run.summary.future_refit_epochs == 1

    paths = dgt.save_forecast_run(run, tmp_path, make_plot=False)
    assert set(paths) == {"json", "checkpoint", "comparison", "future"}
    assert all(path.is_file() for path in paths.values())
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    assert checkpoint["run_id"] == run.result.run_id
    assert checkpoint["model_config"]["symbols"] == ["GP", "DSEX"]
