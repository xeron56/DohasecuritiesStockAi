from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from dohasecuritiesstockai.api.analysis import (
    StockAnalysisBuilder,
    _market_price,
    _rows_through,
)
from dohasecuritiesstockai.api.app import app
from dohasecuritiesstockai.api.models import AnalysisJob
from dohasecuritiesstockai.api.repository import AnalysisRepository


def test_health_contract() -> None:
    response = TestClient(app).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "market": "bangladesh_dse",
        "schema_version": "1.0",
    }


def test_profit_factor_is_bounded_and_grounded() -> None:
    factor = StockAnalysisBuilder._profit_factor([2.0, 3.0, 2.5, 4.0])

    assert factor.key == "profitability"
    assert factor.metrics[0].display_value == "4/4"
    assert all(0 <= metric.score <= 10 for metric in factor.metrics)


def test_interim_dividend_is_not_treated_as_full_year() -> None:
    rows = [
        {"year": 2025, "cash_dividend": "200", "is_interim": False},
        {"year": 2026, "cash_dividend": "75", "is_interim": True},
    ]

    result = StockAnalysisBuilder._dividend_by_year(rows, face_value=10, cutoff_year=2026)

    assert result == {2025: 20.0}


def test_market_closed_quote_falls_back_to_previous_close() -> None:
    quote = {"ltp": 0, "close": 0, "ycp": 258.7}

    assert _market_price(quote) == 258.7


def test_ai_evidence_rows_respect_analysis_date() -> None:
    rows = [
        {"date": "2026-08-09", "eps_basic": "10"},
        {"date": "2026-08-11", "eps_basic": "99"},
        {"year": 2025, "eps_basic": "8"},
        {"year": 2027, "eps_basic": "100"},
    ]

    assert _rows_through(rows, date(2026, 8, 10)) == [rows[0], rows[2]]


def test_historical_snapshot_uses_last_candle_at_cutoff(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-08-09", "2026-08-10"]),
            "Close": [256.6, 258.6],
            "Open": [256.0, 257.0],
            "High": [258.0, 259.0],
            "Low": [255.5, 257.0],
            "Volume": [160_000, 146_000],
        }
    )
    monkeypatch.setattr(
        "dohasecuritiesstockai.api.analysis.fetch_dse_ohlcv",
        lambda symbol, start, end: frame,
    )

    snapshot = StockAnalysisBuilder._historical_snapshot("GP", date(2026, 8, 10))

    assert snapshot["latest_price"] == 258.6
    assert snapshot["previous_close"] == 256.6
    assert snapshot["change_percent"] == 0.78
    assert snapshot["as_of"] == datetime(2026, 8, 10, tzinfo=timezone.utc)
    assert snapshot["technical_evidence"]["latest_volume"] == 146_000
    assert snapshot["technical_evidence"]["average_volume_20"] == 153_000
    assert snapshot["technical_evidence"]["recent_bars"][-1]["date"] == "2026-08-10"


def test_job_repository_round_trip(tmp_path: Path) -> None:
    repository = AnalysisRepository(tmp_path)
    now = datetime.now(timezone.utc)
    job = AnalysisJob(
        job_id="abc123",
        symbol="GP",
        analysis_date=date(2026, 8, 9),
        status="queued",
        created_at=now,
        updated_at=now,
    )

    repository.save_job(job)

    assert repository.get_job(job.job_id) == job
