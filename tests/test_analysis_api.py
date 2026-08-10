from datetime import date, datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from dohasecuritiesstockai.api.analysis import StockAnalysisBuilder
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
