from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import dohasecuritiesstockai.api.app as api_app
from dohasecuritiesstockai.dashboard import opportunity_dashboard_url
from dohasecuritiesstockai.opportunity_screener.ai import OpportunityAIReviewer
from dohasecuritiesstockai.opportunity_screener.repository import OpportunityRepository
from dohasecuritiesstockai.opportunity_screener.schema import (
    OpportunityCandidate,
    OpportunityFactorScores,
    OpportunityMethodology,
    OpportunityMetrics,
    OpportunityScanResult,
)
from dohasecuritiesstockai.opportunity_screener.scoring import (
    coarse_shortlist,
    score_finalist,
)


def _row(
    symbol: str,
    *,
    sector: str = "Engineering",
    category: str = "A",
    eps: float = 10,
    volume: float = 10_000,
) -> dict:
    return {
        "s": symbol,
        "n": f"{symbol} Limited",
        "sec": sector,
        "c": category,
        "lp": 50,
        "eps": eps,
        "nav": 40,
        "pe": 5 if eps > 0 else -5,
        "mc": 500,
        "de": 0.2,
        "dh": 45,
        "dy": 4,
        "roe": 20,
        "vm20": volume,
    }


def _quote(symbol: str, instrument: str = "EQ") -> dict:
    return {
        "stock_code": f"{symbol}'PB",
        "instrument": instrument,
        "volume": 10_000,
        "value": 500_000,
        "trades": 100,
    }


def _evidence() -> dict:
    return {
        "annual_financials": [
            {"year": 2021, "eps_basic": "5"},
            {"year": 2022, "eps_basic": "6"},
            {"year": 2023, "eps_basic": "7"},
            {"year": 2024, "eps_basic": "8"},
            {"year": 2025, "eps_basic": "10"},
        ],
        "quarterly_financials": [
            {"fiscal_year": 2026, "quarter": "Q2", "eps_basic": "3"}
        ],
        "nav_history": [
            {"year": 2021, "nav_per_share": "30"},
            {"year": 2025, "nav_per_share": "40"},
        ],
        "cash_flow_history": [
            {"fiscal_year": 2025, "quarter": "Annual", "nocfps": "12"}
        ],
        "ownership_history": [{"date": "2026-07-31", "sponsor_director": "45"}],
        "price_history": [
            {"date": f"2025-08-{day:02d}", "close": 40 + day / 10, "volume": 10_000}
            for day in range(1, 29)
        ]
        + [
            {"date": f"2026-07-{day:02d}", "close": 47 + day / 10, "volume": 12_000}
            for day in range(1, 29)
        ],
        "missing": [],
    }


def test_coarse_screen_excludes_non_equity_distress_losses_and_thin_volume() -> None:
    rows = [
        _row("GOOD1"),
        _row("GOOD2", volume=12_000),
        _row("FUND", sector="Mutual Funds"),
        _row("DISTRESS", category="Z"),
        _row("LOSS", eps=-2),
        _row("THIN", volume=500),
    ]
    quotes = [
        _quote("GOOD1"),
        _quote("GOOD2"),
        _quote("FUND", "MF"),
        _quote("DISTRESS"),
        _quote("LOSS"),
        _quote("THIN"),
    ]

    finalists, excluded, eligible = coarse_shortlist(rows, quotes, 2)

    assert {row["symbol"] for row in finalists} == {"GOOD1", "GOOD2"}
    assert eligible == 2
    assert excluded == {
        "non_equity": 1,
        "distressed_category": 1,
        "non_positive_earnings": 1,
        "thin_liquidity": 1,
    }


def test_detailed_scoring_is_reproducible_and_explains_growth() -> None:
    row = {
        "symbol": "GOOD",
        "company_name": "Good Limited",
        "sector": "Engineering",
        "category": "A",
        "price": 50,
        "eps": 10,
        "nav": 40,
        "pe": 5,
        "market_cap": 500,
        "de": 0.2,
        "director_holdings": 45,
        "dividend_yield": 4,
        "roe": 20,
        "vol_ma20": 10_000,
        "sector_median_pe": 10,
        "coarse_quality": 80,
        "coarse_valuation": 85,
        "coarse_safety": 90,
        "underfollowed": 75,
    }

    candidate = score_finalist(row, _evidence())

    assert candidate.symbol == "GOOD"
    assert candidate.score > 70
    assert candidate.research_label == "Research first"
    assert candidate.metrics.eps_growth_percent is not None
    assert candidate.metrics.eps_growth_percent > 10
    assert any("EPS grew" in reason for reason in candidate.why_it_ranked)


def _candidate() -> OpportunityCandidate:
    factors = OpportunityFactorScores(
        quality_growth=80,
        valuation=75,
        financial_safety=70,
        momentum=60,
        underfollowed=65,
        data_quality=90,
    )
    return OpportunityCandidate(
        rank=1,
        symbol="GOOD",
        company_name="Good Limited",
        sector="Engineering",
        category="A",
        score=75,
        research_label="Research first",
        factors=factors,
        metrics=OpportunityMetrics(current_price=50, pe_ratio=5),
        why_it_ranked=["Evidence-backed reason"],
        red_flags=["Evidence-backed risk"],
    )


def test_ai_reviews_only_supplied_finalists_with_structured_output() -> None:
    captured: dict[str, object] = {}
    output = {
        "reviews": [
            {
                "symbol": "GOOD",
                "verdict": "Research first",
                "confidence": "medium",
                "thesis": "Profitable and inexpensive on supplied figures.",
                "what_market_may_be_missing": "Execution may improve.",
                "multi_year_path": "Watch audited growth over several reporting periods.",
                "valuation_discipline": "Do not rely on nominal share price.",
                "catalysts": ["Earnings growth", "Cash conversion"],
                "risks": ["Illiquidity", "Margin pressure"],
                "checkpoints": ["Annual EPS", "Operating cash flow"],
            }
        ]
    }
    structured = SimpleNamespace(
        invoke=lambda messages: captured.setdefault("messages", messages) and output
    )
    llm = SimpleNamespace(with_structured_output=lambda schema: structured)
    reviewer = OpportunityAIReviewer(
        {"llm_provider": "test", "deep_think_llm": "test-model"},
        llm=llm,
    )

    reviews = reviewer.review(
        [_candidate()],
        {"GOOD": {"annual_financials": [{"year": 2025, "eps_basic": 10}]}},
        5,
    )

    assert reviews["GOOD"].verdict == "Research first"
    messages = captured["messages"]
    assert "Never guarantee profit" in messages[0]["content"]
    assert '"symbol": "GOOD"' in messages[1]["content"]
    assert '"symbol": "OTHER"' not in messages[1]["content"]


def test_repository_api_and_dashboard_url_round_trip(tmp_path: Path, monkeypatch) -> None:
    result = OpportunityScanResult(
        scan_id="opportunity-20260813-abc123",
        as_of=date(2026, 8, 13),
        generated_at="2026-08-13T10:00:00+06:00",
        horizon_years=5,
        ai_enabled=False,
        candidates=[_candidate()],
        methodology=OpportunityMethodology(
            weights={"quality_growth": 0.3},
            initial_universe=399,
            eligible_universe=150,
            detailed_finalists=16,
        ),
        sources=[],
        disclaimer="Research only.",
    )
    repository = OpportunityRepository(tmp_path)

    repository.save(result)

    assert repository.get(result.scan_id) == result
    assert repository.latest() == result
    assert repository.get("../unsafe") is None

    monkeypatch.setitem(api_app.DEFAULT_CONFIG, "results_dir", str(tmp_path))
    client = TestClient(api_app.create_app())
    assert client.get("/api/v1/opportunities/latest").json()["scan_id"] == result.scan_id
    assert client.get(f"/api/v1/opportunities/{result.scan_id}").status_code == 200
    assert client.get("/api/v1/opportunities/..%2Funsafe").status_code == 404

    assert opportunity_dashboard_url("0.0.0.0", 8000, result.scan_id) == (
        "http://127.0.0.1:8000/?view=opportunities&run=opportunity-20260813-abc123"
    )
