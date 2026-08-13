"""Orchestration for deterministic screening, detailed evidence, and AI review."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from dohasecuritiesstockai.default_config import DEFAULT_CONFIG

from .ai import OpportunityAIReviewer
from .data import OpportunityDataCollector
from .schema import (
    OpportunityMethodology,
    OpportunityScanResult,
    OpportunitySource,
)
from .scoring import FACTOR_WEIGHTS, coarse_shortlist, score_finalist

ProgressCallback = Callable[[str], None]


def run_opportunity_scan(
    *,
    horizon_years: int = 5,
    limit: int = 8,
    finalist_count: int = 16,
    use_ai: bool = True,
    config: dict[str, Any] | None = None,
    collector: OpportunityDataCollector | None = None,
    reviewer: OpportunityAIReviewer | None = None,
    progress: ProgressCallback | None = None,
) -> OpportunityScanResult:
    """Run one current DSE scan and return a persistable research shortlist."""

    if not 2 <= horizon_years <= 10:
        raise ValueError("horizon_years must be between 2 and 10.")
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20.")
    if not limit <= finalist_count <= 50:
        raise ValueError("finalist_count must be between limit and 50.")

    cfg = config or DEFAULT_CONFIG
    data = collector or OpportunityDataCollector()
    notify = progress or (lambda _message: None)
    now = datetime.now(ZoneInfo("Asia/Dhaka"))
    as_of = now.date()

    notify("Loading the DSE equity universe and bulk fundamentals…")
    screener_rows, quote_rows = data.universe()
    finalists, excluded, eligible_count = coarse_shortlist(
        screener_rows,
        quote_rows,
        finalist_count,
    )
    if not finalists:
        raise ValueError("No DSE companies passed the safety and liquidity filters.")

    notify(f"Collecting detailed evidence for {len(finalists)} finalists…")
    evidence_by_symbol: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(6, len(finalists))) as executor:
        futures = {
            executor.submit(data.finalist_evidence, row["symbol"], as_of): row
            for row in finalists
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                evidence_by_symbol[row["symbol"]] = future.result()
            except Exception:
                evidence_by_symbol[row["symbol"]] = {
                    "missing": [
                        "company",
                        "annual_financials",
                        "quarterly_financials",
                        "nav_history",
                        "cash_flow_history",
                        "dividend_history",
                        "ownership_history",
                        "loan_status",
                        "balance_sheet",
                        "recent_disclosures",
                        "price_history",
                    ]
                }

    scored = [
        score_finalist(row, evidence_by_symbol[row["symbol"]])
        for row in finalists
    ]
    scored.sort(key=lambda candidate: candidate.score, reverse=True)
    candidates = [
        candidate.model_copy(update={"rank": index})
        for index, candidate in enumerate(scored[:limit], start=1)
    ]

    provider = None
    model = None
    if use_ai:
        notify(f"Running one grounded AI review for {len(candidates)} candidates…")
        ai = reviewer or OpportunityAIReviewer(cfg)
        reviews = ai.review(candidates, evidence_by_symbol, horizon_years)
        candidates = [
            candidate.model_copy(update={"ai_review": reviews[candidate.symbol]})
            for candidate in candidates
        ]
        provider = ai.provider
        model = ai.model

    scan_id = f"opportunity-{as_of:%Y%m%d}-{uuid.uuid4().hex[:10]}"
    return OpportunityScanResult(
        scan_id=scan_id,
        as_of=as_of,
        generated_at=now,
        horizon_years=horizon_years,
        ai_enabled=use_ai,
        ai_provider=provider,
        ai_model=model,
        candidates=candidates,
        methodology=OpportunityMethodology(
            weights=FACTOR_WEIGHTS,
            initial_universe=len(screener_rows),
            eligible_universe=eligible_count,
            detailed_finalists=len(finalists),
            excluded_counts=excluded,
            notes=[
                "Nominal share price is not treated as cheapness; valuation is relative to earnings and assets.",
                "Funds, bonds, Z-category names, loss-making companies, unusable prices, and the thinnest liquidity are excluded before ranking.",
                "AI reviews only the final shortlist and cannot change deterministic figures, ranks, or scores.",
            ],
        ),
        sources=[
            OpportunitySource(
                name="Doha Securities DSE gateway",
                detail="Authenticated read-only screener, market, fundamentals, disclosures, and candle endpoints discovered in the supplied web-ui source.",
            ),
            OpportunitySource(
                name="Transparent factor calculations",
                detail="Reproducible quality/growth, valuation, safety, momentum, under-followed, and data-quality scores.",
            ),
            *(
                [
                    OpportunitySource(
                        name=f"{provider}:{model}",
                        detail="One structured evidence-only review of the deterministic finalists.",
                    )
                ]
                if use_ai
                else []
            ),
        ],
        disclaimer=(
            "Research shortlist only—not personalized investment advice or a promise of profit. "
            "Small and less-followed shares can be illiquid and can lose most of their value. "
            "Verify the latest audited filings, disclosures, liquidity, and suitability with a "
            "licensed professional before investing money."
        ),
    )
