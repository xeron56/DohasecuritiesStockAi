"""Versioned contracts for the long-term DSE opportunity screener."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ScreenerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OpportunityFactorScores(ScreenerModel):
    quality_growth: float = Field(ge=0, le=100)
    valuation: float = Field(ge=0, le=100)
    financial_safety: float = Field(ge=0, le=100)
    momentum: float = Field(ge=0, le=100)
    underfollowed: float = Field(ge=0, le=100)
    data_quality: float = Field(ge=0, le=100)


class OpportunityMetrics(ScreenerModel):
    current_price: float | None = None
    market_cap_raw: float | None = Field(
        default=None,
        validation_alias=AliasChoices("market_cap_raw", "market_cap_crore"),
    )
    eps_ttm: float | None = None
    nav_per_share: float | None = None
    pe_ratio: float | None = None
    price_to_book: float | None = None
    roe_percent: float | None = None
    debt_to_equity: float | None = None
    dividend_yield_percent: float | None = None
    director_holdings_percent: float | None = None
    average_volume_20d: float | None = None
    eps_growth_percent: float | None = None
    nav_growth_percent: float | None = None
    cash_conversion: float | None = None
    twelve_month_return_percent: float | None = None
    distance_from_52w_high_percent: float | None = None


class OpportunityAIReview(ScreenerModel):
    verdict: Literal["Research first", "Watch", "Avoid", "Insufficient evidence"]
    confidence: Literal["low", "medium", "high"]
    thesis: str
    what_market_may_be_missing: str
    multi_year_path: str
    valuation_discipline: str
    catalysts: list[str] = Field(default_factory=list, max_length=5)
    risks: list[str] = Field(default_factory=list, max_length=6)
    checkpoints: list[str] = Field(default_factory=list, max_length=6)


class OpportunityCandidate(ScreenerModel):
    rank: int = Field(ge=1)
    symbol: str
    company_name: str
    sector: str
    category: str
    score: float = Field(ge=0, le=100)
    research_label: Literal["Research first", "Watch", "Avoid", "Insufficient evidence"]
    factors: OpportunityFactorScores
    metrics: OpportunityMetrics
    why_it_ranked: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    evidence_periods: dict[str, str] = Field(default_factory=dict)
    ai_review: OpportunityAIReview | None = None


class OpportunityMethodology(ScreenerModel):
    weights: dict[str, float]
    initial_universe: int = Field(ge=0)
    eligible_universe: int = Field(ge=0)
    detailed_finalists: int = Field(ge=0)
    excluded_counts: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class OpportunitySource(ScreenerModel):
    name: str
    detail: str


class OpportunityScanResult(ScreenerModel):
    schema_version: Literal["1.0"] = "1.0"
    scan_id: str
    as_of: date
    generated_at: datetime
    horizon_years: int = Field(ge=2, le=10)
    ai_enabled: bool
    ai_provider: str | None = None
    ai_model: str | None = None
    candidates: list[OpportunityCandidate]
    methodology: OpportunityMethodology
    sources: list[OpportunitySource]
    disclaimer: str
