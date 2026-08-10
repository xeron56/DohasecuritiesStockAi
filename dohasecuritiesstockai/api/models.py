"""Versioned API contracts consumed by the Angular stock-analysis screen."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BilingualText(APIModel):
    en: str
    bn: str


class StockOption(APIModel):
    symbol: str
    name: str
    sector: str = ""
    latest_price: float | None = None
    change_percent: float | None = None


class MarketSnapshot(APIModel):
    latest_price: float
    change: float | None = None
    change_percent: float | None = None
    previous_close: float | None = None
    fifty_two_week_low: float | None = None
    fifty_two_week_high: float | None = None
    as_of: datetime


class ScoreMetric(APIModel):
    key: str
    label: BilingualText
    display_value: str
    score: float = Field(ge=0, le=10)


class FactorCard(APIModel):
    key: str
    status: Literal["positive", "caution", "negative", "neutral"]
    title: BilingualText
    subtitle: BilingualText
    explanation: BilingualText
    metrics: list[ScoreMetric]


class ValuationMethod(APIModel):
    key: str
    label: BilingualText
    value: float | None
    available: bool


class ValuationSummary(APIModel):
    verdict: Literal["looks_cheap", "fair", "looks_expensive", "insufficient_data"]
    verdict_label: BilingualText
    current_price: float
    rough_estimate: float | None
    fair_range_low: float | None
    fair_range_high: float | None
    confidence: Literal["low", "medium", "high"]
    summary: BilingualText
    methods: list[ValuationMethod]


class ReportSection(APIModel):
    key: str
    title: BilingualText
    summary: BilingualText
    bullets: list[BilingualText] = Field(default_factory=list)


class AgentReports(APIModel):
    market_report: str = ""
    news_report: str = ""
    fundamentals_report: str = ""
    investment_plan: str = ""
    final_trade_decision: str = ""
    raw_state: dict[str, Any] = Field(default_factory=dict)


class EvidenceSource(APIModel):
    name: str
    source_type: Literal["dse_api", "agent_state", "calculation"]
    detail: str


class StockAnalysis(APIModel):
    schema_version: Literal["1.0"] = "1.0"
    analysis_id: str
    symbol: str
    company_name: str
    sector: str
    analysis_date: date
    generated_at: datetime
    market: MarketSnapshot
    fundamental_score: int = Field(ge=0, le=100)
    score_label: BilingualText
    headline: BilingualText
    takeaways: list[BilingualText]
    in_depth_title: BilingualText
    in_depth_snippet: BilingualText
    valuation: ValuationSummary
    factors: list[FactorCard]
    report_sections: list[ReportSection]
    agent_reports: AgentReports
    sources: list[EvidenceSource]
    disclaimer: BilingualText


class AnalysisRequest(APIModel):
    symbol: str
    analysis_date: date | None = None
    force: bool = False


class AnalysisJob(APIModel):
    job_id: str
    symbol: str
    analysis_date: date
    status: Literal["queued", "running", "completed", "failed"]
    created_at: datetime
    updated_at: datetime
    analysis_url: str | None = None
    message: str = ""
