from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

from dohasecuritiesstockai.api.ai_analysis import (
    AIStockAnalysisGenerator,
    AIStockResearchOutput,
    _evidence_for_ai,
    _narrative_violations,
)
from dohasecuritiesstockai.api.models import (
    AgentReports,
    BilingualText,
    EvidenceSource,
    FactorCard,
    MarketSnapshot,
    ScoreMetric,
    StockAnalysis,
    ValuationMethod,
    ValuationSummary,
)


def _text(value: str) -> dict[str, str]:
    return {"en": value, "bn": f"বাংলা {value}"}


def _ai_output() -> AIStockResearchOutput:
    factors = {
        key: {"score": 8, "rationale": _text(f"{key} evidence")}
        for key in (
            "profitability",
            "financial_health",
            "business_quality",
            "valuation",
            "dividend",
        )
    }
    section_keys = (
        "company",
        "business_model",
        "profitability",
        "financial_safety",
        "valuation",
        "dividends",
        "moat",
        "bull_case",
        "risks",
        "suitability",
    )
    return AIStockResearchOutput.model_validate(
        {
            "factors": factors,
            "score_confidence": "high",
            "score_rationale": _text("Grounded score rationale"),
            "headline": _text("Strong evidence-backed profile"),
            "takeaways": [_text("One"), _text("Two"), _text("Three")],
            "in_depth_title": _text("AI in-depth research"),
            "in_depth_summary": _text("Yearly evidence was reviewed."),
            "valuation": {
                "weights": {
                    "historical_pe": 1,
                    "peer_pe": 0,
                    "historical_pb": 0,
                    "dividend_yield": 0,
                },
                "confidence": "medium",
                "summary": _text("The earnings-history method is most reliable."),
            },
            "sections": [
                {
                    "key": key,
                    "summary": _text(f"Detailed {key} analysis"),
                    "body": [_text(f"Evidence-led {key} explanation")],
                    "bullets": (
                        [
                            _text(f"{key} evidence point one"),
                            _text(f"{key} evidence point two"),
                            _text(f"{key} evidence point three"),
                        ]
                        if key in {"bull_case", "risks"}
                        else []
                    ),
                }
                for key in section_keys
            ],
            "trader": {
                "rating": "Overweight",
                "action": _text("Accumulate only with entry discipline"),
                "confidence": "medium",
                "executive_summary": _text("Use staged entries and defined risk."),
                "investment_thesis": _text("Fundamentals are constructive."),
                "entry_strategy": _text("Wait for confirmation."),
                "risk_controls": _text("Limit exposure and define invalidation."),
                "catalysts": [
                    _text("Improving earnings"),
                    _text("Stable dividend coverage"),
                ],
                "invalidation_conditions": [
                    _text("Earnings deterioration"),
                    _text("Dividend coverage weakens"),
                ],
                "time_horizon": _text("Six to twelve months"),
            },
            "data_quality": _text("Annual evidence is available; some fields are missing."),
        }
    )


def _baseline() -> StockAnalysis:
    labels = {
        "profitability": "Profitability",
        "financial_health": "Financial health",
        "business_quality": "Business quality",
        "valuation": "Valuation",
        "dividend": "Dividend",
    }
    factors = [
        FactorCard(
            key=key,
            status="caution",
            title=BilingualText(en=label, bn=label),
            subtitle=BilingualText(en="Baseline", bn="Baseline"),
            explanation=BilingualText(en="Calculated", bn="Calculated"),
            metrics=[
                ScoreMetric(
                    key="calculated",
                    label=BilingualText(en="Calculated", bn="Calculated"),
                    display_value="5/10",
                    score=5,
                )
            ],
        )
        for key, label in labels.items()
    ]
    methods = [
        ValuationMethod(
            key=key,
            label=BilingualText(en=key, bn=key),
            value=value,
            available=True,
        )
        for key, value in zip(
            ("historical_pe", "peer_pe", "historical_pb", "dividend_yield"),
            (100, 200, 300, 400),
            strict=True,
        )
    ]
    return StockAnalysis(
        analysis_id="GP-2026-08-10",
        symbol="GP",
        company_name="Grameenphone Ltd.",
        sector="Telecom",
        analysis_date=date(2026, 8, 10),
        generated_at=datetime.now(timezone.utc),
        market=MarketSnapshot(
            latest_price=50,
            as_of=datetime.now(timezone.utc),
        ),
        fundamental_score=50,
        score_label=BilingualText(en="Mixed", bn="মিশ্র"),
        headline=BilingualText(en="Baseline", bn="Baseline"),
        takeaways=[],
        in_depth_title=BilingualText(en="Baseline", bn="Baseline"),
        in_depth_snippet=BilingualText(en="Baseline", bn="Baseline"),
        valuation=ValuationSummary(
            verdict="fair",
            verdict_label=BilingualText(en="Fair price", bn="ন্যায্য দাম"),
            current_price=50,
            rough_estimate=250,
            fair_range_low=200,
            fair_range_high=300,
            confidence="high",
            summary=BilingualText(en="Calculated", bn="Calculated"),
            methods=methods,
        ),
        factors=factors,
        report_sections=[],
        agent_reports=AgentReports(),
        sources=[
            EvidenceSource(name="DSE", source_type="dse_api", detail="read-only")
        ],
        disclaimer=BilingualText(en="Educational only", bn="শিক্ষামূলক"),
    )


def test_ai_enhancement_recomputes_score_and_weighted_value() -> None:
    captured: dict[str, object] = {}
    output = _ai_output()
    structured = SimpleNamespace(
        invoke=lambda messages: captured.setdefault("messages", messages) and output
    )
    llm = SimpleNamespace(with_structured_output=lambda schema: structured)
    generator = AIStockAnalysisGenerator(
        {"llm_provider": "test", "deep_think_llm": "test-model"},
        llm=llm,
    )

    result = generator.enhance(
        _baseline(),
        {
            "market_snapshot": {"latest_price": 50},
            "annual_financial_performance": [
                {"year": 2025, "eps_basic": "12.4"}
            ],
        },
    )

    assert result.fundamental_score == 80
    assert result.score_label.en == "Very good"
    assert result.valuation.rough_estimate == 100
    assert result.valuation.fair_range_low == 80
    assert result.valuation.fair_range_high == 120
    assert result.valuation.verdict == "looks_cheap"
    assert result.ai_research is not None
    assert result.ai_research.mode == "ai_fundamental"
    assert result.ai_research.trader_report.rating == "Overweight"
    assert len(result.report_sections) == 10
    assert result.report_sections[0].title.en == "What does this company do?"
    assert result.report_sections[-1].title.en == "So, is it for you?"
    assert result.report_sections[0].body[0].en == "Evidence-led company explanation"
    assert all(factor.metrics[0].key == "ai_factor_assessment" for factor in result.factors)

    messages = captured["messages"]
    assert "Never invent" in messages[0]["content"]
    assert "Never state, estimate, or repeat an overall" in messages[0]["content"]
    assert "Never state a weighted" in messages[0]["content"]
    assert '"latest_price": 50' in messages[1]["content"]
    assert '"year": 2025' in messages[1]["content"]
    assert "annual_financial_performance" in messages[1]["content"]
    assert "ai_fundamental" in messages[1]["content"]
    assert '"fundamental_score"' not in messages[1]["content"]


def test_dedicated_research_provider_is_recorded() -> None:
    structured = SimpleNamespace(invoke=lambda messages: _ai_output())
    llm = SimpleNamespace(with_structured_output=lambda schema: structured)
    generator = AIStockAnalysisGenerator(
        {
            "llm_provider": "openrouter",
            "deep_think_llm": "provider/default",
            "research_llm_provider": "openai",
            "research_llm_model": "gpt-5.4",
        },
        llm=llm,
    )

    result = generator.enhance(_baseline(), {"market_snapshot": {"latest_price": 50}})

    assert result.ai_research is not None
    assert result.ai_research.provider == "openai"
    assert result.ai_research.model == "gpt-5.4"
    assert result.ai_research.prompt_version == "company-analysis-v2"


def test_ai_evidence_includes_date_bounded_company_datasets() -> None:
    evidence = {
        "market_snapshot": {"latest_price": 50},
        "annual_financial_performance": [{"year": 2025}],
        "quarterly_performance": [{"quarter": "Q3"}],
        "shareholding_history": [{"government": 50.35}],
        "dividend_history": [{"year": 2025, "cash_dividend": 160}],
        "nav_history": [{"year": 2025, "nav_per_share": 302.87}],
        "operating_cash_flow_per_share_history": [{"nocfps": -39.83}],
    }

    filtered = _evidence_for_ai(evidence)

    assert filtered == evidence


def test_ai_evidence_rejects_unexpected_keys() -> None:
    filtered = _evidence_for_ai(
        {
            "market_snapshot": {"latest_price": 50},
            "access_token": "must-not-leave-process",
        }
    )

    assert filtered == {"market_snapshot": {"latest_price": 50}}


def test_multi_agent_state_changes_ai_research_mode() -> None:
    output = _ai_output()
    structured = SimpleNamespace(invoke=lambda messages: output)
    llm = SimpleNamespace(with_structured_output=lambda schema: structured)
    generator = AIStockAnalysisGenerator(
        {"llm_provider": "test", "deep_think_llm": "test-model"},
        llm=llm,
    )

    result = generator.enhance(
        _baseline(),
        {},
        {"final_trade_decision": "**Rating**: Hold"},
    )

    assert result.ai_research is not None
    assert result.ai_research.mode == "multi_agent_synthesis"


def test_narrative_guard_retries_structured_output_once() -> None:
    bad_payload = _ai_output().model_dump()
    bad_payload["headline"] = {"en": "Old score 72/100", "bn": "পুরোনো ৭২/১০০"}
    bad = AIStockResearchOutput.model_validate(bad_payload)
    good = _ai_output()
    responses = iter([bad, good])
    calls: list[object] = []

    def invoke(messages):
        calls.append(messages)
        return next(responses)

    structured = SimpleNamespace(invoke=invoke)
    llm = SimpleNamespace(with_structured_output=lambda schema: structured)
    generator = AIStockAnalysisGenerator(
        {"llm_provider": "test", "deep_think_llm": "test-model"},
        llm=llm,
    )

    result = generator.enhance(_baseline(), {})

    assert result.headline.en == good.headline.en
    assert len(calls) == 2
    assert _narrative_violations(good) == []


def test_reference_format_requires_bull_and_risk_bullets() -> None:
    payload = _ai_output().model_dump()
    next(section for section in payload["sections"] if section["key"] == "risks")[
        "bullets"
    ] = []

    try:
        AIStockResearchOutput.model_validate(payload)
    except ValueError as exc:
        assert "risks must contain 3-6 evidence bullets" in str(exc)
    else:
        raise AssertionError("Risk analysis without evidence bullets must be rejected")
