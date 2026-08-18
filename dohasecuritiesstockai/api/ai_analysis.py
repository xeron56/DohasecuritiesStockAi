"""Grounded AI synthesis for the score dashboard and full analysis page."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from dohasecuritiesstockai.default_config import DEFAULT_CONFIG
from dohasecuritiesstockai.llm_clients import create_llm_client

from .models import (
    AIResearch,
    AITraderReport,
    BilingualText,
    EvidenceSource,
    ReportSection,
    ScoreMetric,
    StockAnalysis,
)
from .prompts.stock_research import (
    AI_STOCK_RESEARCH_SYSTEM_PROMPT,
    AI_STOCK_RESEARCH_TASK_TEMPLATE,
)

_SECTION_KEYS = (
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
_SECTION_TITLES = {
    "company": BilingualText(
        en="What does this company do?", bn="কোম্পানিটি কী করে?"
    ),
    "business_model": BilingualText(
        en="How does it make money?", bn="এটি কীভাবে আয় করে?"
    ),
    "profitability": BilingualText(
        en="Is it actually making money?", bn="কোম্পানিটি কি সত্যিই মুনাফা করছে?"
    ),
    "financial_safety": BilingualText(
        en="Is it financially safe?", bn="এটি কি আর্থিকভাবে নিরাপদ?"
    ),
    "valuation": BilingualText(
        en="How do we judge if the price is reasonable?",
        bn="দামটি যুক্তিসঙ্গত কি না আমরা কীভাবে বিচার করি?",
    ),
    "dividends": BilingualText(
        en="Does it reward shareholders?", bn="এটি কি শেয়ারহোল্ডারদের পুরস্কৃত করে?"
    ),
    "moat": BilingualText(en="What makes it special?", bn="কী এটিকে বিশেষ করে তোলে?"),
    "bull_case": BilingualText(en="Why it could do well", bn="কেন এটি ভালো করতে পারে"),
    "risks": BilingualText(en="What could go wrong", bn="কী ভুল হতে পারে"),
    "suitability": BilingualText(en="So, is it for you?", bn="তাহলে, এটি কি আপনার জন্য?"),
}
_FACTOR_WEIGHTS = {
    "profitability": 2.5,
    "financial_health": 2.5,
    "business_quality": 2.0,
    "valuation": 1.5,
    "dividend": 1.5,
}
_METHOD_KEYS = ("historical_pe", "peer_pe", "historical_pb", "dividend_yield")
_AI_EVIDENCE_KEYS = frozenset(
    {
        "symbol",
        "company",
        "analysis_date",
        "market_snapshot",
        "annual_financial_performance",
        "quarterly_performance",
        "balance_sheet_history",
        "shareholding_history",
        "dividend_history",
        "nav_history",
        "loan_status",
        "operating_cash_flow_per_share_history",
        "price_momentum",
        "technical_evidence",
        "calculated_factors",
        "valuation_anchors",
    }
)
_NARRATIVE_PATTERNS = {
    "overall /100 score": re.compile(
        r"(?:\d+(?:\.\d+)?\s*/\s*100|[০-৯]+\s*/\s*১০০)", re.IGNORECASE
    ),
    "not-yet-calculated aggregate valuation": re.compile(
        r"(?:rough estimate|fair range|weighted (?:rough )?estimate|"
        r"আনুমানিক মূল্য|ন্যায্য পরিসর)",
        re.IGNORECASE,
    ),
    "unsupported scaling for raw gateway financials": re.compile(
        r"(?:paid-up capital|reserve(?:s| and surplus)?|net profit|annual profit|"
        r"cash balance|cash & cash equivalents|total equity|total assets|liabilit(?:y|ies))"
        r"[^.\n]{0,80}\b(?:crore|million|billion|lakh)\b",
        re.IGNORECASE,
    ),
}


class AIText(BaseModel):
    en: str = Field(min_length=1)
    bn: str = Field(min_length=1)

    def api_text(self) -> BilingualText:
        return BilingualText(en=self.en.strip(), bn=self.bn.strip())


class AIFactorAssessment(BaseModel):
    score: float = Field(ge=0, le=10)
    rationale: AIText


class AIFactorAssessments(BaseModel):
    profitability: AIFactorAssessment
    financial_health: AIFactorAssessment
    business_quality: AIFactorAssessment
    valuation: AIFactorAssessment
    dividend: AIFactorAssessment


class AIValuationWeights(BaseModel):
    historical_pe: float = Field(ge=0, le=1)
    peer_pe: float = Field(ge=0, le=1)
    historical_pb: float = Field(ge=0, le=1)
    dividend_yield: float = Field(ge=0, le=1)


class AIValuationAssessment(BaseModel):
    weights: AIValuationWeights
    confidence: Literal["low", "medium", "high"]
    summary: AIText


class AIReportSectionOutput(BaseModel):
    key: Literal[
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
    ]
    summary: AIText
    body: list[AIText] = Field(min_length=1, max_length=4)
    bullets: list[AIText] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def enforce_reference_format(self):
        if self.key in {"bull_case", "risks"}:
            if not 3 <= len(self.bullets) <= 6:
                raise ValueError(f"{self.key} must contain 3-6 evidence bullets")
        elif self.bullets:
            raise ValueError("bullets are only allowed for bull_case and risks")
        return self


class AITraderOutput(BaseModel):
    rating: Literal["Buy", "Overweight", "Hold", "Underweight", "Sell"]
    action: AIText
    confidence: Literal["low", "medium", "high"]
    executive_summary: AIText
    investment_thesis: AIText
    entry_strategy: AIText
    risk_controls: AIText
    catalysts: list[AIText] = Field(min_length=2, max_length=5)
    invalidation_conditions: list[AIText] = Field(min_length=2, max_length=5)
    time_horizon: AIText


class AIStockResearchOutput(BaseModel):
    factors: AIFactorAssessments
    score_confidence: Literal["low", "medium", "high"]
    score_rationale: AIText
    headline: AIText
    takeaways: list[AIText] = Field(min_length=3, max_length=5)
    in_depth_title: AIText
    in_depth_summary: AIText
    valuation: AIValuationAssessment
    sections: list[AIReportSectionOutput] = Field(min_length=10, max_length=10)
    trader: AITraderOutput
    data_quality: AIText

    @model_validator(mode="after")
    def require_every_section(self):
        keys = [section.key for section in self.sections]
        if len(set(keys)) != len(keys) or set(keys) != set(_SECTION_KEYS):
            raise ValueError(f"sections must contain exactly: {', '.join(_SECTION_KEYS)}")
        return self


def _provider_kwargs(
    config: dict[str, Any], provider_override: str | None = None
) -> dict[str, Any]:
    provider = str(provider_override or config.get("llm_provider", "")).lower()
    kwargs: dict[str, Any] = {}
    if provider == "google" and config.get("google_thinking_level"):
        kwargs["thinking_level"] = config["google_thinking_level"]
    elif provider == "openai" and config.get("openai_reasoning_effort"):
        kwargs["reasoning_effort"] = config["openai_reasoning_effort"]
    elif provider == "anthropic" and config.get("anthropic_effort"):
        kwargs["effort"] = config["anthropic_effort"]
    temperature = config.get("temperature")
    kwargs["temperature"] = 0.1 if temperature in (None, "") else float(temperature)
    retries = config.get("llm_max_retries")
    if retries not in (None, ""):
        kwargs["max_retries"] = max(0, int(retries))
    return kwargs


def _clip(value: Any, limit: int = 16_000) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "\n[truncated]"


def _agent_evidence(state: dict[str, Any] | None) -> dict[str, str]:
    if not state:
        return {}
    keys = (
        "market_report",
        "news_report",
        "fundamentals_report",
        "investment_plan",
        "trader_investment_decision",
        "trader_investment_plan",
        "final_trade_decision",
    )
    return {key: _clip(state.get(key)) for key in keys if state.get(key)}


def _evidence_for_ai(evidence: dict[str, Any]) -> dict[str, Any]:
    """Allow only the date-bounded DSE datasets needed by company analysis."""

    return {
        key: value
        for key, value in evidence.items()
        if key in _AI_EVIDENCE_KEYS
    }


def _extract_json(content: str) -> dict[str, Any]:
    stripped = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    else:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("AI research response was not a JSON object")
    return payload


def _narrative_violations(result: AIStockResearchOutput) -> list[str]:
    """Find prose that claims values calculated only after the model response."""

    violations: list[str] = []

    def walk(value: Any, path: str = "output") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif isinstance(value, str):
            for label, pattern in _NARRATIVE_PATTERNS.items():
                if pattern.search(value):
                    violations.append(f"{path}: {label}")

    walk(result.model_dump(mode="json"))
    return violations


class AIStockAnalysisGenerator:
    """Generate and validate one structured AI research synthesis."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        llm: Any | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.provider = str(
            self.config.get("research_llm_provider")
            or self.config["llm_provider"]
        )
        self.model = str(
            self.config.get("research_llm_model")
            or self.config["deep_think_llm"]
        )
        if llm is None:
            research_backend = self.config.get("research_llm_backend_url")
            backend_url = (
                research_backend
                if research_backend not in (None, "")
                else self.config.get("backend_url")
                if self.provider == str(self.config["llm_provider"])
                else None
            )
            client = create_llm_client(
                provider=self.provider,
                model=self.model,
                base_url=backend_url,
                **_provider_kwargs(self.config, self.provider),
            )
            llm = client.get_llm()
        self.llm = llm

    def _invoke(self, messages: list[dict[str, str]]) -> AIStockResearchOutput:
        try:
            structured = self.llm.with_structured_output(AIStockResearchOutput)
            result = structured.invoke(messages)
            if result is not None:
                parsed = AIStockResearchOutput.model_validate(result)
                violations = _narrative_violations(parsed)
                if not violations:
                    return parsed
                correction = {
                    "role": "user",
                    "content": (
                        "Your prior structured response violated these grounding rules:\n- "
                        + "\n- ".join(violations)
                        + "\nReturn a corrected full structured object. Preserve supported evidence, "
                        "but remove every prohibited aggregate claim or unsupported unit.\n\n"
                        "PRIOR RESPONSE:\n"
                        + parsed.model_dump_json()
                    ),
                }
                corrected = AIStockResearchOutput.model_validate(
                    structured.invoke([*messages, correction])
                )
                remaining = _narrative_violations(corrected)
                if remaining:
                    raise ValueError(
                        "AI research narrative remained ungrounded after correction: "
                        + "; ".join(remaining)
                    )
                return corrected
        except (NotImplementedError, AttributeError):
            pass

        json_messages = [
            *messages,
            {
                "role": "user",
                "content": (
                    "Structured-output binding is unavailable. Return only one JSON object "
                    "that validates against the requested schema; no markdown fences or prose."
                ),
            },
        ]
        response = self.llm.invoke(json_messages)
        parsed = AIStockResearchOutput.model_validate(_extract_json(response.content))
        violations = _narrative_violations(parsed)
        if violations:
            raise ValueError(
                "AI research narrative violated grounding rules: " + "; ".join(violations)
            )
        return parsed

    def enhance(
        self,
        analysis: StockAnalysis,
        evidence: dict[str, Any],
        agent_state: dict[str, Any] | None = None,
    ) -> StockAnalysis:
        mode = "multi_agent_synthesis" if agent_state else "ai_fundamental"
        prompt_evidence = {
            **_evidence_for_ai(evidence),
            "multi_agent_reports": _agent_evidence(agent_state),
        }
        task = AI_STOCK_RESEARCH_TASK_TEMPLATE.format(
            symbol=analysis.symbol,
            analysis_date=analysis.analysis_date.isoformat(),
            mode=mode,
            evidence_json=json.dumps(prompt_evidence, ensure_ascii=False, default=str),
        )
        result = self._invoke(
            [
                {"role": "system", "content": AI_STOCK_RESEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": task},
            ]
        )
        return self._apply(analysis, result, mode)

    def _apply(
        self,
        analysis: StockAnalysis,
        result: AIStockResearchOutput,
        mode: Literal["ai_fundamental", "multi_agent_synthesis"],
    ) -> StockAnalysis:
        factor_outputs = result.factors.model_dump()
        factor_scores = {
            key: float(value["score"])
            for key, value in factor_outputs.items()
        }
        score = round(
            sum(factor_scores[key] * weight for key, weight in _FACTOR_WEIGHTS.items())
        )
        score = max(0, min(100, score))
        if score >= 75:
            score_label = BilingualText(en="Very good", bn="খুব ভালো")
        elif score >= 60:
            score_label = BilingualText(en="Good", bn="ভালো")
        elif score >= 45:
            score_label = BilingualText(en="Mixed", bn="মিশ্র")
        else:
            score_label = BilingualText(en="Weak", bn="দুর্বল")

        factors = []
        for factor in analysis.factors:
            assessment = getattr(result.factors, factor.key)
            ai_metric = ScoreMetric(
                key="ai_factor_assessment",
                label=BilingualText(en="AI factor judgment", bn="এআই ফ্যাক্টর মূল্যায়ন"),
                display_value=f"{assessment.score:.1f}/10",
                score=assessment.score,
            )
            factors.append(
                factor.model_copy(
                    update={
                        "status": (
                            "positive"
                            if assessment.score >= 7
                            else "caution"
                            if assessment.score >= 4
                            else "negative"
                        ),
                        "explanation": assessment.rationale.api_text(),
                        "metrics": [ai_metric, *factor.metrics],
                    }
                )
            )

        valuation, method_weights = self._weighted_valuation(analysis, result)
        ordered_sections = {section.key: section for section in result.sections}
        sections = [
            ReportSection(
                key=key,
                title=_SECTION_TITLES[key],
                summary=ordered_sections[key].summary.api_text(),
                body=[paragraph.api_text() for paragraph in ordered_sections[key].body],
                bullets=[bullet.api_text() for bullet in ordered_sections[key].bullets],
            )
            for key in _SECTION_KEYS
        ]
        trader = result.trader
        ai_research = AIResearch(
            provider=self.provider,
            model=self.model,
            mode=mode,
            generated_at=datetime.now(timezone.utc),
            score_confidence=result.score_confidence,
            score_rationale=result.score_rationale.api_text(),
            data_quality=result.data_quality.api_text(),
            valuation_method_weights=method_weights,
            trader_report=AITraderReport(
                rating=trader.rating,
                action=trader.action.api_text(),
                confidence=trader.confidence,
                executive_summary=trader.executive_summary.api_text(),
                investment_thesis=trader.investment_thesis.api_text(),
                entry_strategy=trader.entry_strategy.api_text(),
                risk_controls=trader.risk_controls.api_text(),
                catalysts=[item.api_text() for item in trader.catalysts],
                invalidation_conditions=[
                    item.api_text() for item in trader.invalidation_conditions
                ],
                time_horizon=trader.time_horizon.api_text(),
            ),
        )
        ai_source = EvidenceSource(
            name=f"{self.provider}:{self.model}",
            source_type="ai_analysis",
            detail=(
                "Structured AI synthesis of the supplied date-bounded DSE evidence; "
                "numeric valuation anchors remain calculation-backed."
            ),
        )
        return analysis.model_copy(
            update={
                "fundamental_score": score,
                "score_label": score_label,
                "headline": result.headline.api_text(),
                "takeaways": [item.api_text() for item in result.takeaways],
                "in_depth_title": result.in_depth_title.api_text(),
                "in_depth_snippet": result.in_depth_summary.api_text(),
                "valuation": valuation,
                "factors": factors,
                "report_sections": sections,
                "ai_research": ai_research,
                "sources": [*analysis.sources, ai_source],
            }
        )

    @staticmethod
    def _weighted_valuation(
        analysis: StockAnalysis,
        result: AIStockResearchOutput,
    ) -> tuple[Any, dict[str, float]]:
        methods = {method.key: method for method in analysis.valuation.methods}
        requested = result.valuation.weights.model_dump()
        usable = {
            key: methods[key].value
            for key in _METHOD_KEYS
            if key in methods and methods[key].available and methods[key].value is not None
        }
        raw_weights = {key: requested[key] if key in usable else 0.0 for key in _METHOD_KEYS}
        total = sum(raw_weights.values())
        if usable and total <= 0:
            raw_weights = {key: (1.0 if key in usable else 0.0) for key in _METHOD_KEYS}
            total = float(len(usable))
        normalized = {
            key: round(raw_weights[key] / total, 4) if total else 0.0
            for key in _METHOD_KEYS
        }
        estimate = (
            sum(float(usable[key]) * normalized[key] for key in usable)
            if usable
            else None
        )
        estimate = round(estimate, 1) if estimate else None
        fair_low = round(estimate * 0.8, 1) if estimate else None
        fair_high = round(estimate * 1.2, 1) if estimate else None
        price = analysis.market.latest_price
        if estimate is None:
            verdict = "insufficient_data"
            label = BilingualText(en="Not enough data", bn="পর্যাপ্ত তথ্য নেই")
        elif price <= estimate * 0.85:
            verdict = "looks_cheap"
            label = BilingualText(en="Looks cheap", bn="সস্তা মনে হচ্ছে")
        elif price >= estimate * 1.15:
            verdict = "looks_expensive"
            label = BilingualText(en="Looks expensive", bn="দামি মনে হচ্ছে")
        else:
            verdict = "fair"
            label = BilingualText(en="Fair price", bn="ন্যায্য দাম")
        valuation = analysis.valuation.model_copy(
            update={
                "verdict": verdict,
                "verdict_label": label,
                "rough_estimate": estimate,
                "fair_range_low": fair_low,
                "fair_range_high": fair_high,
                "confidence": result.valuation.confidence,
                "summary": result.valuation.summary.api_text(),
            }
        )
        return valuation, normalized
