"""One-call grounded AI review for deterministic opportunity finalists."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dohasecuritiesstockai.default_config import DEFAULT_CONFIG
from dohasecuritiesstockai.llm_clients import create_llm_client

from .schema import OpportunityAIReview, OpportunityCandidate

SYSTEM_PROMPT = """You are a cautious senior equity-research reviewer for the
Dhaka Stock Exchange. The application has already filtered and ranked a small
candidate set using deterministic calculations. Review only the supplied JSON.

NON-NEGOTIABLE RULES
1. Do not browse, call tools, add a company, or use unstated memory.
2. Preserve every reported figure and period. If evidence conflicts, say so.
3. A low nominal share price is not proof that a company is cheap. Evaluate
   valuation relative to earnings, assets, cash flow, quality, and risk.
4. Never guarantee profit, claim a company will become popular, give an exact
   future share price, or present a best/base/bull return percentage.
5. Do not convert raw gateway amounts into crore/million unless the field name
   explicitly supplies that unit.
6. Treat illiquidity, weak cash conversion, leverage, falling earnings, unusual
   ownership changes, missing evidence, and adverse disclosures as risks.
7. Verdicts mean research priority, not personalized buy/sell instructions.
8. Give concrete two-to-five-year checkpoints that would confirm or invalidate
   the thesis. Keep language plain enough for a beginner.

Return exactly one structured review for every supplied symbol and no others.
"""

TASK_TEMPLATE = """Review these DSE research candidates for a {horizon}-year
research horizon. Explain what the deterministic screen may have found, where it
may be wrong, and what the investor must verify before risking money.

FINALIST EVIDENCE JSON:
{evidence_json}
"""


class AIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidateReviewOutput(AIModel):
    symbol: str
    verdict: Literal["Research first", "Watch", "Avoid", "Insufficient evidence"]
    confidence: Literal["low", "medium", "high"]
    thesis: str = Field(min_length=1)
    what_market_may_be_missing: str = Field(min_length=1)
    multi_year_path: str = Field(min_length=1)
    valuation_discipline: str = Field(min_length=1)
    catalysts: list[str] = Field(min_length=2, max_length=5)
    risks: list[str] = Field(min_length=2, max_length=6)
    checkpoints: list[str] = Field(min_length=2, max_length=6)


class OpportunityAIOutput(AIModel):
    reviews: list[CandidateReviewOutput] = Field(min_length=1, max_length=20)


def _provider_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    provider = str(config.get("llm_provider", "")).lower()
    kwargs: dict[str, Any] = {}
    if provider == "google" and config.get("google_thinking_level"):
        kwargs["thinking_level"] = config["google_thinking_level"]
    elif provider == "openai" and config.get("openai_reasoning_effort"):
        kwargs["reasoning_effort"] = config["openai_reasoning_effort"]
    elif provider == "anthropic" and config.get("anthropic_effort"):
        kwargs["effort"] = config["anthropic_effort"]
    kwargs["temperature"] = 0.1
    retries = config.get("llm_max_retries")
    if retries not in (None, ""):
        kwargs["max_retries"] = max(0, int(retries))
    return kwargs


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
        raise ValueError("Opportunity AI response was not a JSON object.")
    return payload


def _compact_evidence(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return str(value)[:500]
    if isinstance(value, list):
        return [_compact_evidence(item, depth + 1) for item in value[-8:]]
    if isinstance(value, dict):
        return {
            str(key): _compact_evidence(item, depth + 1)
            for key, item in list(value.items())[:40]
            if key not in {"price_history", "missing"}
        }
    if isinstance(value, str):
        return value[:1_500]
    return value


class OpportunityAIReviewer:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        llm: Any | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.provider = str(self.config["llm_provider"])
        self.model = str(self.config["deep_think_llm"])
        if llm is None:
            client = create_llm_client(
                provider=self.provider,
                model=self.model,
                base_url=self.config.get("backend_url"),
                **_provider_kwargs(self.config),
            )
            llm = client.get_llm()
        self.llm = llm

    def _invoke(self, messages: list[dict[str, str]]) -> OpportunityAIOutput:
        try:
            structured = self.llm.with_structured_output(OpportunityAIOutput)
            result = structured.invoke(messages)
            if result is not None:
                return OpportunityAIOutput.model_validate(result)
        except (NotImplementedError, AttributeError):
            pass
        response = self.llm.invoke(
            [
                *messages,
                {
                    "role": "user",
                    "content": "Return only one JSON object matching the requested schema.",
                },
            ]
        )
        return OpportunityAIOutput.model_validate(_extract_json(response.content))

    def review(
        self,
        candidates: list[OpportunityCandidate],
        evidence_by_symbol: dict[str, dict[str, Any]],
        horizon_years: int,
    ) -> dict[str, OpportunityAIReview]:
        packets = [
            {
                "candidate": candidate.model_dump(mode="json", exclude={"ai_review"}),
                "source_evidence": _compact_evidence(
                    evidence_by_symbol.get(candidate.symbol, {})
                ),
            }
            for candidate in candidates
        ]
        task = TASK_TEMPLATE.format(
            horizon=horizon_years,
            evidence_json=json.dumps(packets, ensure_ascii=False, default=str),
        )
        result = self._invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task},
            ]
        )
        expected = {candidate.symbol for candidate in candidates}
        actual = [review.symbol.strip().upper() for review in result.reviews]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(
                "Opportunity AI response must review each finalist exactly once."
            )
        return {
            review.symbol.strip().upper(): OpportunityAIReview(
                verdict=review.verdict,
                confidence=review.confidence,
                thesis=review.thesis.strip(),
                what_market_may_be_missing=review.what_market_may_be_missing.strip(),
                multi_year_path=review.multi_year_path.strip(),
                valuation_discipline=review.valuation_discipline.strip(),
                catalysts=[item.strip() for item in review.catalysts],
                risks=[item.strip() for item in review.risks],
                checkpoints=[item.strip() for item in review.checkpoints],
            )
            for review in result.reviews
        }
