"""Grounded DSE scoring and valuation for the stock-analysis REST contract.

The multi-agent state remains available verbatim.  This module adds a stable,
deterministic presentation layer so visible scores and valuation figures can be
reproduced from DSE disclosures instead of being invented by an LLM.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

from dohasecuritiesstockai.dataflows.dse import (
    DSEClient,
    _unwrap,
    fetch_dse_ohlcv,
    normalize_dse_symbol,
)

from .models import (
    AgentReports,
    BilingualText,
    EvidenceSource,
    FactorCard,
    MarketSnapshot,
    ReportSection,
    ScoreMetric,
    StockAnalysis,
    StockOption,
    ValuationMethod,
    ValuationSummary,
)


def text(en: str, bn: str) -> BilingualText:
    return BilingualText(en=en, bn=bn)


def _number(value: Any) -> float | None:
    if value in (None, "", "-", "N/A"):
        return None
    try:
        number = float(str(value).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value: float, low: float = 0, high: float = 10) -> float:
    return round(max(low, min(high, value)), 1)


def _mean(values: list[float], default: float = 5) -> float:
    return statistics.fmean(values) if values else default


def _median(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value) and value > 0]
    return round(statistics.median(clean), 2) if clean else None


def _year(row: dict[str, Any]) -> int:
    raw = row.get("year", row.get("fiscal_year", 0))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _annual_rows(rows: Any, cutoff_year: int) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    annual = [
        row
        for row in rows
        if isinstance(row, dict)
        and 0 < _year(row) <= cutoff_year
        and str(row.get("quarter", "Annual")).lower() == "annual"
    ]
    return sorted(annual, key=_year)


def _metric(
    key: str,
    en: str,
    bn: str,
    display_value: str,
    score: float,
) -> ScoreMetric:
    return ScoreMetric(
        key=key,
        label=text(en, bn),
        display_value=display_value,
        score=_clamp(score),
    )


def _status(score: float) -> str:
    if score >= 7:
        return "positive"
    if score >= 4:
        return "caution"
    return "negative"


def list_dse_stocks(client: DSEClient | None = None) -> list[StockOption]:
    """Return the current MAIN/PUBLIC DSE universe used by the selector."""

    payload = (client or DSEClient()).get(
        "market",
        "/stock_details/all-without-pagination",
        params={"board": "MAIN", "group": "PUBLIC"},
    )
    rows = _unwrap(payload)
    if not isinstance(rows, list):
        return []
    options: list[StockOption] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            symbol = normalize_dse_symbol(str(row.get("stock_code") or row.get("instrument")))
        except ValueError:
            continue
        options.append(
            StockOption(
                symbol=symbol,
                name=str(row.get("securityName") or symbol).strip(),
                sector=str(row.get("sector") or "").strip(),
                latest_price=_number(row.get("ltp")),
                change_percent=_number(row.get("changePercentage")),
            )
        )
    return sorted(options, key=lambda option: option.symbol)


class StockAnalysisBuilder:
    """Build the user-facing report from read-only DSE data and an agent state."""

    FUNDAMENTAL_PATHS = {
        "company": "company_details",
        "financial": "financial_performance",
        "quarterly": "quarterly_performance",
        "shareholding": "share_holding",
        "dividends": "dividend_information",
        "loan": "loan_status",
        "nav": "nav_per_year",
        "nocfps": "nocfps_history_quarter",
    }

    def __init__(self, client: DSEClient | None = None) -> None:
        self.client = client or DSEClient()

    def _fundamental(self, path: str, symbol: str) -> Any:
        return _unwrap(
            self.client.get("analytics", f"/fundamentals/{path}/{symbol}")
        )

    def _quote(self, symbol: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        payload = self.client.get(
            "market",
            "/stock_details/all-without-pagination",
            params={"board": "MAIN", "group": "PUBLIC"},
        )
        rows = _unwrap(payload)
        universe = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
        for row in universe:
            try:
                if normalize_dse_symbol(str(row.get("stock_code") or "")) == symbol:
                    return row, universe
            except ValueError:
                continue
        raise LookupError(f"No live DSE quote was returned for {symbol}")

    @staticmethod
    def _latest_balance(balance: Any, cutoff_year: int) -> dict[str, float]:
        if not isinstance(balance, dict):
            return {}
        columns = balance.get("columns")
        histories = balance.get("year_wise_data")
        if not isinstance(columns, list) or not isinstance(histories, list):
            return {}
        candidates: list[tuple[int, list[Any]]] = []
        for item in histories:
            if not isinstance(item, dict):
                continue
            for raw_year, values in item.items():
                try:
                    parsed_year = int(str(raw_year)[:4])
                except (TypeError, ValueError):
                    continue
                if parsed_year <= cutoff_year and isinstance(values, list):
                    candidates.append((parsed_year, values))
        if not candidates:
            return {}
        _, values = max(candidates, key=lambda pair: pair[0])
        return {
            str(column): number
            for column, raw in zip(columns, values, strict=False)
            if (number := _number(raw)) is not None
        }

    @staticmethod
    def _dividend_by_year(rows: Any, face_value: float, cutoff_year: int) -> dict[int, float]:
        totals: defaultdict[int, float] = defaultdict(float)
        if not isinstance(rows, list):
            return {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            year = _year(row)
            rate = _number(row.get("cash_dividend"))
            # An interim payment is not a full-year policy signal and would
            # understate trailing yield when it is the newest record.
            if (
                0 < year <= cutoff_year
                and not bool(row.get("is_interim"))
                and rate is not None
                and rate > 0
            ):
                totals[year] += rate * face_value / 100
        return dict(totals)

    def _peer_pe(
        self,
        symbol: str,
        sector: str,
        universe: list[dict[str, Any]],
        cutoff_year: int,
    ) -> float | None:
        values: list[float] = []
        peers = [
            row
            for row in universe
            if str(row.get("sector") or "").strip().lower() == sector.strip().lower()
        ][:12]
        for row in peers:
            try:
                peer = normalize_dse_symbol(str(row.get("stock_code") or ""))
            except ValueError:
                continue
            if peer == symbol:
                continue
            price = _number(row.get("ltp"))
            if not price or price <= 0:
                continue
            try:
                financial = self._fundamental("financial_performance", peer)
            except Exception:
                continue
            annual = _annual_rows(financial, cutoff_year)
            eps = _number(annual[-1].get("eps_basic")) if annual else None
            if eps and eps > 0:
                pe = price / eps
                if 0 < pe < 100:
                    values.append(pe)
        return _median(values)

    def build(
        self,
        symbol: str,
        analysis_date: date,
        agent_state: dict[str, Any] | None = None,
    ) -> StockAnalysis:
        symbol = normalize_dse_symbol(symbol)
        cutoff_year = analysis_date.year
        quote, universe = self._quote(symbol)
        data = {
            key: self._fundamental(path, symbol)
            for key, path in self.FUNDAMENTAL_PATHS.items()
        }
        balance = _unwrap(
            self.client.get("analytics", f"/api/balance_sheet/balance-sheet/{symbol}")
        )
        company = data["company"] if isinstance(data["company"], dict) else {}
        price = _number(quote.get("ltp")) or _number(quote.get("close"))
        if not price or price <= 0:
            raise LookupError(f"No usable DSE price was returned for {symbol}")

        financial = _annual_rows(data["financial"], cutoff_year)
        quarterly = _annual_rows(data["quarterly"], cutoff_year)
        annual_eps = [
            value
            for row in financial
            if (value := _number(row.get("eps_basic"))) is not None
        ]
        if quarterly:
            latest_quarterly_year = _year(quarterly[-1])
            if not financial or latest_quarterly_year > _year(financial[-1]):
                value = _number(quarterly[-1].get("eps_basic"))
                if value is not None:
                    annual_eps.append(value)
        latest_eps = annual_eps[-1] if annual_eps else None

        nav_rows = [
            row
            for row in (data["nav"] if isinstance(data["nav"], list) else [])
            if isinstance(row, dict) and 0 < _year(row) <= cutoff_year
        ]
        nav_rows.sort(key=_year)
        latest_nav = _number(nav_rows[-1].get("nav_per_share")) if nav_rows else None
        face_value = _number(company.get("face_value")) or 10
        dividends = self._dividend_by_year(data["dividends"], face_value, cutoff_year)

        profit_factor = self._profit_factor(annual_eps)
        balance_values = self._latest_balance(balance, cutoff_year)
        health_factor = self._health_factor(
            balance_values, data["loan"], data["nocfps"], latest_eps
        )
        business_factor = self._business_factor(
            annual_eps, data["shareholding"], company, cutoff_year
        )
        valuation, valuation_factor = self._valuation(
            symbol,
            price,
            latest_eps,
            latest_nav,
            financial,
            quarterly,
            dividends,
            face_value,
            str(quote.get("sector") or company.get("sector") or ""),
            universe,
            cutoff_year,
        )
        dividend_factor = self._dividend_factor(dividends, price)
        factors = [
            profit_factor,
            health_factor,
            business_factor,
            valuation_factor,
            dividend_factor,
        ]
        factor_scores = {factor.key: _mean([metric.score for metric in factor.metrics]) for factor in factors}
        score = round(
            factor_scores["profitability"] * 2.5
            + factor_scores["financial_health"] * 2.5
            + factor_scores["business_quality"] * 2
            + factor_scores["valuation"] * 1.5
            + factor_scores["dividend"] * 1.5
        )
        score = int(max(0, min(100, score)))

        momentum = self._momentum(symbol, analysis_date)
        score_label, headline = self._headline(score, momentum)
        report_sections = self._sections(
            score,
            profit_factor,
            health_factor,
            business_factor,
            valuation_factor,
            dividend_factor,
            valuation,
            momentum,
        )
        takeaways = [
            health_factor.subtitle,
            dividend_factor.subtitle,
            text(
                f"The current DSE price is Tk {price:,.1f}; the valuation is {valuation.verdict_label.en.lower()}.",
                f"বর্তমান ডিএসই দাম ৳{price:,.1f}; মূল্যায়নে শেয়ারটি {valuation.verdict_label.bn}।",
            ),
        ]
        company_name = str(company.get("name") or quote.get("securityName") or symbol)
        in_depth_title = text(
            f"{company_name} combines {health_factor.title.en.lower()} with {profit_factor.title.en.lower()} — valuation and execution still matter.",
            f"{company_name}-এর {health_factor.title.bn} এবং {profit_factor.title.bn}—তবে মূল্যায়ন ও বাস্তবায়ন গুরুত্বপূর্ণ।",
        )
        in_depth_snippet = text(
            "This report brings together DSE price history, company disclosures, dividends, ownership and the complete multi-agent debate. Open the full analysis to see the evidence, risks and final trading view.",
            "এই প্রতিবেদনে ডিএসই মূল্য ইতিহাস, কোম্পানি প্রকাশনা, লভ্যাংশ, মালিকানা এবং সম্পূর্ণ মাল্টি-এজেন্ট বিতর্ক একত্র করা হয়েছে। প্রমাণ, ঝুঁকি ও চূড়ান্ত ট্রেডিং মত দেখতে পূর্ণ বিশ্লেষণ খুলুন।",
        )

        state = agent_state or {}
        agent_reports = AgentReports(
            market_report=str(state.get("market_report") or ""),
            news_report=str(state.get("news_report") or ""),
            fundamentals_report=str(state.get("fundamentals_report") or ""),
            investment_plan=str(
                state.get("investment_plan")
                or state.get("trader_investment_decision")
                or ""
            ),
            final_trade_decision=str(state.get("final_trade_decision") or ""),
            raw_state=state,
        )
        low_52, high_52 = self._parse_range(company.get("fifty_two_weeks_moving_range"))
        return StockAnalysis(
            analysis_id=f"{symbol}-{analysis_date}",
            symbol=symbol,
            company_name=company_name,
            sector=str(company.get("sector") or quote.get("sector") or ""),
            analysis_date=analysis_date,
            generated_at=datetime.now(timezone.utc),
            market=MarketSnapshot(
                latest_price=price,
                change=_number(quote.get("change")),
                change_percent=_number(quote.get("changePercentage")),
                previous_close=_number(quote.get("ycp")),
                fifty_two_week_low=low_52,
                fifty_two_week_high=high_52,
                as_of=datetime.now(timezone.utc),
            ),
            fundamental_score=score,
            score_label=score_label,
            headline=headline,
            takeaways=takeaways,
            in_depth_title=in_depth_title,
            in_depth_snippet=in_depth_snippet,
            valuation=valuation,
            factors=factors,
            report_sections=report_sections,
            agent_reports=agent_reports,
            sources=[
                EvidenceSource(
                    name="Doha Securities DSE gateway",
                    source_type="dse_api",
                    detail="Live quote, candles, company fundamentals, balance sheet, ownership and dividends (read-only).",
                ),
                EvidenceSource(
                    name="TradingAgents full state",
                    source_type="agent_state",
                    detail="Market, news and fundamentals reports, researcher debate, risk debate and final decision.",
                ),
                EvidenceSource(
                    name="Transparent presentation calculations",
                    source_type="calculation",
                    detail="Reproducible 0–10 factor scores and four educational valuation methods.",
                ),
            ],
            disclaimer=text(
                "Educational analysis only. Fair value is a rough estimate from available figures, not a price target or investment advice.",
                "শুধু শিক্ষামূলক বিশ্লেষণ। ন্যায্য মূল্য উপলভ্য তথ্যের আনুমানিক হিসাব; এটি মূল্য লক্ষ্য বা বিনিয়োগ পরামর্শ নয়।",
            ),
        )

    @staticmethod
    def _profit_factor(eps_values: list[float]) -> FactorCard:
        usable = eps_values[-10:]
        profitable = sum(value > 0 for value in usable)
        profitable_score = profitable / len(usable) * 10 if usable else 0
        transitions = [new > old for old, new in zip(usable, usable[1:], strict=False)]
        growth_score = sum(transitions) / len(transitions) * 10 if transitions else 5
        score = _mean([profitable_score, growth_score])
        if profitable_score >= 8 and growth_score < 5:
            title = text("Profit goes up and down", "মুনাফা ওঠানামা করে")
            subtitle = text("Profitable, but growth is uneven.", "লাভজনক, তবে প্রবৃদ্ধি অসম।")
        elif score >= 7:
            title = text("Consistent profits", "ধারাবাহিক মুনাফা")
            subtitle = text("Earnings have generally moved in the right direction.", "আয় সাধারণত সঠিক দিকে এগিয়েছে।")
        else:
            title = text("Earnings need attention", "আয়ে নজর দেওয়া দরকার")
            subtitle = text("Profitability or earnings growth is not yet dependable.", "লাভজনকতা বা আয় প্রবৃদ্ধি এখনো নির্ভরযোগ্য নয়।")
        return FactorCard(
            key="profitability",
            status=_status(score),
            title=title,
            subtitle=subtitle,
            explanation=text(
                "The score compares profitable reported years and how often annual EPS improved.",
                "স্কোরে লাভজনক বছর এবং বার্ষিক ইপিএস কতবার বেড়েছে তা তুলনা করা হয়েছে।",
            ),
            metrics=[
                _metric("profitable_years", "Profitable years", "লাভজনক বছর", f"{profitable}/{len(usable)}", profitable_score),
                _metric("earnings_growth", "Earnings growth", "আয় প্রবৃদ্ধি", f"{sum(transitions)}/{len(transitions)} improving", growth_score),
            ],
        )

    @staticmethod
    def _health_factor(
        balance: dict[str, float], loan: Any, nocfps_rows: Any, latest_eps: float | None
    ) -> FactorCard:
        equity = balance.get("Total Equity") or balance.get("Total Owners Equity & Liabilities")
        cash = balance.get("Cash & cash equivalents", 0)
        debt_keys = (
            "Lease liabilities",
            "Long term loans payable within one year/ Loans and Borrowings",
            "Obligation under finance & operating leases",
        )
        debt = sum(max(0, balance.get(key, 0)) for key in debt_keys)
        if isinstance(loan, dict):
            disclosed_loan = sum(
                max(0, _number(loan.get(key)) or 0)
                for key in ("short_term_loan", "long_term_loan")
            )
            debt = max(debt, disclosed_loan)
        debt_to_equity = debt / equity if equity and equity > 0 else None
        low_debt_score = _clamp(10 - max(0, (debt_to_equity or 0) - 0.1) * 10) if equity else 5

        annual_nocfps = _annual_rows(nocfps_rows, 9999)
        latest_nocfps = _number(annual_nocfps[-1].get("nocfps")) if annual_nocfps else None
        cash_conversion = (
            latest_nocfps / latest_eps
            if latest_nocfps is not None and latest_eps and latest_eps > 0
            else None
        )
        conversion_score = _clamp((cash_conversion or 0) * 8) if cash_conversion is not None else 5
        cushion = cash / debt if debt > 0 else (1 if cash > 0 else None)
        cushion_score = _clamp((cushion or 0) * 10) if cushion is not None else 5
        score = _mean([low_debt_score, conversion_score, cushion_score])
        title = (
            text("Strong money health", "শক্তিশালী আর্থিক স্বাস্থ্য")
            if score >= 7
            else text("Mixed money health", "মিশ্র আর্থিক স্বাস্থ্য")
            if score >= 4
            else text("Financial pressure", "আর্থিক চাপ")
        )
        subtitle = (
            text("Low debt and useful cash support.", "কম ঋণ ও নগদ সহায়তা রয়েছে।")
            if score >= 7
            else text("Debt, cash flow or liquidity deserves a closer look.", "ঋণ, নগদ প্রবাহ বা তারল্য আরও খতিয়ে দেখা দরকার।")
        )
        return FactorCard(
            key="financial_health",
            status=_status(score),
            title=title,
            subtitle=subtitle,
            explanation=text(
                "DSE balance-sheet debt, cash and annual operating cash flow per share are compared with equity and EPS.",
                "ডিএসই ব্যালান্স শিটের ঋণ, নগদ ও বার্ষিক অপারেটিং ক্যাশ ফ্লো প্রতি শেয়ারকে ইকুইটি ও ইপিএসের সঙ্গে তুলনা করা হয়েছে।",
            ),
            metrics=[
                _metric("low_debt", "Low debt", "কম ঋণ", f"{debt_to_equity:.2f}× D/E" if debt_to_equity is not None else "N/A", low_debt_score),
                _metric("cash_from_profit", "Cash from profit", "মুনাফা থেকে নগদ", f"{cash_conversion:.2f}×" if cash_conversion is not None else "N/A", conversion_score),
                _metric("cash_cushion", "Cash cushion", "নগদ সুরক্ষা", f"{cushion:.2f}× debt" if cushion is not None else "N/A", cushion_score),
            ],
        )

    @staticmethod
    def _business_factor(
        eps_values: list[float], shareholding: Any, company: dict[str, Any], cutoff_year: int
    ) -> FactorCard:
        usable = eps_values[-10:]
        positive_score = sum(value > 0 for value in usable) / len(usable) * 10 if usable else 0
        if len(usable) > 1 and statistics.fmean(abs(value) for value in usable) > 0:
            variation = statistics.pstdev(usable) / statistics.fmean(abs(value) for value in usable)
            stability_score = _clamp(10 - variation * 10)
        else:
            stability_score = 5
        sponsor_values = [
            value
            for row in (shareholding if isinstance(shareholding, list) else [])
            if isinstance(row, dict) and (value := _number(row.get("sponsor_director"))) is not None
        ]
        sponsor_stability = _clamp(10 - (max(sponsor_values) - min(sponsor_values)) * 2) if sponsor_values else 5
        listing_year = _number(company.get("listing_year"))
        listing_age = cutoff_year - int(listing_year) if listing_year else 0
        tenure_score = _clamp(listing_age / 2)
        score = _mean([positive_score, stability_score, sponsor_stability, tenure_score])
        title = text("Strong business", "শক্তিশালী ব্যবসা") if score >= 7 else text("Established but mixed business", "প্রতিষ্ঠিত তবে মিশ্র ব্যবসা") if score >= 4 else text("Business quality needs proof", "ব্যবসার মানের আরও প্রমাণ দরকার")
        return FactorCard(
            key="business_quality",
            status=_status(score),
            title=title,
            subtitle=text(
                "Longevity, earnings resilience and stable sponsor ownership support the business-quality view.",
                "দীর্ঘ ইতিহাস, আয়ের স্থিতি ও স্পনসর মালিকানার স্থায়িত্ব ব্যবসার মানকে সমর্থন করে।",
            ),
            explanation=text(
                "This is an evidence-based proxy because the available endpoint does not provide complete revenue-margin or market-share history for every company.",
                "সব কোম্পানির পূর্ণ রাজস্ব-মার্জিন বা বাজার অংশীদারির ইতিহাস না থাকায় এটি প্রমাণভিত্তিক বিকল্প সূচক।",
            ),
            metrics=[
                _metric("earnings_resilience", "Earnings resilience", "আয়ের স্থিতিস্থাপকতা", f"{sum(value > 0 for value in usable)}/{len(usable)} positive", positive_score),
                _metric("earnings_stability", "Earnings stability", "আয়ের স্থিতি", "Annual EPS", stability_score),
                _metric("sponsor_stability", "Sponsor stability", "স্পনসর স্থিতি", f"{sponsor_values[-1]:.1f}% held" if sponsor_values else "N/A", sponsor_stability),
                _metric("listing_tenure", "Listing tenure", "তালিকাভুক্তির বয়স", f"{listing_age} years" if listing_year else "N/A", tenure_score),
            ],
        )

    def _valuation(
        self,
        symbol: str,
        price: float,
        latest_eps: float | None,
        latest_nav: float | None,
        financial: list[dict[str, Any]],
        quarterly: list[dict[str, Any]],
        dividends: dict[int, float],
        face_value: float,
        sector: str,
        universe: list[dict[str, Any]],
        cutoff_year: int,
    ) -> tuple[ValuationSummary, FactorCard]:
        del face_value
        historical_pe = _median(
            [value for row in financial if (value := _number(row.get("pe_ratio"))) is not None]
        )
        current_pe = price / latest_eps if latest_eps and latest_eps > 0 else None
        own_profit = latest_eps * historical_pe if latest_eps and historical_pe else None
        peer_pe = self._peer_pe(symbol, sector, universe, cutoff_year) if sector else None
        peer_profit = latest_eps * peer_pe if latest_eps and peer_pe else None

        historical_pb_values: list[float] = []
        annual_prices: dict[int, float] = {}
        annual_navs: dict[int, float] = {}
        for row in quarterly:
            year = _year(row)
            year_price = _number(row.get("market_price_end_period"))
            year_nav = _number(row.get("nav"))
            if year_price and year_nav and year_nav > 0:
                historical_pb_values.append(year_price / year_nav)
                annual_prices[year] = year_price
                annual_navs[year] = year_nav
        historical_pb = _median(historical_pb_values)
        current_pb = price / latest_nav if latest_nav and latest_nav > 0 else None
        asset_value = latest_nav * historical_pb if latest_nav and historical_pb else None

        yields = [
            dps / annual_prices[year]
            for year, dps in dividends.items()
            if year in annual_prices and annual_prices[year] > 0 and dps > 0
        ]
        historical_yield = _median(yields)
        latest_dps = dividends[max(dividends)] if dividends else None
        dividend_value = latest_dps / historical_yield if latest_dps and historical_yield else None

        raw_methods = [own_profit, peer_profit, asset_value, dividend_value]
        available = [value for value in raw_methods if value is not None and 0 < value < price * 10]
        estimate = _median(available)
        if len(available) >= 2:
            ordered = sorted(available)
            fair_low = round(ordered[0], 1)
            fair_high = round(ordered[-1], 1)
        elif estimate:
            fair_low, fair_high = round(estimate * 0.9, 1), round(estimate * 1.1, 1)
        else:
            fair_low = fair_high = None

        if estimate is None:
            verdict = "insufficient_data"
            verdict_label = text("Not enough data", "পর্যাপ্ত তথ্য নেই")
        elif price <= estimate * 0.85:
            verdict = "looks_cheap"
            verdict_label = text("Looks cheap", "সস্তা মনে হচ্ছে")
        elif price >= estimate * 1.15:
            verdict = "looks_expensive"
            verdict_label = text("Looks expensive", "দামি মনে হচ্ছে")
        else:
            verdict = "fair"
            verdict_label = text("Fair price", "ন্যায্য দাম")
        confidence = "high" if len(available) == 4 else "medium" if len(available) >= 2 else "low"
        summary = text(
            f"Based on the company's usual earnings and asset multiples, comparable-company earnings and dividend history; {confidence} confidence.",
            f"কোম্পানির স্বাভাবিক আয় ও সম্পদ গুণিতক, তুলনীয় কোম্পানির আয় এবং লভ্যাংশ ইতিহাসের ভিত্তিতে; আস্থার মাত্রা { {'high': 'উচ্চ', 'medium': 'মাঝারি', 'low': 'কম'}[confidence] }।",
        )
        methods = [
            ValuationMethod(key="historical_pe", label=text("Its own usual price vs profit", "নিজস্ব স্বাভাবিক মূল্য বনাম মুনাফা"), value=round(own_profit, 1) if own_profit else None, available=own_profit is not None),
            ValuationMethod(key="peer_pe", label=text("Priced like similar companies (profit)", "একই ধরনের কোম্পানির মতো মূল্য (মুনাফা)"), value=round(peer_profit, 1) if peer_profit else None, available=peer_profit is not None),
            ValuationMethod(key="historical_pb", label=text("Its own usual price vs asset value", "নিজস্ব স্বাভাবিক মূল্য বনাম সম্পদ মূল্য"), value=round(asset_value, 1) if asset_value else None, available=asset_value is not None),
            ValuationMethod(key="dividend_yield", label=text("Based on the dividend it pays", "প্রদত্ত লভ্যাংশের ভিত্তিতে"), value=round(dividend_value, 1) if dividend_value else None, available=dividend_value is not None),
        ]
        pe_score = _clamp(5 + ((historical_pe / current_pe) - 1) * 10) if current_pe and historical_pe else 5
        pb_score = _clamp(5 + ((historical_pb / current_pb) - 1) * 10) if current_pb and historical_pb else 5
        factor_score = _mean([pe_score, pb_score])
        factor = FactorCard(
            key="valuation",
            status=_status(factor_score),
            title=verdict_label,
            subtitle=text(
                "Price is compared with the company's own history and available DSE peers.",
                "দামকে কোম্পানির নিজস্ব ইতিহাস ও উপলভ্য ডিএসই সমকক্ষের সঙ্গে তুলনা করা হয়েছে।",
            ),
            explanation=summary,
            metrics=[
                _metric("pe_value", "P/E value", "পি/ই মূল্য", f"{current_pe:.1f}× vs {historical_pe:.1f}× usual" if current_pe and historical_pe else "N/A", pe_score),
                _metric("pb_value", "P/B value", "পি/বি মূল্য", f"{current_pb:.1f}× vs {historical_pb:.1f}× usual" if current_pb and historical_pb else "N/A", pb_score),
            ],
        )
        return (
            ValuationSummary(
                verdict=verdict,
                verdict_label=verdict_label,
                current_price=price,
                rough_estimate=round(estimate, 1) if estimate else None,
                fair_range_low=fair_low,
                fair_range_high=fair_high,
                confidence=confidence,
                summary=summary,
                methods=methods,
            ),
            factor,
        )

    @staticmethod
    def _dividend_factor(dividends: dict[int, float], price: float) -> FactorCard:
        years = sorted(dividends)
        latest_dps = dividends[years[-1]] if years else None
        dividend_yield = latest_dps / price * 100 if latest_dps and price else None
        if years:
            span = min(10, years[-1] - years[0] + 1)
            recent_start = years[-1] - span + 1
            paid_count = sum(year in dividends for year in range(recent_start, years[-1] + 1))
            consistency_score = paid_count / span * 10
        else:
            paid_count = span = 0
            consistency_score = 0
        values = [dividends[year] for year in years[-6:]]
        transitions = [new >= old for old, new in zip(values, values[1:], strict=False)]
        growth_score = sum(transitions) / len(transitions) * 10 if transitions else 5
        yield_score = _clamp((dividend_yield or 0) * 1.25) if dividend_yield is not None else 0
        score = _mean([growth_score, consistency_score, yield_score])
        if consistency_score >= 8 and (dividend_yield or 0) >= 4:
            title = text("Reliable dividend", "নির্ভরযোগ্য লভ্যাংশ")
            subtitle = text("Has paid cash regularly in recent years.", "সাম্প্রতিক বছরগুলোতে নিয়মিত নগদ লভ্যাংশ দিয়েছে।")
        elif years:
            title = text("Some dividend", "কিছু লভ্যাংশ")
            subtitle = text("Pays cash, but growth or consistency is mixed.", "নগদ দেয়, তবে প্রবৃদ্ধি বা ধারাবাহিকতা মিশ্র।")
        else:
            title = text("No recent cash dividend", "সাম্প্রতিক নগদ লভ্যাংশ নেই")
            subtitle = text("No usable cash-dividend record was returned.", "ব্যবহারযোগ্য নগদ লভ্যাংশের রেকর্ড পাওয়া যায়নি।")
        return FactorCard(
            key="dividend",
            status=_status(score),
            title=title,
            subtitle=subtitle,
            explanation=text(
                "Cash dividend percentages are converted to taka per share using face value, then compared for consistency, growth and current yield.",
                "নগদ লভ্যাংশ শতাংশকে অভিহিত মূল্য দিয়ে শেয়ারপ্রতি টাকায় রূপান্তর করে ধারাবাহিকতা, প্রবৃদ্ধি ও বর্তমান ফলন তুলনা করা হয়েছে।",
            ),
            metrics=[
                _metric("dividend_growth", "Dividend growth", "লভ্যাংশ প্রবৃদ্ধি", f"{sum(transitions)}/{len(transitions)} non-declining" if transitions else "N/A", growth_score),
                _metric("pays_consistently", "Pays consistently", "ধারাবাহিকভাবে দেয়", f"{paid_count}/{span} years" if span else "N/A", consistency_score),
                _metric("dividend_yield", "Dividend yield", "লভ্যাংশ ফলন", f"{dividend_yield:.1f}%" if dividend_yield is not None else "N/A", yield_score),
            ],
        )

    @staticmethod
    def _momentum(symbol: str, analysis_date: date) -> dict[str, float | bool | None]:
        try:
            frame = fetch_dse_ohlcv(
                symbol,
                (analysis_date - timedelta(days=180)).isoformat(),
                analysis_date.isoformat(),
            )
            closes = [float(value) for value in frame["Close"].tail(60)]
            if len(closes) < 2:
                return {"return_30d": None, "quiet": False}
            start = closes[-22] if len(closes) >= 22 else closes[0]
            change = (closes[-1] / start - 1) * 100 if start else 0
            return {"return_30d": round(change, 1), "quiet": abs(change) < 5}
        except Exception:
            return {"return_30d": None, "quiet": False}

    @staticmethod
    def _headline(score: int, momentum: dict[str, float | bool | None]) -> tuple[BilingualText, BilingualText]:
        if score >= 75:
            label = text("Very good", "খুব ভালো")
            base = text("Strong company with a solid overall profile", "সামগ্রিকভাবে শক্তিশালী কোম্পানি")
        elif score >= 60:
            label = text("Good", "ভালো")
            base = text("Good company with a few things to watch", "ভালো কোম্পানি, তবে কিছু বিষয় নজরে রাখা দরকার")
        elif score >= 45:
            label = text("Mixed", "মিশ্র")
            base = text("A mixed company — strengths and risks are balanced", "মিশ্র কোম্পানি—শক্তি ও ঝুঁকি কাছাকাছি")
        else:
            label = text("Weak", "দুর্বল")
            base = text("The numbers need more improvement", "পরিসংখ্যানে আরও উন্নতি প্রয়োজন")
        if momentum.get("quiet") and score >= 60:
            base = text("Good company, but the share is quiet", "ভালো কোম্পানি, তবে শেয়ারটি শান্ত")
        return label, base

    @staticmethod
    def _sections(
        score: int,
        profit: FactorCard,
        health: FactorCard,
        business: FactorCard,
        valuation_factor: FactorCard,
        dividend: FactorCard,
        valuation: ValuationSummary,
        momentum: dict[str, float | bool | None],
    ) -> list[ReportSection]:
        current_return = momentum.get("return_30d")
        return [
            ReportSection(key="overview", title=text("Company at a glance", "এক নজরে কোম্পানি"), summary=text(f"The evidence-based fundamental score is {score}/100.", f"প্রমাণভিত্তিক মৌলিক স্কোর {score}/১০০।"), bullets=[health.subtitle, business.subtitle]),
            ReportSection(key="earnings", title=text("Profit and earnings", "মুনাফা ও আয়"), summary=profit.explanation, bullets=[profit.title, profit.subtitle]),
            ReportSection(key="financial_health", title=text("Financial health", "আর্থিক স্বাস্থ্য"), summary=health.explanation, bullets=[health.title, health.subtitle]),
            ReportSection(key="business", title=text("Business quality", "ব্যবসার মান"), summary=business.explanation, bullets=[business.title, business.subtitle]),
            ReportSection(key="valuation", title=text("Price and valuation", "দাম ও মূল্যায়ন"), summary=valuation.summary, bullets=[valuation_factor.title, text(f"Rough estimate: Tk {valuation.rough_estimate:,.1f}" if valuation.rough_estimate else "A rough estimate could not be calculated.", f"আনুমানিক মূল্য: ৳{valuation.rough_estimate:,.1f}" if valuation.rough_estimate else "আনুমানিক মূল্য হিসাব করা যায়নি।")]),
            ReportSection(key="dividend", title=text("Dividend", "লভ্যাংশ"), summary=dividend.explanation, bullets=[dividend.title, dividend.subtitle]),
            ReportSection(key="price_behavior", title=text("Recent share behavior", "সাম্প্রতিক শেয়ার আচরণ"), summary=text(f"The roughly one-month DSE price move is {current_return:+.1f}%." if isinstance(current_return, (int, float)) else "Recent price momentum was unavailable.", f"প্রায় এক মাসে ডিএসই দামের পরিবর্তন {current_return:+.1f}%।" if isinstance(current_return, (int, float)) else "সাম্প্রতিক দামের গতি পাওয়া যায়নি।"), bullets=[]),
            ReportSection(key="bottom_line", title=text("Bottom line", "সারকথা"), summary=text("Use the score as a research summary, then read the original agent evidence and final decision before acting.", "স্কোরকে গবেষণার সারাংশ হিসেবে ব্যবহার করুন; সিদ্ধান্তের আগে মূল এজেন্ট প্রমাণ ও চূড়ান্ত মত পড়ুন।"), bullets=[]),
        ]

    @staticmethod
    def _parse_range(value: Any) -> tuple[float | None, float | None]:
        if not value:
            return None, None
        parts = str(value).replace("–", "-").split("-")
        if len(parts) != 2:
            return None, None
        return _number(parts[0]), _number(parts[1])
