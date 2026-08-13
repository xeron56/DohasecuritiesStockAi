"""Pure filtering and factor scoring for the long-horizon DSE shortlist."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any

from dohasecuritiesstockai.dataflows.dse import normalize_dse_symbol

from .schema import (
    OpportunityCandidate,
    OpportunityFactorScores,
    OpportunityMetrics,
)

FACTOR_WEIGHTS = {
    "quality_growth": 0.30,
    "valuation": 0.25,
    "financial_safety": 0.20,
    "momentum": 0.10,
    "underfollowed": 0.10,
    "data_quality": 0.05,
}

NON_EQUITY_TERMS = (
    "mutual fund",
    "mutfund",
    "bond",
    "debenture",
    "treasury",
)


def number(value: Any) -> float | None:
    if value in (None, "", "-", "N/A"):
        return None
    try:
        result = float(str(value).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def clamp(value: float, low: float = 0, high: float = 100) -> float:
    return round(max(low, min(high, value)), 1)


def percentile(value: float | None, values: list[float], *, higher_is_better: bool = True) -> float:
    clean = sorted(item for item in values if math.isfinite(item))
    if value is None or not clean:
        return 50.0
    below = sum(item < value for item in clean)
    equal = sum(item == value for item in clean)
    rank = (below + equal * 0.5) / len(clean) * 100
    return clamp(rank if higher_is_better else 100 - rank)


def quantile(values: list[float], fraction: float) -> float:
    clean = sorted(item for item in values if item > 0 and math.isfinite(item))
    if not clean:
        return 0
    index = min(len(clean) - 1, max(0, round((len(clean) - 1) * fraction)))
    return clean[index]


def _quote_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            symbol = normalize_dse_symbol(row.get("stock_code") or "")
        except ValueError:
            continue
        result[symbol] = row
    return result


def _normalized_row(row: dict[str, Any], quote: dict[str, Any] | None) -> dict[str, Any]:
    symbol = normalize_dse_symbol(row.get("s") or "")
    quote = quote or {}
    return {
        "symbol": symbol,
        "company_name": str(row.get("n") or symbol).strip(),
        "sector": str(row.get("sec") or quote.get("sector") or "Unknown").strip(),
        "category": str(row.get("c") or quote.get("category") or "").strip().upper(),
        "instrument": str(quote.get("instrument") or "EQ").strip().upper(),
        "price": number(row.get("lp")),
        "eps": number(row.get("eps")),
        "nav": number(row.get("nav")),
        "pe": number(row.get("pe")),
        "market_cap": number(row.get("mc")),
        "de": number(row.get("de")),
        "director_holdings": number(row.get("dh")),
        "dividend_yield": number(row.get("dy")),
        "roe": number(row.get("roe")),
        "vol_ma20": number(row.get("vm20")) or number(quote.get("previousDayVolume")),
        "current_volume": number(quote.get("volume")),
        "current_value": number(quote.get("value")),
        "trades": number(quote.get("trades")),
    }


def coarse_shortlist(
    screener_rows: list[dict[str, Any]],
    quote_rows: list[dict[str, Any]],
    finalist_count: int,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    """Filter the bulk universe and return diversified deterministic finalists."""

    quotes = _quote_map(quote_rows)
    normalized: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for row in screener_rows:
        try:
            symbol = normalize_dse_symbol(row.get("s") or "")
            normalized.append(_normalized_row(row, quotes.get(symbol)))
        except ValueError:
            excluded["invalid_symbol"] += 1

    liquid_values = [
        row["vol_ma20"]
        for row in normalized
        if row.get("vol_ma20") is not None and row["vol_ma20"] > 0
    ]
    liquidity_floor = max(1_000.0, quantile(liquid_values, 0.10))
    eligible: list[dict[str, Any]] = []
    for row in normalized:
        sector = row["sector"].lower()
        if row["instrument"] != "EQ" or any(term in sector for term in NON_EQUITY_TERMS):
            excluded["non_equity"] += 1
        elif row["category"] == "Z":
            excluded["distressed_category"] += 1
        elif not row["price"] or row["price"] <= 0 or not row["nav"] or row["nav"] <= 0:
            excluded["unusable_price_or_nav"] += 1
        elif not row["eps"] or row["eps"] <= 0 or not row["pe"] or row["pe"] <= 0:
            excluded["non_positive_earnings"] += 1
        elif not row["market_cap"] or row["market_cap"] <= 0:
            excluded["missing_market_cap"] += 1
        elif not row["vol_ma20"] or row["vol_ma20"] < liquidity_floor:
            excluded["thin_liquidity"] += 1
        else:
            eligible.append(row)

    roe_values = [row["roe"] for row in eligible if row["roe"] is not None]
    de_values = [row["de"] for row in eligible if row["de"] is not None and row["de"] >= 0]
    pb_values = [row["price"] / row["nav"] for row in eligible]
    yields = [row["dividend_yield"] for row in eligible if row["dividend_yield"] is not None]
    caps = [row["market_cap"] for row in eligible]
    volumes = [row["vol_ma20"] for row in eligible]
    sector_pe: defaultdict[str, list[float]] = defaultdict(list)
    for row in eligible:
        if row["pe"] and row["pe"] < 100:
            sector_pe[row["sector"]].append(row["pe"])

    for row in eligible:
        earnings_yield = row["eps"] / row["price"] * 100
        row["price_to_book"] = row["price"] / row["nav"]
        row["sector_median_pe"] = (
            statistics.median(sector_pe[row["sector"]])
            if sector_pe[row["sector"]]
            else None
        )
        row["coarse_quality"] = clamp(
            percentile(row["roe"], roe_values) * 0.65
            + clamp(earnings_yield * 5) * 0.35
        )
        pe_comparison = (
            clamp((row["sector_median_pe"] / row["pe"]) * 50)
            if row["sector_median_pe"] and row["pe"]
            else 50
        )
        row["coarse_valuation"] = clamp(
            pe_comparison * 0.45
            + percentile(row["price_to_book"], pb_values, higher_is_better=False) * 0.35
            + percentile(row["dividend_yield"], yields) * 0.20
        )
        row["coarse_safety"] = clamp(
            percentile(row["de"], de_values, higher_is_better=False) * 0.75
            + (100 if row["category"] == "A" else 65) * 0.25
        )
        row["underfollowed"] = clamp(
            percentile(row["market_cap"], caps, higher_is_better=False) * 0.65
            + percentile(row["vol_ma20"], volumes, higher_is_better=False) * 0.35
        )
        row["coarse_score"] = clamp(
            row["coarse_quality"] * 0.35
            + row["coarse_valuation"] * 0.30
            + row["coarse_safety"] * 0.20
            + row["underfollowed"] * 0.15
        )

    ordered = sorted(eligible, key=lambda item: item["coarse_score"], reverse=True)
    diversified: list[dict[str, Any]] = []
    sector_counts: Counter[str] = Counter()
    per_sector = max(2, math.ceil(finalist_count / 4))
    for row in ordered:
        if sector_counts[row["sector"]] >= per_sector:
            continue
        diversified.append(row)
        sector_counts[row["sector"]] += 1
        if len(diversified) >= finalist_count:
            break
    if len(diversified) < finalist_count:
        chosen = {row["symbol"] for row in diversified}
        diversified.extend(
            row for row in ordered if row["symbol"] not in chosen
        )
    return diversified[:finalist_count], dict(excluded), len(eligible)


def _year(row: dict[str, Any]) -> int:
    for key in ("year", "fiscal_year"):
        try:
            return int(str(row.get(key) or "")[:4])
        except ValueError:
            pass
    return 0


def _series(rows: Any, field: str, *, annual_only: bool = False) -> list[tuple[int, float]]:
    if not isinstance(rows, list):
        return []
    values: list[tuple[int, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if annual_only and str(row.get("quarter") or "Annual").lower() != "annual":
            continue
        year = _year(row)
        value = number(row.get(field))
        if year and value is not None:
            values.append((year, value))
    return sorted(dict(values).items())


def _cagr(values: list[tuple[int, float]]) -> float | None:
    usable = [(year, value) for year, value in values if value > 0]
    if len(usable) < 2:
        return None
    first_year, first = usable[0]
    last_year, last = usable[-1]
    years = last_year - first_year
    if years <= 0:
        return None
    return (pow(last / first, 1 / years) - 1) * 100


def _growth_score(value: float | None) -> float:
    if value is None:
        return 50
    return clamp(50 + value * 2.0)


def _latest_period(rows: Any) -> str | None:
    if not isinstance(rows, list) or not rows:
        return None
    row = rows[-1]
    if not isinstance(row, dict):
        return None
    for key in ("date", "year", "fiscal_year"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    return None


def score_finalist(
    row: dict[str, Any],
    evidence: dict[str, Any],
    rank: int = 1,
) -> OpportunityCandidate:
    annual_eps = _series(evidence.get("annual_financials"), "eps_basic")
    nav_values = _series(evidence.get("nav_history"), "nav_per_share")
    annual_cash = _series(evidence.get("cash_flow_history"), "nocfps", annual_only=True)
    eps_growth = _cagr(annual_eps[-6:])
    nav_growth = _cagr(nav_values[-6:])
    positive_eps_ratio = (
        sum(value > 0 for _, value in annual_eps) / len(annual_eps) * 100
        if annual_eps
        else 50
    )
    latest_eps = annual_eps[-1][1] if annual_eps else row["eps"]
    latest_cash = annual_cash[-1][1] if annual_cash else None
    cash_conversion = (
        latest_cash / latest_eps
        if latest_cash is not None and latest_eps and latest_eps > 0
        else None
    )
    conversion_score = (
        clamp(cash_conversion * 70) if cash_conversion is not None else 50
    )

    prices = [
        (item.get("date"), number(item.get("close")), number(item.get("volume")))
        for item in evidence.get("price_history", [])
        if isinstance(item, dict)
    ]
    prices = [item for item in prices if item[1] is not None and item[1] > 0]
    current_price = prices[-1][1] if prices else row["price"]
    year_start = prices[-253][1] if len(prices) >= 253 else prices[0][1] if prices else None
    return_12m = (
        (current_price / year_start - 1) * 100
        if current_price and year_start and year_start > 0
        else None
    )
    recent_prices = [value for _, value, _ in prices[-253:]]
    high_52 = max(recent_prices) if recent_prices else None
    distance_high = (
        (current_price / high_52 - 1) * 100
        if current_price and high_52 and high_52 > 0
        else None
    )
    sma_200 = statistics.fmean(recent_prices[-200:]) if len(recent_prices) >= 200 else None
    trend_score = 70 if current_price and sma_200 and current_price >= sma_200 else 35
    return_score = clamp(50 + (return_12m or 0) * 1.2)
    if return_12m is not None and return_12m > 80:
        return_score = clamp(100 - (return_12m - 80) * 0.8)
    drawdown_score = clamp(75 + (distance_high or -20) * 1.2)
    momentum_score = clamp(return_score * 0.45 + trend_score * 0.35 + drawdown_score * 0.20)

    quality_growth = clamp(
        row["coarse_quality"] * 0.35
        + _growth_score(eps_growth) * 0.30
        + _growth_score(nav_growth) * 0.15
        + positive_eps_ratio * 0.10
        + conversion_score * 0.10
    )
    financial_safety = clamp(row["coarse_safety"] * 0.65 + conversion_score * 0.35)
    missing = list(evidence.get("missing") or [])
    expected = 11
    available = max(0, expected - len(set(missing)))
    data_quality = clamp(available / expected * 100)
    factors = OpportunityFactorScores(
        quality_growth=quality_growth,
        valuation=row["coarse_valuation"],
        financial_safety=financial_safety,
        momentum=momentum_score,
        underfollowed=row["underfollowed"],
        data_quality=data_quality,
    )
    score = clamp(
        sum(getattr(factors, key) * weight for key, weight in FACTOR_WEIGHTS.items())
    )

    pe = current_price / latest_eps if current_price and latest_eps and latest_eps > 0 else row["pe"]
    nav = nav_values[-1][1] if nav_values else row["nav"]
    pb = current_price / nav if current_price and nav and nav > 0 else None
    why: list[str] = []
    red_flags: list[str] = []
    if pe and row.get("sector_median_pe") and pe < row["sector_median_pe"]:
        why.append(f"P/E {pe:.1f}× is below the sector median {row['sector_median_pe']:.1f}×.")
    if eps_growth is not None and eps_growth > 5:
        why.append(f"Reported annual EPS grew about {eps_growth:.1f}% per year across available history.")
    if row["roe"] is not None and row["roe"] >= 15:
        why.append(f"ROE is {row['roe']:.1f}%, a useful profitability signal.")
    if pb is not None and pb < 1.5:
        why.append(f"Price-to-book is {pb:.2f}× based on the latest available NAV.")
    if row["underfollowed"] >= 65:
        why.append("Market-cap and trading-activity percentiles suggest the company is less followed than many eligible peers.")
    if cash_conversion is not None and cash_conversion >= 0.8:
        why.append(f"Annual operating cash flow per share is {cash_conversion:.2f}× annual EPS.")
    if not why:
        why.append("The combined factor score is stronger than the eligible-universe alternatives reviewed in this scan.")

    if pe is not None and pe > 30:
        red_flags.append(f"P/E is elevated at {pe:.1f}×.")
    if row["de"] is not None and row["de"] > 1.5:
        red_flags.append(f"Debt-to-equity is high at {row['de']:.2f}×.")
    if row["roe"] is not None and row["roe"] < 8:
        red_flags.append(f"ROE is modest at {row['roe']:.1f}%.")
    if eps_growth is not None and eps_growth < 0:
        red_flags.append(f"Annual EPS declined about {abs(eps_growth):.1f}% per year across available history.")
    if cash_conversion is not None and cash_conversion <= 0:
        red_flags.append("Latest annual operating cash flow per share was not positive.")
    if return_12m is not None and return_12m < -25:
        red_flags.append(f"The share fell {abs(return_12m):.1f}% over the available one-year window; this may be a value trap.")
    if row["underfollowed"] >= 80:
        red_flags.append("Less-followed shares can be harder to buy or sell near the displayed price.")
    if missing:
        red_flags.append("Some evidence endpoints were unavailable; treat the ranking with lower confidence.")

    if data_quality < 40:
        label = "Insufficient evidence"
    elif score >= 70 and not any("not positive" in item for item in red_flags):
        label = "Research first"
    elif score >= 55:
        label = "Watch"
    else:
        label = "Avoid"

    periods = {
        key: value
        for key, value in {
            "annual_financials": _latest_period(evidence.get("annual_financials")),
            "quarterly_financials": _latest_period(evidence.get("quarterly_financials")),
            "nav": _latest_period(evidence.get("nav_history")),
            "cash_flow": _latest_period(evidence.get("cash_flow_history")),
            "ownership": _latest_period(evidence.get("ownership_history")),
            "price": prices[-1][0] if prices else None,
        }.items()
        if value
    }
    return OpportunityCandidate(
        rank=rank,
        symbol=row["symbol"],
        company_name=row["company_name"],
        sector=row["sector"],
        category=row["category"],
        score=score,
        research_label=label,
        factors=factors,
        metrics=OpportunityMetrics(
            current_price=current_price,
            market_cap_raw=row["market_cap"],
            eps_ttm=latest_eps,
            nav_per_share=nav,
            pe_ratio=pe,
            price_to_book=pb,
            roe_percent=row["roe"],
            debt_to_equity=row["de"],
            dividend_yield_percent=row["dividend_yield"],
            director_holdings_percent=row["director_holdings"],
            average_volume_20d=row["vol_ma20"],
            eps_growth_percent=eps_growth,
            nav_growth_percent=nav_growth,
            cash_conversion=cash_conversion,
            twelve_month_return_percent=return_12m,
            distance_from_52w_high_percent=distance_high,
        ),
        why_it_ranked=why[:6],
        red_flags=red_flags[:8],
        missing_evidence=missing,
        evidence_periods=periods,
    )
