"""Read-only DSE evidence collection for opportunity screening."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import requests

from dohasecuritiesstockai.dataflows.dse import (
    DSEAPIError,
    DSEClient,
    _filter_to_date,
    _news_timestamp,
    _paged_content,
    _unwrap,
    fetch_dse_ohlcv,
    normalize_dse_symbol,
)
from dohasecuritiesstockai.dataflows.errors import (
    VendorNotConfiguredError,
    VendorRateLimitError,
)

FUNDAMENTAL_PATHS = {
    "company": "company_details",
    "annual_financials": "financial_performance",
    "quarterly_financials": "quarterly_performance",
    "nav_history": "nav_per_year",
    "cash_flow_history": "nocfps_history_quarter",
    "dividend_history": "dividend_information",
    "ownership_history": "share_holding",
    "loan_status": "loan_status",
}

SCREENER_ADDITIONAL_FIELDS = [
    "name",
    "sector",
    "last_price",
    "category",
    "sma_200",
    "vol_ma_20",
]


def _clean_news(article: dict[str, Any]) -> dict[str, Any]:
    published = _news_timestamp(article)
    return {
        "date": published.date().isoformat() if published is not None else None,
        "title": str(article.get("title") or article.get("headline") or "Untitled"),
        "category": article.get("categories") or article.get("category") or [],
        "text": str(
            article.get("newsText")
            or article.get("content")
            or article.get("summary")
            or ""
        )[:1_500],
        "reference": str(article.get("reference") or article.get("url") or ""),
    }


class OpportunityDataCollector:
    """Collect bulk universe data, then richer evidence for finalists only."""

    def __init__(self, client: DSEClient | None = None) -> None:
        self.client = client or DSEClient()

    def _post_analytics(self, path: str, payload: dict[str, Any]) -> Any:
        url = f"{self.client.service_urls['analytics']}/{path.lstrip('/')}"
        token = self.client._access_token()
        if not token:
            raise VendorNotConfiguredError(
                "DSE authentication is not configured for the screener request."
            )
        timeout, verify = self.client._request_settings()
        try:
            response = self.client.session.post(
                url,
                json=payload,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                timeout=timeout,
                verify=verify,
            )
        except requests.RequestException as exc:
            raise DSEAPIError(f"DSE request failed for {path}: {exc}") from exc
        if response.status_code in (401, 403):
            raise VendorNotConfiguredError(
                "DSE authentication failed for the screener request."
            )
        if response.status_code == 429:
            raise VendorRateLimitError("The DSE gateway rate-limited the screener request.")
        if response.status_code >= 400:
            raise DSEAPIError(
                f"DSE gateway returned HTTP {response.status_code} for {path}."
            )
        try:
            result = response.json()
        except ValueError as exc:
            raise DSEAPIError(f"DSE gateway returned non-JSON data for {path}.") from exc
        if (
            isinstance(result, dict)
            and isinstance(result.get("code"), int)
            and result["code"] >= 400
        ):
            raise DSEAPIError(
                f"DSE gateway error {result['code']} for {path}: "
                f"{result.get('message') or 'request failed'}"
            )
        return result

    def universe(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        payload = self._post_analytics(
            "/api/screener",
            {
                "filters": [],
                "field_group": "Fundamentals",
                "additional_fields": SCREENER_ADDITIONAL_FIELDS,
                "sort_by": "symbol",
                "sort_dir": "asc",
                "pattern": None,
                "sentiment": None,
                "watchList": None,
            },
        )
        screener_rows = _unwrap(payload)
        quote_payload = self.client.get(
            "market",
            "/stock_details/all-without-pagination",
            params={"board": "MAIN", "group": "PUBLIC"},
        )
        quote_rows = _unwrap(quote_payload)
        if not isinstance(screener_rows, list) or not screener_rows:
            raise DSEAPIError("DSE screener returned no universe rows.")
        if not isinstance(quote_rows, list):
            quote_rows = []
        return (
            [row for row in screener_rows if isinstance(row, dict)],
            [row for row in quote_rows if isinstance(row, dict)],
        )

    def finalist_evidence(self, symbol: str, as_of: date) -> dict[str, Any]:
        canonical = normalize_dse_symbol(symbol)
        cutoff = as_of.isoformat()
        evidence: dict[str, Any] = {}
        missing: list[str] = []

        for key, path in FUNDAMENTAL_PATHS.items():
            try:
                value = _unwrap(
                    self.client.get(
                        "analytics",
                        f"/fundamentals/{path}/{canonical}",
                    )
                )
                value = _filter_to_date(value, cutoff)
                if isinstance(value, list):
                    value = value[-12:]
                evidence[key] = value
                if value in (None, {}, []):
                    missing.append(key)
            except Exception:
                evidence[key] = None
                missing.append(key)

        try:
            balance = _unwrap(
                self.client.get(
                    "analytics",
                    f"/api/balance_sheet/balance-sheet/{canonical}",
                )
            )
            evidence["balance_sheet"] = balance
            if balance in (None, {}, []):
                missing.append("balance_sheet")
        except Exception:
            evidence["balance_sheet"] = None
            missing.append("balance_sheet")

        news_start = as_of - timedelta(days=540)
        try:
            news_payload = self.client.get(
                "market",
                "/news/all/filter",
                params={
                    "page": 1,
                    "size": 12,
                    "stockCode": canonical,
                    "from": news_start.isoformat(),
                    "to": cutoff,
                },
            )
            articles = []
            for article in _paged_content(news_payload):
                published = _news_timestamp(article)
                if published is not None and published.date() > as_of:
                    continue
                articles.append(_clean_news(article))
            evidence["recent_disclosures"] = articles[:12]
            if not articles:
                missing.append("recent_disclosures")
        except Exception:
            evidence["recent_disclosures"] = []
            missing.append("recent_disclosures")

        price_start = as_of - timedelta(days=1_100)
        try:
            frame = fetch_dse_ohlcv(
                canonical,
                price_start.isoformat(),
                cutoff,
                client=self.client,
            )
            evidence["price_history"] = [
                {
                    "date": row.Date.date().isoformat(),
                    "close": float(row.Close),
                    "volume": float(row.Volume),
                }
                for row in frame.itertuples()
            ]
        except Exception:
            evidence["price_history"] = []
            missing.append("price_history")

        evidence["missing"] = sorted(set(missing))
        return evidence
