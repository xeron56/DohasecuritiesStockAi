"""Dhaka Stock Exchange data vendor backed by the Doha Securities gateway.

The endpoint paths and response shapes in this module come from the Angular
application shipped in ``../web-ui``.  The gateway is protected by the same
OAuth bearer token used by that application.  Set ``DSE_ACCESS_TOKEN`` at run
time; tokens are never persisted or included in exception messages.

All historical endpoints are filtered again client-side to the requested
analysis date.  That defensive boundary is important for backtests: even when
an API accepts ``from``/``to`` parameters, a malformed or overly broad response
must not leak future candles, news, or financial records into an older run.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
from stockstats import wrap

from .config import get_config
from .errors import (
    NoMarketDataError,
    VendorError,
    VendorNotConfiguredError,
    VendorRateLimitError,
)
from .stockstats_utils import _assert_ohlcv_not_stale, _clean_dataframe
from .utils import safe_ticker_component

DEFAULT_DSE_GATEWAY_URL = "http://192.168.3.32:9071"
DSE_ACCESS_TOKEN_ENV = "DSE_ACCESS_TOKEN"
DSE_EMAIL_OR_PHONE_ENV = "DSE_EMAIL_OR_PHONE"
DSE_PASSWORD_ENV = "DSE_PASSWORD"
DSE_AUTH_GRANT_TYPE_ENV = "DSE_AUTH_GRANT_TYPE"
DSE_AUTH_CLIENT_ID_ENV = "DSE_AUTH_CLIENT_ID"
DSE_AUTH_SCOPE_ENV = "DSE_AUTH_SCOPE"
DSE_INDEX_SYMBOLS = frozenset({"DSEX", "DSES", "DS30"})

DEFAULT_DSE_AUTH_PATH = "/usermanagementservice/v1/auth_server/token/custom-flow"
DEFAULT_DSE_AUTH_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:mobile-application"
DEFAULT_DSE_AUTH_CLIENT_ID = "oms"
DEFAULT_DSE_AUTH_SCOPE = "android"

_DSE_SUFFIXES = (".DSE", ".DH")
_SYMBOL_RE = re.compile(r"^[A-Z0-9&._-]{1,32}$")
_DATE_KEYS = ("timestamp", "date", "published_at", "created_at", "event_date")
_YEAR_KEYS = ("year", "fiscal_year", "listing_year")
_AUTH_CACHE: dict[tuple[str, str, str, str], tuple[str, float]] = {}
_AUTH_CACHE_LOCK = threading.Lock()


class DSEAPIError(VendorError):
    """The DSE gateway returned an unexpected non-authentication failure."""


def normalize_dse_symbol(symbol: str) -> str:
    """Return the base DSE trading code accepted by analytics endpoints.

    The OMS UI uses board-qualified values such as ``GP'PB`` while analytics
    routes use ``GP``.  ``.DSE`` and ``.DH`` are accepted as convenient input
    aliases.  The strict character check also prevents path traversal when a
    symbol is interpolated into a REST path.
    """

    value = str(symbol or "").strip().upper()
    if "'" in value:
        value = value.split("'", 1)[0]
    for suffix in _DSE_SUFFIXES:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    if not value or not _SYMBOL_RE.fullmatch(value):
        raise ValueError(
            f"Invalid DSE symbol {symbol!r}. Use a base trading code such as "
            "GP, BRACBANK, or SQURPHARMA."
        )
    return value


def is_dse_market(config: dict[str, Any] | None = None) -> bool:
    """Whether the active configuration is intentionally DSE-backed."""

    cfg = config or get_config()
    profile = str(cfg.get("market_profile", "")).strip().lower()
    if profile:
        return profile == "bangladesh_dse"
    vendors = cfg.get("data_vendors", {})
    return any(
        "dse" in [part.strip().lower() for part in str(vendors.get(key, "")).split(",")]
        for key in ("core_stock_apis", "technical_indicators", "fundamental_data", "news_data")
    )


def _service_urls(config: dict[str, Any]) -> dict[str, str]:
    gateway = str(config.get("dse_gateway_url") or DEFAULT_DSE_GATEWAY_URL).rstrip("/")
    return {
        "market": str(config.get("dse_market_data_url") or f"{gateway}/marketdataservice").rstrip(
            "/"
        ),
        "analytics": str(
            config.get("dse_analytics_url") or f"{gateway}/market-analytics-service/v1"
        ).rstrip("/"),
    }


class DSEClient:
    """Small authenticated JSON client for the DSE/OMS gateway."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config or get_config()
        self.service_urls = _service_urls(self.config)
        self.session = session or requests.Session()

    def _request_settings(self) -> tuple[float, bool]:
        return (
            float(self.config.get("dse_request_timeout", 30)),
            bool(self.config.get("dse_verify_ssl", True)),
        )

    @staticmethod
    def _token_from_payload(payload: Any) -> str:
        """Extract an access token without ever serializing the response."""

        if isinstance(payload, str):
            return payload.strip()
        if not isinstance(payload, dict):
            return ""
        for key in ("access_token", "accessToken", "token", "id_token", "idToken"):
            token = payload.get(key)
            if isinstance(token, str) and token.strip():
                return token.strip()
        for key in ("data", "result", "auth", "authentication"):
            token = DSEClient._token_from_payload(payload.get(key))
            if token:
                return token
        return ""

    @staticmethod
    def _expires_in(payload: Any) -> float:
        if not isinstance(payload, dict):
            return 900
        for key in ("expires_in", "expiresIn", "expires"):
            raw = payload.get(key)
            try:
                return max(60, float(raw))
            except (TypeError, ValueError):
                pass
        for key in ("data", "result", "auth", "authentication"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                value = DSEClient._expires_in(nested)
                if value != 900:
                    return value
        return 900

    def _login_with_credentials(self) -> str:
        """Exchange DSE login credentials for a process-local bearer token."""

        email_or_phone = os.environ.get(DSE_EMAIL_OR_PHONE_ENV, "").strip()
        password = os.environ.get(DSE_PASSWORD_ENV, "")
        if not email_or_phone or not password:
            return ""

        gateway = str(self.config.get("dse_gateway_url") or DEFAULT_DSE_GATEWAY_URL).rstrip("/")
        auth_path = str(self.config.get("dse_auth_path") or DEFAULT_DSE_AUTH_PATH)
        grant_type = os.environ.get(DSE_AUTH_GRANT_TYPE_ENV, DEFAULT_DSE_AUTH_GRANT_TYPE)
        client_id = os.environ.get(DSE_AUTH_CLIENT_ID_ENV, DEFAULT_DSE_AUTH_CLIENT_ID)
        scope = os.environ.get(DSE_AUTH_SCOPE_ENV, DEFAULT_DSE_AUTH_SCOPE)
        cache_key = (gateway, client_id, scope, email_or_phone)

        with _AUTH_CACHE_LOCK:
            cached = _AUTH_CACHE.get(cache_key)
            if cached and cached[1] > time.monotonic():
                return cached[0]

            timeout, verify = self._request_settings()
            try:
                response = self.session.post(
                    f"{gateway}/{auth_path.lstrip('/')}",
                    json={
                        "grantType": grant_type,
                        "authorizationClientId": client_id,
                        "scope": scope,
                        "emailOrPhone": email_or_phone,
                        "password": password,
                    },
                    headers={"Accept": "application/json", "Content-Type": "application/json"},
                    timeout=timeout,
                    verify=verify,
                )
            except requests.RequestException as exc:
                raise DSEAPIError("DSE credential login request failed.") from exc

            if response.status_code in (400, 401, 403):
                raise VendorNotConfiguredError(
                    "DSE credential login was rejected. Check DSE_EMAIL_OR_PHONE, "
                    "DSE_PASSWORD, and the mobile authentication settings."
                )
            if response.status_code == 429:
                raise VendorRateLimitError("The DSE gateway rate-limited credential login.")
            if response.status_code >= 400:
                raise DSEAPIError(f"DSE credential login returned HTTP {response.status_code}.")
            try:
                payload = response.json()
            except ValueError as exc:
                raise DSEAPIError("DSE credential login returned non-JSON data.") from exc

            token = self._token_from_payload(payload)
            if not token:
                raise DSEAPIError(
                    "DSE credential login succeeded but the response contained no access token."
                )
            # Refresh before the advertised expiry. When the response omits an
            # expiry, use a conservative 15-minute in-memory lifetime.
            lifetime = self._expires_in(payload)
            _AUTH_CACHE[cache_key] = (token, time.monotonic() + max(30, lifetime - 60))
            return token

    def _access_token(self) -> str:
        explicit = os.environ.get(DSE_ACCESS_TOKEN_ENV, "").strip()
        return explicit or self._login_with_credentials()

    def get(
        self,
        service: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        if service not in self.service_urls:
            raise ValueError(f"Unknown DSE service {service!r}")

        url = f"{self.service_urls[service]}/{path.lstrip('/')}"
        token = self._access_token()
        if not token:
            raise VendorNotConfiguredError(
                "DSE authentication is not configured. Set DSE_ACCESS_TOKEN, or leave it "
                "empty and set both DSE_EMAIL_OR_PHONE and DSE_PASSWORD for automatic login."
            )
        headers = {"Accept": "application/json"}
        headers["Authorization"] = f"Bearer {token}"

        timeout, verify = self._request_settings()
        try:
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
                verify=verify,
            )
        except requests.RequestException as exc:
            raise DSEAPIError(f"DSE request failed for {path}: {exc}") from exc

        if response.status_code in (401, 403):
            raise VendorNotConfiguredError(
                "DSE authentication failed: the access token is expired or is not authorized "
                "for this route. Refresh DSE_ACCESS_TOKEN or verify the configured DSE login."
            )
        if response.status_code == 429:
            raise VendorRateLimitError("The DSE gateway rate-limited this request.")
        if response.status_code >= 400:
            raise DSEAPIError(f"DSE gateway returned HTTP {response.status_code} for {path}.")

        try:
            payload = response.json()
        except ValueError as exc:
            raise DSEAPIError(f"DSE gateway returned non-JSON data for {path}.") from exc

        if isinstance(payload, dict):
            code = payload.get("code")
            if isinstance(code, int) and code >= 400:
                message = payload.get("message") or "request failed"
                raise DSEAPIError(f"DSE gateway error {code} for {path}: {message}")
        return payload


def _unwrap(payload: Any) -> Any:
    """Unwrap the gateway's common ``{code, message, data}`` envelope."""

    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def _parse_date(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.tz_convert(None)


def _timestamp_to_date(value: Any) -> pd.Timestamp | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _parse_date(value)
    if numeric > 10_000_000_000:  # tolerate millisecond timestamps
        numeric /= 1000
    try:
        return pd.Timestamp(datetime.fromtimestamp(numeric, tz=timezone.utc)).tz_localize(None)
    except (OSError, OverflowError, ValueError):
        return None


def _validate_range(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.to_datetime(start_date, format="%Y-%m-%d", errors="raise")
    end = pd.to_datetime(end_date, format="%Y-%m-%d", errors="raise")
    if start > end:
        raise ValueError(f"start_date {start_date} must not be after end_date {end_date}")
    return start, end


def fetch_dse_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    client: DSEClient | None = None,
) -> pd.DataFrame:
    """Fetch and normalize daily DSE candles into a capitalized OHLCV frame."""

    start, end = _validate_range(start_date, end_date)
    canonical = normalize_dse_symbol(symbol)
    day_span = max(1, (end - start).days + 1)
    symbol_type = "index" if canonical in DSE_INDEX_SYMBOLS else "stock"
    payload = (client or DSEClient()).get(
        "analytics",
        "/candlesticks/limited",
        params={
            "symbol": canonical,
            "type": symbol_type,
            "resolution": "1D",
            "from": start_date,
            "to": end_date,
            # Trading days are fewer than calendar days, but a generous limit
            # avoids truncating long windows if the API applies it before dates.
            "limit": max(64, day_span + 32),
        },
    )
    candles = _unwrap(payload)
    if not isinstance(candles, dict):
        raise NoMarketDataError(symbol, canonical, "DSE returned no candle object")

    arrays = {name: candles.get(name) for name in ("t", "o", "h", "l", "c")}
    volume = candles.get("v")
    if volume is None:
        volume = candles.get("vo")
    arrays["v"] = volume
    lengths = [len(value) for value in arrays.values() if isinstance(value, list)]
    if len(lengths) != 6 or not lengths or min(lengths) == 0:
        raise NoMarketDataError(symbol, canonical, "DSE returned no complete OHLCV arrays")
    if len(set(lengths)) != 1:
        raise DSEAPIError(f"DSE returned mismatched OHLCV array lengths for {canonical}.")

    rows: list[dict[str, Any]] = []
    for index in range(lengths[0]):
        row_date = _timestamp_to_date(arrays["t"][index])
        if row_date is None or row_date.normalize() < start or row_date.normalize() > end:
            continue
        rows.append(
            {
                "Date": row_date.normalize(),
                "Open": arrays["o"][index],
                "High": arrays["h"][index],
                "Low": arrays["l"][index],
                "Close": arrays["c"][index],
                "Volume": arrays["v"][index],
            }
        )

    frame = _clean_dataframe(pd.DataFrame(rows)) if rows else pd.DataFrame()
    if frame.empty:
        raise NoMarketDataError(
            symbol, canonical, f"no DSE candles between {start_date} and {end_date}"
        )
    frame = frame.sort_values("Date").drop_duplicates("Date", keep="last")
    _assert_ohlcv_not_stale(frame, end_date, symbol, canonical)
    return frame.reset_index(drop=True)


def _cache_is_fresh(path: Path, curr_date: pd.Timestamp) -> bool:
    if curr_date.date() < pd.Timestamp.now().date():
        return True
    return time.time() - path.stat().st_mtime <= 900


def load_dse_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """Load five years of DSE candles with a per-date, token-safe disk cache."""

    canonical = normalize_dse_symbol(symbol)
    cutoff = pd.to_datetime(curr_date, format="%Y-%m-%d", errors="raise")
    start = cutoff - pd.DateOffset(years=5)
    config = get_config()
    cache_dir = Path(config["data_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"{safe_ticker_component(canonical)}-DSE-data-through-{cutoff:%Y-%m-%d}.csv"
    )

    data: pd.DataFrame | None = None
    if cache_path.exists() and _cache_is_fresh(cache_path, cutoff):
        cached = pd.read_csv(cache_path, on_bad_lines="skip", encoding="utf-8")
        if not cached.empty and {"Date", "Open", "High", "Low", "Close", "Volume"}.issubset(
            cached.columns
        ):
            data = _clean_dataframe(cached)

    if data is None:
        data = fetch_dse_ohlcv(
            canonical,
            start.strftime("%Y-%m-%d"),
            cutoff.strftime("%Y-%m-%d"),
        )
        data.to_csv(cache_path, index=False, encoding="utf-8")

    data = data[data["Date"] <= cutoff].sort_values("Date")
    if data.empty:
        raise NoMarketDataError(symbol, canonical, f"no rows on or before {curr_date}")
    _assert_ohlcv_not_stale(data, curr_date, symbol, canonical)
    return data.reset_index(drop=True)


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    data = fetch_dse_ohlcv(symbol, start_date, end_date)
    canonical = normalize_dse_symbol(symbol)
    rounded = data.copy()
    for column in ("Open", "High", "Low", "Close"):
        rounded[column] = rounded[column].round(2)
    return (
        f"# Dhaka Stock Exchange OHLCV for {canonical} from {start_date} to {end_date}\n"
        f"# Currency: BDT | Records: {len(rounded)} | Source: Doha Securities DSE gateway\n\n"
        + rounded.to_csv(index=False)
    )


_INDICATOR_DESCRIPTIONS = {
    "close_50_sma": "50-day simple moving average (medium-term trend).",
    "close_200_sma": "200-day simple moving average (long-term trend).",
    "close_10_ema": "10-day exponential moving average (short-term trend).",
    "macd": "MACD momentum line.",
    "macds": "MACD signal line.",
    "macdh": "MACD histogram.",
    "rsi": "Relative Strength Index; interpret with trend context rather than mechanically.",
    "boll": "20-day Bollinger middle band.",
    "boll_ub": "Bollinger upper band.",
    "boll_lb": "Bollinger lower band.",
    "atr": "Average True Range (price volatility in BDT).",
    "vwma": "Volume-weighted moving average.",
    "mfi": "Money Flow Index using DSE price and volume.",
}


def get_indicator(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int = 30,
) -> str:
    """Calculate stockstats indicators exclusively from DSE candles."""

    name = indicator.strip().lower()
    if name not in _INDICATOR_DESCRIPTIONS:
        raise ValueError(
            f"Indicator {indicator} is not supported. Choose from: {list(_INDICATOR_DESCRIPTIONS)}"
        )
    data = load_dse_ohlcv(symbol, curr_date)
    stock_df = wrap(data.copy())
    stock_df[name]  # trigger stockstats calculation
    date_column = "Date" if "Date" in stock_df.columns else "date"
    dates = pd.to_datetime(stock_df[date_column], errors="coerce")
    cutoff = pd.to_datetime(curr_date)
    start = cutoff - pd.Timedelta(days=max(0, int(look_back_days)))

    lines = []
    for row_index in range(len(stock_df)):
        row_date = dates.iloc[row_index]
        if pd.isna(row_date) or row_date < start or row_date > cutoff:
            continue
        value = stock_df.iloc[row_index][name]
        rendered = "N/A" if pd.isna(value) else f"{float(value):.4f}"
        lines.append(f"{row_date:%Y-%m-%d}: {rendered}")
    if not lines:
        raise NoMarketDataError(symbol, normalize_dse_symbol(symbol), "no indicator rows")
    return (
        f"## {name} from DSE candles ({start:%Y-%m-%d} to {cutoff:%Y-%m-%d})\n\n"
        + "\n".join(lines)
        + f"\n\n{_INDICATOR_DESCRIPTIONS[name]}"
    )


def _record_is_on_or_before(record: dict[str, Any], cutoff: pd.Timestamp) -> bool:
    for key in _DATE_KEYS:
        if key in record and record[key] not in (None, ""):
            parsed = _parse_date(record[key])
            return parsed is None or parsed.normalize() <= cutoff
    for key in _YEAR_KEYS:
        if key in record and record[key] not in (None, ""):
            year = _extract_year(record[key])
            return year is None or year <= cutoff.year
    return True


def _extract_year(value: Any) -> int | None:
    """Extract a four-digit year from DSE values such as ``2025-26`` or ``FY 2025``."""

    match = re.search(r"(?:19|20)\d{2}", str(value))
    return int(match.group()) if match else None


def _filter_to_date(value: Any, curr_date: str | None) -> Any:
    if not curr_date:
        return value
    cutoff = pd.to_datetime(curr_date, format="%Y-%m-%d", errors="raise").normalize()
    if isinstance(value, list):
        return [
            item
            for item in value
            if not isinstance(item, dict) or _record_is_on_or_before(item, cutoff)
        ]
    if isinstance(value, dict) and not _record_is_on_or_before(value, cutoff):
        return {}
    return value


def _json_section(title: str, value: Any) -> str:
    if value in (None, {}, []):
        return f"## {title}\n\nNo DSE record was returned."
    return f"## {title}\n\n```json\n{json.dumps(value, ensure_ascii=False, indent=2, default=str)}\n```"


def _get_fundamental_path(
    path: str,
    symbol: str,
    curr_date: str | None,
    *,
    client: DSEClient | None = None,
) -> Any:
    canonical = normalize_dse_symbol(symbol)
    payload = (client or DSEClient()).get(
        "analytics", f"/fundamentals/{path}/{quote(canonical, safe='')}"
    )
    return _filter_to_date(_unwrap(payload), curr_date)


def get_fundamentals(ticker: str, curr_date: str | None = None) -> str:
    """Return DSE company profile plus the core disclosed financial histories."""

    canonical = normalize_dse_symbol(ticker)
    client = DSEClient()
    sections = [
        ("Company profile", "company_details"),
        ("Annual financial performance history", "financial_performance"),
        ("Quarterly performance history", "quarterly_performance"),
        ("Shareholding history", "share_holding"),
        ("Dividend history", "dividend_information"),
        ("Loan status history", "loan_status"),
        ("NAV per year", "nav_per_year"),
    ]
    rendered = [
        f"# DSE fundamentals for {canonical}",
        f"Analysis cutoff: {curr_date or 'current'} | Currency: BDT where applicable",
        "Source: Doha Securities market analytics APIs discovered in the bundled Angular UI.",
    ]
    usable = False
    for title, path in sections:
        value = _get_fundamental_path(path, canonical, curr_date, client=client)
        usable = usable or value not in (None, {}, [])
        rendered.append(_json_section(title, value))
    if not usable:
        raise NoMarketDataError(ticker, canonical, "DSE returned no fundamental records")
    return "\n\n".join(rendered)


def get_balance_sheet(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    """Return the year-wise balance sheet exposed by the DSE analytics API."""

    del freq  # The Angular-discovered route is year-wise and has no frequency parameter.
    canonical = normalize_dse_symbol(ticker)
    payload = DSEClient().get(
        "analytics",
        f"/api/balance_sheet/balance-sheet/{quote(canonical, safe='')}",
    )
    data = _unwrap(payload)
    if isinstance(data, dict) and curr_date:
        cutoff_year = pd.to_datetime(curr_date).year
        histories = data.get("year_wise_data")
        if isinstance(histories, list):
            data = dict(data)
            data["year_wise_data"] = [
                item
                for item in histories
                if not isinstance(item, dict)
                or not any(_extract_year(year) is not None for year in item)
                or any(
                    parsed_year is not None and parsed_year <= cutoff_year
                    for parsed_year in (_extract_year(year) for year in item)
                )
            ]
    if data in (None, {}, []):
        raise NoMarketDataError(ticker, canonical, "DSE returned no balance sheet")
    return "\n\n".join(
        [f"# DSE balance sheet for {canonical}", _json_section("Year-wise balance sheet", data)]
    )


def get_cashflow(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    """Return DSE NOCFPS history, the available cash-flow-per-share disclosure.

    The Angular application exposes NOCFPS history but no full cash-flow
    statement route.  The report says so explicitly to prevent the model from
    treating this proxy as a complete statement.
    """

    del freq
    canonical = normalize_dse_symbol(ticker)
    value = _get_fundamental_path("nocfps_history_quarter", canonical, curr_date)
    if value in (None, {}, []):
        raise NoMarketDataError(ticker, canonical, "DSE returned no NOCFPS history")
    return "\n\n".join(
        [
            f"# DSE cash-flow disclosure for {canonical}",
            "The bundled Angular API exposes quarterly NOCFPS (net operating cash flow "
            "per share), not a complete cash-flow statement. Do not infer missing line items.",
            _json_section("Quarterly NOCFPS history", value),
        ]
    )


def get_income_statement(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    """Return annual performance plus quarterly EPS/profit disclosures."""

    del freq
    canonical = normalize_dse_symbol(ticker)
    client = DSEClient()
    sections = [
        ("Annual financial performance", "financial_performance"),
        ("Quarterly performance", "quarterly_performance"),
        ("Quarterly EPS history", "eps_history_quarter"),
        ("Quarterly cumulative profit history", "profit_history_quarter"),
    ]
    values = [
        (title, _get_fundamental_path(path, canonical, curr_date, client=client))
        for title, path in sections
    ]
    if not any(value not in (None, {}, []) for _, value in values):
        raise NoMarketDataError(ticker, canonical, "DSE returned no earnings history")
    return "\n\n".join(
        [f"# DSE earnings disclosures for {canonical}"]
        + [_json_section(title, value) for title, value in values]
    )


def _paged_content(payload: Any) -> list[dict[str, Any]]:
    data = _unwrap(payload)
    if isinstance(data, dict):
        data = data.get("content", data.get("items", data.get("results", [])))
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _news_timestamp(article: dict[str, Any]) -> pd.Timestamp | None:
    for key in ("timestamp", "publishedAt", "published_at", "created_at", "date"):
        if key in article:
            parsed = _parse_date(article[key])
            if parsed is not None:
                return parsed
    return None


def _format_news(
    articles: list[dict[str, Any]],
    *,
    title: str,
    start_date: str,
    end_date: str,
) -> str:
    start, end = _validate_range(start_date, end_date)
    end_exclusive = end + pd.Timedelta(days=1)
    filtered = []
    for article in articles:
        published = _news_timestamp(article)
        if published is not None and not (start <= published < end_exclusive):
            continue
        filtered.append((article, published))
    if not filtered:
        return f"No DSE news found for {title} between {start_date} and {end_date}."

    lines = [f"# {title}", f"Window: {start_date} to {end_date} (inclusive)"]
    for article, published in filtered:
        headline = article.get("title") or article.get("headline") or "Untitled disclosure"
        text = article.get("newsText") or article.get("content") or article.get("summary") or ""
        stock_code = article.get("stockCode") or article.get("stock_code") or "Market-wide"
        categories = article.get("categories") or article.get("category") or []
        reference = article.get("reference") or article.get("url") or ""
        date_label = published.strftime("%Y-%m-%d %H:%M") if published is not None else "Unknown"
        lines.extend(
            [
                "",
                f"## {headline}",
                f"- Published: {date_label}",
                f"- DSE symbol: {stock_code}",
                f"- Categories: {categories}",
                f"- Reference: {reference or 'N/A'}",
                "",
                str(text).strip(),
            ]
        )
    return "\n".join(lines)


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    canonical = normalize_dse_symbol(ticker)
    limit = int(get_config().get("news_article_limit", 20))
    payload = DSEClient().get(
        "market",
        "/news/all/filter",
        params={
            "page": 1,
            "size": limit,
            "stockCode": canonical,
            "from": start_date,
            "to": end_date,
        },
    )
    return _format_news(
        _paged_content(payload),
        title=f"DSE company news for {canonical}",
        start_date=start_date,
        end_date=end_date,
    )


def get_global_news(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """Return exchange-wide DSE news; no foreign publisher/search fallback."""

    config = get_config()
    days = int(
        look_back_days if look_back_days is not None else config.get("global_news_lookback_days", 7)
    )
    article_limit = int(limit if limit is not None else config.get("global_news_article_limit", 10))
    end = pd.to_datetime(curr_date, format="%Y-%m-%d", errors="raise")
    start = end - pd.Timedelta(days=max(0, days))
    start_date = start.strftime("%Y-%m-%d")
    payload = DSEClient().get(
        "market",
        "/news/all/filter",
        params={
            "page": 1,
            "size": article_limit,
            "from": start_date,
            "to": curr_date,
        },
    )
    return _format_news(
        _paged_content(payload),
        title="DSE exchange-wide market news",
        start_date=start_date,
        end_date=curr_date,
    )


def get_insider_transactions(ticker: str, curr_date: str | None = None) -> str:
    """Report sponsor/director holding changes from DSE shareholding history.

    The Angular source exposes ownership percentages, not individual insider
    trade filings.  We calculate disclosed period-to-period changes and label
    the limitation rather than inventing transaction dates or quantities.
    """

    canonical = normalize_dse_symbol(ticker)
    value = _get_fundamental_path("share_holding", canonical, curr_date)
    if not isinstance(value, list) or not value:
        return f"No DSE sponsor/director shareholding history reported for {canonical}."

    sorted_rows = sorted(value, key=lambda row: str(row.get("date", "")))
    changes = []
    previous: float | None = None
    for row in sorted_rows:
        raw = row.get("sponsor_director")
        try:
            current = float(str(raw).replace("%", "").strip())
        except (TypeError, ValueError):
            current = None
        changes.append(
            {
                "date": row.get("date"),
                "sponsor_director_pct": raw,
                "change_percentage_points": (
                    None if current is None or previous is None else round(current - previous, 4)
                ),
            }
        )
        if current is not None:
            previous = current
    return "\n\n".join(
        [
            f"# DSE sponsor/director ownership disclosures for {canonical}",
            "The available API provides periodic sponsor/director ownership percentages, "
            "not individual insider transaction filings. Changes below are percentage-point "
            "differences between disclosures and must not be described as a specific trade.",
            _json_section("Disclosed ownership changes", changes),
        ]
    )


def resolve_identity(ticker: str) -> dict[str, str]:
    """Best-effort deterministic DSE identity for cross-agent context."""

    canonical = normalize_dse_symbol(ticker)
    payload = DSEClient().get(
        "analytics",
        f"/fundamentals/company_details/{quote(canonical, safe='')}",
    )
    data = _unwrap(payload)
    if not isinstance(data, dict):
        return {"exchange": "Dhaka Stock Exchange", "quote_type": "EQUITY"}
    identity = {
        "company_name": str(data.get("name") or "").strip(),
        "sector": str(data.get("sector") or "").strip(),
        "industry": str(data.get("type_of_instrument") or "").strip(),
        "exchange": "Dhaka Stock Exchange",
        "quote_type": str(data.get("type_of_instrument") or "EQUITY").strip(),
    }
    return {key: value for key, value in identity.items() if value}


def fetch_return_series(symbol: str, start_date: str, end_date: str) -> pd.Series:
    """Close-price series used for post-decision performance reflection."""

    frame = fetch_dse_ohlcv(symbol, start_date, end_date)
    return frame.set_index("Date")["Close"].astype(float).sort_index()
