"""DSE candle loading for every resolution exposed by the OMS chart API."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from dohasecuritiesstockai.dataflows.dse import (
    DSE_INDEX_SYMBOLS,
    DSEAPIError,
    DSEClient,
    normalize_dse_symbol,
)
from dohasecuritiesstockai.dataflows.errors import NoMarketDataError


@dataclass(frozen=True)
class Resolution:
    key: str
    server_value: str
    label: str
    seconds: int
    timesfm_frequency: int


RESOLUTIONS: dict[str, Resolution] = {
    "1min": Resolution("1min", "1", "1 minute", 60, 0),
    "15min": Resolution("15min", "15", "15 minutes", 15 * 60, 0),
    "30min": Resolution("30min", "30", "30 minutes", 30 * 60, 0),
    "1h": Resolution("1h", "1H", "1 hour", 60 * 60, 0),
    "1d": Resolution("1d", "1D", "1 day", 24 * 60 * 60, 0),
    "1w": Resolution("1w", "1W", "1 week", 7 * 24 * 60 * 60, 1),
    "1mo": Resolution("1mo", "1M", "1 month", 30 * 24 * 60 * 60, 1),
    "1y": Resolution("1y", "12M", "1 year", 365 * 24 * 60 * 60, 2),
}

_ALIASES = {
    "1": "1min",
    "minute": "1min",
    "1minute": "1min",
    "15": "15min",
    "30": "30min",
    "hour": "1h",
    "1hour": "1h",
    "d": "1d",
    "day": "1d",
    "daily": "1d",
    "w": "1w",
    "week": "1w",
    "weekly": "1w",
    "m": "1mo",
    "month": "1mo",
    "monthly": "1mo",
    "12m": "1y",
    "year": "1y",
    "yearly": "1y",
}


@dataclass(frozen=True)
class MarketCandle:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def normalize_resolution(value: str) -> Resolution:
    raw = str(value or "").strip()
    # TradingView distinguishes one minute ("1") from one month ("1M").
    # Preserve that uppercase month spelling before normalizing CLI aliases.
    normalized = "1mo" if raw == "1M" else raw.lower()
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in RESOLUTIONS:
        choices = ", ".join(RESOLUTIONS)
        raise ValueError(f"Unsupported resolution {value!r}. Choose one of: {choices}.")
    return RESOLUTIONS[normalized]


def _timestamp(value: Any) -> datetime:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise DSEAPIError(f"DSE returned an invalid candle timestamp: {value!r}.") from exc
    if numeric > 10_000_000_000:
        numeric /= 1000
    try:
        return datetime.fromtimestamp(numeric, timezone.utc)
    except (OSError, OverflowError, ValueError) as exc:
        raise DSEAPIError(f"DSE returned an out-of-range candle timestamp: {value!r}.") from exc


def _number(value: Any, field: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise DSEAPIError(f"DSE returned a non-numeric {field} value.") from exc
    if not math.isfinite(numeric):
        raise DSEAPIError(f"DSE returned a non-finite {field} value.")
    return numeric


def _unwrap(payload: Any) -> Any:
    return payload.get("data") if isinstance(payload, dict) and "data" in payload else payload


def fetch_dse_candles(
    symbol: str,
    resolution: str,
    *,
    start: date | None = None,
    end: date | None = None,
    limit: int = 100_000,
    client: DSEClient | None = None,
) -> tuple[Resolution, list[MarketCandle]]:
    """Fetch the complete chart history returned by the DSE analytics server."""

    canonical = normalize_dse_symbol(symbol)
    resolved = normalize_resolution(resolution)
    start_date = start or date(1990, 1, 1)
    end_date = end or date.today()
    if start_date > end_date:
        raise ValueError("start must not be after end.")
    if limit < 1:
        raise ValueError("limit must be at least 1.")

    payload = (client or DSEClient()).get(
        "analytics",
        "/candlesticks/limited",
        params={
            "symbol": canonical,
            "type": "index" if canonical in DSE_INDEX_SYMBOLS else "stock",
            "resolution": resolved.server_value,
            "from": start_date.isoformat(),
            "to": end_date.isoformat(),
            "limit": limit,
        },
    )
    data = _unwrap(payload)
    if not isinstance(data, dict):
        raise NoMarketDataError(symbol, canonical, "DSE returned no candle object")

    arrays = {key: data.get(key) for key in ("t", "o", "h", "l", "c")}
    arrays["v"] = data.get("v") if data.get("v") is not None else data.get("vo")
    if any(not isinstance(values, list) for values in arrays.values()):
        raise NoMarketDataError(symbol, canonical, "DSE returned incomplete OHLCV arrays")
    lengths = {key: len(values) for key, values in arrays.items()}
    if not lengths or min(lengths.values()) == 0:
        raise NoMarketDataError(symbol, canonical, "DSE returned empty OHLCV arrays")
    if len(set(lengths.values())) != 1:
        raise DSEAPIError(f"DSE returned mismatched OHLCV array lengths for {canonical}.")

    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)
    by_time: dict[datetime, MarketCandle] = {}
    for index in range(lengths["t"]):
        candle_time = _timestamp(arrays["t"][index])
        if candle_time < start_dt or candle_time > end_dt:
            continue
        candle = MarketCandle(
            time=candle_time,
            open=_number(arrays["o"][index], "open"),
            high=_number(arrays["h"][index], "high"),
            low=_number(arrays["l"][index], "low"),
            close=_number(arrays["c"][index], "close"),
            volume=_number(arrays["v"][index], "volume"),
        )
        if candle.close > 0:
            by_time[candle_time] = candle

    candles = [by_time[key] for key in sorted(by_time)]
    if not candles:
        raise NoMarketDataError(
            symbol,
            canonical,
            f"no {resolved.label} candles between {start_date} and {end_date}",
        )
    return resolved, candles
