"""Unit coverage for the authenticated Dhaka Stock Exchange vendor."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from dohasecuritiesstockai.dataflows import dse, interface
from dohasecuritiesstockai.dataflows.errors import VendorNotConfiguredError


class _Response:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _Session:
    def __init__(self, response: _Response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


class _LoginSession:
    def __init__(self, login_response: _Response, get_response: _Response):
        self.login_response = login_response
        self.get_response = get_response
        self.post_calls = []
        self.get_calls = []

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return self.login_response

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return self.get_response


def _epoch(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def test_normalize_dse_symbol_accepts_ui_and_exchange_aliases():
    assert dse.normalize_dse_symbol("gp") == "GP"
    assert dse.normalize_dse_symbol("GP'PB") == "GP"
    assert dse.normalize_dse_symbol("gp.dse") == "GP"
    assert dse.normalize_dse_symbol("GP.DH") == "GP"


@pytest.mark.parametrize("value", ["", "../GP", "GP/../../x", "GP?x=1"])
def test_normalize_dse_symbol_rejects_unsafe_values(value):
    with pytest.raises(ValueError):
        dse.normalize_dse_symbol(value)


def test_client_sends_bearer_without_exposing_it(monkeypatch):
    monkeypatch.setenv("DSE_ACCESS_TOKEN", "secret-token")
    session = _Session(_Response(200, {"code": 200, "data": ["GP"]}))
    client = dse.DSEClient(session=session)

    assert client.get("market", "/market/symbols/sorted") == {
        "code": 200,
        "data": ["GP"],
    }
    _, kwargs = session.calls[0]
    assert kwargs["headers"]["Authorization"] == "Bearer secret-token"


def test_client_explains_missing_access_token(monkeypatch):
    monkeypatch.delenv("DSE_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("DSE_EMAIL_OR_PHONE", raising=False)
    monkeypatch.delenv("DSE_PASSWORD", raising=False)
    session = _Session(_Response(401, {}))
    client = dse.DSEClient(session=session)

    with pytest.raises(VendorNotConfiguredError, match="DSE_ACCESS_TOKEN") as exc:
        client.get("market", "/market/summary")
    assert "secret" not in str(exc.value).lower()


def test_client_can_login_with_credentials_and_keep_token_in_memory(monkeypatch):
    monkeypatch.delenv("DSE_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("DSE_EMAIL_OR_PHONE", "user@example.invalid")
    monkeypatch.setenv("DSE_PASSWORD", "test-password")
    monkeypatch.delenv("DSE_AUTH_GRANT_TYPE", raising=False)
    monkeypatch.delenv("DSE_AUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("DSE_AUTH_SCOPE", raising=False)
    dse._AUTH_CACHE.clear()
    session = _LoginSession(
        _Response(200, {"data": {"access_token": "generated-token", "expires_in": 3600}}),
        _Response(200, {"code": 200, "data": ["GP"]}),
    )
    client = dse.DSEClient(session=session)

    assert client.get("market", "/market/symbols/sorted")["data"] == ["GP"]
    assert client.get("market", "/market/symbols/sorted")["data"] == ["GP"]
    assert len(session.post_calls) == 1
    _, login_kwargs = session.post_calls[0]
    assert login_kwargs["json"] == {
        "grantType": "urn:ietf:params:oauth:grant-type:mobile-application",
        "authorizationClientId": "oms",
        "scope": "android",
        "emailOrPhone": "user@example.invalid",
        "password": "test-password",
    }
    assert all(
        kwargs["headers"]["Authorization"] == "Bearer generated-token"
        for _, kwargs in session.get_calls
    )
    dse._AUTH_CACHE.clear()


def test_fetch_dse_ohlcv_parses_parallel_arrays_and_date_bounds(monkeypatch):
    monkeypatch.setattr(dse, "_assert_ohlcv_not_stale", lambda *args, **kwargs: None)
    payload = {
        "code": 200,
        "data": {
            "t": [_epoch("2026-07-31"), _epoch("2026-08-03"), _epoch("2026-08-04")],
            "o": [99, 100, 101],
            "h": [100, 103, 104],
            "l": [98, 99, 100],
            "c": [99.5, 102, 103],
            "v": [1000, 1200, 1300],
        },
    }
    client = dse.DSEClient(session=_Session(_Response(200, payload)))

    frame = dse.fetch_dse_ohlcv("GP'PB", "2026-08-01", "2026-08-04", client=client)

    assert list(frame.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert frame["Date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-08-03", "2026-08-04"]
    assert frame["Close"].tolist() == [102, 103]


def test_fetch_dse_ohlcv_rejects_mismatched_arrays(monkeypatch):
    payload = {
        "data": {
            "t": [_epoch("2026-08-03")],
            "o": [100, 101],
            "h": [103],
            "l": [99],
            "c": [102],
            "v": [1200],
        }
    }
    client = dse.DSEClient(session=_Session(_Response(200, payload)))
    with pytest.raises(dse.DSEAPIError, match="mismatched"):
        dse.fetch_dse_ohlcv("GP", "2026-08-01", "2026-08-04", client=client)


def test_news_formatter_excludes_future_articles():
    articles = [
        {"timestamp": "2026-08-03T10:00:00Z", "title": "In range", "newsText": "A"},
        {"timestamp": "2026-08-05T10:00:00Z", "title": "Future", "newsText": "B"},
    ]
    report = dse._format_news(
        articles,
        title="DSE company news for GP",
        start_date="2026-08-01",
        end_date="2026-08-04",
    )
    assert "In range" in report
    assert "Future" not in report


def test_fundamental_filter_excludes_future_dates_and_years():
    records = [
        {"date": "2025-12-31", "eps_basic": "4.2"},
        {"date": "2027-01-01", "eps_basic": "9.9"},
        {"year": 2024, "cash_dividend": "10"},
        {"year": 2027, "cash_dividend": "99"},
    ]
    filtered = dse._filter_to_date(records, "2026-08-09")
    assert filtered == [records[0], records[2]]


def test_balance_sheet_year_filter_understands_fiscal_year_labels(monkeypatch):
    payload = {
        "data": {
            "columns": ["Assets"],
            "year_wise_data": [
                {"FY 2025-26": [100]},
                {"FY 2027-28": [200]},
            ],
        }
    }
    monkeypatch.setattr(dse.DSEClient, "get", lambda *args, **kwargs: payload)
    report = dse.get_balance_sheet("GP", curr_date="2026-08-09")
    assert "FY 2025-26" in report
    assert "FY 2027-28" not in report


def test_dse_is_registered_for_all_primary_tools():
    methods = (
        "get_stock_data",
        "get_indicators",
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
        "get_news",
        "get_global_news",
        "get_insider_transactions",
    )
    assert all("dse" in interface.VENDOR_METHODS[method] for method in methods)


def test_get_indicator_uses_supplied_dse_frame(monkeypatch):
    dates = pd.bdate_range("2025-01-01", periods=220)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": range(100, 320),
            "High": range(101, 321),
            "Low": range(99, 319),
            "Close": range(100, 320),
            "Volume": [1000] * 220,
        }
    )
    monkeypatch.setattr(dse, "load_dse_ohlcv", lambda symbol, curr_date: frame)
    report = dse.get_indicator("GP", "close_50_sma", dates[-1].strftime("%Y-%m-%d"), 10)
    assert "DSE candles" in report
    assert "close_50_sma" in report
