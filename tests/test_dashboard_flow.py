from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import cli.main as cli_main
from dohasecuritiesstockai.api.app import create_app
from dohasecuritiesstockai.dashboard import dashboard_url
from dohasecuritiesstockai.graph.trading_graph import TradingAgentsGraph


def test_dashboard_url_targets_exact_cli_result() -> None:
    assert dashboard_url("0.0.0.0", 8000, "BRACBANK", "2026-08-09") == (
        "http://127.0.0.1:8000/?symbol=BRACBANK&date=2026-08-09"
    )


def test_open_ui_env_flag_is_strict(monkeypatch) -> None:
    monkeypatch.setenv("TRADINGAGENTS_OPEN_UI_AFTER_ANALYSIS", "yes")
    assert cli_main._env_flag("TRADINGAGENTS_OPEN_UI_AFTER_ANALYSIS") is True

    monkeypatch.setenv("TRADINGAGENTS_OPEN_UI_AFTER_ANALYSIS", "sometimes")
    try:
        cli_main._env_flag("TRADINGAGENTS_OPEN_UI_AFTER_ANALYSIS")
    except ValueError as exc:
        assert "TRADINGAGENTS_OPEN_UI_AFTER_ANALYSIS" in str(exc)
    else:
        raise AssertionError("An invalid UI flag must fail loudly")


def test_cli_state_is_saved_in_api_safe_format(tmp_path: Path) -> None:
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    graph.config = {"results_dir": str(tmp_path)}
    graph.log_states_dict = {}
    graph.ticker = None
    graph.curr_state = None
    final_state = {
        "company_of_interest": "GP",
        "trade_date": "2026-08-09",
        "market_report": "market",
        "sentiment_report": "",
        "news_report": "news",
        "fundamentals_report": "fundamentals",
        "investment_debate_state": {
            "bull_history": "bull",
            "bear_history": "bear",
            "history": "debate",
            "current_response": "response",
            "judge_decision": "research decision",
        },
        "trader_investment_plan": "trade plan",
        "risk_debate_state": {
            "aggressive_history": "aggressive",
            "conservative_history": "conservative",
            "neutral_history": "neutral",
            "history": "risk debate",
            "judge_decision": "portfolio decision",
        },
        "investment_plan": "investment plan",
        "final_trade_decision": "BUY",
        "messages": [object()],
    }

    path = graph.save_state("2026-08-09", final_state, ticker="GP")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["final_trade_decision"] == "BUY"
    assert "messages" not in payload


def test_api_serves_built_angular_root(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "index.html").write_text("<html>dashboard</html>", encoding="utf-8")
    monkeypatch.setenv("TRADINGAGENTS_UI_DIST_DIR", str(tmp_path))

    response = TestClient(create_app()).get("/")

    assert response.status_code == 200
    assert "dashboard" in response.text
