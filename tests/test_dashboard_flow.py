from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import cli.main as cli_main
import dohasecuritiesstockai.dashboard_cli as dashboard_cli
from dohasecuritiesstockai.api.app import create_app
from dohasecuritiesstockai.dashboard import dashboard_url, prepare_dashboard_analysis
from dohasecuritiesstockai.graph.trading_graph import TradingAgentsGraph


def test_dashboard_url_targets_exact_cli_result() -> None:
    assert dashboard_url("0.0.0.0", 8000, "BRACBANK", "2026-08-09") == (
        "http://127.0.0.1:8000/?symbol=BRACBANK&date=2026-08-09"
    )


def test_lightweight_dashboard_command_skips_agent_state(tmp_path: Path, monkeypatch) -> None:
    calls: dict[str, object] = {}
    analysis = SimpleNamespace(
        symbol="GP",
        analysis_date=date(2026, 8, 10),
        fundamental_score=72,
        score_label=SimpleNamespace(en="Good"),
        ai_research=None,
    )

    def fake_prepare(
        symbol, analysis_date, agent_state, results_dir, *, use_ai, config
    ):
        calls["prepare"] = (
            symbol,
            analysis_date,
            agent_state,
            results_dir,
            use_ai,
            config,
        )
        return analysis, tmp_path / "GP.json"

    def fake_launch(symbol, analysis_date, *, host, port):
        calls["launch"] = (symbol, analysis_date, host, port)

    monkeypatch.setattr(dashboard_cli, "prepare_dashboard_analysis", fake_prepare)
    monkeypatch.setattr(dashboard_cli, "launch_dashboard", fake_launch)
    monkeypatch.setitem(dashboard_cli.DEFAULT_CONFIG, "results_dir", str(tmp_path))

    dashboard_cli.show_dashboard(
        "gp",
        analysis_date="2026-08-10",
        use_ai=True,
        open_ui=True,
        host="0.0.0.0",
        port=8123,
    )

    assert calls["prepare"] == (
        "gp",
        date(2026, 8, 10),
        None,
        str(tmp_path),
        True,
        dashboard_cli.DEFAULT_CONFIG,
    )
    assert calls["launch"] == ("GP", date(2026, 8, 10), "0.0.0.0", 8123)


def test_prepare_dashboard_analysis_accepts_lightweight_mode(tmp_path: Path, monkeypatch) -> None:
    built = SimpleNamespace(symbol="GP", analysis_date=date(2026, 8, 10))
    calls: dict[str, object] = {}

    class FakeBuilder:
        def build(self, symbol, analysis_date, agent_state):
            calls["build"] = (symbol, analysis_date, agent_state)
            return built

    class FakeRepository:
        def __init__(self, results_dir):
            calls["results_dir"] = results_dir

        def save_analysis(self, analysis):
            calls["saved"] = analysis
            return tmp_path / "analysis.json"

    monkeypatch.setattr("dohasecuritiesstockai.dashboard.StockAnalysisBuilder", FakeBuilder)
    monkeypatch.setattr("dohasecuritiesstockai.dashboard.AnalysisRepository", FakeRepository)

    result, path = prepare_dashboard_analysis(
        "GP", "2026-08-10", None, tmp_path, use_ai=False
    )

    assert result is built
    assert path == tmp_path / "analysis.json"
    assert calls["build"] == ("GP", date(2026, 8, 10), None)


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
