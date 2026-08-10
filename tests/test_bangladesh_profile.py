"""Bangladesh profile defaults and feature gates."""

from dohasecuritiesstockai.default_config import DEFAULT_CONFIG
from dohasecuritiesstockai.graph.trading_graph import TradingAgentsGraph


def test_bangladesh_profile_is_the_default():
    assert DEFAULT_CONFIG["market_profile"] == "bangladesh_dse"
    assert DEFAULT_CONFIG["data_vendors"]["core_stock_apis"] == "dse"
    assert DEFAULT_CONFIG["data_vendors"]["technical_indicators"] == "dse"
    assert DEFAULT_CONFIG["data_vendors"]["fundamental_data"] == "dse"
    assert DEFAULT_CONFIG["data_vendors"]["news_data"] == "dse"
    assert DEFAULT_CONFIG["dse_auth_path"].endswith("/token/custom-flow")


def test_social_and_foreign_enrichments_are_off_by_default():
    assert DEFAULT_CONFIG["social_media_enabled"] is False
    assert DEFAULT_CONFIG["macro_data_enabled"] is False
    assert DEFAULT_CONFIG["prediction_markets_enabled"] is False
    assert "social" not in DEFAULT_CONFIG["default_analysts"]


def test_news_tool_node_respects_bangladesh_feature_gates():
    graph = object.__new__(TradingAgentsGraph)
    graph.config = DEFAULT_CONFIG.copy()
    nodes = graph._create_tool_nodes()
    news_tools = set(nodes["news"].tools_by_name)
    assert {"get_news", "get_global_news", "get_insider_transactions"} <= news_tools
    assert "get_macro_indicators" not in news_tools
    assert "get_prediction_markets" not in news_tools


def test_openrouter_and_native_gemini_are_configurable():
    assert DEFAULT_CONFIG["llm_provider"] == "openrouter"
    assert DEFAULT_CONFIG["quick_think_llm"].startswith("google/")
    assert DEFAULT_CONFIG["deep_think_llm"].startswith("google/")
