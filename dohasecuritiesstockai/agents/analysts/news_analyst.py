from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from dohasecuritiesstockai.agents.utils.agent_utils import (
    get_global_news,
    get_insider_transactions,
    get_instrument_context_from_state,
    get_language_instruction,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
)
from dohasecuritiesstockai.dataflows.config import get_config
from dohasecuritiesstockai.dataflows.dse import is_dse_market


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)
        config = get_config()

        tools = [get_news, get_global_news, get_insider_transactions]
        if config.get("macro_data_enabled", True):
            tools.append(get_macro_indicators)
        if config.get("prediction_markets_enabled", True):
            tools.append(get_prediction_markets)

        if is_dse_market(config):
            system_message = (
                f"You are a Bangladesh capital-market news researcher. Analyze {asset_label}-specific "
                "and exchange-wide Dhaka Stock Exchange disclosures over the requested window. "
                "Use get_news(ticker, start_date, end_date) for stock-wise DSE news, "
                "get_global_news(curr_date, look_back_days, limit) for exchange-wide DSE news, "
                "and get_insider_transactions(ticker, curr_date) for disclosed sponsor/director ownership "
                "changes. Do not request, cite, or invent Bloomberg, Reuters, Reddit, StockTwits, "
                "FRED, Polymarket, or other foreign/social sources. Distinguish official disclosures "
                "from interpretation, respect the analysis-date cutoff, and state when the DSE API "
                "has no record. Provide evidence-backed Bangladesh-market catalysts and risks."
                " Append a Markdown table organizing the key news, dates, likely impact, and caveats."
                + get_language_instruction()
            )
        else:
            optional_tools = ""
            if config.get("macro_data_enabled", True):
                optional_tools += (
                    ", get_macro_indicators(indicator, curr_date, look_back_days) "
                    "for grounded macro series"
                )
            if config.get("prediction_markets_enabled", True):
                optional_tools += (
                    ", and get_prediction_markets(topic, limit) for market-implied probabilities"
                )
            system_message = (
                f"You are a news researcher analyzing recent trading-relevant news. Use "
                f"get_news for {asset_label}-specific news, get_global_news for broader news, "
                f"and get_insider_transactions for disclosed insider activity{optional_tools}. "
                "Provide specific insights supported by retrieved evidence and append a Markdown table."
                + get_language_instruction()
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
