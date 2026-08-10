from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from dohasecuritiesstockai.agents.utils.agent_utils import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_instrument_context_from_state,
    get_language_instruction,
)
from dohasecuritiesstockai.dataflows.config import get_config
from dohasecuritiesstockai.dataflows.dse import is_dse_market


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        dse_mode = is_dse_market(get_config())

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        market_instruction = (
            " This is a Dhaka Stock Exchange company. Treat amounts as BDT where the API says or "
            "the field convention implies BDT, and preserve percentages/units exactly. The DSE "
            "cash-flow tool exposes quarterly NOCFPS rather than a complete cash-flow statement; "
            "never invent missing statement lines. Use DSE company profile, annual and quarterly "
            "performance, shareholding, dividends, NAV, loan status, balance sheet, EPS, NOCFPS, "
            "and cumulative-profit histories, respecting the analysis-date cutoff."
            if dse_mode
            else ""
        )
        system_message = (
            "You are a fundamentals researcher. Produce a comprehensive, evidence-backed report "
            "covering company identity, financial performance and history, capital structure, "
            "ownership, dividends, earnings quality, and material risks. Use get_fundamentals for "
            "the comprehensive disclosure set and get_balance_sheet, get_cashflow, and "
            "get_income_statement for the available statement-specific views."
            + market_instruction
            + " State missing or incomplete fields rather than estimating them. Append a Markdown "
            "table summarizing the key metrics, periods, trends, and caveats."
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
