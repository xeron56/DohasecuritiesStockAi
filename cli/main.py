import datetime
import os
import sys
import time
from collections import deque
from functools import wraps
from pathlib import Path

import typer
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from cli.stats_handler import StatsCallbackHandler
from cli.theme import CLI_THEME
from cli.utils import (
    ask_anthropic_effort,
    ask_gemini_thinking_config,
    ask_glm_region,
    ask_minimax_region,
    ask_openai_reasoning_effort,
    ask_output_language,
    ask_qwen_region,
    confirm_ollama_endpoint,
    detect_asset_type,
    ensure_api_key,
    get_ticker,
    prompt_openai_compatible_url,
    resolve_backend_url,
    select_analysts,
    select_deep_thinking_agent,
    select_llm_provider,
    select_research_depth,
    select_shallow_thinking_agent,
)
from dohasecuritiesstockai.dataflows.errors import VendorError
from dohasecuritiesstockai.default_config import DEFAULT_CONFIG
from dohasecuritiesstockai.graph.analyst_execution import (
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
    get_initial_analyst_node,
    sync_analyst_tracker_from_chunk,
)
from dohasecuritiesstockai.graph.trading_graph import TradingAgentsGraph
from dohasecuritiesstockai.reporting import write_report_tree

console = Console(theme=CLI_THEME, highlight=False)

# The product and Python package share one canonical name. Legacy executable
# aliases remain available for existing shell scripts.
PRODUCT_NAME = "DohasecuritiesStockAi"
PRODUCT_DISPLAY_NAME = "Doha Securities Stock AI"
PRODUCT_TAGLINE = "DSE Multi-Agent Stock Analysis"

WORKFLOW_TEAMS = {
    "Analyst Desk": [
        "Market Analyst",
        "Sentiment Analyst",
        "News Analyst",
        "Fundamentals Analyst",
    ],
    "Research Desk": ["Bull Researcher", "Bear Researcher", "Research Manager"],
    "Trade Desk": ["Trader"],
    "Risk Desk": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
    "Portfolio Desk": ["Portfolio Manager"],
}

# prompt_toolkit's win32 output module is importable only on Windows (it asserts
# the platform at import time), so gate on the platform rather than catching the
# failure — that way a genuinely broken prompt_toolkit on Windows still surfaces
# instead of silently disabling the handler below. Off Windows this stays an
# empty tuple, which `except` accepts and never matches (#1138).
if sys.platform == "win32":  # pragma: no cover - platform dependent
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError

    _NO_CONSOLE_ERRORS: tuple[type[BaseException], ...] = (NoConsoleScreenBufferError,)
else:
    _NO_CONSOLE_ERRORS = ()

app = typer.Typer(
    name=PRODUCT_NAME,
    help=f"{PRODUCT_DISPLAY_NAME}: {PRODUCT_TAGLINE}",
    add_completion=True,  # Enable shell completion
)


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a strict boolean flag without silently accepting misspellings."""

    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw!r}")


# Create a deque to store recent messages with a maximum length
class MessageBuffer:
    # Fixed teams that always run (not user-selectable)
    FIXED_AGENTS = {
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Analyst name mapping
    ANALYST_MAPPING = {
        "market": "Market Analyst",
        "social": "Sentiment Analyst",
        "news": "News Analyst",
        "fundamentals": "Fundamentals Analyst",
    }

    # Report section mapping: section -> (analyst_key for filtering, finalizing_agent)
    # analyst_key: which analyst selection controls this section (None = always included)
    # finalizing_agent: which agent must be "completed" for this report to count as done
    REPORT_SECTIONS = {
        "market_report": ("market", "Market Analyst"),
        "sentiment_report": ("social", "Sentiment Analyst"),
        "news_report": ("news", "News Analyst"),
        "fundamentals_report": ("fundamentals", "Fundamentals Analyst"),
        "investment_plan": (None, "Research Manager"),
        "trader_investment_plan": (None, "Trader"),
        "final_trade_decision": (None, "Portfolio Manager"),
    }

    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None  # Store the complete final report
        self.agent_status = {}
        self.current_agent = None
        self.report_sections = {}
        self.selected_analysts = []
        self.run_context = {}
        self._processed_message_ids = set()

    def init_for_analysis(self, selected_analysts, run_context=None):
        """Initialize agent status and report sections based on selected analysts.

        Args:
            selected_analysts: List of analyst type strings (e.g., ["market", "news"])
        """
        self.selected_analysts = [a.lower() for a in selected_analysts]
        self.run_context = dict(run_context or {})

        # Build agent_status dynamically
        self.agent_status = {}

        # Add selected analysts
        for analyst_key in self.selected_analysts:
            if analyst_key in self.ANALYST_MAPPING:
                self.agent_status[self.ANALYST_MAPPING[analyst_key]] = "pending"

        # Add fixed teams
        for team_agents in self.FIXED_AGENTS.values():
            for agent in team_agents:
                self.agent_status[agent] = "pending"

        # Build report_sections dynamically
        self.report_sections = {}
        for section, (analyst_key, _) in self.REPORT_SECTIONS.items():
            if analyst_key is None or analyst_key in self.selected_analysts:
                self.report_sections[section] = None

        # Reset other state
        self.current_report = None
        self.final_report = None
        self.current_agent = None
        self.messages.clear()
        self.tool_calls.clear()
        self._processed_message_ids.clear()

    def get_completed_reports_count(self):
        """Count reports that are finalized (their finalizing agent is completed).

        A report is considered complete when:
        1. The report section has content (not None), AND
        2. The agent responsible for finalizing that report has status "completed"

        This prevents interim updates (like debate rounds) from counting as completed.
        """
        count = 0
        for section in self.report_sections:
            if section not in self.REPORT_SECTIONS:
                continue
            _, finalizing_agent = self.REPORT_SECTIONS[section]
            # Report is complete if it has content AND its finalizing agent is done
            has_content = self.report_sections.get(section) is not None
            agent_done = self.agent_status.get(finalizing_agent) == "completed"
            if has_content and agent_done:
                count += 1
        return count

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        # For the panel display, only show the most recently updated section
        latest_section = None
        latest_content = None

        # Find the most recently updated section
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content

        if latest_section and latest_content:
            # Format the current section for display
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment",
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Portfolio Management Decision",
            }
            self.current_report = f"### {section_titles[latest_section]}\n{latest_content}"

        # Update the final complete report
        self._update_final_report()

    def _update_final_report(self):
        report_parts = []

        # Analyst Team Reports - use .get() to handle missing sections
        analyst_sections = [
            "market_report",
            "sentiment_report",
            "news_report",
            "fundamentals_report",
        ]
        if any(self.report_sections.get(section) for section in analyst_sections):
            report_parts.append("## Analyst Team Reports")
            if self.report_sections.get("market_report"):
                report_parts.append(f"### Market Analysis\n{self.report_sections['market_report']}")
            if self.report_sections.get("sentiment_report"):
                report_parts.append(
                    f"### Social Sentiment\n{self.report_sections['sentiment_report']}"
                )
            if self.report_sections.get("news_report"):
                report_parts.append(f"### News Analysis\n{self.report_sections['news_report']}")
            if self.report_sections.get("fundamentals_report"):
                report_parts.append(
                    f"### Fundamentals Analysis\n{self.report_sections['fundamentals_report']}"
                )

        # Research Team Reports
        if self.report_sections.get("investment_plan"):
            report_parts.append("## Research Team Decision")
            report_parts.append(f"{self.report_sections['investment_plan']}")

        # Trading Team Reports
        if self.report_sections.get("trader_investment_plan"):
            report_parts.append("## Trading Team Plan")
            report_parts.append(f"{self.report_sections['trader_investment_plan']}")

        # Portfolio Management Decision
        if self.report_sections.get("final_trade_decision"):
            report_parts.append("## Portfolio Management Decision")
            report_parts.append(f"{self.report_sections['final_trade_decision']}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None


message_buffer = MessageBuffer()


def create_layout():
    """Build the live analysis desk: run map, activity feed, and live brief."""

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=5),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_row(
        Layout(name="run_map", size=42),
        Layout(name="workspace"),
    )
    layout["workspace"].split_column(
        Layout(name="activity", size=15),
        Layout(name="analysis"),
    )
    return layout


def create_live_display(layout: Layout) -> Live:
    """Create the live renderer on the same themed console as every panel.

    Rich resolves named styles at render time. Using its implicit global
    console here leaves custom styles such as ``brand`` unknown, which can stop
    the auto-refresh thread after the empty Layout placeholder is painted.
    """

    return Live(
        layout,
        console=console,
        refresh_per_second=8,
        vertical_overflow="crop",
    )


def format_tokens(n):
    """Format token count for display."""
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def _active_agent() -> str | None:
    return next(
        (agent for agent, status in message_buffer.agent_status.items() if status == "in_progress"),
        None,
    )


def _phase_status(agents: list[str]) -> str:
    statuses = [message_buffer.agent_status.get(agent, "pending") for agent in agents]
    if any(status == "error" for status in statuses):
        return "error"
    if statuses and all(status == "completed" for status in statuses):
        return "completed"
    if any(status == "in_progress" for status in statuses):
        return "in_progress"
    return "pending"


def _status_text(status: str, *, compact: bool = False) -> Text:
    labels = {
        "pending": ("○", "QUEUED", "stage.pending"),
        "in_progress": ("●", "RUNNING", "stage.active"),
        "completed": ("✓", "DONE", "stage.done"),
        "error": ("×", "ERROR", "danger"),
    }
    mark, label, style = labels.get(status, ("·", status.upper(), "muted"))
    return Text(mark if compact else f"{mark} {label}", style=style)


def _render_live_header() -> Panel:
    context = message_buffer.run_context
    ticker = str(context.get("ticker") or "DSE")
    analysis_date = str(context.get("analysis_date") or "—")
    provider = str(context.get("llm_provider") or "—").upper()
    active_agent = _active_agent()
    is_complete = bool(message_buffer.agent_status) and all(
        status == "completed" for status in message_buffer.agent_status.values()
    )

    header = Table.grid(expand=True)
    header.add_column(ratio=2)
    header.add_column(justify="right", ratio=1)
    header.add_row(
        Text.assemble(
            ("DOHA SECURITIES", "brand"),
            ("  /  ", "muted"),
            ("INTELLIGENCE DESK", "label"),
        ),
        Text(ticker, style="value"),
    )
    header.add_row(
        Text(
            "DSE multi-agent research, trading and risk orchestration",
            style="muted",
        ),
        Text(f"{analysis_date}  ·  {provider}", style="muted"),
    )
    state = (
        Text("✓ RUN COMPLETE", style="success")
        if is_complete
        else Text(f"● {active_agent or 'INITIALIZING'}", style="stage.active")
    )
    return Panel(
        header,
        title=state,
        title_align="right",
        border_style="#334155",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _render_run_map() -> Panel:
    run_map = Table.grid(expand=True, padding=(0, 1))
    run_map.add_column(width=3, justify="right")
    run_map.add_column(ratio=1)
    run_map.add_column(width=10, justify="right")

    stage_number = 0
    for team, configured_agents in WORKFLOW_TEAMS.items():
        agents = [agent for agent in configured_agents if agent in message_buffer.agent_status]
        if not agents:
            continue

        stage_number += 1
        phase_status = _phase_status(agents)
        complete_count = sum(
            message_buffer.agent_status.get(agent) == "completed" for agent in agents
        )
        number_style = "brand" if phase_status == "in_progress" else "muted"
        team_style = "value" if phase_status == "in_progress" else "label"
        run_map.add_row(
            Text(f"{stage_number:02}", style=number_style),
            Text(team.upper(), style=team_style),
            Text(f"{complete_count}/{len(agents)}", style="muted"),
        )
        for agent in agents:
            status = message_buffer.agent_status.get(agent, "pending")
            agent_style = "value" if status == "in_progress" else "muted"
            run_map.add_row(
                "",
                Text(f"  {agent}", style=agent_style),
                _status_text(status),
            )
        run_map.add_row("", "", "")

    return Panel(
        run_map,
        title=Text(" RUN MAP ", style="label"),
        subtitle=Text("ANALYZE → DECIDE → CONTROL", style="muted"),
        border_style="#334155",
        box=box.ROUNDED,
        padding=(1, 1),
    )


def _activity_style(message_type: str) -> str:
    return {
        "Tool": "activity.tool",
        "Agent": "activity.agent",
        "Data": "activity.agent",
        "System": "activity.system",
        "User": "activity.user",
    }.get(message_type, "label")


def _render_activity_feed() -> Panel:
    events = []
    for timestamp, tool_name, args in message_buffer.tool_calls:
        events.append((timestamp, "Tool", f"{tool_name}  {format_tool_args(args)}"))
    for timestamp, message_type, content in message_buffer.messages:
        content_text = str(content or "").replace("\n", " ")
        if len(content_text) > 180:
            content_text = content_text[:177] + "..."
        events.append((timestamp, message_type, content_text))
    events.sort(key=lambda event: event[0], reverse=True)

    feed = Table.grid(expand=True, padding=(0, 1))
    feed.add_column(width=8, style="muted")
    feed.add_column(width=9)
    feed.add_column(ratio=1, overflow="fold")
    for timestamp, message_type, content in events[:9]:
        feed.add_row(
            timestamp,
            Text(message_type.upper(), style=_activity_style(message_type)),
            Text(content, overflow="fold"),
        )
    if not events:
        feed.add_row("", Text("SYSTEM", style="activity.system"), "Preparing run…")

    return Panel(
        feed,
        title=Text(" ACTIVITY FEED ", style="label"),
        border_style="#334155",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _render_live_brief(spinner_text: str | None) -> Panel:
    content = (
        Markdown(message_buffer.current_report)
        if message_buffer.current_report
        else Spinner(
            "dots2",
            text=Text(
                f"  {spinner_text or 'Waiting for the first analyst brief…'}",
                style="muted",
            ),
            style="brand",
        )
    )
    return Panel(
        content,
        title=Text(" LIVE BRIEF ", style="label"),
        subtitle=Text("LATEST COMPLETED OUTPUT", style="muted"),
        border_style="#334155",
        box=box.ROUNDED,
        padding=(1, 2),
    )


def _render_metrics(stats_handler=None, start_time=None) -> Panel:
    agents_completed = sum(status == "completed" for status in message_buffer.agent_status.values())
    agents_total = len(message_buffer.agent_status)
    reports_completed = message_buffer.get_completed_reports_count()
    reports_total = len(message_buffer.report_sections)
    values = [
        ("AGENTS", f"{agents_completed}/{agents_total}"),
        ("REPORTS", f"{reports_completed}/{reports_total}"),
    ]

    if stats_handler:
        stats = stats_handler.get_stats()
        tokens = (
            f"{format_tokens(stats['tokens_in'])}↑ {format_tokens(stats['tokens_out'])}↓"
            if stats["tokens_in"] > 0 or stats["tokens_out"] > 0
            else "—"
        )
        values.extend(
            [
                ("LLM", str(stats["llm_calls"])),
                ("TOOLS", str(stats["tool_calls"])),
                ("TOKENS", tokens),
            ]
        )
    if start_time:
        elapsed = time.time() - start_time
        values.append(("ELAPSED", f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"))

    metrics = Table.grid(expand=True)
    for _ in values:
        metrics.add_column(justify="center")
    metrics.add_row(
        *[Text.assemble((f"{label} ", "muted"), (value, "value")) for label, value in values]
    )
    return Panel(
        metrics,
        border_style="#334155",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def update_display(layout, spinner_text=None, stats_handler=None, start_time=None):
    """Refresh every region of the live intelligence desk."""

    layout["header"].update(_render_live_header())
    layout["run_map"].update(_render_run_map())
    layout["activity"].update(_render_activity_feed())
    layout["analysis"].update(_render_live_brief(spinner_text))
    layout["footer"].update(_render_metrics(stats_handler, start_time))


def _compact_panel_width() -> int:
    return max(36, min(92, console.size.width - 4))


def render_startup_header() -> Panel:
    """Render a compact launch identity without the old oversized ASCII logo."""

    title = Text.assemble(
        ("DOHA SECURITIES", "brand"),
        ("  /  ", "muted"),
        ("STOCK AI", "brand.secondary"),
    )
    pipeline = Text.assemble(
        ("01 ANALYZE", "brand"),
        (
            "  ─  02 RESEARCH  ─  03 TRADE  ─  04 RISK  ─  05 PORTFOLIO",
            "muted",
        ),
    )
    return Panel(
        Group(
            title,
            Text("DSE multi-agent market intelligence desk", style="muted"),
            Text(""),
            pipeline,
        ),
        title=Text(" MARKET INTELLIGENCE CONSOLE ", style="label"),
        subtitle=Text("DECISION SUPPORT · READ-ONLY MARKET DATA", style="muted"),
        border_style="#334155",
        box=box.ROUNDED,
        padding=(1, 2),
        width=_compact_panel_width(),
    )


def render_setup_step(
    number: int,
    title: str,
    description: str,
    default: str | None = None,
) -> Panel:
    """Render one compact setup card with consistent hierarchy."""

    content = Table.grid(expand=True)
    content.add_column(width=5)
    content.add_column(ratio=1)
    content.add_row(Text(f"{number:02}", style="brand"), Text(title.upper(), style="value"))
    content.add_row("", Text(description, style="muted"))
    if default:
        content.add_row(
            "",
            Text.assemble(("DEFAULT  ", "label"), (default, "brand.secondary")),
        )
    return Panel(
        content,
        border_style="#334155",
        box=box.ROUNDED,
        padding=(0, 1),
        width=_compact_panel_width(),
    )


def _print_setup_step(*args, **kwargs) -> None:
    console.print(Align.center(render_setup_step(*args, **kwargs)))


def _print_setup_value(
    label: str,
    value: str,
    *,
    source: str | None = None,
) -> None:
    line = Text.assemble(
        ("  ✓  ", "success"),
        (f"{label.upper():<13}", "label"),
        (value, "value"),
    )
    if source:
        line.append("  ·  ", style="muted")
        line.append(source.upper(), style="muted")
    console.print(Align.center(Align.left(line, width=_compact_panel_width())))


def render_run_brief(selections: dict) -> Panel:
    """Summarize the configured run before the live workspace takes over."""

    analysts = ", ".join(analyst.value for analyst in selections["analysts"])
    rows = [
        ("SYMBOL", selections["ticker"], "DATE", selections["analysis_date"]),
        ("ANALYSTS", analysts, "DEPTH", f"{selections['research_depth']} rounds"),
        (
            "PROVIDER",
            selections["llm_provider"].upper(),
            "LANGUAGE",
            selections["output_language"],
        ),
    ]
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(width=11, style="label")
    table.add_column(ratio=1, style="value")
    table.add_column(width=11, style="label")
    table.add_column(ratio=1, style="value")
    for row in rows:
        table.add_row(*[str(value) for value in row])
    return Panel(
        table,
        title=Text(" RUN BRIEF ", style="brand"),
        subtitle=Text("CONFIGURATION LOCKED · STARTING ANALYSIS", style="muted"),
        border_style="#22d3ee",
        box=box.ROUNDED,
        padding=(1, 2),
        width=_compact_panel_width(),
    )


def render_completion_card(selections: dict, timing_summary: str) -> Panel:
    """Render a calm handoff from the live workspace to report actions."""

    heading = Text.assemble(
        ("✓  RUN COMPLETE", "success"),
        ("  /  ", "muted"),
        (str(selections["ticker"]), "value"),
    )
    details = Table.grid(expand=True, padding=(0, 1))
    details.add_column(width=12, style="label")
    details.add_column(ratio=1, style="value")
    details.add_row("AS-OF DATE", str(selections["analysis_date"]))
    details.add_row("STATUS", "Decision chain complete")
    details.add_row("TIMING", timing_summary)
    return Panel(
        Group(heading, Text(""), details),
        title=Text(" ANALYSIS HANDOFF ", style="label"),
        subtitle=Text("REPORT ACTIONS", style="muted"),
        border_style="#34d399",
        box=box.ROUNDED,
        padding=(1, 2),
        width=_compact_panel_width(),
    )


def _print_notice(title: str, message: str, *, level: str = "warning") -> None:
    style = {"success": "#34d399", "error": "#fb7185"}.get(level, "#fbbf24")
    title_style = {"success": "success", "error": "danger"}.get(level, "warning")
    console.print(
        Align.center(
            Panel(
                Text(message),
                title=Text(f" {title.upper()} ", style=title_style),
                border_style=style,
                box=box.ROUNDED,
                padding=(0, 1),
                width=_compact_panel_width(),
            )
        )
    )


def get_user_selections():
    """Get all user selections before starting the analysis display."""
    console.print(Align.center(render_startup_header()))
    console.print()

    def thinking_value_or_prompt(env_var, config_key, label, box_title, box_body, prompt_fn):
        """Return the env-configured reasoning/thinking value, or prompt for it.

        When ``env_var`` is set the interactive choice is skipped and the value
        the env overlay placed on DEFAULT_CONFIG is used — mirroring the
        env-precedence rule applied to the other selection steps.
        """
        if os.environ.get(env_var):
            value = DEFAULT_CONFIG[config_key]
            _print_setup_value(label, str(value), source="environment")
            return value
        _print_setup_step(8, box_title.replace("Step 8: ", ""), box_body)
        return prompt_fn()

    # Step 1: Ticker symbol
    _print_setup_step(
        1,
        "Instrument",
        "Enter a Dhaka Stock Exchange trading code, such as GP or BRACBANK.",
        "GP",
    )
    selected_ticker = get_ticker()
    _print_setup_value("Instrument", selected_ticker)
    has_dse_credentials = bool(
        os.environ.get("DSE_EMAIL_OR_PHONE") and os.environ.get("DSE_PASSWORD")
    )
    if (
        DEFAULT_CONFIG.get("market_profile") == "bangladesh_dse"
        and not os.environ.get("DSE_ACCESS_TOKEN")
        and not has_dse_credentials
    ):
        _print_notice(
            "DSE login required",
            "Set DSE_ACCESS_TOKEN, or configure both DSE_EMAIL_OR_PHONE and "
            "DSE_PASSWORD before starting market-data analysis.",
        )
    asset_type = detect_asset_type(selected_ticker)
    # Only announce when it's not the default stock path, to avoid printing
    # "stock" on every run.
    if asset_type.value != "stock":
        _print_setup_value("Asset class", asset_type.value)

    # Step 2: Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    _print_setup_step(
        2,
        "As-of date",
        "Choose the market-information cutoff date in YYYY-MM-DD format.",
        default_date,
    )
    analysis_date = get_analysis_date()
    _print_setup_value("As-of date", analysis_date)

    # Step 3: Output language (skipped when set via TRADINGAGENTS_OUTPUT_LANGUAGE)
    if os.environ.get("TRADINGAGENTS_OUTPUT_LANGUAGE"):
        output_language = DEFAULT_CONFIG["output_language"]
        _print_setup_value("Language", output_language, source="environment")
    else:
        _print_setup_step(
            3,
            "Report language",
            "Choose the language used for every analyst brief and final decision.",
        )
        output_language = ask_output_language()
        _print_setup_value("Language", output_language)

    # Step 4: Select analysts
    _print_setup_step(
        4,
        "Analyst desk",
        "Build the specialist team that will investigate the instrument.",
    )
    selected_analysts = select_analysts(
        asset_type,
        social_media_enabled=DEFAULT_CONFIG.get("social_media_enabled", False),
    )
    _print_setup_value(
        "Analysts",
        ", ".join(analyst.value for analyst in selected_analysts),
    )

    # Step 5: Research depth (skipped when both round counts are set via env).
    # Research depth maps to the debate + risk round counts; when both are
    # supplied through TRADINGAGENTS_MAX_DEBATE_ROUNDS / _MAX_RISK_ROUNDS we keep
    # the run non-interactive and honor the env values (#977).
    depth_from_env = bool(os.environ.get("TRADINGAGENTS_MAX_DEBATE_ROUNDS")) and bool(
        os.environ.get("TRADINGAGENTS_MAX_RISK_ROUNDS")
    )
    if depth_from_env:
        selected_research_depth = DEFAULT_CONFIG["max_debate_rounds"]
        _print_setup_value(
            "Research",
            f"{DEFAULT_CONFIG['max_debate_rounds']} debate / "
            f"{DEFAULT_CONFIG['max_risk_discuss_rounds']} risk rounds",
            source="environment",
        )
    else:
        _print_setup_step(
            5,
            "Research depth",
            "Set the number of debate and risk-challenge rounds.",
        )
        selected_research_depth = select_research_depth()
        _print_setup_value("Research", f"{selected_research_depth} rounds")

    # Step 6: LLM Provider (skipped when set via TRADINGAGENTS_LLM_PROVIDER).
    # The backend URL comes from TRADINGAGENTS_LLM_BACKEND_URL when set,
    # otherwise the provider's default endpoint — the same value the menu
    # would have picked.
    provider_from_env = bool(os.environ.get("TRADINGAGENTS_LLM_PROVIDER"))
    if provider_from_env:
        selected_llm_provider = DEFAULT_CONFIG["llm_provider"].lower()
        backend_url = resolve_backend_url(
            selected_llm_provider, env_url=DEFAULT_CONFIG["backend_url"]
        )
        _print_setup_value("LLM provider", selected_llm_provider, source="environment")
        _print_setup_value("Endpoint", str(backend_url), source="resolved")
        # Still confirm/persist the API key so the run doesn't fail later.
        ensure_api_key(selected_llm_provider)
    else:
        _print_setup_step(
            6,
            "Intelligence engine",
            "Select the model provider that will power the analyst desk.",
        )
        selected_llm_provider, backend_url = select_llm_provider()

        # Providers with regional endpoints prompt for the region as a secondary
        # step so the main dropdown stays clean (mainland China and international
        # accounts cannot share API keys).
        if selected_llm_provider == "qwen":
            selected_llm_provider, backend_url = ask_qwen_region()
        elif selected_llm_provider == "minimax":
            selected_llm_provider, backend_url = ask_minimax_region()
        elif selected_llm_provider == "glm":
            selected_llm_provider, backend_url = ask_glm_region()

        # Honor an explicit env backend URL even when the provider was chosen
        # interactively, so it isn't overwritten by the menu default (#978).
        backend_url = resolve_backend_url(
            selected_llm_provider, backend_url, env_url=DEFAULT_CONFIG["backend_url"]
        )

        # The generic OpenAI-compatible endpoint has no default; ask for it if
        # neither the menu nor the environment supplied one.
        if selected_llm_provider == "openai_compatible" and not backend_url:
            backend_url = prompt_openai_compatible_url()

        # For Ollama, surface the resolved endpoint (OLLAMA_BASE_URL vs default)
        # before model selection so it's obvious where we're connecting.
        if selected_llm_provider == "ollama":
            confirm_ollama_endpoint(backend_url or "")

        # Confirm the provider's API key is present; prompt the user to paste
        # one and persist it to .env if it's missing, so the analysis run
        # doesn't fail later at the first API call.
        ensure_api_key(selected_llm_provider)
        _print_setup_value("LLM provider", selected_llm_provider)
        _print_setup_value("Endpoint", str(backend_url or "SDK managed"), source="resolved")

    # Step 7: Thinking agents (skipped when either model is set via environment)
    if os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM") or os.environ.get(
        "TRADINGAGENTS_DEEP_THINK_LLM"
    ):
        selected_shallow_thinker = DEFAULT_CONFIG["quick_think_llm"]
        selected_deep_thinker = DEFAULT_CONFIG["deep_think_llm"]
        thinker_source = "environment"
    else:
        _print_setup_step(
            7,
            "Thinking models",
            "Assign fast and deep models to the appropriate reasoning work.",
        )
        selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
        selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)
        thinker_source = None
    _print_setup_value("Quick model", selected_shallow_thinker, source=thinker_source)
    _print_setup_value("Deep model", selected_deep_thinker, source=thinker_source)

    # Step 8: Provider-specific reasoning/thinking configuration. Each knob is
    # settable via its TRADINGAGENTS_* env var; when that var is set (or the
    # provider itself came from env) the prompt is skipped and the configured
    # value is used — same env-precedence rule as the steps above. None = each
    # provider's own default.
    thinking_level = None
    reasoning_effort = None
    anthropic_effort = None

    provider_lower = selected_llm_provider.lower()
    if provider_from_env:
        thinking_level = DEFAULT_CONFIG["google_thinking_level"]
        reasoning_effort = DEFAULT_CONFIG["openai_reasoning_effort"]
        anthropic_effort = DEFAULT_CONFIG["anthropic_effort"]
    elif provider_lower == "google":
        thinking_level = thinking_value_or_prompt(
            "TRADINGAGENTS_GOOGLE_THINKING_LEVEL",
            "google_thinking_level",
            "Gemini thinking mode",
            "Step 8: Thinking Mode",
            "Configure Gemini thinking mode",
            ask_gemini_thinking_config,
        )
    elif provider_lower == "openai":
        reasoning_effort = thinking_value_or_prompt(
            "TRADINGAGENTS_OPENAI_REASONING_EFFORT",
            "openai_reasoning_effort",
            "Reasoning effort",
            "Step 8: Reasoning Effort",
            "Configure OpenAI reasoning effort level",
            ask_openai_reasoning_effort,
        )
    elif provider_lower == "anthropic":
        anthropic_effort = thinking_value_or_prompt(
            "TRADINGAGENTS_ANTHROPIC_EFFORT",
            "anthropic_effort",
            "Claude effort",
            "Step 8: Effort Level",
            "Configure Claude effort level",
            ask_anthropic_effort,
        )

    selections = {
        "ticker": selected_ticker,
        "asset_type": asset_type.value,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
        "google_thinking_level": thinking_level,
        "openai_reasoning_effort": reasoning_effort,
        "anthropic_effort": anthropic_effort,
        "output_language": output_language,
    }
    console.print()
    console.print(Align.center(render_run_brief(selections)))
    console.print()
    return selections


def get_analysis_date():
    """Get the analysis date from user input."""
    while True:
        date_str = typer.prompt("", default=datetime.datetime.now().strftime("%Y-%m-%d"))
        try:
            # Validate date format and ensure it's not in the future
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                _print_notice(
                    "Invalid date",
                    "The analysis date cannot be in the future.",
                    level="error",
                )
                continue
            return date_str
        except ValueError:
            _print_notice(
                "Invalid date",
                "Use the YYYY-MM-DD format, for example 2026-08-17.",
                level="error",
            )


def save_report_to_disk(final_state, ticker: str, save_path: Path):
    """Save the complete analysis report to disk (shared CLI/API writer)."""
    return write_report_tree(final_state, ticker, save_path)


def display_complete_report(final_state):
    """Display the complete analysis dossier without redundant section panels."""

    def report_panel(stage: str, desk: str, title: str, content: str, accent: str):
        panel_title = Text.assemble(
            (f" {stage}  ", "muted"),
            (f"{desk.upper()}  /  ", "label"),
            (title.upper(), "value"),
            (" ", "muted"),
        )
        return Panel(
            Markdown(content),
            title=panel_title,
            title_align="left",
            border_style=accent,
            box=box.ROUNDED,
            padding=(1, 2),
        )

    console.print()
    console.print(
        Rule(
            Text.assemble(
                ("ANALYSIS DOSSIER", "brand"),
                ("  /  ", "muted"),
                ("COMPLETE DECISION CHAIN", "label"),
            ),
            style="#334155",
        )
    )

    analyst_reports = [
        ("Market Analyst", final_state.get("market_report")),
        ("Sentiment Analyst", final_state.get("sentiment_report")),
        ("News Analyst", final_state.get("news_report")),
        ("Fundamentals Analyst", final_state.get("fundamentals_report")),
    ]
    for title, content in analyst_reports:
        if content:
            console.print(report_panel("01", "Analyst Desk", title, content, "#22d3ee"))

    debate = final_state.get("investment_debate_state") or {}
    research_reports = [
        ("Bull Researcher", debate.get("bull_history")),
        ("Bear Researcher", debate.get("bear_history")),
        ("Research Manager", debate.get("judge_decision")),
    ]
    for title, content in research_reports:
        if content:
            console.print(report_panel("02", "Research Desk", title, content, "#a78bfa"))

    if final_state.get("trader_investment_plan"):
        console.print(
            report_panel(
                "03",
                "Trade Desk",
                "Trader",
                final_state["trader_investment_plan"],
                "#fbbf24",
            )
        )

    risk = final_state.get("risk_debate_state") or {}
    risk_reports = [
        ("Aggressive Analyst", risk.get("aggressive_history")),
        ("Conservative Analyst", risk.get("conservative_history")),
        ("Neutral Analyst", risk.get("neutral_history")),
    ]
    for title, content in risk_reports:
        if content:
            console.print(report_panel("04", "Risk Desk", title, content, "#fb7185"))

    if risk.get("judge_decision"):
        console.print(
            report_panel(
                "05",
                "Portfolio Desk",
                "Portfolio Manager",
                risk["judge_decision"],
                "#34d399",
            )
        )


def update_research_team_status(status):
    """Update status for research team members (not Trader)."""
    research_team = ["Bull Researcher", "Bear Researcher", "Research Manager"]
    for agent in research_team:
        message_buffer.update_agent_status(agent, status)


# Ordered list of analysts for status transitions
ANALYST_ORDER = ["market", "social", "news", "fundamentals"]
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Sentiment Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}


def update_analyst_statuses(message_buffer, chunk, wall_time_tracker=None):
    """Update analyst statuses based on accumulated report state.

    Logic:
    - Store new report content from the current chunk if present
    - Check accumulated report_sections (not just current chunk) for status
    - Analysts with reports = completed
    - First analyst without report = in_progress
    - Remaining analysts without reports = pending
    - When all analysts done, set Bull Researcher to in_progress
    """
    selected = message_buffer.selected_analysts
    found_active = False

    if wall_time_tracker is not None:
        sync_analyst_tracker_from_chunk(wall_time_tracker, chunk)

    for analyst_key in ANALYST_ORDER:
        if analyst_key not in selected:
            continue

        agent_name = ANALYST_AGENT_NAMES[analyst_key]
        report_key = ANALYST_REPORT_MAP[analyst_key]

        # Capture new report content from current chunk
        if chunk.get(report_key):
            message_buffer.update_report_section(report_key, chunk[report_key])

        # Determine status from accumulated sections, not just current chunk
        has_report = bool(message_buffer.report_sections.get(report_key))

        if has_report:
            message_buffer.update_agent_status(agent_name, "completed")
        elif not found_active:
            message_buffer.update_agent_status(agent_name, "in_progress")
            found_active = True
        else:
            message_buffer.update_agent_status(agent_name, "pending")

    # When all analysts complete, transition research team to in_progress
    if (
        not found_active
        and selected
        and message_buffer.agent_status.get("Bull Researcher") == "pending"
    ):
        message_buffer.update_agent_status("Bull Researcher", "in_progress")


def extract_content_string(content):
    """Extract string content from various message formats.
    Returns None if no meaningful text content is found.
    """
    import ast

    def is_empty(val):
        """Check if value is empty using Python's truthiness."""
        if val is None or val == "":
            return True
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return True
            try:
                return not bool(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                return False  # Can't parse = real text
        return not bool(val)

    if is_empty(content):
        return None

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        text = content.get("text", "")
        return text.strip() if not is_empty(text) else None

    if isinstance(content, list):
        text_parts = [
            item.get("text", "").strip()
            if isinstance(item, dict) and item.get("type") == "text"
            else (item.strip() if isinstance(item, str) else "")
            for item in content
        ]
        result = " ".join(t for t in text_parts if t and not is_empty(t))
        return result if result else None

    return str(content).strip() if not is_empty(content) else None


def classify_message_type(message) -> tuple[str, str | None]:
    """Classify LangChain message into display type and extract content.

    Returns:
        (type, content) - type is one of: User, Agent, Data, Control
                        - content is extracted string or None
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    content = extract_content_string(getattr(message, "content", None))

    if isinstance(message, HumanMessage):
        if content and content.strip() == "Continue":
            return ("Control", content)
        return ("User", content)

    if isinstance(message, ToolMessage):
        return ("Data", content)

    if isinstance(message, AIMessage):
        return ("Agent", content)

    # Fallback for unknown types
    return ("System", content)


def format_tool_args(args, max_length=80) -> str:
    """Format tool arguments for terminal display."""
    result = str(args)
    if len(result) > max_length:
        return result[: max_length - 3] + "..."
    return result


def _build_run_config(selections: dict, checkpoint: bool | None) -> dict:
    """Assemble the run config from interactive selections, honoring env precedence.

    Round counts and checkpoint follow "explicit env/flag wins": an env-applied
    value on DEFAULT_CONFIG is preserved unless the user overrode it on the CLI.
    """
    config = DEFAULT_CONFIG.copy()
    # Research depth sets both round counts, but an explicit env override
    # (TRADINGAGENTS_MAX_DEBATE_ROUNDS / _MAX_RISK_ROUNDS) wins over the
    # interactive selection — leave the env-applied value in place (#977).
    if not os.environ.get("TRADINGAGENTS_MAX_DEBATE_ROUNDS"):
        config["max_debate_rounds"] = selections["research_depth"]
    if not os.environ.get("TRADINGAGENTS_MAX_RISK_ROUNDS"):
        config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    # Provider-specific thinking configuration
    config["google_thinking_level"] = selections.get("google_thinking_level")
    config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")
    config["anthropic_effort"] = selections.get("anthropic_effort")
    config["output_language"] = selections.get("output_language", "English")
    # --checkpoint/--no-checkpoint overrides only when explicitly given; omitting
    # the flag preserves TRADINGAGENTS_CHECKPOINT_ENABLED / the default (#976).
    if checkpoint is not None:
        config["checkpoint_enabled"] = checkpoint
    return config


def run_analysis(checkpoint: bool | None = None, *, open_ui: bool = False):
    # First get all user selections
    selections = get_user_selections()

    config = _build_run_config(selections, checkpoint)

    # Create stats callback handler for tracking LLM/tool calls
    stats_handler = StatsCallbackHandler()

    # Normalize analyst selection to predefined order (selection is a 'set', order is fixed)
    selected_set = {analyst.value for analyst in selections["analysts"]}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]
    analyst_execution_plan = build_analyst_execution_plan(selected_analyst_keys)
    analyst_wall_time_tracker = AnalystWallTimeTracker(analyst_execution_plan)

    # Initialize the graph with callbacks bound to LLMs
    graph = TradingAgentsGraph(
        selected_analyst_keys,
        config=config,
        debug=True,
        callbacks=[stats_handler],
    )

    # Initialize message buffer with selected analysts
    message_buffer.init_for_analysis(
        selected_analyst_keys,
        run_context={
            "ticker": selections["ticker"],
            "analysis_date": selections["analysis_date"],
            "llm_provider": selections["llm_provider"],
        },
    )

    # Track start time for elapsed display
    start_time = time.time()

    # Create result directory
    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)

        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # Replace newlines with spaces
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")

        return wrapper

    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)

        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")

        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)

        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if (
                section_name in obj.report_sections
                and obj.report_sections[section_name] is not None
            ):
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    text = (
                        "\n".join(str(item) for item in content)
                        if isinstance(content, list)
                        else content
                    )
                    with open(report_dir / file_name, "w", encoding="utf-8") as f:
                        f.write(text)

        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(
        message_buffer, "update_report_section"
    )

    # Populate every leaf before Live mounts. This prevents Rich's diagnostic
    # Layout placeholders from ever becoming the first visible frame.
    layout = create_layout()
    spinner_text = f"Analyzing {selections['ticker']} on {selections['analysis_date']}…"
    update_display(
        layout,
        spinner_text,
        stats_handler=stats_handler,
        start_time=start_time,
    )

    with create_live_display(layout) as live:
        # Add initial messages
        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        if selections["asset_type"] != "stock":
            message_buffer.add_message("System", f"Detected asset type: {selections['asset_type']}")
        message_buffer.add_message("System", f"Analysis date: {selections['analysis_date']}")
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout, stats_handler=stats_handler, start_time=start_time)
        live.refresh()

        # Update agent status to in_progress for the first analyst
        first_analyst = get_initial_analyst_node(analyst_execution_plan)
        message_buffer.update_agent_status(first_analyst, "in_progress")
        analyst_wall_time_tracker.mark_started(selected_analyst_keys[0])
        update_display(layout, spinner_text, stats_handler=stats_handler, start_time=start_time)
        live.refresh()

        # Initialize state and get graph args with callbacks.
        # Resolve the instrument identity once here so all agents anchor to
        # the real company (#814); the CLI builds state directly rather than
        # going through propagate(), so this must happen on the CLI path too.
        instrument_context = graph.resolve_instrument_context(
            selections["ticker"], selections["asset_type"]
        )
        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"],
            selections["analysis_date"],
            asset_type=selections["asset_type"],
            instrument_context=instrument_context,
        )
        # Pass callbacks to graph config for tool execution tracking
        # (LLM tracking is handled separately via LLM constructor)
        args = graph.propagator.get_graph_args(callbacks=[stats_handler])

        # Stream the analysis
        trace = []
        for chunk in graph.graph.stream(init_agent_state, **args):
            # Process all messages in chunk, deduplicating by message ID
            for message in chunk.get("messages", []):
                msg_id = getattr(message, "id", None)
                if msg_id is not None:
                    if msg_id in message_buffer._processed_message_ids:
                        continue
                    message_buffer._processed_message_ids.add(msg_id)

                msg_type, content = classify_message_type(message)
                if content and content.strip():
                    message_buffer.add_message(msg_type, content)

                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        if isinstance(tool_call, dict):
                            message_buffer.add_tool_call(tool_call["name"], tool_call["args"])
                        else:
                            message_buffer.add_tool_call(tool_call.name, tool_call.args)

            # Update analyst statuses based on report state (runs on every chunk)
            update_analyst_statuses(
                message_buffer,
                chunk,
                wall_time_tracker=analyst_wall_time_tracker,
            )

            # Research Team - Handle Investment Debate State
            if chunk.get("investment_debate_state"):
                debate_state = chunk["investment_debate_state"]
                bull_hist = debate_state.get("bull_history", "").strip()
                bear_hist = debate_state.get("bear_history", "").strip()
                judge = debate_state.get("judge_decision", "").strip()

                # Only update status when there's actual content
                if bull_hist or bear_hist:
                    update_research_team_status("in_progress")
                if bull_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Bull Researcher Analysis\n{bull_hist}"
                    )
                if bear_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Bear Researcher Analysis\n{bear_hist}"
                    )
                if judge:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Research Manager Decision\n{judge}"
                    )
                    update_research_team_status("completed")
                    message_buffer.update_agent_status("Trader", "in_progress")

            # Trading Team
            if chunk.get("trader_investment_plan"):
                message_buffer.update_report_section(
                    "trader_investment_plan", chunk["trader_investment_plan"]
                )
                if message_buffer.agent_status.get("Trader") != "completed":
                    message_buffer.update_agent_status("Trader", "completed")
                    message_buffer.update_agent_status("Aggressive Analyst", "in_progress")

            # Risk Management Team - Handle Risk Debate State
            if chunk.get("risk_debate_state"):
                risk_state = chunk["risk_debate_state"]
                agg_hist = risk_state.get("aggressive_history", "").strip()
                con_hist = risk_state.get("conservative_history", "").strip()
                neu_hist = risk_state.get("neutral_history", "").strip()
                judge = risk_state.get("judge_decision", "").strip()

                if agg_hist:
                    if message_buffer.agent_status.get("Aggressive Analyst") != "completed":
                        message_buffer.update_agent_status("Aggressive Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Aggressive Analyst Analysis\n{agg_hist}"
                    )
                if con_hist:
                    if message_buffer.agent_status.get("Conservative Analyst") != "completed":
                        message_buffer.update_agent_status("Conservative Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Conservative Analyst Analysis\n{con_hist}"
                    )
                if neu_hist:
                    if message_buffer.agent_status.get("Neutral Analyst") != "completed":
                        message_buffer.update_agent_status("Neutral Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Neutral Analyst Analysis\n{neu_hist}"
                    )
                if judge and message_buffer.agent_status.get("Portfolio Manager") != "completed":
                    message_buffer.update_agent_status("Portfolio Manager", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Portfolio Manager Decision\n{judge}"
                    )
                    message_buffer.update_agent_status("Aggressive Analyst", "completed")
                    message_buffer.update_agent_status("Conservative Analyst", "completed")
                    message_buffer.update_agent_status("Neutral Analyst", "completed")
                    message_buffer.update_agent_status("Portfolio Manager", "completed")

            # Update the display
            update_display(layout, stats_handler=stats_handler, start_time=start_time)
            live.refresh()

            trace.append(chunk)

        # Streamed chunks are per-node deltas, not full state. Merge them
        # so every report field populated across the run is present.
        final_state = {}
        for chunk in trace:
            final_state.update(chunk)

        # Update all agent statuses to completed
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "System", f"Completed analysis for {selections['analysis_date']}"
        )
        message_buffer.add_message("System", analyst_wall_time_tracker.format_summary())

        # Update final report sections
        for section in message_buffer.report_sections:
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        update_display(layout, stats_handler=stats_handler, start_time=start_time)
        live.refresh()

    # The interactive CLI streams the graph directly for live rendering. Save
    # its merged final state explicitly so the API/UI consumes this exact run
    # instead of starting the multi-agent workflow a second time.
    state_log_path = graph.save_state(
        selections["analysis_date"],
        final_state,
        ticker=selections["ticker"],
    )

    # Post-analysis prompts (outside Live context for clean interaction)
    timing_summary = analyst_wall_time_tracker.format_summary()
    console.print()
    console.print(Align.center(render_completion_card(selections, timing_summary)))
    console.print()

    # Prompt to save report
    save_choice = typer.prompt("Save report?", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = Path.cwd() / "reports" / f"{selections['ticker']}_{timestamp}"
        save_path_str = typer.prompt(
            "Save path (press Enter for default)", default=str(default_path)
        ).strip()
        save_path = Path(save_path_str)
        try:
            report_file = save_report_to_disk(final_state, selections["ticker"], save_path)
            _print_notice(
                "Report saved",
                f"{save_path.resolve()}\nComplete report: {report_file.name}",
                level="success",
            )
        except Exception as e:
            _print_notice("Report save failed", str(e), level="error")

    # Prompt to display full report
    display_choice = typer.prompt("\nDisplay full report on screen?", default="Y").strip().upper()
    if display_choice in ("Y", "YES", ""):
        display_complete_report(final_state)

    if open_ui:
        from dohasecuritiesstockai.api.repository import AnalysisRepository
        from dohasecuritiesstockai.dashboard import (
            dashboard_url,
            launch_dashboard,
            prepare_dashboard_analysis,
        )

        _print_notice(
            "Dashboard",
            "Preparing the interactive analysis workspace…",
            level="success",
        )
        try:
            persisted_state = AnalysisRepository.load_state(state_log_path)
            analysis, analysis_path = prepare_dashboard_analysis(
                selections["ticker"],
                selections["analysis_date"],
                persisted_state,
                config["results_dir"],
                use_ai=False,
            )
            host = os.environ.get("TRADINGAGENTS_API_HOST", "127.0.0.1")
            port = int(os.environ.get("TRADINGAGENTS_API_PORT", "8000"))
            url = dashboard_url(host, port, analysis.symbol, analysis.analysis_date)
            _print_notice(
                "Dashboard ready",
                f"Opening {url}\nData: {analysis_path}\nKeep this terminal open; Ctrl+C stops the UI.",
                level="success",
            )
            launch_dashboard(
                analysis.symbol,
                analysis.analysis_date,
                host=host,
                port=port,
            )
        except Exception as exc:
            # DSE and dashboard helpers deliberately redact credentials/tokens
            # from their exception messages, so surface the actionable stage
            # error without losing the already completed CLI report.
            _print_notice("Dashboard launch failed", str(exc), level="error")


@app.command()
def analyze(
    checkpoint: bool | None = typer.Option(
        None,
        "--checkpoint/--no-checkpoint",
        help="Enable/disable checkpoint-resume (save state after each node so a "
        "crashed run can resume). Omit to honor TRADINGAGENTS_CHECKPOINT_ENABLED.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
    open_ui: bool | None = typer.Option(
        None,
        "--open-ui/--no-open-ui",
        help="After analysis, prepare and open the Angular dashboard. Omit to honor "
        "TRADINGAGENTS_OPEN_UI_AFTER_ANALYSIS.",
    ),
):
    """Run Doha Securities Stock AI: DSE Multi-Agent Stock Analysis."""

    if clear_checkpoints:
        from dohasecuritiesstockai.graph.checkpointer import clear_all_checkpoints

        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")
    try:
        should_open_ui = (
            _env_flag("TRADINGAGENTS_OPEN_UI_AFTER_ANALYSIS") if open_ui is None else open_ui
        )
        run_analysis(checkpoint=checkpoint, open_ui=should_open_ui)
    except _NO_CONSOLE_ERRORS:
        # A terminal with no console buffer cannot host the interactive prompts.
        # Emit one actionable line on stderr instead of a prompt_toolkit
        # traceback; plain text, since rich may not render here either (#1138).
        typer.echo(
            "Error: no Windows console available. The interactive CLI needs a real "
            "console buffer — run it from Windows Terminal, PowerShell, or cmd.exe "
            "rather than a piped or embedded terminal.",
            err=True,
        )
        raise typer.Exit(code=1) from None
    except VendorError as exc:
        # Market-data failures are expected operational states. Present the
        # actionable, already-redacted vendor message without a wall of stack
        # frames; unexpected programming errors still propagate normally.
        _print_notice("Market data unavailable", str(exc), level="error")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
