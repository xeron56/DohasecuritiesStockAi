"""Visual-structure regressions for the redesigned terminal experience."""

from __future__ import annotations

from rich.console import Console
from typer.testing import CliRunner

import cli.main as main
from cli.models import AnalystType
from cli.theme import CLI_THEME
from dohasecuritiesstockai.dataflows.errors import VendorError


def _preview_console(*, width: int = 120, height: int = 42) -> Console:
    return Console(
        theme=CLI_THEME,
        width=width,
        height=height,
        record=True,
        color_system=None,
        highlight=False,
    )


def test_startup_is_compact_and_drops_the_ascii_banner(monkeypatch):
    preview = _preview_console(width=120)
    monkeypatch.setattr(main, "console", preview)

    preview.print(main.render_startup_header())
    preview.print(
        main.render_setup_step(
            1,
            "Instrument",
            "Enter a Dhaka Stock Exchange trading code.",
            "GP",
        )
    )
    output = preview.export_text()

    assert "DOHA SECURITIES" in output
    assert "01 ANALYZE" in output
    assert "INSTRUMENT" in output
    assert "DEFAULT" in output
    assert "____" not in output
    assert max(len(line) for line in output.splitlines()) <= 92


def test_run_brief_surfaces_the_decisions_that_define_the_run(monkeypatch):
    preview = _preview_console(width=120)
    monkeypatch.setattr(main, "console", preview)
    selections = {
        "ticker": "BRACBANK",
        "analysis_date": "2026-08-17",
        "analysts": [
            AnalystType.MARKET,
            AnalystType.NEWS,
            AnalystType.FUNDAMENTALS,
        ],
        "research_depth": 3,
        "llm_provider": "openrouter",
        "output_language": "English",
    }

    preview.print(main.render_run_brief(selections))
    output = preview.export_text()

    assert "RUN BRIEF" in output
    assert "BRACBANK" in output
    assert "OPENROUTER" in output
    assert "market, news, fundamentals" in output


def test_live_workspace_has_distinct_run_activity_and_brief_regions(monkeypatch):
    preview = _preview_console(width=140, height=44)
    monkeypatch.setattr(main, "console", preview)
    buffer = main.MessageBuffer()
    buffer.init_for_analysis(
        ["market", "news", "fundamentals"],
        run_context={
            "ticker": "BRACBANK",
            "analysis_date": "2026-08-17",
            "llm_provider": "openrouter",
        },
    )
    buffer.update_agent_status("Market Analyst", "in_progress")
    buffer.add_message("System", "Analysis date locked")
    buffer.add_tool_call("get_stock_data", {"symbol": "BRACBANK"})
    monkeypatch.setattr(main, "message_buffer", buffer)

    layout = main.create_layout()
    main.update_display(layout, spinner_text="Analyzing BRACBANK…")
    preview.print(layout)
    output = preview.export_text()

    assert "INTELLIGENCE DESK" in output
    assert "BRACBANK" in output
    assert "RUN MAP" in output
    assert "ACTIVITY FEED" in output
    assert "LIVE BRIEF" in output
    assert "Market Analyst" in output
    assert "Messages & Tools" not in output
    assert "Layout(name=" not in output


def test_live_renderer_uses_the_same_themed_console(monkeypatch):
    preview = _preview_console(width=120, height=40)
    monkeypatch.setattr(main, "console", preview)

    live = main.create_live_display(main.create_layout())

    assert live.console is preview
    assert live.refresh_per_second == 8


def test_expected_market_data_failure_is_presented_without_a_traceback(monkeypatch):
    def fail_cleanly(*_args, **_kwargs):
        raise VendorError("DSE gateway timed out")

    monkeypatch.setattr(main, "run_analysis", fail_cleanly)
    result = CliRunner().invoke(main.app, [])

    assert result.exit_code == 1
    assert "MARKET DATA UNAVAILABLE" in result.output
    assert "DSE gateway timed out" in result.output
    assert "Traceback" not in result.output
