"""Lightweight dashboard command that never starts the multi-agent graph."""

from __future__ import annotations

from datetime import date
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from dohasecuritiesstockai.dashboard import (
    dashboard_url,
    launch_dashboard,
    prepare_dashboard_analysis,
)
from dohasecuritiesstockai.default_config import DEFAULT_CONFIG

console = Console()


def show_dashboard(
    symbol: Annotated[
        str,
        typer.Argument(help="DSE symbol, for example GP or BRACBANK."),
    ],
    analysis_date: Annotated[
        str | None,
        typer.Option("--date", "-d", help="Dashboard data date (YYYY-MM-DD)."),
    ] = None,
    use_ai: Annotated[
        bool,
        typer.Option(
            "--ai/--no-ai",
            help="Use the configured AI for score, valuation weighting, and trader report.",
        ),
    ] = True,
    open_ui: Annotated[
        bool,
        typer.Option(
            "--open-ui/--no-open-ui",
            help="Serve and open the dashboard after fetching its data.",
        ),
    ] = True,
    host: Annotated[
        str,
        typer.Option(help="Local dashboard/API bind host."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(min=1, max=65535, help="Local dashboard/API port."),
    ] = 8000,
) -> None:
    """Fetch only the DSE score-page data, then open the local dashboard."""

    try:
        requested_date = date.fromisoformat(analysis_date) if analysis_date else date.today()
        if requested_date > date.today():
            raise ValueError("--date cannot be in the future.")
    except ValueError as exc:
        console.print(f"[bold red]Invalid dashboard date:[/bold red] {exc}")
        raise typer.Exit(code=2) from None

    console.print(
        Panel.fit(
            f"[bold]DSE score dashboard[/bold]\n"
            f"{symbol.strip().upper()} · {requested_date.isoformat()} · "
            f"{'grounded AI research' if use_ai else 'calculated data only'}",
            border_style="cyan",
        )
    )
    try:
        status = (
            "Collecting yearly DSE evidence and running the configured AI analyst…"
            if use_ai
            else "Requesting the dashboard fields from the DSE data APIs…"
        )
        with console.status(status):
            analysis, saved_path = prepare_dashboard_analysis(
                symbol,
                requested_date,
                agent_state=None,
                results_dir=DEFAULT_CONFIG["results_dir"],
                use_ai=use_ai,
                config=DEFAULT_CONFIG,
            )
    except Exception as exc:
        console.print(f"[bold red]Dashboard data request failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    console.print(
        f"[green]✓ Dashboard data saved:[/green] {saved_path}\n"
        f"[green]✓ Fundamental score:[/green] "
        f"{analysis.fundamental_score}/100 ({analysis.score_label.en})"
    )
    if analysis.ai_research is not None:
        console.print(
            f"[green]✓ AI analyst:[/green] {analysis.ai_research.provider} / "
            f"{analysis.ai_research.model}\n"
            f"[green]✓ Trader view:[/green] "
            f"{analysis.ai_research.trader_report.rating}"
        )
    if not open_ui:
        return

    url = dashboard_url(host, port, analysis.symbol, analysis.analysis_date)
    console.print(
        f"[green]✓ Opening:[/green] {url}\n"
        "[dim]Keep this terminal open; press Ctrl+C to stop the UI.[/dim]"
    )
    try:
        launch_dashboard(
            analysis.symbol,
            analysis.analysis_date,
            host=host,
            port=port,
        )
    except Exception as exc:
        console.print(f"[bold red]Dashboard launch failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from None


def main() -> None:
    typer.run(show_dashboard)


if __name__ == "__main__":
    main()
