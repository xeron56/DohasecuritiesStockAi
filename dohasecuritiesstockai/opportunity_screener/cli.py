"""Independent CLI for current long-horizon DSE opportunity research."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=False)

from dohasecuritiesstockai.dashboard import (  # noqa: E402
    launch_opportunity_dashboard,
    opportunity_dashboard_url,
)
from dohasecuritiesstockai.default_config import DEFAULT_CONFIG  # noqa: E402

from .pipeline import run_opportunity_scan  # noqa: E402
from .repository import OpportunityRepository  # noqa: E402

console = Console()


def _format(value: float | None, suffix: str = "") -> str:
    return "—" if value is None else f"{value:,.1f}{suffix}"


def opportunities(
    horizon_years: Annotated[
        int,
        typer.Option(
            "--horizon",
            min=2,
            max=10,
            help="Research horizon in years; this is not a return guarantee.",
        ),
    ] = 5,
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=20, help="Candidates shown in the result."),
    ] = 8,
    finalists: Annotated[
        int,
        typer.Option(
            "--finalists",
            min=1,
            max=50,
            help="Bulk-screen finalists that receive detailed evidence collection.",
        ),
    ] = 16,
    use_ai: Annotated[
        bool,
        typer.Option(
            "--ai/--no-ai",
            help="Use one structured AI call to critique the deterministic shortlist.",
        ),
    ] = True,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Also write result JSON to this path."),
    ] = None,
    open_ui: Annotated[
        bool,
        typer.Option("--open-ui/--no-open-ui", help="Build and open the dedicated UI."),
    ] = True,
    host: Annotated[str, typer.Option(help="Local UI/API bind host.")] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(min=1, max=65535, help="Local UI/API port."),
    ] = 8000,
) -> None:
    """Find less-followed DSE companies worth deeper multi-year research."""

    if finalists < limit:
        console.print("[bold red]--finalists must be at least --limit.[/bold red]")
        raise typer.Exit(code=2)
    console.print(
        Panel.fit(
            "[bold]DSE long-term opportunity research[/bold]\n"
            f"{horizon_years}-year horizon · {limit} results · "
            f"{'grounded AI review' if use_ai else 'deterministic screen only'}\n"
            "[yellow]Research shortlist—not a promise of profit or a buy instruction.[/yellow]",
            border_style="cyan",
        )
    )
    try:
        with console.status("Loading DSE evidence…") as status:
            result = run_opportunity_scan(
                horizon_years=horizon_years,
                limit=limit,
                finalist_count=finalists,
                use_ai=use_ai,
                config=DEFAULT_CONFIG,
                progress=status.update,
            )
    except Exception as exc:
        console.print(f"[bold red]Opportunity scan failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    repository = OpportunityRepository(DEFAULT_CONFIG["results_dir"])
    saved_path = repository.save(result)
    if output is not None:
        repository.save(result, output)

    table = Table(title="Long-term research shortlist", header_style="bold cyan")
    table.add_column("#", justify="right")
    table.add_column("Symbol", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Priority")
    table.add_column("Price", justify="right")
    table.add_column("P/E", justify="right")
    table.add_column("EPS growth", justify="right")
    table.add_column("AI review")
    for candidate in result.candidates:
        table.add_row(
            str(candidate.rank),
            candidate.symbol,
            f"{candidate.score:.1f}",
            candidate.research_label,
            _format(candidate.metrics.current_price, " BDT"),
            _format(candidate.metrics.pe_ratio, "×"),
            _format(candidate.metrics.eps_growth_percent, "%"),
            candidate.ai_review.verdict if candidate.ai_review else "Not run",
        )
    console.print(table)
    console.print(
        f"[green]Saved:[/green] {saved_path}\n"
        f"[dim]{result.methodology.initial_universe} screened · "
        f"{result.methodology.eligible_universe} eligible · "
        f"{result.methodology.detailed_finalists} detailed finalists[/dim]\n"
        f"[yellow]{result.disclaimer}[/yellow]"
    )
    if output is not None:
        console.print(f"[green]Additional output:[/green] {output}")
    if not open_ui:
        return

    url = opportunity_dashboard_url(host, port, result.scan_id)
    console.print(
        f"[green]Opening:[/green] {url}\n"
        "[dim]Keep this terminal open; press Ctrl+C to stop the UI.[/dim]"
    )
    try:
        launch_opportunity_dashboard(result.scan_id, host=host, port=port)
    except Exception as exc:
        console.print(f"[bold red]UI launch failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from None


def main() -> None:
    typer.run(opportunities)


if __name__ == "__main__":
    main()
