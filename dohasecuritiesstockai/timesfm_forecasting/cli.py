"""Independent `dohasecuritiesstockai-predict` command (no multi-agent graph startup)."""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dohasecuritiesstockai.default_config import DEFAULT_CONFIG

from .pipeline import run_stock_prediction
from .repository import PredictionRepository

console = Console()

_LOOKBACK_RE = re.compile(r"^(?P<count>[1-9]\d*)(?P<unit>d|w|mo|y)$")


def _lookback_start(end_date: date, lookback: str) -> date | None:
    value = str(lookback or "").strip().lower()
    if value in {"all", "max"}:
        return None
    match = _LOOKBACK_RE.fullmatch(value)
    if match is None:
        raise ValueError("lookback must look like 30d, 26w, 12mo, 1y, or max.")

    count = int(match.group("count"))
    unit = match.group("unit")
    if unit == "d":
        return end_date - timedelta(days=count)
    if unit == "w":
        return end_date - timedelta(weeks=count)
    if unit == "y":
        try:
            return end_date.replace(year=end_date.year - count)
        except ValueError:
            return end_date.replace(year=end_date.year - count, day=28)

    month_index = end_date.year * 12 + end_date.month - 1 - count
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    day = min(end_date.day, monthrange(year, month)[1])
    return end_date.replace(year=year, month=month, day=day)


def predict(
    symbol: Annotated[str, typer.Argument(help="DSE symbol, for example GP or BRACBANK.")],
    resolution: Annotated[
        str,
        typer.Option(
            "--resolution",
            "-r",
            help="Bar interval: 1min, 15min, 30min, 1h, 1d, 1w, 1mo, or 1y.",
        ),
    ] = "1d",
    split_ratio: Annotated[
        float,
        typer.Option("--split", min=0.1, max=0.9, help="Fraction used as model context."),
    ] = 0.5,
    future_steps: Annotated[
        int,
        typer.Option("--future-steps", min=1, help="Number of future bars to forecast."),
    ] = 12,
    start: Annotated[
        str | None,
        typer.Option(help="Optional first historical date (YYYY-MM-DD)."),
    ] = None,
    end: Annotated[
        str | None,
        typer.Option(help="Optional final historical date (YYYY-MM-DD)."),
    ] = None,
    lookback: Annotated[
        str | None,
        typer.Option(
            "--lookback",
            help="Historical range ending at --end or today: 30d, 26w, 12mo, 1y, or max.",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(min=1, help="Maximum bars requested from the chart API."),
    ] = 100_000,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Also write the chart-ready JSON to this path."),
    ] = None,
    open_ui: Annotated[
        bool,
        typer.Option("--open-ui/--no-open-ui", help="Build and serve the Angular result UI."),
    ] = True,
    host: Annotated[str, typer.Option(help="Local UI/API bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535, help="Local UI/API port.")] = 8000,
) -> None:
    """Backtest TimesFM on half of a stock history and forecast the next bars."""

    console.print(
        Panel.fit(
            f"[bold]TimesFM stock forecast[/bold]\n"
            f"{symbol.upper()} · {resolution} · "
            f"{split_ratio:.0%}/{1 - split_ratio:.0%} context/holdout"
            f"{f' · {lookback} history' if lookback else ''}",
            border_style="cyan",
        )
    )
    try:
        start_date = date.fromisoformat(start) if start else None
        end_date = date.fromisoformat(end) if end else None
        if start_date is not None and lookback is not None:
            raise ValueError("Use either --start or --lookback, not both.")
        if lookback is not None:
            end_date = end_date or date.today()
            start_date = _lookback_start(end_date, lookback)
        with console.status("Fetching DSE candles and running TimesFM 2.0 500M on the GPU…"):
            result = run_stock_prediction(
                symbol,
                resolution,
                split_ratio=split_ratio,
                future_steps=future_steps,
                start=start_date,
                end=end_date,
                limit=limit,
            )
    except Exception as exc:
        console.print(f"[bold red]Prediction failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    repository = PredictionRepository(DEFAULT_CONFIG["results_dir"])
    saved_path = repository.save(result)
    if output is not None:
        repository.save(result, output)

    metrics = result.metrics
    table = Table(title="Held-out accuracy", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Accuracy score (100 - sMAPE)", f"{metrics.accuracy_score:.2f}%")
    table.add_row("MAE", f"{metrics.mae:.4f} BDT")
    table.add_row("RMSE", f"{metrics.rmse:.4f} BDT")
    table.add_row("Directional accuracy", f"{metrics.directional_accuracy_percent or 0:.2f}%")
    table.add_row("80% interval coverage", f"{metrics.interval_80_coverage_percent:.2f}%")
    table.add_row("Skill vs last-price baseline", f"{metrics.skill_vs_naive_percent or 0:.2f}%")
    console.print(table)
    console.print(
        f"[green]Saved:[/green] {saved_path}\n"
        f"[dim]{result.data.context_points} context bars · "
        f"{result.data.holdout_points} held-out bars · "
        f"{len(result.future)} future bars · {result.model.device}[/dim]"
    )
    if output is not None:
        console.print(f"[green]Additional output:[/green] {output}")

    if open_ui:
        from dohasecuritiesstockai.dashboard import launch_prediction_dashboard

        console.print("[cyan]Starting the standalone Angular prediction UI…[/cyan]")
        launch_prediction_dashboard(result.run_id, host=host, port=port)


def main() -> None:
    typer.run(predict)


if __name__ == "__main__":
    main()
