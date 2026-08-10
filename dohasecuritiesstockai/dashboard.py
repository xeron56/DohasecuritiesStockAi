"""Prepare a completed CLI run and launch its local Angular dashboard."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import webbrowser
from datetime import date
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import uvicorn

from dohasecuritiesstockai.api.analysis import StockAnalysisBuilder
from dohasecuritiesstockai.api.models import StockAnalysis
from dohasecuritiesstockai.api.repository import AnalysisRepository


class DashboardLaunchError(RuntimeError):
    """Raised when the local dashboard cannot be prepared or started."""


def prepare_dashboard_analysis(
    symbol: str,
    analysis_date: date | str,
    agent_state: dict[str, Any],
    results_dir: str | Path,
) -> tuple[StockAnalysis, Path]:
    """Build and save the presentation model from the exact completed CLI state."""

    parsed_date = (
        analysis_date
        if isinstance(analysis_date, date)
        else date.fromisoformat(str(analysis_date))
    )
    analysis = StockAnalysisBuilder().build(symbol, parsed_date, agent_state=agent_state)
    path = AnalysisRepository(results_dir).save_analysis(analysis)
    return analysis, path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_ui_dist(ui_root: Path) -> Path | None:
    candidates = (
        ui_root / "dist" / "analysis-ui" / "browser",
        ui_root / "dist" / "analysis-ui",
    )
    return next((path for path in candidates if (path / "index.html").is_file()), None)


def _latest_frontend_source_mtime(ui_root: Path) -> float:
    inputs = [
        ui_root / "angular.json",
        ui_root / "package.json",
        ui_root / "package-lock.json",
        ui_root / "tsconfig.json",
        ui_root / "tsconfig.app.json",
    ]
    inputs.extend(path for path in (ui_root / "src").rglob("*") if path.is_file())
    return max((path.stat().st_mtime for path in inputs if path.exists()), default=0)


def ensure_ui_build(project_root: Path | None = None) -> Path:
    """Install locked Angular dependencies when needed and build stale sources."""

    root = project_root or _project_root()
    ui_root = root / "analysis-ui"
    if not (ui_root / "package.json").is_file():
        raise DashboardLaunchError(f"Angular project not found at {ui_root}")

    npm = shutil.which("npm")
    if npm is None:
        raise DashboardLaunchError(
            "npm is required to build the Angular dashboard. Install Node.js/npm and retry."
        )

    if not (ui_root / "node_modules" / ".bin" / "ng").exists():
        try:
            subprocess.run(
                [npm, "ci", "--no-audit", "--no-fund"],
                cwd=ui_root,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise DashboardLaunchError("Angular dependency installation failed.") from exc

    dist = _find_ui_dist(ui_root)
    if dist is None or (dist / "index.html").stat().st_mtime < _latest_frontend_source_mtime(
        ui_root
    ):
        try:
            subprocess.run([npm, "run", "build:production"], cwd=ui_root, check=True)
        except subprocess.CalledProcessError as exc:
            raise DashboardLaunchError("Angular production build failed.") from exc
        dist = _find_ui_dist(ui_root)

    if dist is None:
        raise DashboardLaunchError("Angular build completed without an index.html output.")
    return dist


def dashboard_url(host: str, port: int, symbol: str, analysis_date: date | str) -> str:
    """Return the browser URL for one exact CLI analysis."""

    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    query = urlencode({"symbol": symbol, "date": str(analysis_date)})
    return f"http://{browser_host}:{port}/?{query}"


def prediction_dashboard_url(host: str, port: int, run_id: str) -> str:
    """Return the standalone Angular URL for one prediction run."""

    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    query = urlencode({"view": "timesfm", "run": run_id})
    return f"http://{browser_host}:{port}/?{query}"


def _url_is_ready(url: str, *, expect_html: bool = False) -> bool:
    try:
        with urlopen(url, timeout=0.5) as response:  # noqa: S310 - local user URL only
            content_type = response.headers.get("Content-Type", "")
            return response.status == 200 and (not expect_html or "text/html" in content_type)
    except (OSError, URLError):
        return False


def _open_when_ready(health_url: str, url: str) -> None:
    for _ in range(100):
        if _url_is_ready(health_url):
            webbrowser.open(url)
            return
        time.sleep(0.2)


def launch_dashboard(
    symbol: str,
    analysis_date: date | str,
    *,
    host: str | None = None,
    port: int | None = None,
) -> str:
    """Build, serve, and open the Angular UI; block until the server stops."""

    resolved_host = host or os.environ.get("TRADINGAGENTS_API_HOST", "127.0.0.1")
    resolved_port = port or int(os.environ.get("TRADINGAGENTS_API_PORT", "8000"))
    url = dashboard_url(resolved_host, resolved_port, symbol, analysis_date)
    browser_root = url.split("?", 1)[0]
    health_url = f"{browser_root.rstrip('/')}/api/v1/health"

    if _url_is_ready(health_url):
        if not _url_is_ready(browser_root, expect_html=True):
            raise DashboardLaunchError(
                f"Port {resolved_port} already has the API without the Angular UI. "
                "Stop that server and run the CLI again."
            )
        webbrowser.open(url)
        return url

    dist = ensure_ui_build()
    os.environ["TRADINGAGENTS_UI_DIST_DIR"] = str(dist)
    opener = threading.Thread(
        target=_open_when_ready,
        args=(health_url, url),
        name="dashboard-browser-opener",
        daemon=True,
    )
    opener.start()
    uvicorn.run(
        "dohasecuritiesstockai.api.app:app",
        host=resolved_host,
        port=resolved_port,
        reload=False,
    )
    return url


def launch_prediction_dashboard(
    run_id: str,
    *,
    host: str | None = None,
    port: int | None = None,
) -> str:
    """Build, serve, and open the standalone forecast result screen."""

    resolved_host = host or os.environ.get("TRADINGAGENTS_API_HOST", "127.0.0.1")
    resolved_port = port or int(os.environ.get("TRADINGAGENTS_API_PORT", "8000"))
    url = prediction_dashboard_url(resolved_host, resolved_port, run_id)
    browser_root = url.split("?", 1)[0]
    health_url = f"{browser_root.rstrip('/')}/api/v1/health"

    if _url_is_ready(health_url):
        if not _url_is_ready(browser_root, expect_html=True):
            raise DashboardLaunchError(
                f"Port {resolved_port} already has the API without the Angular UI. "
                "Stop that server and run the prediction command again."
            )
        webbrowser.open(url)
        return url

    dist = ensure_ui_build()
    os.environ["TRADINGAGENTS_UI_DIST_DIR"] = str(dist)
    opener = threading.Thread(
        target=_open_when_ready,
        args=(health_url, url),
        name="timesfm-dashboard-browser-opener",
        daemon=True,
    )
    opener.start()
    uvicorn.run(
        "dohasecuritiesstockai.api.app:app",
        host=resolved_host,
        port=resolved_port,
        reload=False,
    )
    return url
