"""Token-safe filesystem persistence for API jobs and presentation reports."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from dohasecuritiesstockai.dataflows.utils import safe_ticker_component

from .models import AnalysisJob, StockAnalysis


class AnalysisRepository:
    def __init__(self, results_dir: str | Path) -> None:
        self.results_dir = Path(results_dir)
        self.api_dir = self.results_dir / "api"
        self.api_dir.mkdir(parents=True, exist_ok=True)

    def _analysis_path(self, symbol: str, analysis_date: date | str) -> Path:
        safe_symbol = safe_ticker_component(symbol.upper())
        return self.api_dir / "analyses" / safe_symbol / f"{analysis_date}.json"

    def save_analysis(self, analysis: StockAnalysis) -> Path:
        path = self._analysis_path(analysis.symbol, analysis.analysis_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
        return path

    def get_analysis(self, symbol: str, analysis_date: date | str) -> StockAnalysis | None:
        path = self._analysis_path(symbol, analysis_date)
        if not path.exists():
            return None
        return StockAnalysis.model_validate_json(path.read_text(encoding="utf-8"))

    def get_latest_analysis(self, symbol: str) -> StockAnalysis | None:
        safe_symbol = safe_ticker_component(symbol.upper())
        directory = self.api_dir / "analyses" / safe_symbol
        if not directory.exists():
            return None
        paths = sorted(directory.glob("*.json"), reverse=True)
        return (
            StockAnalysis.model_validate_json(paths[0].read_text(encoding="utf-8"))
            if paths
            else None
        )

    def save_job(self, job: AnalysisJob) -> None:
        path = self.api_dir / "jobs" / f"{job.job_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(job.model_dump_json(indent=2), encoding="utf-8")

    def get_job(self, job_id: str) -> AnalysisJob | None:
        path = self.api_dir / "jobs" / f"{job_id}.json"
        if not path.exists():
            return None
        return AnalysisJob.model_validate_json(path.read_text(encoding="utf-8"))

    def find_state_log(self, symbol: str, analysis_date: date | None = None) -> Path | None:
        safe_symbol = safe_ticker_component(symbol.upper())
        directory = self.results_dir / safe_symbol / "TradingAgentsStrategy_logs"
        if not directory.exists():
            return None
        if analysis_date:
            exact = directory / f"full_states_log_{analysis_date}.json"
            if exact.exists():
                return exact
        candidates = sorted(directory.glob("full_states_log_*.json"), reverse=True)
        return candidates[0] if candidates else None

    @staticmethod
    def load_state(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"State log {path} is not a JSON object")
        return payload
