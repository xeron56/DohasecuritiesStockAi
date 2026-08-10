"""Background orchestration for REST-triggered TradingAgents runs."""

from __future__ import annotations

import copy
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone

from dohasecuritiesstockai.default_config import DEFAULT_CONFIG
from dohasecuritiesstockai.graph.trading_graph import TradingAgentsGraph

from .analysis import StockAnalysisBuilder
from .models import AnalysisJob, StockAnalysis
from .repository import AnalysisRepository


class AnalysisService:
    """Runs at most one graph per worker and persists every observable state."""

    def __init__(
        self,
        repository: AnalysisRepository,
        *,
        max_workers: int = 1,
    ) -> None:
        self.repository = repository
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers), thread_name_prefix="stock-analysis"
        )
        self._lock = threading.Lock()

    def submit(self, symbol: str, analysis_date: date, *, force: bool = False) -> AnalysisJob:
        now = datetime.now(timezone.utc)
        job = AnalysisJob(
            job_id=uuid.uuid4().hex,
            symbol=symbol,
            analysis_date=analysis_date,
            status="queued",
            created_at=now,
            updated_at=now,
            message="Analysis queued.",
        )
        self.repository.save_job(job)
        self.executor.submit(self._run, job.job_id, force)
        return job

    def _update(self, job: AnalysisJob, **updates: object) -> AnalysisJob:
        updated = job.model_copy(
            update={**updates, "updated_at": datetime.now(timezone.utc)}
        )
        with self._lock:
            self.repository.save_job(updated)
        return updated

    def _run(self, job_id: str, force: bool) -> None:
        job = self.repository.get_job(job_id)
        if job is None:
            return
        try:
            job = self._update(job, status="running", message="Collecting DSE evidence.")
            if not force:
                existing = self.repository.get_analysis(job.symbol, job.analysis_date)
                if existing:
                    self._update(
                        job,
                        status="completed",
                        analysis_url=f"/api/v1/analyses/{job.symbol}/{job.analysis_date}",
                        message="Existing analysis loaded.",
                    )
                    return

            state_path = None if force else self.repository.find_state_log(
                job.symbol, job.analysis_date
            )
            if state_path is None:
                job = self._update(
                    job,
                    message="Running the DSE multi-agent research workflow.",
                )
                config = copy.deepcopy(DEFAULT_CONFIG)
                graph = TradingAgentsGraph(
                    selected_analysts=config.get("default_analysts"),
                    debug=False,
                    config=config,
                )
                final_state, _ = graph.propagate(job.symbol, job.analysis_date.isoformat())
                state = final_state
            else:
                state = self.repository.load_state(state_path)

            job = self._update(job, message="Calculating the presentation score and valuation.")
            analysis = StockAnalysisBuilder().build(
                job.symbol, job.analysis_date, agent_state=state
            )
            self.repository.save_analysis(analysis)
            self._update(
                job,
                status="completed",
                analysis_url=f"/api/v1/analyses/{job.symbol}/{job.analysis_date}",
                message="Analysis completed.",
            )
        except Exception as exc:
            # Exception text can identify the failed stage but the DSE client is
            # deliberately designed never to include credentials or tokens.
            self._update(job, status="failed", message=str(exc)[:500])

    def build_from_latest_state(self, symbol: str) -> StockAnalysis | None:
        state_path = self.repository.find_state_log(symbol)
        if state_path is None:
            return None
        state = self.repository.load_state(state_path)
        raw_date = state.get("trade_date")
        try:
            analysis_date = date.fromisoformat(str(raw_date))
        except ValueError:
            analysis_date = date.today()
        existing = self.repository.get_analysis(symbol, analysis_date)
        if existing:
            return existing
        analysis = StockAnalysisBuilder().build(symbol, analysis_date, state)
        self.repository.save_analysis(analysis)
        return analysis
