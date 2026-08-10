"""FastAPI application exposing DSE stock research to the Angular client."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Load the project-local secret file before importing DEFAULT_CONFIG or the
# background service, because both resolve TRADINGAGENTS_* overrides at import.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=False)

from dohasecuritiesstockai.dataflows.dse import normalize_dse_symbol  # noqa: E402
from dohasecuritiesstockai.default_config import DEFAULT_CONFIG  # noqa: E402
from dohasecuritiesstockai.timesfm_forecasting.repository import PredictionRepository  # noqa: E402
from dohasecuritiesstockai.timesfm_forecasting.schema import TimesFMPredictionResult  # noqa: E402

from .analysis import list_dse_stocks  # noqa: E402
from .models import (  # noqa: E402
    AnalysisJob,
    AnalysisRequest,
    StockAnalysis,
    StockOption,
)
from .repository import AnalysisRepository  # noqa: E402
from .service import AnalysisService  # noqa: E402


def _origins() -> list[str]:
    raw = os.environ.get(
        "TRADINGAGENTS_API_CORS_ORIGINS",
        "http://localhost:4200,http://127.0.0.1:4200",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _ui_dist_directory() -> Path | None:
    """Return the built Angular browser directory when it is available."""

    configured = os.environ.get("TRADINGAGENTS_UI_DIST_DIR")
    candidates = (
        [Path(configured).expanduser()] if configured else []
    ) + [
        _PROJECT_ROOT / "analysis-ui" / "dist" / "analysis-ui" / "browser",
        _PROJECT_ROOT / "analysis-ui" / "dist" / "analysis-ui",
    ]
    return next((path for path in candidates if (path / "index.html").is_file()), None)


def create_app() -> FastAPI:
    app = FastAPI(
        title="TradingAgents DSE Analysis API",
        version="1.0.0",
        description=(
            "Read-only DSE evidence, background multi-agent analysis jobs, and a "
            "stable presentation contract for the Angular research screen."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
    )

    repository = AnalysisRepository(DEFAULT_CONFIG["results_dir"])
    prediction_repository = PredictionRepository(DEFAULT_CONFIG["results_dir"])
    service = AnalysisService(
        repository,
        max_workers=int(os.environ.get("TRADINGAGENTS_API_WORKERS", "1")),
    )
    router = APIRouter(prefix="/api/v1")

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "market": "bangladesh_dse", "schema_version": "1.0"}

    @router.get("/stocks", response_model=list[StockOption])
    def stocks(q: str = Query(default="", max_length=50)) -> list[StockOption]:
        try:
            options = list_dse_stocks()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        needle = q.strip().lower()
        if not needle:
            return options
        return [
            option
            for option in options
            if needle in option.symbol.lower()
            or needle in option.name.lower()
            or needle in option.sector.lower()
        ]

    @router.get("/predictions/latest", response_model=TimesFMPredictionResult)
    def latest_prediction(
        symbol: str | None = Query(default=None, max_length=32),
        resolution: str | None = Query(default=None, max_length=16),
    ) -> TimesFMPredictionResult:
        canonical = None
        if symbol:
            try:
                canonical = normalize_dse_symbol(symbol)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        result = prediction_repository.latest(canonical, resolution)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail="No TimesFM prediction run matches this request.",
            )
        return result

    @router.get("/predictions/{run_id}", response_model=TimesFMPredictionResult)
    def prediction(run_id: str) -> TimesFMPredictionResult:
        result = prediction_repository.get(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Prediction run not found.")
        return result

    @router.post(
        "/analyses",
        response_model=AnalysisJob,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_analysis(request: AnalysisRequest) -> AnalysisJob:
        try:
            symbol = normalize_dse_symbol(request.symbol)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return service.submit(
            symbol,
            request.analysis_date or date.today(),
            force=request.force,
        )

    @router.get("/analyses/jobs/{job_id}", response_model=AnalysisJob)
    def get_job(job_id: str) -> AnalysisJob:
        job = repository.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Analysis job not found.")
        return job

    @router.get("/analyses/{symbol}/latest", response_model=StockAnalysis)
    def latest_analysis(symbol: str) -> StockAnalysis:
        try:
            canonical = normalize_dse_symbol(symbol)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        analysis = repository.get_latest_analysis(canonical)
        if analysis is None:
            try:
                analysis = service.build_from_latest_state(canonical)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
        if analysis is None:
            raise HTTPException(
                status_code=404,
                detail="No completed analysis exists yet. Submit an analysis job first.",
            )
        return analysis

    @router.get("/analyses/{symbol}/{analysis_date}", response_model=StockAnalysis)
    def dated_analysis(symbol: str, analysis_date: date) -> StockAnalysis:
        try:
            canonical = normalize_dse_symbol(symbol)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        analysis = repository.get_analysis(canonical, analysis_date)
        if analysis is None:
            raise HTTPException(status_code=404, detail="Completed analysis not found.")
        return analysis

    app.include_router(router)
    ui_dist = _ui_dist_directory()
    if ui_dist is not None:
        # Register this after every API route so same-origin /api/v1 and /docs
        # keep precedence while the Angular production bundle owns the root.
        app.mount("/", StaticFiles(directory=ui_dist, html=True), name="analysis-ui")
    return app


app = create_app()
