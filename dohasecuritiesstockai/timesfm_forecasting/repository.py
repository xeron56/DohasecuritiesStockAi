"""Filesystem persistence for chart-ready prediction runs."""

from __future__ import annotations

from pathlib import Path

from .schema import TimesFMPredictionResult


class PredictionRepository:
    def __init__(self, results_dir: str | Path) -> None:
        self.directory = Path(results_dir) / "api" / "predictions"
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        result: TimesFMPredictionResult,
        output_path: str | Path | None = None,
    ) -> Path:
        path = Path(output_path) if output_path else self.directory / f"{result.run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return path

    def get(self, run_id: str) -> TimesFMPredictionResult | None:
        if not run_id or Path(run_id).name != run_id:
            return None
        path = self.directory / f"{run_id}.json"
        if not path.is_file():
            return None
        return TimesFMPredictionResult.model_validate_json(path.read_text(encoding="utf-8"))

    def latest(
        self,
        symbol: str | None = None,
        resolution: str | None = None,
    ) -> TimesFMPredictionResult | None:
        candidates = sorted(
            self.directory.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True
        )
        for path in candidates:
            result = TimesFMPredictionResult.model_validate_json(path.read_text(encoding="utf-8"))
            if symbol and result.symbol.upper() != symbol.upper():
                continue
            if resolution and result.data.requested_resolution != resolution:
                continue
            return result
        return None
