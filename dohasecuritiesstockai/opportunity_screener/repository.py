"""Token-safe filesystem persistence for opportunity scans."""

from __future__ import annotations

from pathlib import Path

from .schema import OpportunityScanResult


class OpportunityRepository:
    def __init__(self, results_dir: str | Path) -> None:
        self.directory = Path(results_dir) / "api" / "opportunities"
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        result: OpportunityScanResult,
        output_path: str | Path | None = None,
    ) -> Path:
        path = Path(output_path) if output_path else self.directory / f"{result.scan_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return path

    def get(self, scan_id: str) -> OpportunityScanResult | None:
        if not scan_id or Path(scan_id).name != scan_id:
            return None
        path = self.directory / f"{scan_id}.json"
        if not path.is_file():
            return None
        return OpportunityScanResult.model_validate_json(path.read_text(encoding="utf-8"))

    def latest(self) -> OpportunityScanResult | None:
        paths = sorted(
            self.directory.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return self.get(paths[0].stem) if paths else None
