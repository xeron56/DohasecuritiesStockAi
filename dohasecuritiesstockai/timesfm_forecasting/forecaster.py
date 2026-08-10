"""TimesFM 2.0 500M PyTorch backend with CUDA and long-horizon chunking."""

from __future__ import annotations

import contextlib
import importlib
import io
import math
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

TIMESFM_CHECKPOINT = "google/timesfm-2.0-500m-pytorch"
TIMESFM_PARAMETERS = 500_000_000
MAX_FORECAST_CHUNK = 1024
MODEL_CONTEXT_LIMIT = 2_048
INPUT_PATCH = 32
OUTPUT_PATCH = 128
_ARCHIVED_SOURCE_ENV = "TRADINGAGENTS_TIMESFM_V2_SOURCE"


class TimesFMDependencyError(RuntimeError):
    """Raised when the archived TimesFM 2.0 inference module is unavailable."""


def _round_up(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def _archived_source_path() -> Path:
    configured = os.environ.get(_ARCHIVED_SOURCE_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "timesfm" / "v1" / "src"


def _load_archived_timesfm() -> ModuleType:
    """Import Google's archived v1/v2 API from the separate cloned module."""

    source = _archived_source_path()
    package = source / "timesfm" / "__init__.py"
    if not package.is_file():
        raise TimesFMDependencyError(
            "The TimesFM 2.0 compatibility source was not found. Clone "
            "https://github.com/google-research/timesfm into the project's "
            "`timesfm` directory, or set TRADINGAGENTS_TIMESFM_V2_SOURCE to "
            "the clone's `v1/src` directory."
        )

    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    existing = sys.modules.get("timesfm")
    if existing is not None and not hasattr(existing, "TimesFmHparams"):
        for name in [
            name for name in sys.modules if name == "timesfm" or name.startswith("timesfm.")
        ]:
            del sys.modules[name]

    try:
        # The archived package prints a migration notice at import time. The
        # CLI already reports the selected checkpoint, so keep startup clean.
        with contextlib.redirect_stdout(io.StringIO()):
            module = importlib.import_module("timesfm")
    except ImportError as exc:
        raise TimesFMDependencyError(
            "TimesFM 2.0 needs the archived runtime helpers. Install the "
            "prediction extra with `uv pip install -e '.[timesfm]'`."
        ) from exc

    required = ("TimesFm", "TimesFmHparams", "TimesFmCheckpoint")
    if any(not hasattr(module, name) for name in required):
        raise TimesFMDependencyError(
            "The loaded TimesFM module does not expose the v2 inference API."
        )
    return module


class TimesFMBackend:
    """One TimesFM 2.0 500M model instance configured for one prediction run."""

    name = "TimesFM"
    version = "2.0"
    checkpoint = TIMESFM_CHECKPOINT
    parameters = TIMESFM_PARAMETERS

    def __init__(self, checkpoint: str = TIMESFM_CHECKPOINT, frequency: int = 0) -> None:
        if frequency not in {0, 1, 2}:
            raise ValueError("TimesFM 2.0 frequency must be 0, 1, or 2.")
        self.checkpoint = checkpoint
        self.frequency = frequency
        self.model: Any | None = None
        self.max_context = 0
        self.max_horizon = 0
        self.device = "uninitialized"
        self.gpu_name: str | None = None

    def prepare(self, context_points: int, requested_horizon: int) -> None:
        try:
            import torch
        except ImportError as exc:
            raise TimesFMDependencyError(
                "PyTorch is not installed. Install the prediction extra with "
                "`uv pip install -e '.[timesfm]'`."
            ) from exc

        timesfm = _load_archived_timesfm()
        use_cuda = torch.cuda.is_available()
        torch.set_float32_matmul_precision("high")

        desired_context = max(INPUT_PATCH, min(context_points, MODEL_CONTEXT_LIMIT))
        self.max_context = min(MODEL_CONTEXT_LIMIT, _round_up(desired_context, INPUT_PATCH))
        requested_chunk = min(MAX_FORECAST_CHUNK, max(1, requested_horizon))
        self.max_horizon = max(OUTPUT_PATCH, _round_up(requested_chunk, OUTPUT_PATCH))

        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu" if use_cuda else "cpu",
                per_core_batch_size=1,
                horizon_len=self.max_horizon,
                input_patch_len=INPUT_PATCH,
                output_patch_len=OUTPUT_PATCH,
                num_layers=50,
                model_dims=1280,
                context_len=self.max_context,
                use_positional_embedding=False,
                point_forecast_mode="median",
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self.checkpoint,
            ),
        )
        if use_cuda:
            self.device = "cuda:0"
            self.gpu_name = torch.cuda.get_device_name(0)
        else:
            self.device = "cpu"
            self.gpu_name = None

    def forecast_open_loop(
        self,
        context: np.ndarray,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Forecast without leaking held-out actual values into later chunks."""

        if self.model is None:
            raise RuntimeError("TimesFMBackend.prepare() must be called before forecasting.")
        if horizon < 1:
            return np.empty(0, dtype=np.float32), np.empty((0, 10)), 0

        rolling_context = np.asarray(context, dtype=np.float32).copy()
        point_chunks: list[np.ndarray] = []
        quantile_chunks: list[np.ndarray] = []
        remaining = horizon
        chunks = 0
        while remaining:
            step = min(remaining, self.max_horizon)
            points, quantiles = self.model.forecast(
                [rolling_context[-self.max_context :]],
                freq=[self.frequency],
                normalize=True,
            )
            point_chunk = np.asarray(points[0, :step], dtype=np.float32)
            quantile_chunk = np.asarray(quantiles[0, :step, :], dtype=np.float32)
            if point_chunk.shape != (step,) or quantile_chunk.shape != (step, 10):
                raise RuntimeError(
                    "TimesFM returned unexpected shapes: "
                    f"points={point_chunk.shape}, quantiles={quantile_chunk.shape}."
                )
            if not np.isfinite(point_chunk).all() or not np.isfinite(quantile_chunk).all():
                raise RuntimeError("TimesFM returned non-finite forecast values.")

            # V2's experimental quantile heads are not crossing-calibrated.
            # Preserve the mean in column 0 and enforce q10 <= ... <= q90.
            quantile_chunk[:, 1:] = np.sort(quantile_chunk[:, 1:], axis=1)
            quantile_chunk = np.maximum(quantile_chunk, 0)
            point_chunk = quantile_chunk[:, 5].copy()

            point_chunks.append(point_chunk)
            quantile_chunks.append(quantile_chunk)
            rolling_context = np.concatenate((rolling_context, point_chunk))
            remaining -= step
            chunks += 1

        return (
            np.concatenate(point_chunks),
            np.concatenate(quantile_chunks),
            chunks,
        )
