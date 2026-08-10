"""REST presentation layer for DSE-backed TradingAgents analyses."""

from __future__ import annotations

from typing import Any


def create_app(*args: Any, **kwargs: Any):
    """Import the FastAPI app lazily so CLI-only runs stay side-effect free."""

    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["create_app"]
