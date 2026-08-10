"""Command-line entry point for the DSE analysis API."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "dohasecuritiesstockai.api.app:app",
        host=os.environ.get("TRADINGAGENTS_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("TRADINGAGENTS_API_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
