"""Dedicated command-line entry point for DGT forecasts."""

from .core import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
