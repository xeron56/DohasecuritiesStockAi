"""Shared visual language for the interactive terminal experience."""

from __future__ import annotations

import questionary
from rich.theme import Theme

CLI_THEME = Theme(
    {
        "brand": "bold #22d3ee",
        "brand.secondary": "bold #60a5fa",
        "muted": "#64748b",
        "label": "bold #94a3b8",
        "value": "bold #f8fafc",
        "success": "bold #34d399",
        "warning": "bold #fbbf24",
        "danger": "bold #fb7185",
        "stage.active": "bold #22d3ee",
        "stage.done": "#34d399",
        "stage.pending": "#64748b",
        "activity.tool": "bold #a78bfa",
        "activity.agent": "bold #60a5fa",
        "activity.system": "bold #94a3b8",
        "activity.user": "bold #34d399",
    }
)


PROMPT_STYLE = questionary.Style(
    [
        ("qmark", "fg:#22d3ee bold"),
        ("question", "bold"),
        ("answer", "fg:#34d399 bold"),
        ("pointer", "fg:#22d3ee bold"),
        ("highlighted", "fg:#22d3ee bold"),
        ("selected", "fg:#34d399"),
        ("checkbox", "fg:#64748b"),
        ("checkbox-selected", "fg:#34d399 bold"),
        ("instruction", "fg:#64748b"),
        ("text", "fg:#f8fafc"),
        ("disabled", "fg:#475569 italic"),
    ]
)

PROMPT_MARK = "›"
PROMPT_POINTER = "▸"
