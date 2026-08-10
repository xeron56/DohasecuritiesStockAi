"""Supported report-output languages."""

OUTPUT_LANGUAGE_CHOICES = (
    ("English (default)", "English"),
    ("Bangla (বাংলা)", "Bangla"),
)

_OUTPUT_LANGUAGE_ALIASES = {
    "english": "English",
    "en": "English",
    "bangla": "Bangla",
    "bengali": "Bangla",
    "বাংলা": "Bangla",
    "bn": "Bangla",
}


def normalize_output_language(value: str | None) -> str:
    """Return the canonical supported language name or raise ``ValueError``."""

    normalized = str(value or "English").strip().casefold()
    language = _OUTPUT_LANGUAGE_ALIASES.get(normalized)
    if language is None:
        raise ValueError("output language must be English or Bangla")
    return language
