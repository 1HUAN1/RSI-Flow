"""Shared, safe filesystem helpers.

Canonical home for the size-limited read/JSON helpers that were previously defined
in context_manager.py and re-implemented ad hoc inside orchestrator.py. Keeping a
single implementation removes that duplication (DRY) and gives one place to test
the size-limit and error-handling behavior.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from sia.config import Config
from sia.logging_setup import get_logger

logger = get_logger(__name__)

# Default size cap, evaluated once at import.
_DEFAULT_MAX_BYTES = Config().MAX_CONTEXT_FILE_SIZE


def file_size_ok(path: str | Path, max_bytes: int) -> tuple[bool, int]:
    """Return ``(within_limit, size_bytes)`` for ``path``.

    Centralizes the ``os.path.getsize`` + ``<=`` comparison used across the codebase.
    A file whose size equals ``max_bytes`` is within the limit (matches prior ``>`` checks).
    """
    size = os.path.getsize(path)
    return size <= max_bytes, size


def safe_read_file(path: str | Path, max_bytes: int = _DEFAULT_MAX_BYTES) -> str | None:
    """Read a file as UTF-8 text, returning None if it exceeds ``max_bytes`` or can't be read."""
    try:
        within_limit, size = file_size_ok(path, max_bytes)
        if not within_limit:
            logger.warning(f"File too large ({size:,} bytes > {max_bytes:,}): {path}")
            return None
        return Path(path).read_text(encoding="utf-8")
    except OSError as e:
        logger.warning(f"Could not read file: {path}: {e}")
        return None


def safe_load_json(path: str | Path, max_bytes: int = _DEFAULT_MAX_BYTES) -> dict | list | None:
    """Load JSON from a file, returning None if it exceeds ``max_bytes`` or can't be parsed."""
    try:
        within_limit, size = file_size_ok(path, max_bytes)
        if not within_limit:
            logger.warning(f"JSON file too large ({size:,} bytes > {max_bytes:,}): {path}")
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not load JSON: {path}: {e}")
        return None


def write_text(path: str | Path, content: str) -> None:
    """Write UTF-8 text to ``path`` (single spelling of the common idiom)."""
    Path(path).write_text(content, encoding="utf-8")
