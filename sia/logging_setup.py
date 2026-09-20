"""Centralized logging configuration for SIA.

Replaces the duplicated ``logging.basicConfig(...)`` blocks that previously lived
in both orchestrator.py and util.py. Configuration is idempotent and uses the same
format/datefmt as before, so log output is unchanged.
"""

from __future__ import annotations

import logging
import os

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_ENV_VAR = "SIA_LOG_LEVEL"
_configured = False


def _resolve_level(level: int | str | None) -> int:
    """Resolve a level from an explicit value, then ``$SIA_LOG_LEVEL``, then INFO.

    Accepts a logging int, a level name ("DEBUG"), or a numeric string ("10").
    Unrecognized values fall back to INFO.
    """
    if level is None:
        level = os.environ.get(_ENV_VAR, "").strip() or None
    if level is None:
        return logging.INFO
    if isinstance(level, int):
        return level
    name = level.upper()
    if name.isdigit():
        return int(name)
    return logging.getLevelNamesMapping().get(name, logging.INFO)


def configure_logging(level: int | str | None = None) -> None:
    """Configure root logging.

    On the first call this installs the handler/format. The level is taken from
    ``level`` if given, else from ``$SIA_LOG_LEVEL``, else INFO. A later call with
    an explicit ``level`` (e.g. a CLI flag parsed after import) updates the root
    level; otherwise subsequent calls are no-ops.
    """
    global _configured
    resolved = _resolve_level(level)
    if _configured:
        if level is not None:
            logging.getLogger().setLevel(resolved)
        return
    logging.basicConfig(level=resolved, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured first."""
    configure_logging()
    return logging.getLogger(name)
