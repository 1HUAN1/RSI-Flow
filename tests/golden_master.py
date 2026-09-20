"""Golden-master test helper.

Characterization tests capture the *current* output of a function as a committed
"golden" file, then assert future runs produce byte-identical output. This locks
business logic (especially the large prompt / context.md strings) so the refactor
can be proven behavior-preserving.

To (re)generate goldens after an intentional change:

    UPDATE_GOLDEN=1 python -m pytest tests/

Volatile substrings (timestamps, temp paths) must be normalized by the caller
before passing text in, so goldens stay deterministic across machines/runs.
"""

from __future__ import annotations

import difflib
import os
import re
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent / "golden"
_UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"

# Matches "YYYY-MM-DD HH:MM:SS" timestamps (datetime.now() output in the code).
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")


def normalize_timestamps(text: str) -> str:
    """Replace concrete timestamps with a stable placeholder."""
    return _TIMESTAMP_RE.sub("<TS>", text)


def normalize_paths(text: str, replacements: dict[str, str]) -> str:
    """Replace volatile absolute paths with stable placeholders.

    For each (path -> placeholder), both the raw and os-resolved forms are
    replaced (macOS temp dirs resolve /var -> /private/var).
    """
    for raw, placeholder in replacements.items():
        for variant in (raw, str(Path(raw).resolve())):
            text = text.replace(variant, placeholder)
    return text


def assert_golden(name: str, actual: str) -> None:
    """Assert `actual` matches committed golden `name` (or write it under UPDATE_GOLDEN)."""
    path = GOLDEN_DIR / name
    if _UPDATE:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
        return
    assert path.exists(), f"Missing golden '{name}'. Generate with: UPDATE_GOLDEN=1 pytest"
    expected = path.read_text(encoding="utf-8")
    if actual != expected:
        diff = "\n".join(
            difflib.unified_diff(
                expected.splitlines(),
                actual.splitlines(),
                fromfile=f"golden/{name}",
                tofile="actual",
                lineterm="",
            )
        )
        raise AssertionError(f"Golden mismatch for '{name}':\n{diff}")
