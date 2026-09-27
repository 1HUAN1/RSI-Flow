"""Name-or-path resolution for JSON config files (providers and profiles).

A config value is either a filesystem **path** (contains a path separator or ends in
``.json``) or a bare **name** that resolves, in order, against the user directory
(``$SIA_<KIND>S_DIR`` else ``./<kind>s``) and then the bundled defaults shipped inside
the wheel (``sia/defaults/<kind>s/*.json``). Used by both ``sia.providers`` and
``sia.profiles`` so the lookup rules stay identical.
"""

from __future__ import annotations

import os
from importlib.resources import files as resource_files
from pathlib import Path


def _looks_like_path(value: str) -> bool:
    """True when ``value`` should be treated as a filesystem path, not a bare name."""
    return value.endswith(".json") or "/" in value or os.sep in value


def user_dir(env_var: str, default_subdir: str) -> Path:
    """User config directory: ``$<env_var>`` if set, else ``./<default_subdir>``."""
    return Path(os.environ.get(env_var, default_subdir))


def read_config_text(name_or_path: str, *, env_var: str, subdir: str, kind: str) -> tuple[str, str]:
    """Resolve ``name_or_path`` to (file_text, source_label) for a config of ``kind``.

    Raises SystemExit with the list of available names when a bare name can't be found.
    """
    if _looks_like_path(name_or_path):
        path = Path(name_or_path)
        if not path.is_file():
            raise SystemExit(f"{kind} file not found: {name_or_path}")
        return path.read_text(encoding="utf-8"), str(path)

    filename = f"{name_or_path}.json"

    candidate = user_dir(env_var, subdir) / filename
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8"), str(candidate)

    bundled = resource_files("sia").joinpath("defaults", subdir, filename)
    if bundled.is_file():
        return bundled.read_text(encoding="utf-8"), f"<bundled>/defaults/{subdir}/{filename}"

    available = ", ".join(available_names(env_var=env_var, subdir=subdir)) or "(none)"
    raise SystemExit(f"Unknown {kind} '{name_or_path}'. Available: {available} (or pass a path to a .json file).")


def available_names(*, env_var: str, subdir: str) -> list[str]:
    """Sorted union of config names from the bundled defaults and the user directory."""
    names: set[str] = set()

    bundled = resource_files("sia").joinpath("defaults", subdir)
    if bundled.is_dir():
        names |= {entry.name[:-5] for entry in bundled.iterdir() if entry.name.endswith(".json")}

    udir = user_dir(env_var, subdir)
    if udir.is_dir():
        names |= {entry.name[:-5] for entry in udir.iterdir() if entry.name.endswith(".json")}

    return sorted(names)
