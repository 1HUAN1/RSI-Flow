"""Content-addressed manifests for complete HarnessForge candidate bundles.

The Task state keeps one JSON file as its harness reference.  Unlike the old
configuration harness, that JSON file contains every source file belonging to
one HarnessForge candidate.  Runtime code materializes the bundle in an
isolated package before importing its native ``builder.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping


PINNED_HARNESSFORGE_COMMIT = "05b3ecadb3c9a7a938f75129ea22b8f2b36cf289"
MANIFEST_KIND = "harnessforge_candidate_bundle"
MANIFEST_SCHEMA_VERSION = 1

# This is the upstream Stage-3 contract, not an editable-file allowlist.
# Additional prompt files, helpers, and package files are retained verbatim.
REQUIRED_BUNDLE_FILES = frozenset(
    {
        "__init__.py",
        "builder.py",
        "Description.md",
        "planning_module/provider.py",
        "action_module/provider.py",
        "memory_module/provider.py",
    }
)

_IGNORED_PARTS = frozenset({".git", "__pycache__", ".pytest_cache", ".mypy_cache"})
_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})
_HARNESS_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _bundle_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Bundle paths must be nonempty POSIX relative paths")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe or non-canonical bundle path: {value!r}")
    if any(part in _IGNORED_PARTS for part in path.parts) or path.suffix in _IGNORED_SUFFIXES:
        raise ValueError(f"Generated/cache files do not belong in a candidate bundle: {value!r}")
    return value


def _canonical_payload(harness_name: str, upstream_commit: str, files: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "harness_name": harness_name,
        "upstream_commit": upstream_commit,
        "files": {path: files[path] for path in sorted(files)},
    }


def _payload_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class HarnessBundleManifest:
    """One immutable, self-contained HarnessForge candidate."""

    harness_name: str
    files: Mapping[str, str]
    upstream_commit: str = PINNED_HARNESSFORGE_COMMIT

    def __post_init__(self) -> None:
        if not isinstance(self.harness_name, str) or not _HARNESS_NAME.fullmatch(self.harness_name):
            raise ValueError("harness_name must be an importable Python identifier")
        if self.upstream_commit != PINNED_HARNESSFORGE_COMMIT:
            raise ValueError(
                "Harness bundle is not bound to the pinned HarnessForge commit "
                f"{PINNED_HARNESSFORGE_COMMIT}"
            )
        if not isinstance(self.files, Mapping):
            raise TypeError("files must map relative paths to complete UTF-8 text")

        normalized: dict[str, str] = {}
        for raw_path, content in self.files.items():
            path = _bundle_path(raw_path)
            if not isinstance(content, str):
                raise TypeError(f"Bundle file {path!r} is not UTF-8 text")
            normalized[path] = content
        missing = sorted(REQUIRED_BUNDLE_FILES - normalized.keys())
        if missing:
            raise ValueError(f"Incomplete HarnessForge candidate; missing files: {missing}")
        object.__setattr__(self, "files", MappingProxyType(dict(sorted(normalized.items()))))

    @property
    def bundle_sha256(self) -> str:
        return _payload_digest(_canonical_payload(self.harness_name, self.upstream_commit, self.files))

    @property
    def candidate_name(self) -> str:
        """Upstream Stage-3 terminology for ``harness_name``."""

        return self.harness_name

    @property
    def identity(self) -> dict[str, Any]:
        """Stable identity recorded by frozen runs and evaluation receipts."""

        return {
            "schema_version": "harnessforge_bundle_v1",
            "kind": MANIFEST_KIND,
            "harness_name": self.harness_name,
            "bundle_sha256": self.bundle_sha256,
            "upstream_commit": self.upstream_commit,
            "source_hashes": {
                name: hashlib.sha256(content.encode("utf-8")).hexdigest()
                for name, content in self.files.items()
            },
        }

    def to_dict(self) -> dict[str, Any]:
        payload = _canonical_payload(self.harness_name, self.upstream_commit, self.files)
        return {**payload, "bundle_sha256": _payload_digest(payload)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HarnessBundleManifest":
        if not isinstance(value, Mapping):
            raise TypeError("Harness manifest must be a JSON object")
        expected = {
            "schema_version",
            "kind",
            "harness_name",
            "upstream_commit",
            "files",
            "bundle_sha256",
        }
        if set(value) != expected:
            raise ValueError(
                "Harness manifest fields must be exactly " + ", ".join(sorted(expected))
            )
        if value.get("schema_version") != MANIFEST_SCHEMA_VERSION or value.get("kind") != MANIFEST_KIND:
            raise ValueError("Unsupported HarnessForge bundle manifest schema")
        manifest = cls(
            harness_name=value.get("harness_name"),
            upstream_commit=value.get("upstream_commit"),
            files=value.get("files"),
        )
        if value.get("bundle_sha256") != manifest.bundle_sha256:
            raise ValueError("Harness bundle digest does not match its embedded files")
        return manifest

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        harness_name: str | None = None,
        upstream_commit: str = PINNED_HARNESSFORGE_COMMIT,
    ) -> "HarnessBundleManifest":
        root = Path(directory).resolve()
        if not root.is_dir():
            raise ValueError(f"Harness candidate directory does not exist: {root}")
        files: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if any(part in _IGNORED_PARTS for part in relative.parts) or path.suffix in _IGNORED_SUFFIXES:
                continue
            if path.is_symlink():
                raise ValueError(f"Harness candidate bundles cannot contain symlinks: {relative}")
            if not path.is_file():
                continue
            try:
                files[relative.as_posix()] = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError(f"Harness candidate file is not UTF-8 text: {relative}") from exc
        return cls(harness_name or root.name, files, upstream_commit)

    def materialize(self, destination: str | Path) -> Path:
        """Write the exact bundle into a new or empty candidate directory."""

        root = Path(destination)
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise FileExistsError(f"Harness materialization target is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        for relative, content in self.files.items():
            target = root.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return root.resolve()


def load_manifest(path: str | Path) -> HarnessBundleManifest:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load HarnessForge bundle manifest: {source}") from exc
    return HarnessBundleManifest.from_dict(value)


def save_manifest(path: str | Path, manifest: HarnessBundleManifest) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def build_manifest(
    directory: str | Path,
    *,
    harness_name: str | None = None,
) -> HarnessBundleManifest:
    """Convenience alias used by the production pipeline."""

    return HarnessBundleManifest.from_directory(directory, harness_name=harness_name)


__all__ = [
    "HarnessBundleManifest",
    "MANIFEST_KIND",
    "MANIFEST_SCHEMA_VERSION",
    "PINNED_HARNESSFORGE_COMMIT",
    "REQUIRED_BUNDLE_FILES",
    "build_manifest",
    "load_manifest",
    "save_manifest",
]

