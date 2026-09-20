"""The target agent's *reference*: where its improvable seed code + deps come from.

A target-agent profile carries an ``agent_reference`` describing the seed the
meta-agent starts from and the feedback-agent iterates on:

- ``"default"`` — the task package's bundled ``reference/`` directory (entrypoint
  ``reference_target_agent.py``). This is the historical behavior.
- ``{"source": "./my_agent.py"}`` — a single user file. Its text is embedded in the
  meta-agent prompt (the "basic" reference).
- ``{"source": "./my_agent_dir/", "entrypoint": "main.py"}`` — a multi-file
  directory. SIA copies it into each generation's working dir and the agent reads it
  with its own tools rather than having it piped into the prompt.

A reference has two separable parts: the **improvable seed source** (stays editable —
SIA rewrites it every generation) and its **dependencies**. Dependencies live in a
``requirements.txt`` *inside the reference* (not a profile field) so the meta/feedback
agents can evolve them across generations; they are installed per generation and
augment the global ``Config.VENV_PACKAGES`` baseline.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from sia.layout import Names

if TYPE_CHECKING:
    from sia.layout import TaskLayout


@dataclass(frozen=True)
class AgentReference:
    """A parsed ``agent_reference`` spec (paths already resolved to absolute)."""

    kind: str  # "default" | "file" | "dir"
    source: Path | None = None  # abs path to the file (file) or directory (dir); None for default
    entrypoint: str | None = None  # filename within the directory (dir only)


DEFAULT_AGENT_REFERENCE = AgentReference(kind="default")


@dataclass(frozen=True)
class ResolvedAgentReference:
    """An ``AgentReference`` resolved against a concrete task, ready to use."""

    inline_seed: str | None  # entrypoint text to embed in the prompt (default/file); None for a dir
    ref_dir: Path | None  # directory whose contents are copied into each gen working dir (dir only)
    entrypoint: str  # filename the agent should treat as the starting point
    requirements: Path | None  # requirements.txt to install + carry forward, if present


def parse_agent_reference(spec: object, base_dir: str | Path | None = None) -> AgentReference:
    """Parse a raw ``agent_reference`` value from a profile JSON into an AgentReference.

    ``base_dir`` is the directory the profile file lives in; a relative ``source``
    resolves against it (falling back to the current working directory).
    """
    if spec is None or spec == "default":
        return DEFAULT_AGENT_REFERENCE
    if not isinstance(spec, dict) or "source" not in spec:
        raise SystemExit('agent_reference must be "default" or an object with a "source" field')

    data = cast("dict[str, Any]", spec)
    base = Path(base_dir) if base_dir else Path.cwd()
    source = Path(data["source"])
    if not source.is_absolute():
        source = base / source
    source = source.resolve()

    if source.is_dir():
        return AgentReference(kind="dir", source=source, entrypoint=data.get("entrypoint"))
    if source.is_file():
        return AgentReference(kind="file", source=source)
    raise SystemExit(f"agent_reference source not found: {source}")


def resolve_agent_reference(ref: AgentReference, task_layout: TaskLayout) -> ResolvedAgentReference:
    """Resolve an AgentReference against a concrete task into a ResolvedAgentReference."""
    if ref.kind == "default":
        ref_dir = Path(task_layout.reference_dir)
        seed = (ref_dir / Names.REFERENCE_AGENT_FILE).read_text(encoding="utf-8")
        reqs = ref_dir / Names.REQUIREMENTS_TXT
        return ResolvedAgentReference(
            inline_seed=seed,
            ref_dir=None,
            entrypoint=Names.REFERENCE_AGENT_FILE,
            requirements=reqs if reqs.is_file() else None,
        )

    if ref.kind == "file":
        assert ref.source is not None
        return ResolvedAgentReference(
            inline_seed=ref.source.read_text(encoding="utf-8"),
            ref_dir=None,
            entrypoint=ref.source.name,
            requirements=None,
        )

    # kind == "dir"
    assert ref.source is not None
    entrypoint = ref.entrypoint or Names.REFERENCE_AGENT_FILE
    if not (ref.source / entrypoint).is_file():
        raise SystemExit(f"agent_reference entrypoint '{entrypoint}' not found in {ref.source}")
    reqs = ref.source / Names.REQUIREMENTS_TXT
    return ResolvedAgentReference(
        inline_seed=None,
        ref_dir=ref.source,
        entrypoint=entrypoint,
        requirements=reqs if reqs.is_file() else None,
    )


def copy_reference_into(resolved: ResolvedAgentReference, gen_dir: str | Path) -> None:
    """Place reference helper files + requirements.txt into a generation working dir.

    For a directory reference the whole reference is copied so the generated
    ``target_agent.py`` can import its helper modules and the agent can read them with
    its tools. For default/file references only a sibling ``requirements.txt`` (if any)
    is carried in — the seed code itself reaches the agent via the prompt.
    """
    dest = Path(gen_dir)
    if resolved.ref_dir is not None:
        for entry in resolved.ref_dir.iterdir():
            if entry.is_dir():
                shutil.copytree(entry, dest / entry.name, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, dest / entry.name)
    elif resolved.requirements is not None:
        shutil.copy2(resolved.requirements, dest / Names.REQUIREMENTS_TXT)
