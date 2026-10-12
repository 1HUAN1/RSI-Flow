"""Data layer for the runs visualizer.

Pure, dependency-light functions that read the ``runs/`` directory tree and turn
it into JSON-serializable models. Nothing here imports FastAPI, so it can be
unit-tested in isolation and reused from other tools.

Directory shape this understands (see orchestrator.py for the writer side)::

    runs/
      run_<id>/
        context.md                 # run-level metadata + per-generation log
        gen_<n>/
          target_agent.py          # the generated agent code
          meta_agent_prompt.txt     # gen 1: the meta-agent prompt
          feedback_agent_prompt.txt # gen >1: the feedback prompt
          improvement.md            # gen >1: the improvement plan
          evaluation_results.json   # scores + per-question details
          evaluation.log            # evaluator stdout/stderr
          target_agent_stdout.log   # agent stdout
          agent_execution/          # execution_q<id>.json per-question chat logs
          openhands_trajectory/     # <session>/events/event-*.json (optional)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# Files we surface as first-class artifacts in the UI (label -> filename).
TEXT_ARTIFACTS: dict[str, str] = {
    "target_agent": "target_agent.py",
    "meta_prompt": "meta_agent_prompt.txt",
    "feedback_prompt": "feedback_agent_prompt.txt",
    "improvement": "improvement.md",
    "eval_log": "evaluation.log",
    "stdout_log": "target_agent_stdout.log",  # or train_stdout.log for weights mode
}

# Alternative filenames to check for stdout_log artifact
_STDOUT_LOG_ALTS = ("target_agent_stdout.log", "train_stdout.log")

# Candidate names for the structured evaluation summary, in priority order.
_EVAL_RESULT_NAMES = ("evaluation_results.json", "results.json")

_RUN_DIR_RE = re.compile(r"^run_(\d+)$")
_GEN_DIR_RE = re.compile(r"^gen_(\d+)$")
_EXEC_FILE_RE = re.compile(r"^execution_q(\d+)\.json$")
_META_RE = re.compile(r"^\*\*(?P<key>[^*]+)\*\*:\s*(?P<value>.*)$")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class EvalSummary(BaseModel):
    """Headline numbers from a generation's evaluation results."""

    total: int | None = None
    correct: int | None = None
    incorrect: int | None = None
    missing: int | None = None
    invalid: int | None = None
    accuracy_percent: float | None = None


class GenerationSummary(BaseModel):
    name: str
    index: int
    eval: EvalSummary | None = None
    has_target_agent: bool = False
    has_improvement: bool = False
    num_trajectories: int = 0


class RunSummary(BaseModel):
    name: str
    index: int
    task: str | None = None
    meta_model: str | None = None
    task_model: str | None = None
    agent_impl: str | None = None
    started: str | None = None
    max_generations: int | None = None
    num_generations: int = 0
    best_accuracy_percent: float | None = None
    generations: list[GenerationSummary] = []


class DomainStat(BaseModel):
    domain: str
    total: int
    correct: int
    accuracy_percent: float


class GenerationDetail(BaseModel):
    name: str
    index: int
    eval: EvalSummary | None = None
    domains: list[DomainStat] = []
    artifacts: list[str] = []
    trajectories: list[int] = []
    has_openhands: bool = False


class RunDetail(BaseModel):
    name: str
    index: int
    task: str | None = None
    meta_model: str | None = None
    task_model: str | None = None
    agent_impl: str | None = None
    started: str | None = None
    max_generations: int | None = None
    # Resolved meta/target agent profiles (full JSON from the run's profiles.json),
    # rendered as-is in the profile chips. None for older runs predating profiles.json.
    profiles: dict[str, Any] | None = None
    context_md: str | None = None
    generations: list[GenerationDetail] = []


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> Any | None:
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _eval_results_path(gen_dir: Path) -> Path | None:
    for name in _EVAL_RESULT_NAMES:
        candidate = gen_dir / name
        if candidate.is_file():
            return candidate
    return None


def _eval_summary(gen_dir: Path) -> EvalSummary | None:
    path = _eval_results_path(gen_dir)
    if path is None:
        return None
    data = _read_json(path)
    if not isinstance(data, dict):
        return None
    pct = data.get("accuracy_percent")
    if pct is None and isinstance(data.get("accuracy"), int | float):
        pct = data["accuracy"] * 100
    return EvalSummary(
        total=data.get("total_questions") or data.get("total"),
        correct=data.get("correct"),
        incorrect=data.get("incorrect"),
        missing=data.get("missing"),
        invalid=data.get("invalid"),
        accuracy_percent=pct,
    )


def _trajectory_ids(gen_dir: Path) -> list[int]:
    exec_dir = gen_dir / "agent_execution"
    if not exec_dir.is_dir():
        return []
    ids: list[int] = []
    for child in exec_dir.iterdir():
        m = _EXEC_FILE_RE.match(child.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def _gen_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        m = _GEN_DIR_RE.match(child.name)
        if m:
            found.append((int(m.group(1)), child))
    return sorted(found, key=lambda t: t[0])


def parse_context_md(text: str) -> dict[str, str]:
    """Extract the leading ``**Key**: value`` metadata block from context.md."""
    meta: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("## "):  # first generation heading => end of header block
            break
        m = _META_RE.match(line)
        if m:
            meta[m.group("key").strip().lower()] = m.group("value").strip()
    return meta


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def list_runs(runs_root: Path) -> list[RunSummary]:
    """Summaries for every ``run_<id>`` directory, newest id first."""
    if not runs_root.is_dir():
        return []
    runs: list[RunSummary] = []
    for child in runs_root.iterdir():
        m = _RUN_DIR_RE.match(child.name) if child.is_dir() else None
        if m:
            runs.append(_run_summary(child, int(m.group(1))))
    runs.sort(key=lambda r: r.index, reverse=True)
    return runs


def _run_summary(run_dir: Path, index: int) -> RunSummary:
    meta: dict[str, str] = {}
    context = run_dir / "context.md"
    if context.is_file():
        meta = parse_context_md(_read_text(context) or "")

    gens: list[GenerationSummary] = []
    best: float | None = None
    for gen_index, gen_dir in _gen_dirs(run_dir):
        ev = _eval_summary(gen_dir)
        gens.append(
            GenerationSummary(
                name=gen_dir.name,
                index=gen_index,
                eval=ev,
                has_target_agent=(gen_dir / TEXT_ARTIFACTS["target_agent"]).is_file(),
                has_improvement=(gen_dir / TEXT_ARTIFACTS["improvement"]).is_file(),
                num_trajectories=len(_trajectory_ids(gen_dir)),
            )
        )
        if ev and ev.accuracy_percent is not None:
            best = ev.accuracy_percent if best is None else max(best, ev.accuracy_percent)

    return RunSummary(
        name=run_dir.name,
        index=index,
        task=meta.get("task"),
        meta_model=meta.get("meta model"),
        task_model=meta.get("task model"),
        agent_impl=meta.get("agent impl"),
        started=meta.get("started"),
        max_generations=_as_int(meta.get("max generations")),
        num_generations=len(gens),
        best_accuracy_percent=best,
        generations=gens,
    )


def get_run(runs_root: Path, run_name: str) -> RunDetail | None:
    run_dir = _safe_child(runs_root, run_name)
    if run_dir is None or not run_dir.is_dir():
        return None
    m = _RUN_DIR_RE.match(run_dir.name)
    if not m:
        return None

    meta: dict[str, str] = {}
    context_md: str | None = None
    context = run_dir / "context.md"
    if context.is_file():
        context_md = _read_text(context)
        meta = parse_context_md(context_md or "")

    generations = [_generation_detail(gen_dir, gi) for gi, gen_dir in _gen_dirs(run_dir)]

    profiles_data = _read_json(run_dir / "profiles.json")
    profiles = profiles_data if isinstance(profiles_data, dict) else None

    return RunDetail(
        name=run_dir.name,
        index=int(m.group(1)),
        task=meta.get("task"),
        meta_model=meta.get("meta model"),
        task_model=meta.get("task model"),
        agent_impl=meta.get("agent impl"),
        started=meta.get("started"),
        max_generations=_as_int(meta.get("max generations")),
        profiles=profiles,
        context_md=context_md,
        generations=generations,
    )


def _generation_detail(gen_dir: Path, index: int) -> GenerationDetail:
    artifacts = []
    for label, fname in TEXT_ARTIFACTS.items():
        # For stdout_log, check both harness and weights mode variants
        if label == "stdout_log":
            if any((gen_dir / alt).is_file() for alt in _STDOUT_LOG_ALTS):
                artifacts.append(label)
        elif (gen_dir / fname).is_file():
            artifacts.append(label)
    return GenerationDetail(
        name=gen_dir.name,
        index=index,
        eval=_eval_summary(gen_dir),
        domains=_domain_stats(gen_dir),
        artifacts=artifacts,
        trajectories=_trajectory_ids(gen_dir),
        has_openhands=(gen_dir / "openhands_trajectory").is_dir(),
    )


def _domain_stats(gen_dir: Path) -> list[DomainStat]:
    path = _eval_results_path(gen_dir)
    data = _read_json(path) if path else None
    if not isinstance(data, dict):
        return []
    details = data.get("details")
    if not isinstance(details, list):
        return []
    buckets: dict[str, list[int]] = {}  # domain -> [total, correct]
    for row in details:
        if not isinstance(row, dict):
            continue
        domain = str(row.get("domain") or "Unknown")
        bucket = buckets.setdefault(domain, [0, 0])
        bucket[0] += 1
        if row.get("is_correct"):
            bucket[1] += 1
    stats = [
        DomainStat(
            domain=domain,
            total=total,
            correct=correct,
            accuracy_percent=(correct / total * 100) if total else 0.0,
        )
        for domain, (total, correct) in buckets.items()
    ]
    stats.sort(key=lambda s: s.domain)
    return stats


def get_eval_details(runs_root: Path, run_name: str, gen_name: str) -> list[dict[str, Any]] | None:
    """The full per-question ``details`` array from evaluation results."""
    gen_dir = _resolve_gen(runs_root, run_name, gen_name)
    if gen_dir is None:
        return None
    path = _eval_results_path(gen_dir)
    data = _read_json(path) if path else None
    if isinstance(data, dict) and isinstance(data.get("details"), list):
        return data["details"]
    return None


def get_artifact_text(runs_root: Path, run_name: str, gen_name: str, label: str) -> str | None:
    """Read one of the known text artifacts (by label, not raw path)."""
    fname = TEXT_ARTIFACTS.get(label)
    gen_dir = _resolve_gen(runs_root, run_name, gen_name)
    if fname is None or gen_dir is None:
        return None

    # For stdout_log, check both harness and weights mode variants
    if label == "stdout_log":
        for alt_fname in _STDOUT_LOG_ALTS:
            alt_path = gen_dir / alt_fname
            if alt_path.is_file():
                return _read_text(alt_path)
        return None

    return _read_text(gen_dir / fname)


def get_trajectory(runs_root: Path, run_name: str, gen_name: str, qid: int) -> list[dict[str, str]] | None:
    """Per-question chat log, normalized to ``[{role, text}]`` turns."""
    gen_dir = _resolve_gen(runs_root, run_name, gen_name)
    if gen_dir is None:
        return None
    path = gen_dir / "agent_execution" / f"execution_q{qid}.json"
    data = _read_json(path)
    if not isinstance(data, list):
        return None
    return [_normalize_turn(msg) for msg in data if isinstance(msg, dict)]


def _normalize_turn(msg: dict[str, Any]) -> dict[str, str]:
    role = str(msg.get("role", "unknown"))
    content = msg.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "\n\n".join(_block_text(block) for block in content).strip()
    else:
        text = "" if content is None else str(content)
    return {"role": role, "text": text}


def _block_text(block: Any) -> str:
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return str(block)
    btype = block.get("type")
    if btype == "text" or "text" in block:
        return str(block.get("text", ""))
    if btype == "tool_use":
        args = json.dumps(block.get("input", {}), indent=2)
        return f"[tool_use: {block.get('name', '?')}]\n{args}"
    if btype == "tool_result":
        return f"[tool_result]\n{_stringify(block.get('content'))}"
    return _stringify(block)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_block_text(v) for v in value)
    return json.dumps(value, indent=2, default=str)


def list_openhands_sessions(runs_root: Path, run_name: str, gen_name: str) -> list[str] | None:
    gen_dir = _resolve_gen(runs_root, run_name, gen_name)
    if gen_dir is None:
        return None
    root = gen_dir / "openhands_trajectory"
    if not root.is_dir():
        return []
    return sorted(child.name for child in root.iterdir() if child.is_dir())


def get_openhands_events(runs_root: Path, run_name: str, gen_name: str, session: str) -> list[dict[str, Any]] | None:
    gen_dir = _resolve_gen(runs_root, run_name, gen_name)
    if gen_dir is None:
        return None
    session_dir = _safe_child(gen_dir / "openhands_trajectory", session)
    if session_dir is None or not session_dir.is_dir():
        return None
    events_dir = session_dir / "events"
    if not events_dir.is_dir():
        return []
    events: list[dict[str, Any]] = []
    for child in sorted(events_dir.iterdir()):
        if child.suffix == ".json":
            data = _read_json(child)
            if isinstance(data, dict):
                events.append(data)
    return events


# --------------------------------------------------------------------------- #
# Path safety
# --------------------------------------------------------------------------- #
def _safe_child(parent: Path, name: str) -> Path | None:
    """Resolve ``parent/name`` and refuse anything that escapes ``parent``."""
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        return None
    try:
        resolved = (parent / name).resolve()
        resolved.relative_to(parent.resolve())
    except (OSError, ValueError):
        return None
    return resolved


def _resolve_gen(runs_root: Path, run_name: str, gen_name: str) -> Path | None:
    run_dir = _safe_child(runs_root, run_name)
    if run_dir is None or not _RUN_DIR_RE.match(run_name):
        return None
    gen_dir = _safe_child(run_dir, gen_name)
    if gen_dir is None or not _GEN_DIR_RE.match(gen_name) or not gen_dir.is_dir():
        return None
    return gen_dir


def _as_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None
