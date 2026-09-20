"""Small, explicit snapshot helpers. Logs and artifacts live in separate trees."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path

from pydantic import BaseModel

from sia.task_meta.types import ArtifactState, TaskAgentState


def jsonable(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    return value


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint_manifest(directory: str | Path) -> list[dict]:
    root = Path(directory).resolve()
    weights = sorted([*root.glob("*.safetensors"), *root.glob("*.bin")])
    if not weights or any(p.stat().st_size == 0 for p in weights):
        raise ValueError("Checkpoint must contain nonempty weight files")
    return [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in weights]


def experience_summary(experience) -> dict:
    """Do not recursively embed previous observations and their whole histories."""
    return {"generation": experience.generation, "experience_id": experience.experience_id,
            "action": experience.chosen_action or experience.decision.get("action"),
            "decision_source": experience.decision.get("decision_source", "model"),
            "deployment_status": experience.deployment_status,
            "candidate_attempts": [{key: attempt.get(key) for key in
                ("attempt_id", "action", "status", "gain", "positive_gain", "deployed")}
                for attempt in experience.candidate_attempts],
            "evaluated_state_before": experience.evaluated_state_before,
            "intervention_base_state": experience.intervention_base_state,
            "evaluated_state_after": experience.evaluated_state_after,
            "rollout_artifact_diff": experience.rollout_artifact_diff,
            "intervention_diff": experience.intervention_diff,
            "delta_interpretation": "observed online evolution difference; not isolated causal benefit",
            "observed_performance_delta": experience.observed_performance_delta,
            "modification": experience.modification, "performance_before": experience.performance_before,
            "performance_after": experience.performance_after, "performance_delta": experience.performance_delta,
            "cost_before": experience.cost_before, "cost_after": experience.cost_after,
            "update_cost": experience.update_cost, "trajectory_before": experience.trajectory_before,
            "trajectory_after": experience.trajectory_after}


def manifest_diff(before: list[dict], after: list[dict]) -> dict:
    old, new = {r["path"]: r for r in before}, {r["path"]: r for r in after}
    return {"added": [new[p] for p in sorted(new.keys() - old.keys())],
            "removed": [old[p] for p in sorted(old.keys() - new.keys())],
            "changed": [{"path": p, "before": old[p], "after": new[p]}
                        for p in sorted(old.keys() & new.keys()) if old[p] != new[p]],
            "unchanged": sorted(p for p in old.keys() & new.keys() if old[p] == new[p])}


def artifact_manifest(directory: str | Path | None) -> list[dict]:
    if directory is None or not Path(directory).exists():
        return []
    root = Path(directory)
    if root.is_symlink():
        raise ValueError("Artifact snapshot cannot be a symlink")
    manifest = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Artifact entries cannot be symlinks")
        if path.is_file():
            if path.name in {"results.json", "agent_execution.json", "meta_decision.json", "task_state.json",
                             "target_agent.py", "target_config.json", "artifact_manifest.json"} or path.suffix == ".log":
                raise ValueError(f"Execution or control file cannot be an artifact: {path.name}")
            manifest.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)})
    return manifest


def clone_task(state: TaskAgentState, generation: int, directory: Path) -> TaskAgentState:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / Path(state.harness_path).name
    shutil.copy2(state.harness_path, target)
    artifact_dir = directory / "artifacts"
    if state.artifacts.directory and Path(state.artifacts.directory).exists():
        artifact_manifest(state.artifacts.directory)
        shutil.copytree(state.artifacts.directory, artifact_dir)
        provenance = Path(state.artifacts.directory).parent / "artifact_provenance.json"
        if provenance.is_file():
            shutil.copy2(provenance, directory / "input_artifact_provenance.json")
    artifacts = ArtifactState(str(artifact_dir), artifact_manifest(artifact_dir)) if artifact_dir.exists() else ArtifactState()
    return TaskAgentState(generation, state.model_ref, str(target), artifacts,
                          state.checkpoint_path, list(state.checkpoint_manifest))
