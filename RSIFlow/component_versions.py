"""Numbered, deduplicated experiment components for snapshots and ablations.

Model weights already live in immutable seed/SFT directories: retain references,
not another weight copy per round. Harness, Artifacts and skill contents are
copied once per distinct version. None of these helpers selects a successor.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(path.read_text())


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def content_hash(path: Path) -> str:
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return fingerprint({p.relative_to(path).as_posix(): content_hash(p)
                        for p in sorted(path.rglob("*")) if p.is_file()
                        and "__pycache__" not in p.parts and p.suffix != ".pyc"})


class ComponentVersions:
    def __init__(self, root: Path, workspace: Path):
        self.root, self.workspace = root.resolve(), workspace.resolve()
        self.index_path = self.root / "index.json"

    def path(self, value) -> Path:
        source = Path(value)
        return (source if source.is_absolute() else self.workspace / source).resolve()

    def register(self, kind: str, *, source: Path | None = None, value=None):
        digest = content_hash(source) if source is not None else fingerprint(value)
        self.root.mkdir(parents=True, exist_ok=True)
        # The index can also be used by concurrent snapshot/debug tools.
        with (self.root / ".index.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            index = read_json(self.index_path) if self.index_path.exists() else {}
            entries = index.setdefault(kind, [])
            for entry in entries:
                if entry["sha256"] == digest:
                    return entry
            version = f"{kind}_{len(entries) + 1:03d}"
            folder = self.root / version
            folder.mkdir(exist_ok=True)
            if source is not None:
                target = folder / ("contents" if source.is_dir() else "contents" + source.suffix)
                if source.is_dir():
                    shutil.copytree(source, target, dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                else:
                    shutil.copy2(source, target)
            else:
                target = folder / "reference.json"
                write_json(target, value)
            entry = {"id": version, "sha256": digest, "path": str(target),
                     "source": str(source) if source is not None else None}
            entries.append(entry)
            write_json(self.index_path, index)
            return entry

    def task(self, state_path: Path):
        state = read_json(state_path)
        model = {key: state.get(key) for key in
                 ("model_ref", "checkpoint_path", "checkpoint_manifest")}
        model["weights_copied"] = False
        model_entry = self.register("model", value=model)
        harness = self.register("harness", source=self.path(state["harness_path"]))
        portable = copy.deepcopy(state)
        portable["harness_path"] = harness["path"]
        components = {"model": model_entry["id"], "harness": harness["id"]}
        artifacts = state.get("artifacts") or {}
        if artifacts.get("directory"):
            entry = self.register("artifacts", source=self.path(artifacts["directory"]))
            portable["artifacts"]["directory"] = entry["path"]
            components["artifacts"] = entry["id"]
        return portable, components, model_entry

    def get(self, kind, version):
        return next(item for item in read_json(self.index_path)[kind] if item["id"] == version)

    def compose(self, template_state: Path, model_id: str, harness_id: str, destination: Path):
        """Build an eval-only state; never activate it or change the experiment."""
        state = read_json(template_state)
        model = read_json(Path(self.get("model", model_id)["path"]))
        for key in ("model_ref", "checkpoint_path", "checkpoint_manifest"):
            state[key] = model[key]
        state["harness_path"] = self.get("harness", harness_id)["path"]
        write_json(destination, state)
        return {"status": "composed", "state_path": str(destination),
                "model_id": model_id, "harness_id": harness_id,
                "artifacts_source_state": str(template_state), "activated": False}


def snapshot(versions: ComponentVersions, destination: Path, states: dict[str, Path],
             skills: Path, references: dict):
    if destination.exists():
        return {"status": "destination_exists", "destination": str(destination)}
    # Register first, so missing inputs do not leave an unusable snapshot folder.
    tasks = {role: versions.task(path) for role, path in states.items()}
    meta = versions.register("meta", source=skills)
    destination.mkdir(parents=True)
    combinations = {}
    for role, (state, components, model) in tasks.items():
        relative = "task/task_state_snapshot.json" if role == "selected" else f"task/{role}_state.json"
        path = destination / relative
        write_json(path, state)
        combinations[role] = {**components, "state_path": str(path),
                              "source_state": str(states[role])}
    checkpoint = destination / "task/checkpoint_reference.json"
    write_json(checkpoint, read_json(Path(tasks["selected"][2]["path"])))
    record = {"versions_root": str(versions.root), "index_path": str(versions.index_path),
              "combinations": combinations, "meta": meta, "references": references}
    write_json(destination / "versions.json", record)
    receipt = {"status": "snapshotted", "destination": str(destination),
               "restorable_task_state_path": combinations["selected"]["state_path"],
               "checkpoint_reference_path": str(checkpoint), "weights_copied": False,
               "skills_snapshot_path": meta["path"], **record,
               "context_reference": references.get("context_reference"),
               "files_sha256": {p.relative_to(destination).as_posix(): content_hash(p)
                                for p in destination.rglob("*") if p.is_file()}}
    write_json(destination / "snapshot_receipt.json", receipt)
    return {**receipt, "receipt_path": str(destination / "snapshot_receipt.json")}
