"""Immutable, self-contained Task/Meta snapshots at committed round boundaries."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

from sia.task_meta.durable import value_hash
from sia.task_meta.harnessforge_manifest import load_manifest, save_manifest
from sia.task_meta.meta_harness.bundle import MetaHarnessBundle, strict_json
from sia.task_meta.sequential_loop import content_identity
from sia.task_meta.storage import artifact_manifest, digest, save_json
from sia.task_meta.types import ArtifactState, MetaAgentState, TaskAgentState


SCHEMA = "task-meta-round-checkpoint-v1"
DIRECTORY = "checkpoint"


def _regular(path: Path, *, directory: bool | None = None) -> bool:
    if path.is_symlink():
        return False
    if directory is True:
        return path.is_dir()
    if directory is False:
        return path.is_file()
    return path.exists()


def _copy_task(staging: Path, published: Path, task: TaskAgentState) -> tuple[dict, dict]:
    target = staging / "task"
    published_target = published / "task"
    target.mkdir()
    source_manifest = load_manifest(task.harness_path)
    manifest_path = save_manifest(target / "harness_manifest.json", source_manifest)
    source_manifest.materialize(target / "harness_bundle")

    artifacts_path = None
    expected_artifacts = artifact_manifest(task.artifacts.directory)
    if task.artifacts.directory:
        source_artifacts = Path(task.artifacts.directory)
        if not _regular(source_artifacts, directory=True):
            raise ValueError("Accepted Task artifact directory is missing or unsafe")
        artifacts_path = target / "artifacts"
        shutil.copytree(source_artifacts, artifacts_path)
        if artifact_manifest(artifacts_path) != expected_artifacts:
            raise ValueError("Round checkpoint Task artifacts changed while copying")

    snapshot = TaskAgentState(
        generation=task.generation,
        model_ref=task.model_ref,
        harness_path=str((published_target / "harness_manifest.json").resolve()),
        artifacts=ArtifactState(
            directory=str((published_target / "artifacts").resolve()) if artifacts_path else None,
            manifest=expected_artifacts,
        ),
        checkpoint_path=task.checkpoint_path,
        checkpoint_manifest=list(task.checkpoint_manifest),
    )
    if digest(manifest_path) != digest(Path(task.harness_path)):
        raise ValueError("Round checkpoint Task content differs from the committed Task")
    save_json(target / "state.json", snapshot)
    return asdict(snapshot), {
        "content_hash": content_identity(task),
        "harness_manifest": "task/harness_manifest.json",
        "harness_manifest_sha256": digest(manifest_path),
        "harness_bundle": "task/harness_bundle",
        "harness_bundle_sha256": source_manifest.bundle_sha256,
        "artifacts": "task/artifacts" if artifacts_path else None,
        "artifact_manifest": expected_artifacts,
        "model_checkpoint_path": task.checkpoint_path,
        "model_checkpoint_manifest": list(task.checkpoint_manifest),
    }


def _copy_meta(staging: Path, published: Path, meta: MetaAgentState) -> tuple[dict, dict]:
    target = staging / "meta"
    published_target = published / "meta"
    target.mkdir()
    skill_record = {"path": None, "sha256": None, "record_count": None}
    if meta.bundle_path:
        source_path = Path(meta.bundle_path)
        source = MetaHarnessBundle(
            source_path, strict_json((source_path / "manifest.json").read_text(encoding="utf-8"))
        ).verify()
        if source.hash != meta.bundle_hash:
            raise ValueError("Committed Meta state and Bundle identity differ")
        bundle_path = target / "bundle"
        shutil.copytree(source.path, bundle_path)
        copied = MetaHarnessBundle(
            bundle_path, strict_json((bundle_path / "manifest.json").read_text(encoding="utf-8"))
        ).verify()
        if copied.hash != source.hash:
            raise ValueError("Round checkpoint Meta Bundle differs from its source")
        snapshot = MetaAgentState(
            model_ref=meta.model_ref,
            harness_path=str((published_target / "bundle/instructions.md").resolve()),
            version=meta.version,
            bundle_hash=meta.bundle_hash,
            bundle_path=str((published_target / "bundle").resolve()),
        )
        principles = bundle_path / "principles.json"
        if principles.is_file():
            skills = target / "skills" / "principles.json"
            skills.parent.mkdir()
            shutil.copy2(principles, skills)
            library = strict_json(skills.read_text(encoding="utf-8"))
            skill_record = {
                "path": "meta/skills/principles.json",
                "sha256": digest(skills),
                "record_count": len(library["records"]),
            }
        bundle_record = {
            "path": "meta/bundle",
            "bundle_hash": copied.hash,
            "bundle_version": copied.version,
            "schema_version": copied.schema_version,
        }
    else:
        source = Path(meta.harness_path)
        if not _regular(source, directory=False):
            raise ValueError("Committed legacy Meta harness is missing or unsafe")
        harness = target / "harness.md"
        shutil.copy2(source, harness)
        snapshot = MetaAgentState(
            meta.model_ref, str((published_target / "harness.md").resolve()), meta.version)
        bundle_record = None
    save_json(target / "state.json", snapshot)
    return asdict(snapshot), {
        "bundle": bundle_record,
        "skills": skill_record,
        "state_content_hash": value_hash(asdict(meta)),
    }


def _expected_manifest(round_dir: Path, round_index: int, task: TaskAgentState,
                       meta: MetaAgentState, task_record: dict, meta_record: dict) -> dict:
    complete = round_dir / "complete.json"
    experience = round_dir / "experience.json"
    deployment = round_dir / "deployment.json"
    for path in (complete, experience, deployment):
        if not _regular(path, directory=False):
            raise ValueError("Round checkpoint requires committed round evidence: " + str(path))
    return {
        "schema_version": SCHEMA,
        "status": "committed",
        "round_index": round_index,
        "task_version": task.version,
        "meta_version": meta.version,
        "round_complete_sha256": digest(complete),
        "experience_sha256": digest(experience),
        "deployment_sha256": digest(deployment),
        "task": task_record,
        "meta": meta_record,
    }


def _load_snapshot(checkpoint: Path) -> tuple[TaskAgentState, MetaAgentState, dict]:
    if not _regular(checkpoint, directory=True):
        raise ValueError("Round checkpoint is missing or unsafe")
    manifest_path = checkpoint / "manifest.json"
    if not _regular(manifest_path, directory=False):
        raise ValueError("Round checkpoint manifest is missing or unsafe")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_value = json.loads((checkpoint / "task/state.json").read_text(encoding="utf-8"))
    task_value["artifacts"] = ArtifactState(**task_value.get("artifacts", {}))
    meta_value = json.loads((checkpoint / "meta/state.json").read_text(encoding="utf-8"))
    return TaskAgentState(**task_value), MetaAgentState(**meta_value), manifest


def load_round_checkpoint(round_dir: str | Path, round_index: int) -> tuple[TaskAgentState, MetaAgentState]:
    """Load independent Task and Meta copies from a committed round boundary."""

    root = Path(round_dir)
    checkpoint = root / DIRECTORY
    task, meta, manifest = _load_snapshot(checkpoint)
    task_record = manifest["task"]
    meta_record = manifest["meta"]
    expected = _expected_manifest(root, round_index, task, meta, task_record, meta_record)
    if manifest != expected:
        raise ValueError("Round checkpoint no longer matches committed round evidence")
    if (content_identity(task) != task_record["content_hash"]
            or digest(Path(task.harness_path)) != task_record["harness_manifest_sha256"]
            or task.checkpoint_path != task_record["model_checkpoint_path"]
            or task.checkpoint_manifest != task_record["model_checkpoint_manifest"]):
        raise ValueError("Round checkpoint Task snapshot failed integrity verification")
    harness = load_manifest(task.harness_path)
    if harness.bundle_sha256 != task_record["harness_bundle_sha256"]:
        raise ValueError("Round checkpoint Harness bundle identity changed")
    materialized = Path(task.harness_path).parent / "harness_bundle"
    if harness.files != type(harness).from_directory(
            materialized, harness_name=harness.harness_name).files:
        raise ValueError("Round checkpoint materialized Harness bundle changed")
    if artifact_manifest(task.artifacts.directory) != task_record["artifact_manifest"]:
        raise ValueError("Round checkpoint Task artifacts changed")
    bundle_record = meta_record["bundle"]
    skills_record = meta_record["skills"]
    if bundle_record:
        bundle_path = Path(meta.bundle_path)
        bundle = MetaHarnessBundle(
            bundle_path, strict_json((bundle_path / "manifest.json").read_text(encoding="utf-8"))
        ).verify()
        if bundle.hash != meta.bundle_hash or bundle.hash != bundle_record["bundle_hash"]:
            raise ValueError("Round checkpoint Meta Bundle identity changed")
        principles = bundle_path / "principles.json"
        if skills_record["path"]:
            skills = checkpoint / skills_record["path"]
            if (not _regular(skills, directory=False) or digest(skills) != skills_record["sha256"]
                    or strict_json(skills.read_text(encoding="utf-8")) != strict_json(principles.read_text(encoding="utf-8"))):
                raise ValueError("Round checkpoint Meta skill library changed")
        elif principles.exists():
            raise ValueError("Round checkpoint omitted the Meta skill library")
    elif not _regular(Path(meta.harness_path), directory=False):
        raise ValueError("Round checkpoint legacy Meta harness is missing")
    context_record = meta_record.get("context")
    if context_record and digest(checkpoint / context_record["path"]) != context_record["sha256"]:
        raise ValueError("Round checkpoint Meta context changed")
    return task, meta


def verify_round_checkpoint(round_dir: str | Path, round_index: int,
                            task: TaskAgentState, meta: MetaAgentState) -> dict:
    """Verify a committed snapshot against the live round boundary."""

    root = Path(round_dir)
    checkpoint = root / DIRECTORY
    saved_task, saved_meta, manifest = _load_snapshot(checkpoint)
    if set(manifest) != {
        "schema_version", "status", "round_index", "task_version", "meta_version",
        "round_complete_sha256", "experience_sha256", "deployment_sha256", "task", "meta",
    }:
        raise ValueError("Round checkpoint manifest has an unexpected schema")
    task_record = manifest["task"]
    meta_record = manifest["meta"]
    expected = _expected_manifest(root, round_index, task, meta, task_record, meta_record)
    if manifest != expected:
        raise ValueError("Round checkpoint no longer matches committed round evidence")
    if (saved_task.generation != task.generation or saved_task.model_ref != task.model_ref
            or saved_task.checkpoint_path != task.checkpoint_path
            or saved_task.checkpoint_manifest != task.checkpoint_manifest
            or saved_task.artifacts.manifest != artifact_manifest(task.artifacts.directory)
            or content_identity(saved_task) != content_identity(task)
            or task_record.get("content_hash") != content_identity(task)
            or digest(Path(saved_task.harness_path)) != task_record.get("harness_manifest_sha256")):
        raise ValueError("Round checkpoint Task snapshot failed integrity verification")
    harness = load_manifest(saved_task.harness_path)
    if harness.bundle_sha256 != task_record.get("harness_bundle_sha256"):
        raise ValueError("Round checkpoint Harness bundle identity changed")
    materialized = Path(saved_task.harness_path).parent / "harness_bundle"
    if harness.files != type(harness).from_directory(
            materialized, harness_name=harness.harness_name).files:
        raise ValueError("Round checkpoint materialized Harness bundle changed")
    if artifact_manifest(saved_task.artifacts.directory) != task_record.get("artifact_manifest"):
        raise ValueError("Round checkpoint Task artifacts changed")
    if (saved_meta.model_ref != meta.model_ref or saved_meta.version != meta.version
            or saved_meta.bundle_hash != meta.bundle_hash
            or meta_record.get("state_content_hash") != value_hash(asdict(meta))):
        raise ValueError("Round checkpoint Meta state differs from the committed Meta")
    bundle_record = meta_record.get("bundle")
    skills_record = meta_record.get("skills", {})
    if bundle_record:
        bundle_path = Path(saved_meta.bundle_path)
        bundle = MetaHarnessBundle(
            bundle_path, strict_json((bundle_path / "manifest.json").read_text(encoding="utf-8"))
        ).verify()
        if bundle.hash != meta.bundle_hash or bundle_record.get("bundle_hash") != bundle.hash:
            raise ValueError("Round checkpoint Meta Bundle identity changed")
        principles = bundle_path / "principles.json"
        if skills_record.get("path"):
            skills = checkpoint / skills_record["path"]
            if (not _regular(skills, directory=False) or digest(skills) != skills_record.get("sha256")
                    or strict_json(skills.read_text(encoding="utf-8")) != strict_json(principles.read_text(encoding="utf-8"))):
                raise ValueError("Round checkpoint Meta skill library changed")
        elif principles.exists():
            raise ValueError("Round checkpoint omitted the Meta skill library")
    elif saved_meta.bundle_path is not None or saved_meta.harness_path is None:
        raise ValueError("Round checkpoint legacy Meta snapshot is invalid")
    context_record = meta_record.get("context")
    if context_record and digest(checkpoint / context_record["path"]) != context_record["sha256"]:
        raise ValueError("Round checkpoint Meta context changed")
    return manifest


def commit_round_checkpoint(round_dir: str | Path, round_index: int,
                            task: TaskAgentState, meta: MetaAgentState) -> dict:
    """Atomically publish one round's executable Task and Meta/skill snapshots."""

    root = Path(round_dir)
    checkpoint = root / DIRECTORY
    if checkpoint.exists() or checkpoint.is_symlink():
        return verify_round_checkpoint(root, round_index, task, meta)
    staging = root / ("." + DIRECTORY + "." + uuid.uuid4().hex + ".tmp")
    staging.mkdir()
    try:
        _, task_record = _copy_task(staging, checkpoint, task)
        _, meta_record = _copy_meta(staging, checkpoint, meta)
        source_context = root / "meta_context.json"
        if source_context.is_file():
            saved_context = staging / "meta/context.json"
            shutil.copy2(source_context, saved_context)
            meta_record["context"] = {
                "path": "meta/context.json", "sha256": digest(saved_context)}
        manifest = _expected_manifest(root, round_index, task, meta, task_record, meta_record)
        save_json(staging / "manifest.json", manifest)
        os.replace(staging, checkpoint)
        return verify_round_checkpoint(root, round_index, task, meta)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


__all__ = ["SCHEMA", "commit_round_checkpoint", "load_round_checkpoint", "verify_round_checkpoint"]
