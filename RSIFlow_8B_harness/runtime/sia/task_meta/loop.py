"""The ordered Task -> evaluation -> Meta learning -> routing state machine."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sia.task_meta.storage import artifact_manifest, digest, manifest_diff, save_json
from sia.task_meta.types import (
    DecisionConstraintError,
    MetaAgentState,
    MetaHarnessUpdate,
    TaskUpdateAction,
)


@dataclass
class BudgetManager:
    max_generations: int
    max_wall_time: float | None = None
    started: float = field(default_factory=time.perf_counter)
    wall_started: float | None = None

    def __post_init__(self):
        if self.max_generations < 1:
            raise ValueError("max_generations must be at least 1")
        if self.max_wall_time is not None and (not math.isfinite(self.max_wall_time) or self.max_wall_time <= 0):
            raise ValueError("max_wall_time must be positive and finite")

    def stop_after(self, completed: int) -> bool:
        return completed >= self.max_generations or (
            self.max_wall_time is not None and self.elapsed() >= self.max_wall_time
        )

    def elapsed(self) -> float:
        return max(0.0, time.time() - self.wall_started) if self.wall_started is not None else time.perf_counter() - self.started

    def persist(self, path, *, resume):
        """A durable run keeps its original UTC deadline, including downtime."""
        protocol = {"max_generations": self.max_generations, "max_wall_time_seconds": self.max_wall_time,
                    "downtime_policy": "included", "clock": "unix_utc"}
        if path.exists():
            if not resume:
                raise ValueError("Budget already exists; use resume")
            saved = json.loads(path.read_text(encoding="utf-8"))
            if any(saved.get(key) != value for key, value in protocol.items()):
                raise ValueError("Durable run budget cannot change during recovery")
            self.wall_started = saved["started_at_epoch_seconds"]
            if not isinstance(self.wall_started, (int, float)) or not math.isfinite(self.wall_started):
                raise ValueError("Invalid persisted budget start")
            expected = None if self.max_wall_time is None else self.wall_started + self.max_wall_time
            if saved.get("deadline_epoch_seconds") != expected:
                raise ValueError("Invalid persisted budget deadline")
        else:
            self.wall_started = time.time()
            save_json(path, {**protocol, "started_at_epoch_seconds": self.wall_started,
                             "deadline_epoch_seconds": None if self.max_wall_time is None
                             else self.wall_started + self.max_wall_time})


def _validate_decision(decision):
    if decision.target_components != [decision.action]:
        raise DecisionConstraintError("target_components must contain exactly the selected action")
    if not decision.requested_changes:
        raise DecisionConstraintError("requested_changes must declare the selected component intervention")
    ids = [change.id for change in decision.requested_changes]
    if len(ids) != len(set(ids)):
        raise DecisionConstraintError("requested change IDs must be unique")
    if any(change.component != decision.action for change in decision.requested_changes):
        raise DecisionConstraintError("Every requested change must belong to the one selected action")
    if decision.action == TaskUpdateAction.HARNESS:
        if len(decision.requested_changes) != 1:
            raise DecisionConstraintError("HARNESS routing declares one complete HarnessForge bundle production")
        change = decision.requested_changes[0]
        if (change.operation, change.target) != ("produce_harness", "harness_bundle"):
            raise DecisionConstraintError(
                "HARNESS routing must use operation=produce_harness, target=harness_bundle, and no preselected module"
            )


def _intervention_diff(before, after):
    return {"model": {"before": before.model_ref, "after": after.model_ref,
                       "checkpoint_before": before.checkpoint_path, "checkpoint_after": after.checkpoint_path},
            "harness": {"before_sha256": digest(Path(before.harness_path)),
                         "after_sha256": digest(Path(after.harness_path))},
            "artifacts": manifest_diff(before.artifacts.manifest, artifact_manifest(after.artifacts.directory))}


def _accept_meta_update(run_dir, meta, update, experiences, path, phase, handler=None, resume=False):
    # Validation is structural only. No quality scoring, selection, or rollback.
    update = MetaHarnessUpdate.model_validate(update)
    if not update.harness.strip():
        raise ValueError("Meta harness cannot be blank")
    from sia.task_meta.durable import value_hash
    if meta.bundle_hash:
        from sia.task_meta.meta import validate_meta_candidate
        validate_meta_candidate(update, meta)
        if phase == "learn_from_experience" and (len(experiences) != 1 or update.experience_id != experiences[0].experience_id):
            raise ValueError("Meta self-update must bind the actual re-evaluated experience")
    binding = {"prior_state": asdict(meta), "prior_harness_sha256": digest(Path(meta.harness_path)),
               "update": update.model_dump(mode="json"), "phase": phase,
               "experience_generations": [e.generation for e in experiences]}
    input_hash = value_hash(binding)
    if resume and path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if not cached.get("accepted_state") or cached.get("input_hash") != input_hash:
            raise ValueError("Recovered Meta update does not match its acceptance receipt")
        accepted = MetaAgentState(**cached["accepted_state"])
        changed = cached.get("version_changed", True)
        if (accepted.version != meta.version + int(changed) or accepted.model_ref != meta.model_ref
                or digest(Path(accepted.harness_path)) != cached.get("accepted_harness_sha256")
                or value_hash(cached["accepted_state"]) != cached.get("accepted_state_hash")):
            raise ValueError("Accepted Meta state integrity check failed")
        if accepted.bundle_path:
            from sia.task_meta.meta_harness.bundle import MetaHarnessBundle
            bundle_path = Path(accepted.bundle_path)
            bundle = MetaHarnessBundle(bundle_path, json.loads((bundle_path / "manifest.json").read_text())).verify()
            if bundle.hash != accepted.bundle_hash:
                raise ValueError("Accepted Meta Bundle identity changed")
        return accepted
    old_version = meta.version
    old_hash = meta.bundle_hash
    old_model = meta.model_ref
    old_harness = Path(meta.harness_path).read_text(encoding="utf-8")
    meta = copy.deepcopy(meta)
    if handler:
        meta = handler(meta, update)
    if meta.model_ref != old_model:
        raise ValueError("Meta update cannot change the frozen model identity")
    changed = meta.bundle_hash != old_hash if old_hash else old_harness.strip() != update.harness.strip()
    if update.status == "NO_CHANGE" and changed:
        raise ValueError("NO_CHANGE cannot advance the Meta version")
    if not old_hash and update.bundle_files:
        raise ValueError("Legacy Markdown Meta requires explicit Bundle migration for executable files")
    if changed:
        meta.version += 1
        meta.harness_path = str(run_dir / "meta" / f"harness_v{meta.version}.md")
        target = Path(meta.harness_path)
        content = update.harness + "\n"
        if target.exists() and target.read_text(encoding="utf-8") != content:
            raise ValueError("Refusing to overwrite an existing Meta snapshot")
        target.write_text(content, encoding="utf-8")
    save_json(path, {"phase": phase, "old_version": old_version, "new_version": meta.version,
                     "version_changed": changed, "change_event": "UPDATED" if changed else "NO_CHANGE",
                     "experience_generations": [e.generation for e in experiences],
                     "input_hash": input_hash, "accepted_harness_sha256": digest(Path(meta.harness_path)),
                     "accepted_state_hash": value_hash(asdict(meta)), "accepted_state": asdict(meta), **update.model_dump()})
    print('[meta-update] ' + json.dumps({'phase': phase, 'status': update.status,
        'G_before': old_version, 'G_after': meta.version, 'bundle_before': old_hash, 'bundle_after': meta.bundle_hash,
        'changed_rules': update.changed_rules, 'rationale': update.rationale,
        'principle_operations': update.five_stage.model_dump(mode='json')['principle_operations'] if update.five_stage else [],
        'harness_bindings': update.five_stage.model_dump(mode='json')['harness_bindings'] if update.five_stage else []}, ensure_ascii=False), flush=True)
    return meta


def _assert_single_component(old, new, action, before_harness, before_artifacts):
    if digest(Path(old.harness_path)) != before_harness or artifact_manifest(old.artifacts.directory) != before_artifacts:
        raise ValueError("Updater mutated an immutable previous generation")
    changed = {
        TaskUpdateAction.HARNESS: digest(Path(new.harness_path)) != before_harness,
        TaskUpdateAction.MODEL: (new.model_ref, new.checkpoint_path, new.checkpoint_manifest)
        != (old.model_ref, old.checkpoint_path, old.checkpoint_manifest),
        TaskUpdateAction.ARTIFACTS: artifact_manifest(new.artifacts.directory) != before_artifacts,
    }
    if any(value for component, value in changed.items() if component != action):
        raise ValueError("Updater changed a component outside the selected action")
    if not changed[action]:
        raise ValueError(f"{action.value} updater produced no actual change")
