"""Apply one declared component change and record what was actually changed."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from pydantic import BaseModel, ConfigDict, Field

from sia.task_meta.storage import artifact_manifest, checkpoint_manifest, clone_task, save_json
from sia.task_meta.types import ArtifactState, DecisionConstraintError, TaskUpdate, TaskUpdateAction


class ArtifactEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)
    content: str | None


class ArtifactChanges(BaseModel):
    model_config = ConfigDict(extra="forbid")
    edits: list[ArtifactEdit] = Field(min_length=1)
    summary: str = Field(min_length=1)


class ModelRequestPlan(BaseModel):
    """G organizes the request; every executable training parameter stays fixed."""
    model_config = ConfigDict(extra="forbid")
    objective: str = Field(min_length=1, max_length=8000)
    preserve_behaviors: list[str] = Field(max_length=20)
    rationale: str = Field(min_length=1, max_length=8000)
    training_method: Literal["positive_sft_lora"] = "positive_sft_lora"
    target: Literal["current_checkpoint"] = "current_checkpoint"


def _cost(start):
    return {"wall_time_seconds": time.monotonic() - start, "api_cost_usd": None, "gpu_hours": None}


def _text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def positive_sft_rows(trajectories):
    """Only complete recorded text dialogues with finite positive terminal rewards."""
    rows = []
    for row in trajectories:
        reward = row.get("terminal_reward")
        messages = row.get("messages")
        if (isinstance(reward, bool) or not isinstance(reward, (int, float))
                or not math.isfinite(reward) or reward <= 0):
            continue
        if (not isinstance(messages, list) or len(messages) < 2
                or not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant"):
            continue
        if not all(isinstance(m, dict) and m.get("role") in {"user", "assistant", "system"}
                   and isinstance(m.get("content"), str) and m["content"].strip() for m in messages):
            continue
        rows.append({"question_id": row["question_id"], "rollout_id": row["rollout_id"],
                     "messages": messages, "terminal_reward": reward})
    return rows


def updater_capabilities(task_state, evaluation, trainer_configured=True, *, sft_profile="legacy_gpqa"):
    from sia.task_meta.sft import select_positive_rows
    from sia.task_meta.seed import seed_capabilities
    positives = select_positive_rows(evaluation.trajectories, profile=sft_profile)
    checkpoint = task_state.checkpoint_path or task_state.model_ref
    reasons = []
    if not positives:
        reasons.append("No complete positive terminal-reward text dialogue is available for SFT")
    if not trainer_configured:
        reasons.append("No fixed SFT trainer is configured")
    if not Path(checkpoint).is_dir():
        reasons.append("Current Task checkpoint is not an available local directory")
    capabilities = {
        "HARNESS": seed_capabilities(task_state.harness_path),
        "ARTIFACTS": {"available": True,
                      "operations": [{"operation": "write_asset", "target": "exact relative asset path"},
                                     {"operation": "delete_asset", "target": "exact existing relative asset path"}],
                      "constraints": ["Only current output assets may be added, edited or deleted",
                                      "No logs, traces, score files, answer tables, harness or checkpoint changes",
                                      "List every intended path in requested_changes; no undeclared extra edits"]},
        "MODEL": {"available": not reasons, "reason": "; ".join(reasons) or None,
                  "positive_samples": len(positives), "current_checkpoint": checkpoint,
                  "length_eligibility": "Checked per complete recorded call before training; overlength calls are excluded and audited, not truncated. One overlength call does not reject other eligible calls.",
                  "operations": [{"operation": "sft", "target": "current_checkpoint"}],
                  "constraints": ["Fixed positive-reward LoRA SFT backend only; no generated training code",
                                  "Train from the current checkpoint and save distinct verified weights",
                                  "No harness or artifact changes; no GRPO, invented labels or empty training"]},
    }
    return capabilities


def validate_decision_targets(decision, action):
    """Reject partial execution of a cross-component or unsupported decision."""
    if decision.action != action or decision.target_components != [action]:
        raise DecisionConstraintError(f"{action.value} updater requires exactly target_components=[{action.value}]")
    requested = decision.requested_changes
    if not requested or any(change.component != action for change in requested):
        raise DecisionConstraintError("Every requested change must name the single selected component")
    if len({change.id for change in requested}) != len(requested):
        raise DecisionConstraintError("Requested change IDs must be unique")
    if len({change.target for change in requested}) != len(requested):
        raise DecisionConstraintError("Each requested target must appear once")
    for change in requested:
        if action == TaskUpdateAction.HARNESS:
            valid = (len(requested) == 1 and change.operation == "produce_harness"
                     and change.target == "harness_bundle")
        elif action == TaskUpdateAction.ARTIFACTS:
            valid = change.operation in {"write_asset", "delete_asset"}
        else:
            valid = change.operation == "sft" and change.target == "current_checkpoint" and len(requested) == 1
        if not valid:
            raise DecisionConstraintError(f"Unsupported {action.value} request: {change.operation} {change.target}")
    return requested


def _audit(decision, applied, files, checks):
    return {"requested_changes": [change.model_dump(mode="json") for change in decision.requested_changes],
            "applied_changes": applied, "unapplied_changes": [], "files": files, "checks": checks,
            "semantic_status": "unverified"}


def artifact_path(root: Path, name: str) -> Path:
    path = PurePosixPath(name)
    forbidden = {"agent_execution", "agent_execution.json", "results.json", "results", "target_agent.py",
                 "target_config.json", "task_state.json", "meta_decision.json", "artifact_manifest.json"}
    if (not name or "\\" in name or ":" in name or path.is_absolute() or ".." in path.parts
            or not path.parts or path.as_posix() != name
            or any(p in forbidden or p.startswith(".") or p.endswith(".log") for p in path.parts)):
        raise ValueError(f"Not a relative reusable artifact path: {name!r}")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes its workspace")
    if any(p.is_symlink() for p in [target, *target.parents] if p != root.parent):
        raise ValueError("Artifact path contains a symlink")
    return target


class ArtifactUpdater:
    def __init__(self, client):
        self.client = client

    def apply(self, task_state, decision, context):
        started = time.monotonic()
        requested = validate_decision_targets(decision, TaskUpdateAction.ARTIFACTS)
        root = context.directory / "artifacts"
        files = {}
        if task_state.artifacts.directory:
            source = Path(task_state.artifacts.directory)
            artifact_manifest(source)
            files = {p.relative_to(source).as_posix(): p.read_text(encoding="utf-8") for p in source.rglob("*") if p.is_file()}
        try:
            for change in requested:
                artifact_path(root, change.target)
                if change.operation == "delete_asset" and change.target not in files:
                    raise ValueError(f"Cannot delete absent asset: {change.target}")
        except ValueError as exc:
            raise DecisionConstraintError(str(exc)) from exc
        prompt = ("Implement exactly the requested ARTIFACTS operations. Return one edit per exact requested path; "
                  "null content means delete_asset, nonnull text means write_asset. Do not edit undeclared paths. "
                  "Assets are general solvers, domain knowledge, scripts, templates or configuration contents. "
                  "No traces, logs, evaluator results, answer lookup tables or Meta/Harness files. Asset text is "
                  "available to next-round Task hooks; it is not automatically marked verified knowledge. "
                  "Keep edited asset content concise. Natural-language effects remain unverified.\nDecision:\n"
                  + decision.model_dump_json() + "\nCurrent output assets (intervention base):\n"
                  + json.dumps(files, ensure_ascii=False) + "\nEvidence:\n"
                  + json.dumps(context.observation.success_examples + context.observation.failure_examples, ensure_ascii=False))
        def candidate(output):
            output = ArtifactChanges.model_validate(output)
            try:
                names = [edit.path for edit in output.edits]
                expected = {change.target: change for change in requested}
                if len(names) != len(set(names)) or set(names) != set(expected):
                    raise ValueError(f"Requested asset paths {sorted(expected)}; returned {sorted(names)}")
                applied, changed_files = [], []
                for edit in output.edits:
                    artifact_path(root, edit.path)
                    change = expected[edit.path]
                    if (edit.content is None) != (change.operation == "delete_asset"):
                        raise ValueError(f"Returned edit does not implement {change.operation}: {edit.path}")
                    old_content = files.get(edit.path)
                    if old_content == edit.content:
                        raise ValueError(f"Requested asset did not change: {edit.path}")
                    before_hash = _text_hash(old_content) if old_content is not None else None
                    after_hash = _text_hash(edit.content) if edit.content is not None else None
                    applied.append({"id": change.id, "operation": change.operation, "target": edit.path,
                                    "before_sha256": before_hash, "after_sha256": after_hash, "semantic_status": "unverified"})
                    changed_files.append({"path": "artifacts/" + edit.path, "before_sha256": before_hash,
                                          "after_sha256": after_hash})
                final_paths = set(files)
                for edit in output.edits:
                    if edit.content is None:
                        final_paths.remove(edit.path)
                    else:
                        final_paths.add(edit.path)
                for name in final_paths:
                    if any(parent.as_posix() in final_paths for parent in PurePosixPath(name).parents):
                        raise ValueError(f"Asset file/directory collision: {name}")
                # Reject shape-changing patches whose execution order could destroy a
                # file or directory before another operation is applied.
                for edit in output.edits:
                    if edit.content is not None and (
                        any(name.startswith(edit.path + "/") for name in files)
                        or any(edit.path.startswith(name + "/") for name in files)
                    ):
                        raise ValueError(f"Asset write crosses an existing file/directory boundary: {edit.path}")
            except ValueError as exc:
                raise DecisionConstraintError(str(exc)) from exc
            return applied, changed_files

        from sia.task_meta.meta import evolution_kwargs
        if getattr(self.client, "supports_evolution", False):
            prompt = "Return all and only the declared current-output asset edits. Null content means deletion; preserve every other Task component."
        output = self.client.complete(prompt, ArtifactChanges, meta_state=context.meta_state,
            operation="artifact_patch", decision_id=decision.decision_id,
            **evolution_kwargs(self.client, context, task_state, decision, files, candidate))
        applied, changed_files = candidate(output)
        new_state = clone_task(task_state, context.generation, context.directory)
        root.mkdir(exist_ok=True)
        for edit in output.edits:
            target = artifact_path(root, edit.path)
            if edit.content is None:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(edit.content, encoding="utf-8")
        new_state.artifacts = ArtifactState(str(root), artifact_manifest(root))
        return new_state, TaskUpdate(TaskUpdateAction.ARTIFACTS, output.summary,
                                     {"edits": output.model_dump()["edits"]}, _cost(started),
                                     **_audit(decision, applied, changed_files,
                                              [{"name": "all_and_only_requested_asset_operations_applied", "passed": True}]))


def served_checkpoint_binding(provider, model_ref):
    """Read trusted loopback serving state, without exposing its API key to proxies."""
    base_url = provider.base_url.rstrip("/")
    parsed = urlsplit(base_url)
    if (parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("MODEL checkpoint verification requires a loopback model endpoint")
    url = base_url.removesuffix("/v1") + "/health"
    request = Request(url, headers={"Authorization": "Bearer " + os.environ.get(provider.api_key_env, "local")})
    with build_opener(ProxyHandler({})).open(request, timeout=30) as response:
        health = json.load(response)
    binding = health.get("bindings", {}).get(model_ref)
    if not health.get("ready") or not binding:
        raise RuntimeError(f"Model service has no ready checkpoint binding for {model_ref}")
    return binding


class ModelUpdater:
    """Execute the fixed real SFT backend directly from a structured training request."""

    def __init__(self, client, task_files, provider, trainer_command=None, training_sandbox="sandboxfusion", timeout=3600,
                 *, sft_profile="legacy_gpqa", supervision="final_assistant", training=None, training_gpu=None):
        self.provider = provider
        self.client = client
        self.trainer_command = trainer_command
        self.timeout = timeout
        self.sft_profile, self.supervision, self.training = sft_profile, supervision, training or {}
        self.training_gpu = training_gpu

    def apply(self, task_state, decision, context):
        if self.training_gpu is not None:
            from sia.task_meta.resources import training_gpu_lease
            with training_gpu_lease(self.training_gpu):
                return self._apply(task_state, decision, context)
        return self._apply(task_state, decision, context)

    def _apply(self, task_state, decision, context):
        started = time.monotonic()
        requested = validate_decision_targets(decision, TaskUpdateAction.MODEL)
        from sia.task_meta.sft import select_positive_rows
        from sia.task_meta.round_evolution import select_sft
        positives = select_sft(self,task_state,context)
        if not positives:
            raise DecisionConstraintError("MODEL/SFT unavailable: no complete positive terminal-reward text dialogues")
        if not self.trainer_command:
            raise DecisionConstraintError("MODEL/SFT unavailable: no fixed trainer is configured")
        old_checkpoint = Path(task_state.checkpoint_path or task_state.model_ref).resolve()
        if not old_checkpoint.is_dir():
            raise DecisionConstraintError("MODEL/SFT requires the current local immutable Task checkpoint")
        before_weights = checkpoint_manifest(old_checkpoint)
        if task_state.checkpoint_manifest and before_weights != task_state.checkpoint_manifest:
            raise ValueError("Current checkpoint changed since its recorded model version")
        # Validate complete supervised examples before MODEL planning or stopping services.
        from transformers import AutoTokenizer
        from sia.task_meta.sft import select_length_eligible
        tokenizer = AutoTokenizer.from_pretrained(old_checkpoint, local_files_only=True, trust_remote_code=False)
        try:
            positives, eligibility = select_length_eligible(tokenizer, positives,
                self.training.get('max_length', 8192), supervision=self.supervision)
        except ValueError as exc:
            raise DecisionConstraintError('MODEL/SFT unavailable before any training: ' + str(exc)) from exc
        save_json(context.directory / 'sft_eligibility.json', eligibility)
        if not positives:
            raise DecisionConstraintError('MODEL/SFT unavailable: no complete positive calls fit the fixed max_length; see sft_eligibility.json')
        minimum = self.training.get('round_protocol',{}).get('minimum_sft_samples',1)
        if len(positives) < minimum:
            raise DecisionConstraintError('MODEL/SFT unavailable: eligible examples below fixed minimum')
        lengths = eligibility['eligible_token_lengths']
        meta_plan = None
        if getattr(self.client, "supports_evolution", False):
            from sia.task_meta.meta import evolution_kwargs
            fixed_request = {"checkpoint_path": str(old_checkpoint), "positive_samples": len(positives),
                             "training_method": "positive_sft_lora", "training": self.training,
                             "sft_profile": self.sft_profile, "supervision": self.supervision,
                             "positive_token_lengths": lengths, "length_eligibility": eligibility}
            plan = self.client.complete(
                "Organize this MODEL request. Preserve the fixed SFT method, checkpoint and training parameters.",
                ModelRequestPlan, meta_state=context.meta_state, operation="model_request", decision_id=decision.decision_id,
                **evolution_kwargs(self.client, context, task_state, decision,
                                   {"fixed_training_request.json": json.dumps(fixed_request)}, ModelRequestPlan.model_validate))
            meta_plan = ModelRequestPlan.model_validate(plan).model_dump(mode="json")
        # A continued run may read H from an immutable parent segment. Training
        # outputs belong to this intervention's successor, never that parent.
        request_dir = Path(context.directory) / "model_update"
        command = [part.replace("{request_dir}", str(request_dir)) for part in self.trainer_command]
        if any("{script}" in part for part in command):
            raise DecisionConstraintError("Fixed SFT trainer_command must not depend on generated train.py")
        request_dir.mkdir(parents=True, exist_ok=False)
        rows = [{"question_id": row["question_id"], "rollout_id": row["rollout_id"],
                 "messages": row.get("messages", []), "terminal_reward": row["terminal_reward"]}
                for row in context.evaluation.trajectories]
        if self.sft_profile == "multidomain":
            rows = list(context.evaluation.trajectories)
        (request_dir / "rollouts.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        (request_dir / "sft_positive.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in positives), encoding="utf-8")
        save_json(request_dir / "training_request.json", {
            "model_ref": task_state.model_ref, "checkpoint_path": str(old_checkpoint),
            "decision": decision.model_dump(mode="json"), "decision_id": decision.decision_id,
            "meta_harness_version": context.meta_state.version if context.meta_state else None,
            "reward_source": "trusted evaluator terminal rewards", "rollouts": "rollouts.jsonl",
            "sft_positive": "sft_positive.jsonl", "group_key": "question_id", "samples": len(rows),
            "positive_samples": len(positives), "training_method": "positive_sft_lora", "status": "prepared",
            "generated_training_code_required": False,
            "sft_profile": self.sft_profile, "supervision": self.supervision, "training": self.training,
            "meta_harness_hash": context.meta_state.bundle_hash if context.meta_state else None,
            "meta_request_plan": meta_plan, "length_eligibility": eligibility,
            "source_binding": json.loads((context.directory/"sft_source_binding.json").read_text()) if self.training.get("round_protocol") else None,
        })
        save_json(request_dir / "checkpoint_before.json", {"path": str(old_checkpoint), "weights": before_weights})
        with (request_dir / "train_stdout.log").open("w", encoding="utf-8") as log:
            environment = {k: v for k, v in os.environ.items()
                           if not any(word in k.upper() for word in ("KEY", "TOKEN", "PASSWORD", "SECRET", "AUTH"))}
            if self.training_gpu is not None:
                environment["CUDA_VISIBLE_DEVICES"] = str(self.training_gpu)
            subprocess.run(command, cwd=request_dir, check=True, stdout=log, stderr=subprocess.STDOUT,
                           timeout=self.timeout, env=environment)
        manifest = json.loads((request_dir / "checkpoint.json").read_text(encoding="utf-8"))
        model_ref = manifest["model_ref"]
        checkpoint = Path(manifest["checkpoint_path"]).resolve()
        if not model_ref or model_ref == task_state.model_ref or not checkpoint.is_dir() or checkpoint == old_checkpoint:
            raise ValueError("Trainer must return a different model_ref and a real checkpoint directory")
        if checkpoint_manifest(old_checkpoint) != before_weights:
            raise ValueError("Trainer mutated the immutable base checkpoint")
        new_weights = checkpoint_manifest(checkpoint)
        if {weight["sha256"] for weight in new_weights} == {weight["sha256"] for weight in before_weights}:
            raise ValueError("MODEL update only renamed/copied unchanged checkpoint weights")
        binding = served_checkpoint_binding(self.provider, model_ref)
        if Path(binding.get("checkpoint_path", "")).resolve() != checkpoint or binding.get("weights") != new_weights:
            raise ValueError("Served checkpoint binding does not match the trained checkpoint path and hashes")
        new_state = clone_task(task_state, context.generation, context.directory)
        new_state.model_ref = model_ref
        new_state.checkpoint_path = str(checkpoint)
        new_state.checkpoint_manifest = new_weights
        manifest.update({"weights": new_weights, "checkpoint_before": str(old_checkpoint),
                         "served_binding": binding, "training_request": str(request_dir / "training_request.json")})
        save_json(request_dir / "checkpoint_verified.json", manifest)
        applied = [{"id": requested[0].id, "operation": "sft", "target": "current_checkpoint",
                    "before_checkpoint": str(old_checkpoint), "after_checkpoint": str(checkpoint),
                    "before_model_ref": task_state.model_ref, "after_model_ref": model_ref,
                    "semantic_status": "unverified"}]
        files = [{"path": str(checkpoint / weight["path"]), "after_sha256": weight["sha256"]} for weight in new_weights]
        checks = [{"name": name, "passed": True} for name in
                  ["positive_terminal_reward_data", "current_checkpoint_training_start", "immutable_base_weights",
                   "new_weights_changed", "served_binding_matches_new_checkpoint"]]
        return new_state, TaskUpdate(TaskUpdateAction.MODEL, manifest.get("training_summary", "Applied real positive-reward SFT"),
                                     manifest, _cost(started), **_audit(decision, applied, files, checks))
