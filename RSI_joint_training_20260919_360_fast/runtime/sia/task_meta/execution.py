"""GPQA runtime adapter, retaining SIA's evaluator and execution-log formats."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

from sia.config import Config
from sia.layout import venv_python_path
from sia.task_meta.gpqa_target import safe_task_id
from sia.task_meta.storage import artifact_manifest, checkpoint_manifest, digest, save_json
from sia.task_meta.types import ArtifactState, EvaluationResult


def result_metrics(rows):
    attempts = len(rows)
    valid = sum(bool(row.get("valid_answer", row.get("model_answer") in {"A", "B", "C", "D"})) for row in rows)
    successes = sum(row["terminal_reward"] for row in rows)
    return {
        "success_rate": successes / attempts if attempts else 0.0,
        "successes": successes, "attempts": attempts,
        "valid_answers": valid, "valid_answer_rate": valid / attempts if attempts else 0.0,
        "parse_failures": sum(bool(row.get("parse_failure")) for row in rows),
        "output_truncations": sum(bool(row.get("output_truncated")) for row in rows),
        "api_errors": sum(bool(row.get("api_error")) for row in rows),
        "missing_trajectories": sum(bool(row.get("missing_trajectory")) for row in rows),
        "parse_failure_events": sum(row.get("parse_failure_count", int(bool(row.get("parse_failure")))) for row in rows),
        "output_truncation_events": sum(row.get("output_truncation_count", int(bool(row.get("output_truncated")))) for row in rows),
        "api_error_events": sum(row.get("api_error_count", int(bool(row.get("api_error")))) for row in rows),
    }


def rollout_artifact_diff(before, after):
    old, new = ({entry["path"]: entry["sha256"] for entry in manifest} for manifest in (before, after))
    return {
        "added": sorted(new.keys() - old.keys()),
        "modified": sorted(path for path in old.keys() & new.keys() if old[path] != new[path]),
        "unchanged_content_reproduced": sorted(path for path in old.keys() & new.keys() if old[path] == new[path]),
        "removed": sorted(old.keys() - new.keys()),
        "lifecycle_removed": sorted(old.keys() - new.keys()),
        "removal_reason": "Only assets produced in this execution become output; unproduced input assets expire.",
        "accidental_overwrites": [],
    }


def input_provenance(gen_dir, active, manifest):
    copied = gen_dir / "input_artifact_provenance.json"
    origin = active.parent / "artifact_provenance.json" if active else None
    source = copied if copied.exists() else origin
    known = json.loads(source.read_text(encoding="utf-8")) if source and source.exists() else []
    by_content = {(row["path"], row["sha256"]): row for row in known}
    rows = []
    for item in manifest:
        record = by_content.get((item["path"], item["sha256"]))
        rows.append(record or {
            **item, "question_id": None, "rollout_id": None, "terminal_reward": None,
            "production_method": "explicit_artifact_update_or_untracked_input",
            "knowledge_verified": False,
        })
    save_json(copied, rows)


def collect_generated_artifacts(work, destination, rows, rollout):
    """Register every current output exactly once, without inheriting input files."""
    produced = work / "artifacts_generated"
    manifest = artifact_manifest(produced)
    declarations = {}
    for row in rows:
        prefix = f"{safe_task_id(row['question_id'])}/rollout_{rollout}/"
        for item in row.get("generated_artifacts", []):
            name = item["path"]
            if name in declarations or not name.startswith(prefix) or any(part in {"", ".", ".."} for part in name.split("/")):
                raise ValueError("Generated artifact has duplicate or incorrect task/rollout ownership")
            declarations[name] = (row, item)
    if {item["path"] for item in manifest} != declarations.keys():
        raise ValueError("Generated artifact files must exactly match per-trajectory provenance declarations")
    provenance = []
    for item in manifest:
        relative = item["path"]
        row, declaration = declarations[relative]
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with (produced / relative).open("rb") as source, target.open("xb") as output:
            shutil.copyfileobj(source, output)
        provenance.append({
            **item, "question_id": row["question_id"], "rollout_id": rollout,
            "attempt": declaration.get("attempt", 0),
            "production_method": declaration.get("production_method", "task_generated"),
            "terminal_reward": row["terminal_reward"], "knowledge_verified": False,
            "source_path": str(produced / relative),
        })
    return provenance


def prepare_gpqa_task(source: Path, destination: Path, batch_size: int) -> dict:
    """Freeze one public/private subset before the first generation."""
    public_file = source / "data/public/diamond_questions.json"
    private_file = source / "data/private/diamond_questions.json"
    if not public_file.is_file() or not private_file.is_file():
        raise ValueError("This MVP execution adapter supports GPQA task directories; other tasks need an adapter")
    public = json.loads(public_file.read_text(encoding="utf-8"))
    if not 1 <= batch_size <= len(public):
        raise ValueError(f"batch_size must be between 1 and {len(public)}")
    public = public[:batch_size]
    ids = [q["id"] for q in public]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task IDs")
    private = [q for q in json.loads(private_file.read_text(encoding="utf-8")) if q["id"] in ids]
    if len(private) != len(public):
        raise ValueError("Public/private task IDs do not match")
    save_json(destination / "data/public/diamond_questions.json", public)
    save_json(destination / "data/private/diamond_questions.json", private)
    shutil.copy2(source / "data/public/evaluate.py", destination / "data/public/evaluate.py")
    if (source / "data/public/task.md").exists():
        shutil.copy2(source / "data/public/task.md", destination / "data/public/task.md")
    manifest = {"task_ids": ids, "batch_size": batch_size, "source": str(source.resolve()),
                "public_sha256": digest(destination / "data/public/diamond_questions.json"),
                "private_sha256": digest(destination / "data/private/diamond_questions.json"),
                "evaluator_sha256": digest(destination / "data/public/evaluate.py")}
    save_json(destination / "batch_manifest.json", manifest)
    return manifest


class SIAExecutor:
    def __init__(self, task_dir, venv_dir, target_provider, rollouts_per_task=8, temperature=0.7, max_tokens=128, seed=42):
        self.task_dir = Path(task_dir).resolve()
        self.venv_dir = venv_dir
        self.provider = target_provider
        self.rollouts = rollouts_per_task
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        if self.rollouts < 1 or self.max_tokens < 1 or self.temperature < 0:
            raise ValueError("Invalid rollout or generation settings")
        self.questions = json.loads((self.task_dir / "data/public/diamond_questions.json").read_text(encoding="utf-8"))
        self.input_hashes = {str(p): digest(p) for p in (self.task_dir / "data").rglob("*") if p.is_file()}

    def service_binding(self, task_state):
        if not task_state.checkpoint_path:
            return None  # Non-checkpoint unit-test adapters do not have a local GPU service.
        health_url = self.provider.base_url.rstrip("/").removesuffix("/v1") + "/health"
        with urlopen(health_url, timeout=10) as response:
            health = json.load(response)
        binding = health.get("bindings", {}).get(task_state.model_ref)
        if not health.get("ready") or not binding:
            raise ValueError("Inference service has no verified binding for the requested Task checkpoint")
        if Path(binding["checkpoint_path"]).resolve() != Path(task_state.checkpoint_path).resolve():
            raise ValueError("Inference service binding points to a different Task checkpoint")
        expected = task_state.checkpoint_manifest or checkpoint_manifest(task_state.checkpoint_path)
        if binding.get("weights") != expected:
            raise ValueError("Inference service loaded weights differ from the recorded Task checkpoint")
        return {**binding, "model_ref": task_state.model_ref, "device": health.get("device"),
                "visible_devices": health.get("visible_devices")}

    def execute(self, task_state, gen_dir):
        from sia.orchestrator import load_agent_execution, run_evaluation

        started = time.monotonic()
        if task_state.checkpoint_manifest and checkpoint_manifest(task_state.checkpoint_path) != task_state.checkpoint_manifest:
            raise ValueError("Model checkpoint content changed since its recorded generation")
        for path, expected in self.input_hashes.items():
            if digest(Path(path)) != expected:
                raise ValueError("Frozen task or evaluator changed across generations")
        gen_dir = Path(gen_dir)
        binding = self.service_binding(task_state)
        save_json(gen_dir / "service_binding.json", binding)
        trajectories, evaluator_results, provenance = [], [], []
        active = Path(task_state.artifacts.directory) if task_state.artifacts.directory else None
        active_hashes = artifact_manifest(active)
        snapshot = gen_dir / "artifacts_input"
        if snapshot.exists():
            raise ValueError("Generation input snapshot already exists")
        if active and active.exists():
            shutil.copytree(active, snapshot)
        else:
            snapshot.mkdir(parents=True)
        evaluated_state = copy.deepcopy(task_state)
        evaluated_state.artifacts = ArtifactState(str(snapshot), active_hashes)
        input_provenance(gen_dir, active, active_hashes)
        save_json(gen_dir / "evaluated_task_state.json", evaluated_state)
        save_json(gen_dir / "task_state.json", evaluated_state)
        destination = gen_dir / "artifacts_generated"
        destination.mkdir(exist_ok=False)
        for rollout in range(self.rollouts):
            work = Path(gen_dir) / "rollouts" / f"rollout_{rollout}"
            work.mkdir(parents=True, exist_ok=False)
            shutil.copytree(snapshot, work / "artifacts_input")
            runtime = {"model_ref": task_state.model_ref, "base_url": self.provider.base_url,
                       "api_key_env": self.provider.api_key_env, "timeout": self.provider.request_timeout_seconds,
                       "temperature": self.temperature, "max_tokens": self.max_tokens, "seed": self.seed + rollout,
                       "rollout_id": rollout, "service_binding": binding}
            save_json(work / "target_config.json", runtime)
            env = os.environ.copy()
            if self.provider.api_key_env != "OPENROUTER_API_KEY":
                env.pop("OPENROUTER_API_KEY", None)
            cmd = [venv_python_path(self.venv_dir), "-u", task_state.harness_path,
                   "--dataset_dir", str(self.task_dir / "data/public"), "--working_dir", str(work)]
            with (work / "target_agent_stdout.log").open("w", encoding="utf-8") as log:
                try:
                    process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                             timeout=3 * self.provider.request_timeout_seconds * len(self.questions) + 60)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(f"Rollout {rollout} timed out") from exc
            if process.returncode:
                raise RuntimeError(f"Target program failed in {work}; see target_agent_stdout.log")
            for path, expected in self.input_hashes.items():
                if digest(Path(path)) != expected:
                    raise ValueError("Target modified the fixed benchmark or evaluator")
            if (artifact_manifest(work / "artifacts_input") != active_hashes
                    or artifact_manifest(snapshot) != active_hashes):
                raise ValueError("Target mutated the frozen artifact input during rollout")
            evaluation = run_evaluation(str(work), str(self.task_dir), self.venv_dir, config=Config())
            if evaluation["status"] != "success":
                raise RuntimeError(f"SIA evaluator failed: {evaluation.get('reason', evaluation['status'])}")
            scores = json.loads((work / "results.json").read_text(encoding="utf-8"))
            evaluator_results.append(scores)
            data, multi = load_agent_execution(str(work))
            if multi:
                data = data["trajectories"]
            if not isinstance(data, list) or any(not isinstance(r, dict) for r in data):
                raise ValueError("Task-meta trajectories must be structured records, one per question")
            by_id = {row.get("question_id"): row for row in data}
            if len(by_id) != len(data):
                raise ValueError("Duplicate question IDs in execution trajectories")
            rewards = {row["question_id"]: int(row.get("is_correct", False)) for row in scores["details"]}
            rollout_rows = []
            for question in self.questions:
                row = dict(by_id.get(question["id"], {"error": "Missing trajectory", "messages": [], "missing_trajectory": True}))
                row.update({"question_id": question["id"], "rollout_id": rollout,
                            "terminal_reward": rewards.get(question["id"], 0)})
                if binding and row.get("model_ref_requested") != task_state.model_ref:
                    raise ValueError("Execution trajectory did not request the recorded Task model")
                if row.get("model_ref_response") not in (None, task_state.model_ref):
                    raise ValueError("Execution trajectory response came from a different Task model")
                trajectories.append(row)
                rollout_rows.append(row)
            provenance.extend({**item, "generation": task_state.generation}
                              for item in collect_generated_artifacts(work, destination, rollout_rows, rollout))
            print(f"[task-meta] generation={task_state.generation} rollout={rollout + 1}/{self.rollouts} done", flush=True)
        if artifact_manifest(active) != active_hashes:
            raise ValueError("Target mutated the input artifact snapshot during rollout")
        if self.service_binding(task_state) != binding:
            raise ValueError("Inference service binding changed during Task execution")
        output_artifacts = ArtifactState(str(destination), artifact_manifest(destination))
        artifact_diff = rollout_artifact_diff(active_hashes, output_artifacts.manifest)
        provenance.sort(key=lambda row: (safe_task_id(row["question_id"]), row["rollout_id"], row["path"]))
        save_json(gen_dir / "artifact_provenance.json", provenance)
        save_json(gen_dir / "rollout_artifact_diff.json", artifact_diff)
        save_json(gen_dir / "output_artifacts.json", output_artifacts)
        performance = {**result_metrics(trajectories), "batch_size": len(self.questions), "rollouts_per_task": self.rollouts,
                       "per_task": [{"question_id": q["id"], **result_metrics([t for t in trajectories if t["question_id"] == q["id"]])}
                                    for q in self.questions]}
        save_json(Path(gen_dir) / "evaluator_results.json", evaluator_results)
        def tokens(name):
            values = [t.get(name) for t in trajectories]
            return sum(values) if all(isinstance(v, int) for v in values) else None
        cost = {"wall_time_seconds": time.monotonic() - started, "input_tokens": tokens("input_tokens"),
                "output_tokens": tokens("output_tokens"), "api_cost_usd": None, "gpu_hours": None}
        return EvaluationResult(performance, trajectories, cost, evaluated_state=evaluated_state,
                                output_artifacts=output_artifacts, artifact_provenance=provenance,
                                rollout_artifact_diff=artifact_diff)
