"""Stage-level Task tools for the Codex-led 8B experiment.

This module executes requests and records observations.  It does not choose a
component, compare a score to a threshold, accept a child, or advance a round.
The local ``runtime`` contains the copied Task engine; the old 4B monolithic
``pipeline.run``/``run_sequential_task_meta`` entrypoints are never called.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class TaskAdapter:
    """An explicit, file-oriented interface to one Task stage at a time."""

    def __init__(self, project_root: str | Path, receipt_root: str | Path,
                 model_path: str | Path | None = None):
        self.project_root = Path(project_root).resolve()
        self.receipt_root = Path(receipt_root).resolve()
        self.model_path = Path(model_path).resolve() if model_path else None
        runtime = str(self.project_root / "runtime")
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        if runtime not in sys.path:
            sys.path.insert(0, runtime)

    def _path(self, value: str | Path) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else self.project_root / path).resolve()

    def _config(self, arguments: dict[str, Any]):
        from sia.task_meta.pipeline import PipelineConfig
        source = self._path(arguments["config_path"])
        value = json.loads(source.read_text(encoding="utf-8"))
        if value.get("schema_version") == 1:
            value["task_checkpoint"] = str(arguments.get("checkpoint_path") or self.model_path or value["task_checkpoint"])
            return PipelineConfig.model_validate(value).checked()
        # Convert the experiment launcher JSON to the runtime config for one stage.
        runtime = self.project_root / value.get("runtime_source", "runtime")
        base = json.loads((runtime / value["base_config"]).read_text(encoding="utf-8"))
        from evolution_protocol import SFT_PRESET, training_quotas
        source_quotas = training_quotas(value)
        quotas = {"tool_use": source_quotas["envscaler"],
                  "code": source_quotas["deepcoder_taco"],
                  "searchqa": sum(count for name, count in source_quotas.items()
                                  if name not in {"envscaler", "deepcoder_taco"})}
        data_dir = value.get("frozen_data_dir") or str(runtime / "data" / value.get("data_release", "rounds"))
        ports = value["ports"]
        base.update(
            mode="full", experiment_scope="multidomain", training_schedule="round_disjoint",
            task_checkpoint=str(arguments.get("checkpoint_path") or self.model_path or base["task_checkpoint"]),
            output_root=value["output_root"], data_dir=data_dir,
            max_generations=value["rounds"], seed=value.get("rollout_seed", value.get("seed", 42)),
            search_dev_fraction=0, rollouts_per_task=1, probe_rollouts=1,
            window_quotas=quotas, probe_per_domain=quotas,
            gpu_execution="phased_four", trainer_python=sys.executable,
            task_base_url=f"http://127.0.0.1:{ports[0]}/v1",
            task_replicas=[{"gpu": i, "base_url": f"http://127.0.0.1:{port}/v1"}
                           for i, port in enumerate(ports)],
            artifact_evaluation=value.get("artifact_evaluation", "direct_submission"),
            training_timeout_seconds=value.get("training_timeout_seconds", 604800),
            round_validation_config=str(self.project_root / value.get("validation_config", "configs/validation.json")),
        )
        base["meta"]["run_mode"] = "full"
        base["training"].update(num_train_epochs=1, max_steps=-1,
                                max_length=value.get("max_length", base["training"]["max_length"]),
                                max_samples=value.get("max_sft_samples", 10000000))
        base["training"].update(SFT_PRESET, seed=value.get("training_seed", 42))
        base["round_protocol"] = {
            "data_dir": data_dir,
            "train_quotas_per_round": quotas,
            "validation_manifest": str(Path(value["validation_vault"]) / "validation/manifest.json"),
            "minimum_sft_samples": value.get("minimum_sft_samples", 1),
            "skip_acebench": value.get("skip_acebench", False),
        }
        return PipelineConfig.model_validate(base).checked()

    def _state(self, arguments: dict[str, Any]):
        from sia.task_meta.durable import load_task
        return load_task(json.loads(self._path(arguments["state_path"]).read_text(encoding="utf-8")))

    def _output(self, arguments: dict[str, Any], operation: str) -> Path:
        if "output_dir" in arguments:
            return self._path(arguments["output_dir"])
        round_number = int(arguments.get("round_number", 0))
        return self.receipt_root / f"round_{round_number:02d}" / operation

    @staticmethod
    def _write(path: Path, value: Any) -> None:
        from sia.task_meta.storage import save_json
        save_json(path, value)

    def call(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Return a factual receipt even when a stage fails; Meta decides what next."""
        stages = {
            "bootstrap": self.bootstrap,
            "rollout": self.rollout,
            "prepare_meta_evidence": self.prepare_meta_evidence,
            "sft": self.sft,
            "harnessforge": self.harnessforge,
            "artifacts": self.artifacts,
            "prepare_evaluation": self.prepare_evaluation,
            "evaluate": self.evaluate,
        }
        if operation not in stages:
            return {"status": "unknown_stage", "available_stages": sorted(stages)}
        started = time.monotonic()
        try:
            result = stages[operation](arguments)
            return {"status": "completed", "stage": operation,
                    "elapsed_seconds": time.monotonic() - started, **result}
        except Exception as exc:
            return {"status": "stage_error", "stage": operation,
                    "error_type": type(exc).__name__, "error": str(exc),
                    "elapsed_seconds": time.monotonic() - started,
                    "output_dir": str(self._output(arguments, operation))}

    def bootstrap(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Create A0 from an explicit checkpoint and the pinned seed manifest."""
        from sia.task_meta.storage import checkpoint_manifest
        from sia.task_meta.types import TaskAgentState
        from sia.task_meta.harnessforge_manifest import load_manifest

        checkpoint = self._path(arguments.get("checkpoint_path") or self.model_path or
                                self._config(arguments).task_checkpoint)
        if not checkpoint.is_dir():
            return {"status": "missing_checkpoint", "checkpoint_path": str(checkpoint)}
        harness = self._path(arguments.get("seed_harness", "seed_harness/harnessforge_base_manifest.json"))
        load_manifest(harness)
        target = self._path(arguments.get("state_path", self.receipt_root / "task_a0.json"))
        state = TaskAgentState(0, str(checkpoint), str(harness),
                               checkpoint_path=str(checkpoint),
                               checkpoint_manifest=checkpoint_manifest(checkpoint))
        self._write(target, asdict(state))
        return {"state_path": str(target), "checkpoint_path": str(checkpoint),
                "harness_path": str(harness)}

    def prepare_meta_evidence(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Summarize all parent outcomes and excerpt 48 real trajectories."""
        from task_differences import failure_fingerprint
        from sia.task_meta.evolution_protocol import meta_routing_rows
        from sia.task_meta.intervention_evidence import decision_sources
        from sia.task_meta.data import sha256_file
        from sia.task_meta.observations import trajectory_statistics

        parent_dir = self._path(arguments["parent_rollout_dir"])
        original = parent_dir / "train_trajectories.jsonl"
        with original.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        selected = meta_routing_rows(rows, maximum=48, per_domain=16,
                                     per_outcome=8, seed=42)
        sources = decision_sources(rows, [])
        output = self._output(arguments, "meta_evidence")
        output.mkdir(parents=True, exist_ok=True)
        summary_path = output / "all_task_statistics.json"
        excerpts_path = output / "routing_sources.json"
        index_path = output / "trajectory_index.json"
        fingerprint_path = output / "failure_fingerprint.json"
        self._write(fingerprint_path, failure_fingerprint(rows))
        self._write(summary_path, trajectory_statistics(rows))
        self._write(excerpts_path, sources)
        self._write(index_path, {
            "full_trajectories_path": str(original),
            "full_trajectories_sha256": sha256_file(original),
            "rows": [{"task_id": row.get("task_id"), "rollout_id": row.get("rollout_id"),
                      "trajectory_id": row.get("trajectory_id")}
                     for row in rows],
        })
        return {"all_tasks": len(rows), "selected_excerpts": len(selected),
                "statistics_path": str(summary_path),
                "routing_sources_path": str(excerpts_path),
                "trajectory_index_path": str(index_path),
                "failure_fingerprint_path": str(fingerprint_path),
                "full_trajectories_path": str(original)}

    def _executor(self, config, state, round_number: int, stage: str, output: Path):
        from sia.task_meta.data import DOMAINS
        from sia.task_meta.pipeline import adapter_factory
        from sia.task_meta.pipeline_execution import MultiDomainExecutor
        from sia.task_meta.round_evolution import RoundStore
        from sia.task_meta.task_client import LocalTaskClient
        from sia.task_meta.durable import task_hash
        from sia.task_meta.evolution_protocol import fingerprint
        from sia.task_meta.storage import artifact_manifest, digest

        data_root = self._path(config.data_dir)
        store = RoundStore(data_root / "tasks.sqlite")
        store.select_round(round_number)
        store.mode = "parent_pre_update" if stage == "parent" else "child_post_update"
        factory, _ = adapter_factory(config)
        executor = MultiDomainExecutor(
            store, factory,
            lambda task, base_url=None: LocalTaskClient(
                task, base_url or config.task_base_url,
                timeout=config.task_timeout, enable_thinking=config.task_enable_thinking),
            quotas=config.window_quotas, rollouts_per_task=config.rollouts_per_task,
            probe_rollouts=config.probe_rollouts, seed=config.seed,
            model_call_limit=config.model_call_limit,
            max_output_tokens=config.max_output_tokens,
            expected_domains=DOMAINS, replicas=config.task_replicas,
        )
        executor.artifact_evaluation = config.artifact_evaluation
        scope = {
            "role": "train_evolution", "purpose": "evolution_train",
            "round_id": round_number, "manifest_hash": store.current["manifest_hash"],
            "collection_stage": store.mode, "task_hash": task_hash(state),
            "harness_sha256": digest(Path(state.harness_path)),
            "checkpoint_manifest": state.checkpoint_manifest,
            "artifacts": artifact_manifest(state.artifacts.directory),
            "protocol_hash": fingerprint({"runtime_config": config.model_dump(),
                                          "stage": "train_evolution_v1"}),
        }

        class Scope:
            root = output.parent
            execution_scope = scope

            @staticmethod
            def observed(result, directory):
                from collections import Counter
                rows = result.trajectories
                members = set(store.members)
                if len(rows) != len(members) or {row["task_id"] for row in rows} != members:
                    raise ValueError("Each parent/child pass must execute the complete current B_r exactly once")
                result.performance.update(feedback_role="train_evolution",
                                          round_id=round_number,
                                          training_manifest=store.current["manifest_hash"],
                                          collection_stage=store.mode,
                                          evaluation_protocol="same_round_full_B_parent_child_v2")
                benchmarks = {}
                for source in sorted({row["source"] for row in rows}):
                    group = [row for row in rows if row["source"] == source]
                    benchmarks[source] = {
                        "count": len(group),
                        "successes": sum(row["verification"].get("success") is True for row in group),
                        "errors": dict(Counter(
                            row.get("execution_error_type") or row.get("error_type") or "none"
                            for row in group)),
                    }
                result.performance["benchmarks"] = benchmarks
                return result

        executor.round_protocol = Scope()
        return executor

    def rollout(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run the selected Task state on one frozen round; return score paths."""
        config = self._config(arguments)
        state = self._state(arguments)
        stage = arguments.get("stage", "parent")
        if stage not in {"parent", "candidate"}:
            raise ValueError("rollout stage must be parent or candidate")
        round_number = int(arguments["round_number"])
        output = self._output(arguments, f"rollout_{stage}")
        output.mkdir(parents=True, exist_ok=True)
        if arguments.get("ensure_services", True):
            from sia.task_meta.gpu_phases import ensure_services
            ensure_services(config, state.checkpoint_path or state.model_ref)
        executor = self._executor(config, state, round_number, stage, output)
        if arguments.get("baseline_dir") and arguments.get("submission_targets"):
            from sia.task_meta.submissions import SubmissionEvaluator
            result = SubmissionEvaluator(
                executor, self._path(arguments["baseline_dir"]),
                arguments["submission_targets"]).execute(state, output)
        else:
            result = executor.execute(state, output)
        performance = output / "performance.json"
        self._write(performance, result.performance)
        self._write(output / "cost.json", result.cost)
        return {"round_number": round_number, "rollout_stage": stage, "output_dir": str(output),
                "state_path": str(self._path(arguments["state_path"])),
                "performance_path": str(performance),
                "trajectories_path": str(output / "train_trajectories.jsonl"),
                "tasks_observed": len(result.trajectories),
                "macro_success": result.performance.get("macro_success"),
                "collection_stage": result.performance.get("collection_stage")}

    def harnessforge(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Materialize the one complete Codex-authored bundle after upstream checks.

        Codex performs fault localization, direction and complete code generation;
        this tool handles the official isolated candidate validation and Task
        state materialization.  Validation diagnostics return to the same Meta.
        """
        from sia.task_meta.harnessforge_manifest import HarnessBundleManifest
        from sia.task_meta.harnessforge_validation import validate_candidate
        from sia.task_meta.storage import clone_task

        candidate = self._path(arguments["candidate_dir"])
        output = self._output(arguments, "harnessforge")
        output.mkdir(parents=True, exist_ok=True)
        reports = candidate.with_name(candidate.name + "_production")
        paths = {
            "localization_report_path": self._path(arguments.get("localization_report_path") or
                                                   reports / "module_localization_report.md"),
            "improvement_direction_path": self._path(arguments.get("improvement_direction_path") or
                                                    reports / "improvement_direction_brief.md"),
        }
        production = {"production_executor": "meta_codex",
                      "tool_role": "executable_validation_and_materialization_only",
                      "max_validation_attempts": 3,
                      **{key: str(path) for key, path in paths.items()}}
        missing = [key for key, path in paths.items()
                   if not path.is_file() or not path.read_text(encoding="utf-8").strip()]
        if missing:
            return {"status": "production_incomplete", "missing": missing, **production,
                    "next_step": "Use the upstream localization and direction templates to write both reports, then generate the complete candidate before validation."}
        reports.mkdir(parents=True, exist_ok=True)
        attempts_path = reports / "validation_attempts.json"
        attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
        if len(attempts) >= 3:
            return {"status": "repair_limit_reached", **production,
                    "validation_attempts_path": str(attempts_path),
                    "next_step": "Keep the parent; record this attempt's failure. Meta decides the next modification attempt."}
        try:
            validation = validate_candidate(candidate).to_dict()
        except (ValueError, FileNotFoundError, SyntaxError) as exc:
            # An incomplete/generated package is a repairable candidate result,
            # not a reason to end the Meta conversation before getting feedback.
            validation = {"verdict": "failed_static", "checks": [],
                          "errors": [f"{type(exc).__name__}: {exc}"]}
        attempt = len(attempts) + 1
        attempt_path = reports / f"validation_{attempt:02d}.json"
        self._write(attempt_path, validation)
        attempts.append({"attempt": attempt, "verdict": validation["verdict"],
                         "validation_path": str(attempt_path)})
        self._write(attempts_path, attempts)
        production.update(validation_attempt=attempt, checks_remaining=3 - attempt,
                          validation_attempts_path=str(attempts_path))
        self._write(output / "production.json", {
            **production, "candidate_dir": str(candidate),
            "stages": ["localization", "improvement_direction", "complete_generation", "validation"],
            "parent_state_path": str(self._path(arguments["state_path"]))})
        self._write(output / "validation.json", validation)
        if validation["verdict"] != "passed":
            return {"status": "validation_failed", "validation_path": str(output / "validation.json"),
                    "candidate_dir": str(candidate), "verdict": validation["verdict"], **production,
                    "next_step": ("Read diagnostics; repair this candidate and call harnessforge again."
                                  if attempt < 3 else "Checks exhausted; retain parent and record failure.")}
        manifest = HarnessBundleManifest.from_directory(candidate)
        manifest_path = output / "candidate_bundle.json"
        self._write(manifest_path, manifest.to_dict())
        parent = self._state(arguments)
        child = clone_task(parent, parent.generation + 1, output / "child")
        child.harness_path = str(manifest_path)
        state_path = output / "candidate_state.json"
        self._write(state_path, asdict(child))
        return {"candidate_state_path": str(state_path), "candidate_manifest_path": str(manifest_path),
                "validation_path": str(output / "validation.json"),
                "bundle_sha256": manifest.bundle_sha256,
                **production}

    def artifacts(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Apply Codex-proposed submission edits; a later rollout re-scores them."""
        from sia.task_meta.durable import value_hash
        from sia.task_meta.storage import artifact_manifest, clone_task
        from sia.task_meta.submissions import baseline_outputs, submission, validate_submission
        from sia.task_meta.types import ArtifactState

        baseline = self._path(arguments["baseline_dir"])
        output = self._output(arguments, "artifacts")
        output.mkdir(parents=True, exist_ok=True)
        rows = baseline_outputs(baseline)
        edits = arguments["edits"]
        if not isinstance(edits, dict) or not edits:
            raise ValueError("artifacts edits must map existing submission paths to payloads")
        parent = self._state(arguments)
        child = clone_task(parent, parent.generation + 1, output / "child")
        asset_root = output / "child" / "artifacts"
        asset_root.mkdir(parents=True, exist_ok=True)
        for name, payload in edits.items():
            row = rows[name]
            validate_submission(payload, row, int(arguments.get("max_tool_calls", 128)))
            if payload == submission(row):
                continue
            target = (asset_root / name).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            self._write(target, {"schema": "task_submission_v1",
                                 "baseline_sha256": value_hash(row),
                                 "task_id": row["task_id"],
                                 "task_source_hash": row["task_source_hash"],
                                 "rollout_id": row["rollout_id"],
                                 "split": row["split"], "seed": row["seed"],
                                 "payload": payload})
        child.artifacts = ArtifactState(str(asset_root), artifact_manifest(asset_root))
        state_path = output / "candidate_state.json"
        self._write(state_path, asdict(child))
        return {"candidate_state_path": str(state_path),
                "submission_targets": list(edits), "baseline_dir": str(baseline),
                "evaluation_mode": "direct_submission"}

    def sft(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Train one LoRA candidate from verified successful parent trajectories."""
        from sia.task_meta.types import EvaluationResult, GenerationContext, MetaDecision
        from sia.task_meta.updaters import ModelUpdater

        config = self._config(arguments)
        state = self._state(arguments)
        output = self._output(arguments, "sft")
        output.mkdir(parents=True, exist_ok=True)
        parent_dir = self._path(arguments["parent_rollout_dir"])
        with (parent_dir / "train_trajectories.jsonl").open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        performance = json.loads((parent_dir / "performance.json").read_text(encoding="utf-8"))
        expected = sum(config.window_quotas.values())
        round_id = performance.get("round_id")
        manifest_hash = performance.get("training_manifest")
        if (len(rows) != expected or len({row.get("task_id") for row in rows}) != expected
                or performance.get("total_rollouts") != expected
                or performance.get("feedback_role") != "train_evolution"
                or performance.get("collection_stage") != "parent_pre_update"
                or any(row.get("source_role") != "train_evolution"
                       or row.get("collection_stage") != "parent_pre_update"
                       or row.get("round_id") != round_id
                       or row.get("manifest_hash") != manifest_hash for row in rows)):
            raise ValueError("SFT requires a complete current parent training rollout")
        decision = MetaDecision.model_validate({
            "action": "MODEL", "diagnosis": arguments.get("diagnosis", "Meta selected MODEL"),
            "evidence": arguments.get("evidence", []), "rationale": arguments.get("rationale", "Meta selected SFT"),
            "proposed_change": arguments.get("proposed_change", "One-epoch positive LoRA SFT"),
            "expected_effect": arguments.get("expected_effect", "Measured by candidate rollout"),
            "target_components": ["MODEL"],
            "requested_changes": [{"id": "model-sft", "component": "MODEL",
                                   "operation": "sft", "target": "current_checkpoint",
                                   "instruction": "Train on verified successful parent trajectories"}],
        })
        context = GenerationContext(
            generation=state.generation + 1, directory=output,
            observation=None, evaluation=EvaluationResult(performance, rows), meta_state=None)
        training = dict(config.training)
        training.pop("round_protocol", None)
        effective = output / "effective_config.json"
        self._write(effective, config.model_dump())
        command = [config.trainer_python, str(self.project_root / "runtime/scripts/train_four_gpu_phase.py"),
                   "--request-dir", "{request_dir}", "--config", str(effective)]
        provider = SimpleNamespace(base_url=config.task_base_url, api_key_env="LOCAL_QWEN_API_KEY")
        updater = ModelUpdater(SimpleNamespace(supports_evolution=False), None, provider, command,
                               timeout=config.training_timeout_seconds + 1800,
                               sft_profile="multidomain", supervision=config.supervision,
                               training=training, training_gpu=None)
        child, update = updater.apply(state, decision, context)
        state_path = output / "candidate_state.json"
        self._write(state_path, asdict(child))
        return {"candidate_state_path": str(state_path),
                "training_metrics_path": str(output / "model_update/training_metrics.json"),
                "checkpoint_path": child.checkpoint_path,
                "positive_samples": update.details.get("positive_samples")}

    def prepare_evaluation(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Pin/read external benchmark inputs once; scores remain report-only."""
        from evaluation_manifest import evaluation_budget
        from validation_sources import freeze, verify

        config = self._config(arguments)
        settings_path = self._path(arguments.get("validation_config_path") or
                                   config.round_validation_config)
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        provenance = Path(config.output_root) / "validation_sources.json"
        if provenance.is_file():
            verify(settings, provenance)
        else:
            freeze(settings, provenance)
        manifest = self._path(arguments.get("manifest_path") or
                              config.round_protocol["validation_manifest"])
        role = arguments.get("role", "independent_validation")
        budget = evaluation_budget(manifest, role)
        return {"validation_sources_path": str(provenance),
                "manifest_path": str(manifest), "role": role,
                "budget": budget}

    def evaluate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call the separate official 300-task evaluator; never feed its score to SFT."""
        output = self._output(arguments, "evaluation")
        output.mkdir(parents=True, exist_ok=True)
        prepared = self.prepare_evaluation(arguments)
        config = self._config(arguments)
        encoded = json.dumps(config.model_dump(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        effective = self.receipt_root / "_effective_configs" / (
            hashlib.sha256(encoded).hexdigest()[:16] + ".json")
        self._write(effective, config.model_dump())
        settings_path = self._path(arguments.get("validation_config_path") or
                                   config.round_validation_config)
        command = [sys.executable, "-u", str(self.project_root / "validate.py"),
                   "--snapshot", str(self._path(arguments["snapshot_path"])),
                   "--pipeline-config", str(effective),
                   "--config", str(settings_path),
                   "--role", prepared["role"],
                   "--manifest", prepared["manifest_path"], "--execute"]
        skip_ace = arguments.get("skip_ace", config.round_protocol.get("skip_acebench", False))
        if skip_ace:
            command.append("--skip-ace")
        with (output / "stdout.log").open("w", encoding="utf-8") as stdout, \
             (output / "stderr.log").open("w", encoding="utf-8") as stderr:
            process = subprocess.run(command, cwd=self.project_root, stdout=stdout, stderr=stderr,
                                     check=False, timeout=arguments.get("timeout_seconds"))
        snapshot = self._path(arguments["snapshot_path"])
        results = snapshot.parent
        return {"status": "exited", "returncode": process.returncode,
                "stdout_path": str(output / "stdout.log"),
                "stderr_path": str(output / "stderr.log"),
                "snapshot_path": str(snapshot),
                "evaluation_output_dir": str(results),
                "metrics_path": str(results / "task_metrics.json") if (results / "task_metrics.json").is_file() else None,
                "complete_path": str(results / "complete.json") if (results / "complete.json").is_file() else None,
                "partial_path": str(results / "code_search_partial_complete.json")
                                if (results / "code_search_partial_complete.json").is_file() else None,
                "skip_ace": skip_ace}
