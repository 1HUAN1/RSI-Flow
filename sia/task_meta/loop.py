"""The ordered Task -> evaluation -> Meta learning -> routing state machine."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic import ValidationError

from sia.layout import RunLayout
from sia.task_meta.storage import artifact_manifest, digest, manifest_diff, save_json
from sia.task_meta.types import (
    ArtifactState,
    DecisionConstraintError,
    GenerationContext,
    ImprovementExperience,
    MetaAgentState,
    MetaDecision,
    MetaHarnessUpdate,
    MetaObservation,
    TaskUpdateAction,
    UpdatePending,
)


def primary_metric(results: dict, name: str) -> float:
    value = results
    for key in name.split("."):
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Metric {name} must be a finite number, got {value!r}")
    return float(value)


def performance_delta(before: dict, after: dict, name: str, mode: str) -> float:
    if mode not in {"min", "max"}:
        raise ValueError("primary_metric_mode must be min or max")
    delta = primary_metric(after, name) - primary_metric(before, name)
    return delta if mode == "max" else -delta


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
        raise DecisionConstraintError("requested_changes must explicitly name the concrete requested edits")
    ids = [change.id for change in decision.requested_changes]
    if len(ids) != len(set(ids)):
        raise DecisionConstraintError("requested change IDs must be unique")
    if any(change.component != decision.action for change in decision.requested_changes):
        raise DecisionConstraintError("Every requested change must belong to the one selected action")


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


def run_task_meta(
    run_dir, task_state, meta_state, executor, meta_agent, updaters,
    max_generations=5, primary_metric_name="success_rate", primary_metric_mode="max", max_wall_time=None,
    *, test_decision_override=None, resume=False, meta_update_handler=None, evaluated_prefix=None,
):
    """Execute N task states, with N-1 decisions/experiences, then consolidate once.

    A wall-time budget is checked at generation boundaries. An in-flight evaluation
    and its feedback are completed before stopping; this is not a hard kill timer.
    """
    budget = BudgetManager(max_generations, max_wall_time)
    if primary_metric_mode not in {"min", "max"}:
        raise ValueError("primary_metric_mode must be min or max")
    if evaluated_prefix is None:
        if task_state.generation != 0 or task_state.artifacts.manifest or artifact_manifest(task_state.artifacts.directory):
            raise ValueError("Task generation 0 must start with empty artifacts")
    elif (not resume or task_state.generation < 1
          or len(evaluated_prefix['scores']) != task_state.generation
          or len(evaluated_prefix['costs']) != task_state.generation
          or len(evaluated_prefix['history']) != task_state.generation
          or [x['generation'] for x in evaluated_prefix['scores']] != list(range(task_state.generation))):
        raise ValueError("Continuation requires an audited evaluated prefix and original deadline")
    if set(updaters) != set(TaskUpdateAction):
        raise ValueError("Exactly the three named updater implementations must be supplied")
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta").mkdir(exist_ok=True)
    if not resume and ((run_dir / "final_state.json").exists() or (run_dir / "meta" / "experiences.jsonl").exists()):
        raise ValueError("Run already has results; use a new run directory")
    if resume and (run_dir / "final_state.json").exists():
        prior_final = json.loads((run_dir / "final_state.json").read_text(encoding="utf-8"))
        if prior_final.get("status") == "completed":
            return prior_final
    experiences_path = run_dir / "meta" / "experiences.jsonl"
    experiences_path.touch()
    durable = bool(getattr(executor, "durable_recovery", False))
    if durable:
        budget.persist(run_dir / "budget.json", resume=resume)
    layout = RunLayout(str(run_dir))
    task = copy.deepcopy(task_state)
    meta = copy.deepcopy(meta_state)
    history, scores, costs = ([], [], []) if evaluated_prefix is None else (
        copy.deepcopy(evaluated_prefix['history']), copy.deepcopy(evaluated_prefix['scores']), copy.deepcopy(evaluated_prefix['costs']))
    previous = None
    last_input, last_output = None, None
    consolidation_status = "not_attempted"
    status, reason = "completed", None
    failure = None
    decision_sources = set()
    try:
        while True:
            gen_dir = Path(layout.gen_dir(task.generation))
            gen_dir.mkdir(parents=True, exist_ok=True)
            save_json(gen_dir / "meta_state_before.json", meta)
            print(f"[task-meta] rollout {task.version}, Meta harness v{meta.version}", flush=True)
            result = executor.execute(copy.deepcopy(task), gen_dir)
            primary_metric(result.performance, primary_metric_name)
            task_in = copy.deepcopy(result.evaluated_state or task)
            task_out = copy.deepcopy(task_in)
            task_out.artifacts = copy.deepcopy(result.output_artifacts or ArtifactState())
            client = getattr(meta_agent, "client", None)
            if client and hasattr(client, "bind_context"):
                from sia.task_meta.durable import task_hash
                client.bind_context(run_dir.name, task.generation, task_hash(task_out))
            if not result.rollout_artifact_diff:
                result.rollout_artifact_diff = manifest_diff(task_in.artifacts.manifest, task_out.artifacts.manifest)
            last_input, last_output = task_in, task_out
            save_json(gen_dir / "task_state.json", task_in)
            save_json(gen_dir / "evaluated_state.json", task_in)
            save_json(gen_dir / "task_state_after_rollout.json", task_out)
            save_json(gen_dir / "artifact_manifest.json", task_out.artifacts.manifest)
            save_json(gen_dir / "artifact_provenance.json", result.artifact_provenance)
            save_json(gen_dir / "rollout_artifact_diff.json", result.rollout_artifact_diff)
            save_json(gen_dir / "results.json", result.performance)
            save_json(gen_dir / "agent_execution.json", result.trajectories)
            save_json(gen_dir / "cost.json", result.cost)
            scores.append({"generation": task.generation, **result.performance})
            costs.append({"generation": task.generation, **result.cost})
            print(f"[task-meta] {task.version}: {primary_metric_name}={primary_metric(result.performance, primary_metric_name):.6f}", flush=True)
            if previous is not None:
                old_input, old_output, old_result, decision, update, intervention, decision_meta = previous
                observed_delta = primary_metric(result.performance, primary_metric_name) - primary_metric(old_result.performance, primary_metric_name)
                experience = ImprovementExperience(
                    generation=old_input.generation,
                    state_before=asdict(old_input), state_after=asdict(task_in),
                    decision=decision.model_dump(mode="json"), modification=asdict(update),
                    performance_before=old_result.performance, performance_after=result.performance,
                    performance_delta=performance_delta(old_result.performance, result.performance, primary_metric_name, primary_metric_mode),
                    cost_before=old_result.cost, cost_after=result.cost, update_cost=update.cost,
                    trajectory_before=str(Path(layout.gen_dir(old_input.generation)) / "agent_execution.json"),
                    trajectory_after=str(gen_dir / "agent_execution.json"),
                    experience_id=f"experience_{old_input.generation}_{task_in.generation}",
                    evaluated_state_before=asdict(old_input), intervention_base_state=asdict(old_output),
                    evaluated_state_after=asdict(task_in), chosen_action=decision.action.value,
                    requested_change=[c.model_dump(mode="json") for c in decision.requested_changes],
                    actual_change=asdict(update), rollout_artifact_diff=old_result.rollout_artifact_diff,
                    intervention_diff=intervention, observed_performance_delta=observed_delta,
                    versions={"task_before": old_input.version, "task_after": task_in.version,
                              "meta_at_decision": decision_meta.version,
                              "meta_harness_sha256": digest(Path(decision_meta.harness_path)),
                              "meta_bundle_hash_at_decision": decision_meta.bundle_hash,
                              "meta_bundle_path_at_decision": decision_meta.bundle_path,
                              "task_harness_before_sha256": digest(Path(old_input.harness_path)),
                              "task_harness_after_sha256": digest(Path(task_in.harness_path)),
                              "probe_before": old_result.performance.get("probe_identity"),
                              "probe_after": result.performance.get("probe_identity")},
                )
                history.append(experience)
                save_json(gen_dir / "improvement_experience.json", experience)
                from sia.task_meta.durable import record_experience
                record_experience(experiences_path, experience)
                learned = meta_agent.learn_from_experience(meta, experience, history, task_in)
                meta = _accept_meta_update(run_dir, meta, learned, [experience], gen_dir / "meta_self_update.json", "learn_from_experience", meta_update_handler, resume)
                print(f"[task-meta] accepted Meta harness v{meta.version}; delta={experience.performance_delta:+.6f}", flush=True)
            save_json(gen_dir / "meta_state_after.json", meta)
            # Reconstruct the entire accepted prefix even if downtime consumed
            # the budget. A started receipt must surface as pending, not vanish
            # behind a fresh early termination and final consolidation.
            replay_intervention = durable and resume and (gen_dir / "intervention_receipt.json").exists()
            if budget.stop_after(task.generation + 1) and not replay_intervention:
                reason = "max_generations" if task.generation + 1 >= max_generations else "max_wall_time"
                break
            from sia.task_meta.observations import build_observation
            from sia.task_meta.updaters import updater_capabilities

            capabilities = copy.deepcopy(getattr(meta_agent, "capabilities", {}))
            capabilities.update(updater_capabilities(task_out, result, capabilities.get("trainer_configured", False),
                                                    sft_profile=capabilities.get("sft_profile", "legacy_gpqa")))
            observation = build_observation(task_in, task_out, meta, result, scores, history, costs,
                                            capabilities, retain_raw=bool(getattr(client, "supports_evolution", False)))
            observation.budget.update({"max_generations": max_generations, "evaluated_generations": len(scores),
                                  "max_wall_time_seconds": max_wall_time,
                                  "elapsed_seconds": budget.elapsed(),
                                  "boundary_budget": True})
            if resume and (gen_dir / "meta_observation.json").exists():
                observation = MetaObservation(**json.loads((gen_dir / "meta_observation.json").read_text(encoding="utf-8")))
            save_json(gen_dir / "meta_observation.json", observation)
            next_dir = Path(layout.gen_dir(task.generation + 1))
            context = GenerationContext(task.generation + 1, next_dir, observation, result, copy.deepcopy(meta))
            before_harness = digest(Path(task_out.harness_path))
            before_artifacts = artifact_manifest(task_out.artifacts.directory)
            feedback, committed = [], False
            for attempt in range(3):
                decision_id = f"generation_{task.generation}_decision_{attempt}"
                try:
                    override = test_decision_override(meta, observation) if test_decision_override and attempt == 0 else None
                    selected = override if override is not None else meta_agent.diagnose_and_route(meta, observation, feedback=feedback)
                    decision = MetaDecision.model_validate(selected)
                    decision.decision_id = decision_id
                    if override is not None:
                        decision.decision_source = "test_override"
                    else:
                        backend_source = getattr(getattr(meta_agent, "client", None), "decision_source", None)
                        if backend_source in {"mock", "test_override"}:
                            decision.decision_source = backend_source
                        elif decision.decision_source not in {"model", "codex_openrouter", "mock", "test_override"}:
                            raise DecisionConstraintError("Unknown decision_source")
                    decision_sources.add(decision.decision_source)
                    save_json(gen_dir / f"meta_decision_attempt_{attempt}.json", decision)
                    _validate_decision(decision)
                    if budget.stop_after(task.generation + 1) and not replay_intervention:
                        reason = "max_wall_time_before_update"
                        break
                    successor, update = updaters[decision.action].apply(copy.deepcopy(task_out), decision, context)
                except (DecisionConstraintError, ValidationError) as exc:
                    feedback.append({"decision_id": decision_id, "constraint": str(exc),
                                     "actual_task_modification_executed": False})
                    save_json(gen_dir / "constraint_feedback.json", feedback)
                    continue
                except UpdatePending as exc:
                    status, reason = "pending_model_update", str(exc)
                    save_json(gen_dir / "task_update.json", {"action": decision.action.value, "status": status, "reason": reason})
                    break
                committed = True
                break
            if not committed:
                if reason is not None:
                    break
                raise DecisionConstraintError("Three invalid uncommitted decisions; see constraint_feedback.json")
            save_json(gen_dir / "meta_decision.json", decision)
            print(f"[task-meta] M_{meta.version} chose {decision.action.value} ({decision.decision_source})", flush=True)
            if update.action != decision.action or successor.generation != task.generation + 1:
                raise ValueError("Updater returned the wrong action or generation")
            _assert_single_component(task_out, successor, decision.action, before_harness, before_artifacts)
            successor.artifacts.manifest = artifact_manifest(successor.artifacts.directory)
            intervention = _intervention_diff(task_out, successor)
            save_json(gen_dir / "intervention_diff.json", intervention)
            save_json(gen_dir / "task_update.json", update)
            previous = (task_in, task_out, result, decision, update, intervention, copy.deepcopy(meta))
            task = successor
        if not durable or status == "completed":
            consolidation_status = "started"
            final_update = meta_agent.final_consolidation(meta, history, last_input)
            meta = _accept_meta_update(run_dir, meta, final_update, history, run_dir / "meta" / "final_consolidation.json", "final_consolidation", meta_update_handler, resume)
            (run_dir / "meta" / "final_meta_summary.md").write_text(final_update.summary or final_update.rationale, encoding="utf-8")
            consolidation_status = "completed"
    except Exception as exc:
        failure = exc
        status, reason = "failed", str(exc)
        if consolidation_status == "started":
            consolidation_status = "failed"
        save_json(run_dir / "failure.json", {"status": "failed", "type": type(exc).__name__, "message": str(exc),
                                           "task_state": asdict(task), "meta_state": asdict(meta), "evaluated_generations": len(scores)})
    final = {"status": status, "stop_reason": reason,
             "task_state": asdict(last_input) if last_input else None,
             "last_evaluated_task_input": asdict(last_input) if last_input else None,
             "last_task_output_state": asdict(last_output) if last_output else None,
             "last_output_artifacts": asdict(last_output.artifacts) if last_output else None,
             "meta_state": asdict(meta), "final_consolidation_status": consolidation_status,
             "generations_executed": len(scores), "experiences": len(history), "performance_history": scores,
             "primary_metric": primary_metric_name, "primary_metric_mode": primary_metric_mode,
             "wall_time_seconds": budget.elapsed(),
             "budget_clock": "unix_utc_including_downtime" if durable else "process_monotonic",
             "delta_interpretation": "Observed online evolution difference, not an isolated causal intervention benefit",
             "decision_mode": "test_override_enabled" if test_decision_override or "test_override" in decision_sources
             else "mock" if "mock" in decision_sources or getattr(getattr(meta_agent, "client", None), "decision_source", None) == "mock"
             else "autonomous"}
    change_events = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(run_dir.glob("gen_*/meta_self_update.json"))]
    consolidation_path = run_dir / "meta/final_consolidation.json"
    if consolidation_path.exists():
        change_events.append(json.loads(consolidation_path.read_text(encoding="utf-8")))
    final["meta_update_events"] = len(change_events)
    final["meta_version_changes"] = sum(bool(e.get("version_changed", True)) for e in change_events)
    final["meta_no_change_events"] = sum(not e.get("version_changed", True) for e in change_events)
    save_json(run_dir / "final_state.json", final)
    evaluated_version = last_input.version if last_input else "none"
    lines = [f"Task-Meta run: {status}", f"Evaluated task: {evaluated_version}; final Meta harness: v{meta.version}",
             f"Stop: {reason}", "", "| Generation | Metric |", "| --- | --- |"]
    lines += [f"| {r['generation']} | {primary_metric(r, primary_metric_name):.6f} |" for r in scores]
    lines += ["", "Meta Harness updates are accepted structurally; their quality was not evaluated."]
    (run_dir / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failure is not None:
        raise failure
    return final
