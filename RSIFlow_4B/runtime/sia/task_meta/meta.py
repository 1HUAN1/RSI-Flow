"""Frozen SIA provider-backed model with structured, evolvable Meta instructions."""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from pydantic import ValidationError

from sia.task_meta import prompts
from sia.task_meta.observations import (
    artifact_evidence,
    artifact_facts,
    current_artifact_sources,
    harness_evidence,
    operation_input,
    trajectory_evidence,
    trajectory_statistics,
)
from sia.task_meta.storage import digest, experience_summary, save_json
from sia.task_meta.types import DecisionConstraintError, MetaDecision, MetaHarnessUpdate, ExpectedMetaDecision, FiveStageMetaUpdate


def _without_credentials(text):
    for key, value in os.environ.items():
        if len(value) >= 8 and key.upper().endswith(("_KEY", "_TOKEN", "_PASSWORD", "_SECRET")):
            text = text.replace(value, "[REDACTED_CREDENTIAL]")
    return text


def _finish_reasons(result):
    reasons = []
    for message in result.all_messages():
        reason = getattr(message, "finish_reason", None)
        details = getattr(message, "provider_details", None) or {}
        if reason or details.get("finish_reason"):
            reasons.append(str(reason or details["finish_reason"]))
    return reasons


class StructuredClient:
    """Typed SIA model calls, all scoped to an auditable current Meta state."""

    def __init__(self, profile, journal: Path, timeout=180, max_tokens=8192):
        from sia.agent_impls.pydantic_ai import _resolve_model

        self.model = _resolve_model(profile.model, profile.provider)
        self.model_name = profile.model
        self.journal = Path(journal)
        self.timeout = timeout
        self.max_tokens = max_tokens
        existing = self.journal / "usage.json"
        self.calls = json.loads(existing.read_text(encoding="utf-8")) if existing.is_file() else []

    def complete(self, prompt, schema, *, meta_state=None, operation=None, decision_id=None, experience_id=None):
        from pydantic_ai import Agent
        from pydantic_ai.usage import UsageLimits

        self.journal.mkdir(parents=True, exist_ok=True)
        used_ids = [int(p.stem.split("_")[1]) for p in self.journal.glob("call_*_prompt.txt")]
        call_id = max(used_ids, default=-1) + 1
        harness_hash = None
        if meta_state is not None:
            harness_path = Path(meta_state.harness_path)
            harness_hash = digest(harness_path)
            harness = harness_path.read_text(encoding="utf-8")
            prompt = (prompt + f"\nCurrent Meta Harness v{meta_state.version} (working rules; engineering contract has priority):\n"
                      + harness)
        prompt = _without_credentials(prompt)
        (self.journal / f"call_{call_id:03d}_prompt.txt").write_text(prompt, encoding="utf-8")
        started = time.monotonic()
        record = {
            "call": call_id, "model": self.model_name, "schema": schema.__name__,
            "operation": operation or "unspecified", "meta_harness_version": getattr(meta_state, "version", None),
            "meta_harness_hash": harness_hash, "decision_id": decision_id, "experience_id": experience_id,
            "input_tokens": None, "output_tokens": None, "requests": None,
            "api_cost_usd": None, "gpu_hours": None,
        }
        output = None
        try:
            agent = Agent(self.model, output_type=schema, retries=1, instructions=prompts.ENGINEERING_CONTRACT)
            result = agent.run_sync(
                prompt, model_settings={"max_tokens": self.max_tokens, "timeout": self.timeout, "temperature": 0.2},
                usage_limits=UsageLimits(request_limit=2),
            )
            usage = result.usage() if callable(result.usage) else result.usage
            record.update({"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                           "requests": usage.requests, "finish_reasons": _finish_reasons(result)})
            if any(reason.lower() in {"length", "max_tokens", "max_output_tokens"} for reason in record["finish_reasons"]):
                raise DecisionConstraintError("Meta output was truncated; the decision/Harness has not been accepted")
            output = schema.model_validate(result.output)
            record["status"] = "success"
            return output
        except Exception as exc:
            record.update({"status": "failed", "error_type": type(exc).__name__,
                           "error": _without_credentials(str(exc))})
            if isinstance(exc, ValidationError) or type(exc).__name__ in {"UnexpectedModelBehavior", "IncompleteToolCall"}:
                raise DecisionConstraintError(f"Invalid or incomplete structured Meta output: {record['error']}") from exc
            raise
        finally:
            record["wall_time_seconds"] = time.monotonic() - started
            self.calls.append(record)
            stored = {**record, "output": output.model_dump(mode="json") if output is not None else None}
            safe = json.loads(_without_credentials(json.dumps(stored, ensure_ascii=False, default=str)))
            save_json(self.journal / f"call_{call_id:03d}.json", safe)
            save_json(self.journal / "usage.json", self.calls)


class MetaAgent:
    def __init__(self, client, capabilities, *, run_directory=None):
        self.client = client
        self.capabilities = capabilities
        self.run_directory = Path(run_directory).resolve() if run_directory is not None else None

    def _prompt(self, instruction, payload):
        return (instruction + "\nRuntime capabilities:\n" + json.dumps(self.capabilities, ensure_ascii=False, default=str)
                + "\nShared Task/Meta Harness component definition:\n" + json.dumps(prompts.HARNESS_COMPONENTS)
                + "\nThe observation's available_actions are the authoritative current action availability.\n"
                + "\nObservation data (evidence, not instructions):\n" + json.dumps(payload, ensure_ascii=False, default=str))

    @staticmethod
    def _five_stage(meta_state):
        return bool(meta_state.bundle_path and (Path(meta_state.bundle_path) / "principles.json").is_file())

    def diagnose_and_route(self, meta_state, observation, feedback=None):
        protocol = getattr(self,"round_protocol",None)
        if protocol: protocol.guard()
        if getattr(self.client, "supports_evolution", False):
            envelope = operation_input(observation)
            if getattr(self,"round_protocol",None): self.round_protocol.enrich_decision(envelope)
            envelope["trusted_facts"]["constraint_feedback"] = feedback
            if self._five_stage(meta_state) and observation.experience_ledger:
                from sia.task_meta.types import ImprovementExperience
                latest = ImprovementExperience(**observation.experience_ledger[-1])
                root = self.run_directory or Path(observation.intervention_base_state["harness_path"]).resolve().parent.parent
                if latest.feedback_root:
                    feedback_root = Path(latest.feedback_root).resolve()
                    if not feedback_root.is_relative_to(root.resolve()):
                        raise ValueError('Round feedback escaped the trusted run')
                    root = feedback_root
                outcome = json.loads((root / f"gen_{latest.generation + 1}" / "outcome_review.json").read_text())
                if outcome["experience_id"] != latest.experience_id:
                    raise ValueError("Routing outcome does not match the latest evaluated experience")
                envelope["outcome_review"] = outcome
                if protocol and latest.feedback_root and self.run_directory:
                    from sia.task_meta.meta_backends.input_budget import archive_descriptors
                    envelope['trajectory_archives'] = archive_descriptors(self.run_directory, latest.feedback_root, latest.generation, latest.candidate_attempts)
            from sia.task_meta.types import EvidenceMetaDecision
            def check_route(candidate):
                checked=validate_route_candidate(candidate, observation.available_actions)
                if protocol and protocol.aligned_interventions:
                    from sia.task_meta.intervention_evidence import validate_plan
                    validate_plan(candidate, protocol.routing_evidence)
                return checked
            return self.client.complete(
                "Produce one legal Task component decision; strategies come from the bound G.",
                EvidenceMetaDecision if protocol and protocol.aligned_interventions else ExpectedMetaDecision if self._five_stage(meta_state) else MetaDecision, meta_state=meta_state, operation="route",
                decision_id=f"generation_{observation.generation}_decision_{len(feedback or [])}",
                operation_input=envelope,
                validate_candidate=check_route,
            )
        payload = asdict(observation)
        payload.pop("raw_trajectories", None)
        payload.pop("experience_ledger", None)
        if payload.get("trajectories"):
            # Compatibility fields refer to the same records; send only one copy.
            payload.pop("success_examples", None)
            payload.pop("failure_examples", None)
        payload["constraint_feedback"] = feedback
        return self.client.complete(self._prompt(prompts.ROUTE, payload), MetaDecision, meta_state=meta_state,
                                    operation="route", decision_id=f"generation_{observation.generation}_decision_{len(feedback or [])}")

    def learn_from_experience(self, meta_state, experience, history, current_task_state):
        protocol = getattr(self,"round_protocol",None)
        if protocol: protocol.guard()
        if getattr(self.client, "supports_evolution", False):
            envelope = experience_input(experience, history, current_task_state, run_directory=self.run_directory)
            if getattr(self,"round_protocol",None): self.round_protocol.enrich_effect(envelope)
            if self._five_stage(meta_state):
                from sia.task_meta.meta_harness.five_stage import paired_outcome
                root = self.run_directory or Path(current_task_state.harness_path).resolve().parent.parent
                envelope["outcome_review"] = paired_outcome(root, experience)
                if protocol and experience.feedback_root:
                    from sia.task_meta.meta_backends.input_budget import archive_descriptors
                    envelope['trajectory_archives'] = archive_descriptors(root, experience.feedback_root, experience.generation, experience.candidate_attempts)
            return self.client.complete(
                "Use the evaluated experience for a grounded Meta content update. Memory-only updates are valid; do not invent conclusions or use NO_CHANGE.",
                FiveStageMetaUpdate if self._five_stage(meta_state) else MetaHarnessUpdate, meta_state=meta_state, operation="learn",
                decision_id=experience.decision.get("decision_id"),
                experience_id=experience.experience_id or f"experience_g{experience.generation}",
                operation_input=envelope,
                validate_candidate=lambda candidate: protocol.validate_meta(candidate,meta_state) if getattr(self,"round_protocol",None) else validate_meta_candidate(candidate, meta_state),
            )
        def evidence(path):
            if not path:
                return {"source_file": None, "trajectories": [], "coverage": {"status": "candidate_unavailable"}}
            rows = json.loads(Path(path).read_text(encoding="utf-8"))
            selected, coverage = trajectory_evidence(rows, max_chars=48000)
            return {"source_file": path, "trajectories": selected, "coverage": coverage}

        latest = asdict(experience)
        if isinstance(latest.get("state_before"), dict):
            latest["state_before"].pop("observation", None)
        payload = {"latest_experience": latest, "history": [experience_summary(e) for e in history],
                   "before_evidence": evidence(experience.trajectory_before),
                   "after_evidence": evidence(experience.trajectory_after), "current_task": asdict(current_task_state)}
        return self.client.complete(
            self._prompt(prompts.LEARN, payload), MetaHarnessUpdate, meta_state=meta_state, operation="learn",
            decision_id=experience.decision.get("decision_id"),
            experience_id=getattr(experience, "experience_id", None) or f"experience_g{experience.generation}",
        )

    def final_consolidation(self, meta_state, history, current_task_state):
        if getattr(self.client, "supports_evolution", False):
            envelope = experience_input(None, history, current_task_state, run_directory=self.run_directory)
            return self.client.complete(
                "Consolidate the completed run; do not create a successor Task or trigger another evaluation.",
                FiveStageMetaUpdate if self._five_stage(meta_state) else MetaHarnessUpdate, meta_state=meta_state, operation="final_consolidation",
                operation_input=envelope,
                validate_candidate=lambda candidate: validate_meta_candidate(candidate, meta_state),
            )
        # A crash can leave an already accepted next version on disk. It is not
        # part of this request's input state during deterministic recovery.
        paths = Path(meta_state.harness_path).parent.glob("harness_v*.md")
        versions = {p.name: p.read_text(encoding="utf-8") for p in sorted(paths)
                    if p.stem.removeprefix("harness_v").isdigit()
                    and int(p.stem.removeprefix("harness_v")) <= meta_state.version}
        payload = {"history": [experience_summary(e) for e in history], "meta_harness_history": versions,
                   "current_task": asdict(current_task_state)}
        return self.client.complete(self._prompt(prompts.CONSOLIDATE, payload), MetaHarnessUpdate,
                                    meta_state=meta_state, operation="final_consolidation")


def validate_route_candidate(candidate, actions):
    """Validate a component choice without preselecting HarnessForge files."""
    from sia.task_meta.loop import _validate_decision
    decision = MetaDecision.model_validate(candidate)
    _validate_decision(decision)
    if decision.diagnosis_kind != "hypothesis":
        raise ValueError("A Meta diagnosis must remain a hypothesis")
    capability = actions.get(decision.action.value, {})
    if not capability.get("available"):
        raise ValueError(f"{decision.action.value} unavailable: {capability.get('reason')}")
    if decision.action.value == "HARNESS":
        # Stage 1 localization, not routing, owns module/file attribution.  The
        # routing decision therefore names the whole independent harness bundle.
        change = decision.requested_changes[0]
        if change.operation != "produce_harness" or change.target != "harness_bundle":
            raise ValueError("HARNESS must request one complete HarnessForge bundle")
        return {"passed": True, "diagnosis_kind": "hypothesis", "scope": "harness_bundle"}
    for change in decision.requested_changes:
        matches = [op for op in capability.get("operations", []) if op.get("operation") == change.operation]
        if not matches or not any(change.target in op.get("targets", [])
                                 or op.get("target") == change.target
                                 or (decision.action.value == "ARTIFACTS" and "relative" in op.get("target", ""))
                                 for op in matches):
            raise ValueError("Requested operation/target is not in the trusted capability set")
        if decision.action.value == "ARTIFACTS":
            from sia.task_meta.updaters import artifact_path
            artifact_path(Path("/bounded-artifact-validation"), change.target)
    return {"passed": True, "diagnosis_kind": "hypothesis"}


def validate_meta_candidate(candidate, meta_state):
    from sia.task_meta.meta_harness.bundle import MetaHarnessBundle, semantic_files, validate_files
    update = MetaHarnessUpdate.model_validate(candidate)
    path = Path(meta_state.bundle_path)
    bundle = MetaHarnessBundle(path, json.loads((path / "manifest.json").read_text())).verify()
    files = {name: (path / name).read_text(encoding="utf-8") for name in bundle.manifest["editable_files"]}
    if not set(update.bundle_files) <= set(files):
        raise ValueError("Meta candidate attempts to change protected paths")
    if "instructions.md" in update.bundle_files and update.bundle_files["instructions.md"] != update.harness:
        raise ValueError(
            "Conflicting instructions in the Meta candidate: harness must contain the complete "
            "instructions.md text, not a version label or filename. Set harness to the intended "
            "full instruction content and omit instructions.md from bundle_files, or make both "
            "values exactly equal. No candidate has been committed.")
    candidate_files = {**files, **update.bundle_files, "instructions.md": update.harness}
    if "principles.json" in files:
        from sia.task_meta.meta_harness.five_stage import materialize
        candidate_files, _ = materialize(update, files)
    validate_files(candidate_files)
    if update.status == "NO_CHANGE" and semantic_files(candidate_files) != semantic_files(files):
        raise ValueError("NO_CHANGE cannot contain a substantive Meta modification")
    return {"passed": True, "acceptance": "structural_only"}


def experience_input(latest, history, current_task_state, *, run_directory=None):
    """Supply only the trusted run's evaluated prefix; never follow model paths."""
    root = Path(run_directory).resolve() if run_directory is not None else Path(current_task_state.harness_path).resolve().parent.parent
    trusted_root = root
    if latest is not None and latest.feedback_root:
        root = Path(latest.feedback_root).resolve()
        if not root.is_relative_to(trusted_root):
            raise ValueError('Round feedback escaped the trusted run')
    ledger = [asdict(item) for item in history]
    rows, sources = [], []
    # Preserve the existing protocol: full before/after transition trajectories
    # and the complete experience ledger. Historical trajectory bodies are not
    # recursively copied into every new Meta request.
    generations = {int(current_task_state.generation)}
    if latest is not None:
        generations.add(int(latest.generation))
    sources_to_read = [(generation, root / f'gen_{generation}' / 'agent_execution.json') for generation in sorted(generations)]
    if latest is not None:
        for attempt in latest.candidate_attempts:
            if attempt.get('trajectory_after'):
                path = Path(attempt['trajectory_after']).resolve()
                if not path.is_relative_to(trusted_root):
                    raise ValueError('Candidate evidence escaped the trusted run')
                if path not in {item[1].resolve() for item in sources_to_read}:
                    sources_to_read.append((current_task_state.generation, path))
    for generation, path in sources_to_read:
        if generation < 0 or generation > current_task_state.generation:
            raise ValueError("Experience index contains an unevaluated generation")
        if path.is_symlink() or path.parent.is_symlink() or not path.resolve().is_relative_to(trusted_root):
            raise ValueError("Experience trajectory source escaped the trusted run")
        if not path.is_file():
            raise ValueError("Missing evaluated trajectory source")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Invalid trusted trajectory source")
        file_hash = digest(path)
        source_name = path.relative_to(trusted_root).as_posix()
        sources.append({"source": source_name, "sha256": file_hash, "count": len(raw)})
        from sia.task_meta.observations import trajectory_evidence
        examples, coverage = trajectory_evidence(raw, max_chars=72000)
        key = lambda row: (row.get("question_id",row.get("task_id")),row.get("rollout_id"))
        selected = {key(row):row for row in examples}
        sources[-1]["meta_sample_count"] = len(selected)
        sources[-1]["meta_sample_policy"] = "deterministic_domain_success_96_bounded_72000_chars"
        sources[-1]["meta_evidence_coverage"] = coverage
        for index, item in enumerate(raw):
            if key(item) not in selected: continue
            item = selected[key(item)]
            if not isinstance(item, dict) or item.get("split", "evolve_train") != "evolve_train":
                raise ValueError("Only recorded training trajectories may enter Meta experience evidence")
            row = copy.deepcopy(item)
            row["source_id"] = f"{source_name}/trajectory_{index}/{file_hash[:16]}"
            row["task_version"] = f"T_{generation}"
            rows.append(row)
    if latest is not None and (not history or asdict(latest) != ledger[-1]):
        raise ValueError("Self-update requires the latest actual ledger experience")
    harness_files, identity = harness_evidence(current_task_state.harness_path)
    assets, _ = artifact_evidence(current_task_state.artifacts)
    provenance_path = (Path(current_task_state.artifacts.directory).parent / "input_artifact_provenance.json"
                       if current_task_state.artifacts.directory else None)
    if provenance_path and provenance_path.is_file():
        assets["provenance"] = json.loads(provenance_path.read_text(encoding="utf-8"))
    else:
        assets["provenance"] = [{**entry, "production_method": "unknown", "terminal_reward": None,
                                  "knowledge_verified": False} for entry in current_task_state.artifacts.manifest]
    return {"raw_trajectories": rows, "experiences": ledger,
            "latest_experience": asdict(latest) if latest is not None else None,
            "task_state": asdict(current_task_state), "decision": asdict(latest)["decision"] if latest else {},
            "current_files": {**harness_files, **current_artifact_sources(current_task_state.artifacts)},
            "trusted_facts": {"sources": sources, "trajectory_statistics": trajectory_statistics(rows),
                              "current_harness_identity": identity, "evaluated_input_artifacts": artifact_facts(assets),
                              "artifact_scope": "evaluated Task input; live post-rollout assets are supplied separately when routing",
                              "trajectory_scope": "current_evaluated_transition" if latest else "last_evaluated_generation",
                              "history_scope": "complete_permitted_experience_ledger",
                              "current_generation": current_task_state.generation,
                              "experience_count": len(ledger), "performance": latest.performance_after if latest else None,
                              "latest_experience_id": latest.experience_id if latest else None,
                              "experience_origin": "evaluated_run_prefix",
                              "scores_are_observations_not_component_causal_effects": True}}


def evolution_kwargs(client, context, task_state, decision, current_files, validate_candidate):
    if not getattr(client, "supports_evolution", False):
        return {}
    if decision.action.value == "ARTIFACTS":
        current_files = {"assets/" + name: content for name, content in current_files.items()}
    envelope=operation_input(context.observation, task_state=task_state, decision=decision, current_files=current_files)
    from sia.task_meta.intervention_evidence import branch_binding
    binding=branch_binding(task_state,decision,context)
    if binding is not None: envelope["trusted_facts"]["intervention_blueprint"]=binding
    return {"operation_input":envelope, "validate_candidate":validate_candidate}
