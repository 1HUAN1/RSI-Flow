"""Auditable observations: complete statistics and bounded, stratified evidence."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path

from sia.task_meta.storage import experience_summary
from sia.task_meta.types import MetaObservation


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, default=str))


def trajectory_id(row, index):
    return f"task={row.get('question_id', '?')}/rollout={row.get('rollout_id', '?')}/index={index}"


def outcome(row):
    """Observed execution conditions, never inferred reasoning-error labels."""
    if row.get("api_failure") or row.get("api_error") or row.get("model_call_failed"):
        return "model_api_failure"
    if row.get("output_truncated") or row.get("finish_reason") in {"length", "max_tokens"}:
        return "truncated_output"
    answer = row.get("model_answer")
    if row.get("parse_failure") or row.get("parse_failed") or answer == "":
        return "invalid_answer"
    if row.get("error"):
        return "execution_failure"
    return "correct_answer" if row.get("terminal_reward", 0) > 0 else "incorrect_answer"


def trajectory_statistics(rows):
    def aggregate(subset):
        conditions = defaultdict(int)
        for row in subset:
            conditions[outcome(row)] += 1
        known_answers = [r for r in subset if "model_answer" in r or "valid_answer" in r]
        valid = sum(bool(r.get("valid_answer", r.get("model_answer") in {"A", "B", "C", "D"}))
                    for r in known_answers)
        def total(field):
            values = [r.get(field) for r in subset]
            return sum(values) if values and all(isinstance(v, (int, float)) for v in values) else None
        return {
            "count": len(subset), "rewarded": sum(r.get("terminal_reward", 0) > 0 for r in subset),
            "observed_outcomes": dict(conditions), "valid_answers": valid,
            "valid_answer_rate": valid / len(known_answers) if known_answers else None,
            "answer_validity_unknown": len(subset) - len(known_answers),
            "parse_failures": sum(bool(r.get("parse_failure", r.get("parse_failed", r.get("model_answer") == ""
                                         and outcome(r) != "model_api_failure"))) for r in subset),
            "output_truncations": sum(bool(r.get("output_truncated") or r.get("finish_reason") in {"length", "max_tokens"}) for r in subset),
            "model_api_failures": sum(outcome(r) == "model_api_failure" for r in subset),
            "execution_errors": sum(bool(r.get("error")) for r in subset),
            "input_tokens": total("input_tokens"), "output_tokens": total("output_tokens"),
            "wall_time_seconds": total("wall_time_seconds"),
        }

    stats = aggregate(rows)
    ids = sorted({r.get("question_id") for r in rows}, key=str)
    stats["per_task"] = [{"question_id": qid, **aggregate([r for r in rows if r.get("question_id") == qid])}
                         for qid in ids]
    return stats


def _clip_fields(value, limit, path, truncated):
    if isinstance(value, str) and len(value) > limit:
        truncated.append({"field": path, "original_characters": len(value), "included_characters": limit,
                          "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()})
        return value[:limit]
    if isinstance(value, dict):
        return {key: _clip_fields(item, limit, f"{path}.{key}", truncated) for key, item in value.items()}
    if isinstance(value, list):
        return [_clip_fields(item, limit, f"{path}[{index}]", truncated) for index, item in enumerate(value)]
    return value


def trajectory_evidence(rows, max_chars=72000):
    """Keep all records when they fit; otherwise round-robin task/outcome strata."""
    all_ids = [trajectory_id(row, index) for index, row in enumerate(rows)]
    truncated = []
    if _size(rows) <= max_chars:
        selected = list(range(len(rows)))
        included_rows = copy.deepcopy(rows)
    else:
        groups = defaultdict(deque)
        for index, row in enumerate(rows):
            groups[(str(row.get("question_id")), outcome(row))].append(index)
        order = []
        keys = sorted(groups)
        while any(groups.values()):
            for key in keys:
                if groups[key]:
                    order.append(groups[key].popleft())
        selected, included_rows, used = [], [], 2
        # Give each stratum an opportunity to contribute necessary content before
        # taking a second example. Long fields are explicitly recorded below.
        field_limit = max(128, min(6000, max_chars // max(1, len(keys) * 8)))
        for index in order:
            row_truncations = []
            bounded = _clip_fields(rows[index], field_limit, all_ids[index], row_truncations)
            row_size = _size(bounded) + 2
            if used + row_size <= max_chars:
                selected.append(index)
                included_rows.append(bounded)
                truncated.extend(row_truncations)
                used += row_size
    return included_rows, {
        "total_trajectories": len(rows), "included_trajectory_ids": [all_ids[i] for i in selected],
        "omitted_trajectory_ids": [identifier for i, identifier in enumerate(all_ids) if i not in selected],
        "truncated_fields": truncated, "selection": "all" if len(selected) == len(rows) and not truncated else "deterministic_task_outcome_strata",
        "character_budget": max_chars, "statistics": trajectory_statistics(rows),
    }


def artifact_evidence(state, max_chars=15000):
    root = Path(state.directory).resolve() if state.directory else None
    entries, truncations = [], []
    content_limit = max_chars // max(1, len(state.manifest))
    for manifest in sorted(state.manifest, key=lambda item: item["path"]):
        entry = copy.deepcopy(manifest)
        path = root / manifest["path"] if root else None
        if path is None or not path.is_file():
            entry["content_status"] = "unavailable"
        elif path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Artifact observation path escaped its snapshot")
        else:
            raw = path.read_bytes()
            actual_hash = hashlib.sha256(raw).hexdigest()
            if actual_hash != manifest["sha256"]:
                raise ValueError("Artifact changed since its manifest was recorded")
            entry["source_file"] = str(path)
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                entry["content_status"] = "binary_not_included"
            else:
                entry["content"] = content[:content_limit]
                entry["content_status"] = "complete" if len(content) <= content_limit else "truncated"
                if len(content) > content_limit:
                    truncations.append({"path": manifest["path"], "sha256": actual_hash,
                                        "original_characters": len(content), "included_characters": content_limit})
        entries.append(entry)
    return {"directory": state.directory, "entries": entries}, truncations


def artifact_diff(before, after):
    old = {r["path"]: r for r in before.manifest}
    new = {r["path"]: r for r in after.manifest}
    return {
        "added": [new[p] for p in sorted(new.keys() - old.keys())],
        "removed": [old[p] for p in sorted(old.keys() - new.keys())],
        "changed": [{"before": old[p], "after": new[p]} for p in sorted(old.keys() & new.keys())
                    if old[p]["sha256"] != new[p]["sha256"]],
        "removal_semantics": "Prior input assets not regenerated this round are absent by lifecycle; not accidental overwrites.",
    }


def harness_evidence(path):
    """Read the actual immutable Task configuration and its fixed interpreter.

    Source names are shared by routing, patching and dependency inspection. The
    registry alone determines editable targets; seeing runtime source grants no
    permission to change it.
    """
    path = Path(path)
    if path.suffix == ".json":
        from sia.task_meta.task_harness import harness_identity, task_harness_sources
        return task_harness_sources(path), harness_identity(path)
    files = {"target_agent.py": path.read_text(encoding="utf-8")}
    return files, {"schema_version": "legacy_python",
                   "source_hashes": {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                                     for name, text in files.items()}}


def current_artifact_sources(state):
    """Only the live intervention-base assets enter the dependency namespace."""
    root = Path(state.directory).resolve() if state.directory else None
    files = {}
    for entry in state.manifest:
        path = root / entry["path"] if root else None
        if path is None or not path.is_file():
            raise ValueError("Current artifact is unavailable")
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Artifact source escaped its snapshot")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Artifact source changed since its manifest was recorded")
        try:
            files["assets/" + entry["path"]] = raw.decode("utf-8")
        except UnicodeDecodeError:
            # The observation explicitly records binary content as unavailable.
            continue
    return files


def artifact_facts(assets):
    """Keep identity, origin, rewards and coverage outside G's evidence filter."""
    result = copy.deepcopy(assets)
    for entry in result.get("entries", []):
        entry.pop("content", None)
    return result


def build_observation(task_in, task_out, meta, result, performance_history, history, costs, capabilities, *, retain_raw=False):
    previous = history[-1] if history else None
    traces, coverage = trajectory_evidence(result.trajectories)
    input_assets, input_truncations = artifact_evidence(task_in.artifacts)
    output_assets, output_truncations = artifact_evidence(task_out.artifacts)
    input_provenance_file = (Path(task_in.artifacts.directory).parent / "input_artifact_provenance.json"
                             if task_in.artifacts.directory else None)
    if input_provenance_file and input_provenance_file.is_file():
        input_assets["provenance"] = json.loads(input_provenance_file.read_text(encoding="utf-8"))
        input_assets["provenance_source_file"] = str(input_provenance_file)
    else:
        input_assets["provenance"] = [{**entry, "production_method": "unknown", "terminal_reward": None,
                                       "knowledge_verified": False} for entry in task_in.artifacts.manifest]
    output_assets["provenance"] = copy.deepcopy(result.artifact_provenance)
    coverage["input_artifact_truncations"] = input_truncations
    coverage["output_artifact_truncations"] = output_truncations
    coverage["asset_content_omissions"] = [
        {"snapshot": name, "path": entry["path"], "reason": entry["content_status"]}
        for name, assets in [("input", input_assets), ("output", output_assets)]
        for entry in assets["entries"] if entry["content_status"] in {"unavailable", "binary_not_included"}
    ]
    actions = capabilities.get("actions", capabilities)
    available = {name: copy.deepcopy(actions.get(name, {})) for name in ("HARNESS", "MODEL", "ARTIFACTS")}
    budget = copy.deepcopy(capabilities.get("budget", {}))
    budget["observed_task_costs"] = copy.deepcopy(costs)
    harness_files, harness_identity = harness_evidence(task_out.harness_path)
    return MetaObservation(
        generation=task_in.generation, task_agent_version=task_in.version, meta_agent_version=meta.version,
        current_performance=copy.deepcopy(result.performance), performance_history=copy.deepcopy(performance_history),
        trajectory_summary=trajectory_statistics(result.trajectories),
        success_examples=[r for r in traces if r.get("terminal_reward", 0) > 0],
        failure_examples=[r for r in traces if r.get("terminal_reward", 0) <= 0],
        current_model_ref=task_in.model_ref,
        current_harness_summary=Path(task_in.harness_path).read_text(encoding="utf-8"),
        current_artifact_manifest=copy.deepcopy(task_out.artifacts.manifest),
        previous_action=previous.decision["action"] if previous else None,
        previous_modification_summary=previous.modification["summary"] if previous else None,
        previous_performance_delta=previous.performance_delta if previous else None,
        improvement_history=[experience_summary(e) for e in history], cost_history=copy.deepcopy(costs),
        evaluated_state=asdict(task_in), intervention_base_state=asdict(task_out),
        input_artifacts=input_assets, output_artifacts=output_assets,
        rollout_artifact_diff=copy.deepcopy(result.rollout_artifact_diff) or artifact_diff(task_in.artifacts, task_out.artifacts),
        trajectories=traces, observation_coverage=coverage, available_actions=available, budget=budget,
        raw_trajectories=copy.deepcopy(result.trajectories) if retain_raw else [],
        experience_ledger=[asdict(e) for e in history] if retain_raw else [],
        current_harness_files=harness_files,
        current_harness_identity=harness_identity,
        current_artifact_files=current_artifact_sources(task_out.artifacts) if retain_raw else {},
    )


def operation_input(observation, *, task_state=None, decision=None, current_files=None):
    """Give G the full permitted index; previews never constrain its selection.

    Only the trusted loop supplies this input. Probe/final trajectories and arbitrary
    filesystem queries are deliberately absent. The fixed summary is not selectable.
    """
    rows = copy.deepcopy(observation.raw_trajectories or observation.trajectories)
    domains = sorted({str(row.get("domain", "legacy")) for row in rows})
    files = {**copy.deepcopy(observation.current_harness_files), **copy.deepcopy(observation.current_artifact_files)}
    for name, content in (current_files or {}).items():
        if name in files and files[name] != content:
            raise ValueError(f"Conflicting current evidence source: {name}")
        files[name] = copy.deepcopy(content)
    return {
        "raw_trajectories": rows,
        "experiences": copy.deepcopy(observation.experience_ledger),
        "trusted_facts": {
            "generation": observation.generation,
            "performance": copy.deepcopy(observation.current_performance),
            "trajectory_statistics": trajectory_statistics(rows),
            "by_domain": {domain: trajectory_statistics([r for r in rows if str(r.get("domain", "legacy")) == domain])
                          for domain in domains},
            "available_actions": copy.deepcopy(observation.available_actions),
            "resources": copy.deepcopy(observation.cost_history),
            "external_budget": copy.deepcopy(observation.budget),
            "evaluated_state": copy.deepcopy(observation.evaluated_state),
            "intervention_base_state": copy.deepcopy(observation.intervention_base_state),
            "rollout_artifact_diff": copy.deepcopy(observation.rollout_artifact_diff),
            "current_harness_identity": copy.deepcopy(observation.current_harness_identity),
            "input_artifacts": artifact_facts(observation.input_artifacts),
            "output_artifacts": artifact_facts(observation.output_artifacts),
            "artifact_lifecycle": "A_in produced the evaluated score; only live A_out is the intervention base. Missing A_in files are not restored by HARNESS.",
        },
        "task_state": asdict(task_state) if task_state is not None else copy.deepcopy(observation.intervention_base_state),
        "decision": decision.model_dump(mode="json") if decision is not None else {},
        "latest_experience": copy.deepcopy(observation.experience_ledger[-1]) if observation.experience_ledger else None,
        "current_files": files,
    }
