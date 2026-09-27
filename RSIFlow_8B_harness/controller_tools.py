"""File-oriented tools for a Codex-led Task/Meta experiment.

The controller is an executor, not a policy. In particular, ``compare_scores``
reports numbers but never accepts a candidate, and ``activate_task`` changes the
active Task only when Meta explicitly asks for that operation.

The Task-specific operations are delegated to ``task_adapter.TaskAdapter``.
Meta may repair or extend the tool library as well as edit candidate Task bundles
and experience files. Tool implementation changes do not change the research protocol.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any
from collections import Counter


TASK_OPERATIONS = {
    "rollout", "run_parent", "run_candidate", "sft", "harnessforge",
    "artifacts", "evaluate", "bootstrap", "prepare_meta_evidence", "prepare_evaluation",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class ControllerTools:
    """Execute one Meta-requested operation and return a JSON-serializable fact receipt."""

    def __init__(self, workspace: str | Path, receipt_root: str | Path | None = None,
                 adapter: Any | None = None):
        self.workspace = Path(workspace).resolve()
        self.receipt_root = Path(receipt_root or self.workspace / "tool_receipts").resolve()
        self.adapter = adapter
        from meta_harness_runtime import MetaHarnessRuntime
        project = Path(__file__).resolve().parent
        run_root = self.receipt_root if self.receipt_root.parent == project else self.receipt_root.parent
        self.meta_program = MetaHarnessRuntime(project, run_root)
        self.meta_round = 0

    def _path(self, value: str | Path) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else self.workspace / path).resolve()

    def _adapter(self) -> Any:
        if self.adapter is None:
            from task_adapter import TaskAdapter
            self.adapter = TaskAdapter(project_root=self.workspace, receipt_root=self.receipt_root)
        return self.adapter

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return failures as receipts so Meta can inspect/repair and keep its session."""
        try:
            return self._execute(request)
        except Exception as exc:
            receipt = {"operation": request.get("operation"), "status": "tool_error",
                       "error_type": type(exc).__name__, "error": str(exc)}
            error_path = self.receipt_root / "errors" / (uuid.uuid4().hex + ".json")
            try:
                _write_json(error_path, {**receipt, "traceback": traceback.format_exc()})
                receipt["error_receipt_path"] = str(error_path)
            except OSError:
                pass
            return receipt

    def _execute(self, request: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a single request. Branching is tool dispatch, never policy."""
        operation = request.get("operation")
        arguments = {key: value for key, value in request.items() if key != "operation"}
        self.meta_round = int(arguments.pop('_meta_round', arguments.get('round_number', arguments.get('round', 0))))
        if operation in TASK_OPERATIONS:
            delegated = "rollout" if operation in {"run_parent", "run_candidate"} else operation
            if operation in {"run_parent", "run_candidate"}:
                arguments.setdefault("stage", "parent" if operation == "run_parent" else "candidate")
            result = self._adapter().call(delegated, arguments)
            if not isinstance(result, dict):
                raise TypeError("Task adapter must return a JSON object")
            return {"operation": operation, **result}
        handlers = {
            "run_command": self._run_command,
            "write_text": self._write_text,
            "apply_patch": self._apply_patch,
            "compare_scores": self._compare_scores,
            "activate_task": self._activate_task,
            "deploy_candidate": self._activate_task,
            "append_skills": self._append_skills,
            "maintain_skills": self._maintain_skills,
            "retrieve_skills": self._retrieve_skills,
            "read_skill": self._read_skill,
            "rebuild_skills_index": self._rebuild_skills_index,
            "skill_usage_report": self._skill_usage_report,
            "record_skill_use": self._record_skill_use,
            "compare_task_differences": self._compare_task_differences,
            "snapshot": self._snapshot,
            "snapshot_task_meta": self._snapshot_task_meta,
            "compose_task": self._compose_task,
            "prepare_validation_snapshot": self._prepare_validation_snapshot,
            "read_json": self._read_json,
            "read_text": self._read_text,
            "materialize_harness": self._materialize_harness,
            "meta_harness_status": self._meta_harness_status,
            "materialize_meta_harness": self._materialize_meta_harness,
            "check_meta_harness": self._check_meta_harness,
            "update_meta_harness": self._update_meta_harness,
            "prepare_meta_review": self._prepare_meta_review,
        }
        if operation not in handlers:
            return {"operation": operation, "status": "unknown_operation",
                    "available_operations": sorted(TASK_OPERATIONS | handlers.keys())}
        return {"operation": operation, **handlers[operation](arguments)}

    def _meta_harness_status(self, arguments):
        return self.meta_program.status(self.meta_round)

    def _materialize_meta_harness(self, arguments):
        return self.meta_program.materialize(self._path(arguments['destination']), self.meta_round)

    def _check_meta_harness(self, arguments):
        return self.meta_program.check(self._path(arguments['candidate_dir']))

    def _update_meta_harness(self, arguments):
        return self.meta_program.update(
            decision=arguments['decision'], round_number=self.meta_round,
            candidate=self._path(arguments['candidate_dir']) if arguments.get('candidate_dir') else None,
            reason=arguments.get('reason', ''), gaps=arguments.get('gaps', []),
            evidence_refs=arguments.get('evidence_refs', []))

    def _prepare_meta_review(self, arguments):
        selections = [_read_json(self._path(p)) for p in arguments.get('selection_paths', [])]
        context = {**arguments, 'round': self.meta_round, 'phase': 'meta_review', 'selections': selections}
        review = self.meta_program.invoke('workflow', 'review', context, round_number=self.meta_round)
        return {'status': 'prepared', 'review': review, **self.meta_program.phase(context)}

    def _materialize_harness(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Give Meta an editable full copy of the selected parent's Harness."""
        state = _read_json(self._path(arguments["state_path"]))
        self._adapter()
        from sia.task_meta.harnessforge_manifest import load_manifest
        manifest = self._path(state["harness_path"])
        destination = self._path(arguments["destination"])
        bundle = load_manifest(manifest)
        reused = destination.exists()
        if not reused:
            bundle.materialize(destination)
        project = Path(__file__).resolve().parent
        production = project / "upstream/HarnessForge_4B/harness_production"
        reports = destination.with_name(destination.name + "_production")
        reports.mkdir(parents=True, exist_ok=True)
        templates = [production / name for name in (
            "01_module_localization.yaml", "02_improvement_directions.yaml", "03_harness_generation.yaml")]
        return {"status": "materialized", "destination": str(destination), "reused_directory": reused,
                "files": sorted(str(p.relative_to(destination)) for p in destination.rglob("*")
                                if p.is_file()), "parent_harness_path": str(manifest),
                "production_templates": [str(path) for path in templates],
                "historical_harnesses_path": str(project / "upstream/HarnessForge_4B/evolved_pairs"),
                "localization_report_path": str(reports / "module_localization_report.md"),
                "improvement_direction_path": str(reports / "improvement_direction_brief.md"),
                "next_steps": [
                    "Read template 1 plus parent source, scores/costs, failed AND successful trajectories; write module_localization_report.md.",
                    "Read template 2 plus diagnosis, historical Harnesses and Meta skills; write improvement_direction_brief.md.",
                    "Read template 3 plus both reports; implement a complete candidate in destination, preserving needed parent files.",
                    "Call harnessforge after both reports and complete generation; at most 3 executable checks of this candidate, repairing only after a failed check."],
                "max_validation_attempts": 3}

    def _write_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Materialize exactly the text supplied by Meta."""
        path = self._path(arguments["path"])
        content = arguments["content"]
        if not isinstance(content, str):
            raise TypeError("write_text content must be a string")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"status": "written", "path": str(path),
                "size_bytes": path.stat().st_size, "sha256": _sha256(path)}

    def _apply_patch(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Apply Meta's unified diff in a fresh subprocess; return its diagnostics."""
        patch_text = arguments["patch"]
        if not isinstance(patch_text, str):
            raise TypeError("apply_patch patch must be a string")
        cwd = self._path(arguments.get("cwd", self.workspace))
        changed_paths = []
        for line in patch_text.splitlines():
            if not line.startswith("+++ "):
                continue
            name = line[4:].split("\t", 1)[0]
            if name == "/dev/null":
                continue
            parts = Path(name).parts
            relative = Path(*parts[1:]) if len(parts) > 1 else Path(name)
            changed_paths.append(cwd / relative)
        completed = subprocess.run(
            ["patch", "-p1", "--batch", "--forward"],
            input=patch_text, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=cwd, check=False,
        )
        changed_files = [{"path": str(path), "exists": path.is_file(),
                          "sha256": _sha256(path) if path.is_file() else None}
                         for path in dict.fromkeys(changed_paths)]
        return {"status": "exited", "returncode": completed.returncode,
                "cwd": str(cwd), "stdout": completed.stdout,
                "stderr": completed.stderr, "changed_files": changed_files,
                "patch_sha256": hashlib.sha256(patch_text.encode("utf-8")).hexdigest()}

    def _run_command(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run an explicit Meta-supplied argv; write large output to files."""
        argv = arguments["argv"]
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise ValueError("run_command requires a non-empty string argv list")
        call_dir = self._path(arguments.get("receipt_dir", self.receipt_root / uuid.uuid4().hex))
        call_dir.mkdir(parents=True, exist_ok=True)
        stdout_path, stderr_path = call_dir / "stdout.log", call_dir / "stderr.log"
        cwd = self._path(arguments.get("cwd", self.workspace))
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in arguments.get("env", {}).items()})
        started = time.time()
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                completed = subprocess.run(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                           timeout=arguments.get("timeout_seconds"), check=False)
            status = "exited"
            returncode = completed.returncode
            error = None
        except subprocess.TimeoutExpired as exc:
            status, returncode, error = "timed_out", None, str(exc)
        except OSError as exc:
            status, returncode, error = "launch_error", None, str(exc)
        result = {"status": status, "argv": argv, "cwd": str(cwd), "returncode": returncode,
                  "error": error, "elapsed_seconds": time.time() - started,
                  "stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}
        if arguments.get("result_json"):
            result_path = self._path(arguments["result_json"])
            result["result_json"] = str(result_path)
            result["result_exists"] = result_path.is_file()
            if result_path.is_file():
                result["result_sha256"] = _sha256(result_path)
        receipt = call_dir / "receipt.json"
        _write_json(receipt, result)
        result["receipt_path"] = str(receipt)
        return result

    def _compare_scores(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Return measured differences and identity fields; leave acceptance to Meta."""
        before_path = self._path(arguments["before"])
        after_path = self._path(arguments["after"])
        before, after = _read_json(before_path), _read_json(after_path)
        metric = arguments.get("metric", "macro_success")
        left, right = before.get(metric), after.get(metric)
        delta = right - left if (type(left) in (int, float) and type(right) in (int, float)) else None
        keys = ("status", "total_rollouts", "probe_identity", "training_manifest", "round_id",
                "feedback_role", "evaluation_protocol")
        result = {"status": "compared", "metric": metric, "before": left, "after": right,
                  "delta": delta, "before_path": str(before_path), "after_path": str(after_path),
                  "before_sha256": _sha256(before_path), "after_sha256": _sha256(after_path),
                  "before_facts": {key: before.get(key) for key in keys},
                  "after_facts": {key: after.get(key) for key in keys},
                  "before_domains": before.get("domains"), "after_domains": after.get("domains")}
        result["matching_facts"] = {
            key: (before[key] == after[key] if key in before and key in after else None)
            for key in ("training_manifest", "round_id", "total_rollouts", "probe_identity")
        }
        result["trajectory_pairing"] = self._trajectory_pairing(before_path, after_path, arguments)
        return result

    def _trajectory_pairing(self, before_path: Path, after_path: Path,
                            arguments: dict[str, Any]) -> dict[str, Any]:
        """Report exact rollout-ID coverage without deciding comparability or acceptance."""
        before_rows = self._path(arguments.get("before_trajectories",
                                               before_path.parent / "train_trajectories.jsonl"))
        after_rows = self._path(arguments.get("after_trajectories",
                                              after_path.parent / "train_trajectories.jsonl"))
        if not before_rows.is_file() or not after_rows.is_file():
            return {"available": False, "before_path": str(before_rows),
                    "after_path": str(after_rows),
                    "before_exists": before_rows.is_file(), "after_exists": after_rows.is_file()}

        def collect(path: Path):
            counts = Counter()
            source_hashes = {}
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for raw in stream:
                    digest.update(raw)
                    if not raw.strip():
                        continue
                    row = json.loads(raw)
                    key = (str(row.get("task_id") or row.get("question_id")),
                           str(row.get("rollout_id", 0)))
                    counts[key] += 1
                    source_hashes.setdefault(key, row.get("task_source_hash"))
            return counts, source_hashes, digest.hexdigest()

        before_counts, before_hashes, before_digest = collect(before_rows)
        after_counts, after_hashes, after_digest = collect(after_rows)
        before_ids, after_ids = set(before_counts), set(after_counts)
        shared = before_ids & after_ids

        def label(key):
            return {"task_id": key[0], "rollout_id": key[1]}

        known = [key for key in shared if before_hashes.get(key) is not None
                 and after_hashes.get(key) is not None]
        mismatched = [key for key in known if before_hashes[key] != after_hashes[key]]
        return {
            "available": True, "before_path": str(before_rows), "after_path": str(after_rows),
            "before_sha256": before_digest, "after_sha256": after_digest,
            "before_rows": sum(before_counts.values()), "after_rows": sum(after_counts.values()),
            "before_unique_ids": len(before_ids), "after_unique_ids": len(after_ids),
            "shared_ids": len(shared), "same_id_set": before_ids == after_ids,
            "duplicate_before_rows": sum(count - 1 for count in before_counts.values()),
            "duplicate_after_rows": sum(count - 1 for count in after_counts.values()),
            "missing_in_after": [label(key) for key in sorted(before_ids - after_ids)],
            "extra_in_after": [label(key) for key in sorted(after_ids - before_ids)],
            "known_source_hash_pairs": len(known),
            "source_hash_mismatches": [label(key) for key in sorted(mismatched)],
        }

    def _activate_task(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Copy the Task state selected *by Meta* to the active-state file."""
        source = self._path(arguments["source_state"])
        target = self._path(arguments["active_state"])
        state = _read_json(source)
        _write_json(target, state)
        return {"status": "activated_as_requested", "source_state": str(source),
                "active_state": str(target), "source_sha256": _sha256(source),
                "active_sha256": _sha256(target), "selection_note": arguments.get("selection_note")}

    def _append_skills(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Append cases/rules/incidents/revisions and rebuild their derived index."""
        path = self._path(arguments["path"])
        if self.meta_program.status(self.meta_round)['fixed_skills']:
            return {'status': 'skills_frozen', 'path': str(path)}
        result = self.meta_program.invoke('memory', 'append', str(path), arguments['entries'], round_number=self.meta_round)
        return {**result, "sha256": _sha256(path)}

    def _retrieve_skills(self, arguments: dict[str, Any]) -> dict[str, Any]:
        from skill_memory import DEFAULT_CONTEXT_CHARS, DEFAULT_PER_CATEGORY, SkillMemory
        fingerprint = (_read_json(self._path(arguments["fingerprint_path"]))
                       if arguments.get("fingerprint_path") else {})
        fingerprint.update(arguments.get("fingerprint", {}))
        retrieval_path = self.receipt_root / "skill_retrievals" / (uuid.uuid4().hex + ".json")
        budget = max(1024, int(arguments.get("max_chars", DEFAULT_CONTEXT_CHARS)))
        # Reserve the receipt envelope as well as the actual retrieved cards.
        overhead = 1200 + len(str(retrieval_path))
        options = {'max_chars': max(512, budget - overhead),
                   'per_category': arguments.get('per_category', DEFAULT_PER_CATEGORY),
                   'components': arguments.get('components'),
                   'include_inactive': arguments.get('include_inactive', False)}
        result = self.meta_program.invoke('memory', 'retrieve', str(self._path(arguments['path'])),
                                          fingerprint, options, round_number=self.meta_round)
        _write_json(retrieval_path, {"fingerprint": fingerprint, "result": result})
        return {**result, "retrieval_path": str(retrieval_path)}

    def _maintain_skills(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.meta_program.status(self.meta_round)['fixed_skills']:
            return {'status': 'skills_frozen', 'path': str(self._path(arguments['path']))}
        return self.meta_program.invoke('memory', 'maintain', str(self._path(arguments['path'])),
                                        arguments['operations'], round_number=self.meta_round)

    def _read_skill(self, arguments: dict[str, Any]) -> dict[str, Any]:
        from skill_memory import SkillMemory
        return SkillMemory(self._path(arguments["path"])).read_skill(
            arguments["ids"], offset_chars=arguments.get("offset_chars", 0),
            max_chars=arguments.get("max_chars", 65536))

    def _rebuild_skills_index(self, arguments: dict[str, Any]) -> dict[str, Any]:
        from skill_memory import SkillMemory
        memory = SkillMemory(self._path(arguments["path"]))
        index = memory.rebuild_index()
        return {"status": "indexed", "index_path": str(memory.index_path),
                "record_count": index["record_count"], "event_count": index["event_count"],
                "counts": index["counts"], "warnings": index["warnings"]}

    def _record_skill_use(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Freeze Meta's cited IDs and prediction before producing the candidate."""
        from skill_memory import SkillMemory
        path = self._path(arguments["path"])
        if path.exists():
            return {"status": "already_recorded", "path": str(path),
                    "usage": _read_json(path), "note": "Original prospective record preserved."}
        decision = _read_json(self._path(arguments["decision_path"])) if arguments.get("decision_path") else arguments
        used = decision.get("relevant_skill_ids", decision.get("used_skill_ids", []))
        used = [used] if isinstance(used, str) else used
        records, events, warnings = SkillMemory(self._path(arguments["skills_path"])).view()
        usage = {"component": decision.get("component"), "round": decision.get("round"),
                 "attempt": decision.get("attempt"), "used_skill_ids": used,
                 "prediction": decision.get("prediction", {}), "rationale": decision.get("rationale"),
                 "trajectory_refs": decision.get("trajectory_refs", []),
                 "skills_path": str(self._path(arguments["skills_path"])),
                 "ledger_event_count_at_decision": len(events),
                 "decision_path": arguments.get("decision_path"),
                 "retrieval_path": arguments.get("retrieval_path"),
                 "unknown_skill_ids": [identifier for identifier in used if identifier not in records],
                 "warnings": warnings,
                 "interpretation": "Cited IDs are Meta's declared use; file reads do not prove understanding."}
        _write_json(path, usage)
        return {"status": "recorded", "path": str(path), "usage": usage}

    def _skill_usage_report(self, arguments: dict[str, Any]) -> dict[str, Any]:
        from skill_memory import SkillMemory
        result = SkillMemory(self._path(arguments["path"])).usage_report()
        report = self._path(arguments.get("output_path", self.receipt_root / "skill_usage_report.json"))
        _write_json(report, result)
        return {"status": "reported", "report_path": str(report),
                "unique_interventions": result["unique_interventions"],
                "first_candidate_measured": result["first_candidate_measured"],
                "first_candidate_positive_rate": result["first_candidate_positive_rate"],
                "prediction_assessed": result["prediction_assessed"],
                "prediction_hit_rate_declared_by_meta": result["prediction_hit_rate_declared_by_meta"]}

    def _compare_task_differences(self, arguments: dict[str, Any]) -> dict[str, Any]:
        from task_differences import compare_task_differences
        return compare_task_differences(
            self._path(arguments["before_trajectories"]), self._path(arguments["after_trajectories"]),
            self._path(arguments["output_dir"]),
            self._path(arguments["skill_use_path"]) if arguments.get("skill_use_path") else None)

    def _snapshot(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Copy explicit Task/Meta paths to an immutable per-round snapshot directory."""
        destination = self._path(arguments["destination"])
        if destination.exists():
            return {"status": "destination_exists", "destination": str(destination)}
        destination.mkdir(parents=True)
        copied = {}
        for label, raw_source in arguments["sources"].items():
            source = self._path(raw_source)
            target = destination / label
            if source.is_dir():
                shutil.copytree(source, target)
                copied[label] = {"source": str(source), "path": str(target), "type": "directory"}
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied[label] = {"source": str(source), "path": str(target), "type": "file",
                                 "sha256": _sha256(target)}
        receipt = destination / "snapshot_receipt.json"
        _write_json(receipt, {"status": "copied", "sources": copied})
        return {"status": "copied", "destination": str(destination), "sources": copied,
                "receipt_path": str(receipt)}

    def _snapshot_task_meta(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Save numbered component references, not another full Task/Meta tree."""
        from component_versions import ComponentVersions, content_hash, snapshot
        destination = self._path(arguments["destination"])
        run_root = (destination.parent.parent if destination.parent.name == "snapshots"
                    else self.receipt_root.parent)
        versions = ComponentVersions(
            self._path(arguments.get("versions_root", run_root / "versions")), self.workspace)
        states = {}
        for role in ("parent", "candidate"):
            if arguments.get(role + "_task_state"):
                states[role] = self._path(arguments[role + "_task_state"])
        states["selected"] = self._path(arguments["active_task_state"])
        # Preserve the small handoff, not the Codex executable or long chat history.
        references = {key: arguments[key] for key in
                      ("context_path", "meta_state_path", "meta_harness_path",
                       "controller_path", "controller_metadata", "codex_thread_metadata")
                      if key in arguments}
        references['meta_program'] = self.meta_program.snapshot_reference(self.meta_round)
        if arguments.get("context_path"):
            context = self._path(arguments["context_path"])
            if context.is_file():
                references["context_reference"] = {
                    "source": arguments["context_path"], "path": str(context),
                    "sha256": content_hash(context)}
        return snapshot(versions, destination, states,
                        self._path(arguments["skills_path"]), references)

    def _compose_task(self, arguments: dict[str, Any]) -> dict[str, Any]:
        from component_versions import ComponentVersions
        versions = ComponentVersions(self._path(arguments["versions_root"]), self.workspace)
        return versions.compose(self._path(arguments["template_state_path"]),
                                arguments["model_id"], arguments["harness_id"],
                                self._path(arguments["destination"]), arguments.get('artifacts_id'))

    def _prepare_validation_snapshot(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Write the report-only evaluator handoff for Meta's selected Task state."""
        task_path = self._path(arguments["task_state_path"])
        meta_path = self._path(arguments["meta_state_path"]) if arguments.get("meta_state_path") else None
        source_round = self._path(arguments["source_round_path"])
        destination = self._path(arguments["destination"])
        if destination.suffix != ".json":
            destination = destination / "round_snapshot.json"
        task_state = _read_json(task_path)
        meta_state = _read_json(meta_path) if meta_path else {"version": 0}
        value = {
            "round": arguments["round"],
            "task_state": task_state,
            "meta_state": meta_state,
            "deployment_status": arguments.get("deployment_status", "meta_selected"),
            "chosen_component": arguments.get("chosen_component"),
            "source_round": str(source_round),
            "source_role": "independent_validation",
            "purpose": "report_only",
        }
        _write_json(destination, value)
        return {"status": "prepared", "snapshot_path": str(destination),
                "snapshot_sha256": _sha256(destination), "task_state_path": str(task_path),
                "task_state_sha256": _sha256(task_path),
                "meta_state_path": str(meta_path) if meta_path else None,
                "source_round_path": str(source_round), "round": arguments["round"]}

    def _read_json(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._path(arguments["path"])
        return {"status": "read", "path": str(path), "sha256": _sha256(path),
                "value": _read_json(path)}

    def _read_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._path(arguments["path"])
        offset = max(0, int(arguments.get("offset_chars", 0)))
        limit = max(1, min(int(arguments.get("max_chars", 65536)), 65536))
        digest = hashlib.sha256()
        decoder = codecs.getincrementaldecoder("utf-8")()
        pieces = []
        total = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                decoded = decoder.decode(chunk)
                start, end = total, total + len(decoded)
                if end > offset and start < offset + limit:
                    pieces.append(decoded[max(0, offset - start):min(len(decoded), offset + limit - start)])
                total = end
            trailing = decoder.decode(b"", final=True)
            if trailing and total < offset + limit and total + len(trailing) > offset:
                pieces.append(trailing[max(0, offset - total):min(len(trailing), offset + limit - total)])
            total += len(trailing)
        content = "".join(pieces)
        next_offset = min(total, offset + len(content))
        return {"status": "read", "path": str(path), "sha256": digest.hexdigest(),
                "content": content, "offset_chars": offset,
                "next_offset_chars": next_offset, "total_chars": total,
                "truncated": next_offset < total}


def execute(request: dict[str, Any], *, workspace: str | Path | None = None,
            receipt_root: str | Path | None = None, adapter: Any | None = None) -> dict[str, Any]:
    return ControllerTools(workspace or Path(__file__).resolve().parent, receipt_root, adapter).execute(request)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--receipts")
    parser.add_argument(
        "--result-file",
        type=Path,
        help="atomically write the final JSON receipt here instead of stdout",
    )
    args = parser.parse_args(argv)
    request = json.load(sys.stdin)
    try:
        result = execute(request, workspace=args.workspace, receipt_root=args.receipts)
    except Exception as exc:
        result = {"operation": request.get("operation"), "status": "tool_error",
                  "error_type": type(exc).__name__, "error": str(exc)}
    if args.result_file is not None:
        _write_json(args.result_file, result)
    else:
        print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
