"""Persistent, advisory progress ledger for the Task/Meta experiment.

The ledger is deliberately not a policy engine.  It reconstructs facts from
``meta_session/turns/*/command.json`` plus successful ``receipt.json`` files,
and never blocks a tool, chooses a component, or decides whether a candidate
should be accepted.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
MAINLINE_GOAL = (
    "Complete A0 independent validation, then B1-B{rounds} evolution rounds.  In each "
    "round Meta tests one candidate per attempt on the same batch; rejection "
    "appends failure experience and returns to component selection. After acceptance "
    "preserve Task/Meta state and run report-only independent validation."
)

A0_MILESTONES = (
    "bootstrap",
    "task_meta_snapshot",
    "validation_snapshot",
    "independent_validation",
)

ROUND_MILESTONES = (
    "parent_rollout",
    "meta_evidence",
    "decision_recorded",
    "candidate_built",
    "candidate_rollout",
    "scores_compared",
    "selection_recorded",
    "skills_appended",
    "context_updated",
    "task_meta_snapshot",
    "validation_snapshot",
    "independent_validation",
)

MILESTONE_PURPOSES = {
    "bootstrap": "Materialize the initial active Task state.",
    "parent_rollout": "Collect the active parent's round-disjoint training outcomes.",
    "meta_evidence": "Expose aggregate outcomes and representative trajectory evidence to Meta.",
    "component_reselection": "Read the rejected attempt and its experience, then choose a component again on the SAME parent/batch; do not independently validate yet.",
    "decision_recorded": "Record Meta's component choice and evidence-grounded rationale.",
    "candidate_built": "Materialize the one candidate selected by Meta for this attempt.",
    "candidate_rollout": "Evaluate the candidate on exactly the parent's round tasks.",
    "scores_compared": "Obtain factual paired parent/candidate score comparisons.",
    "selection_recorded": "Record Meta's explicit accept-or-retain decision and supporting facts.",
    "skills_appended": "Preserve bounded component experience and general principles.",
    "context_updated": "Preserve the concise evidence-linked handoff for the next round.",
    "task_meta_snapshot": "Snapshot the selected Task state and persistent Meta state.",
    "validation_snapshot": "Prepare a frozen report-only independent-validation handoff.",
    "independent_validation": "Complete report-only independent benchmark evaluation.",
}

_BAD_STATUSES = {
    "running", "submitted", "waiting", "job_failed",
    "tool_error", "tool_process_error", "invalid_tool_receipt", "unknown_operation",
    "timed_out", "launch_error", "missing_checkpoint", "failed", "error", "validation_failed",
}
_ROUND_RE = re.compile(r"(?:^|[/_-])(?:round[_-]?|b)([0-9]+)(?:$|[/_.-])", re.IGNORECASE)
_A0_RE = re.compile(r"(?:^|[/_-])a0(?:$|[/_.-])", re.IGNORECASE)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _successful(operation: str | None, receipt: dict[str, Any]) -> bool:
    """Recognize an executed operation without imposing domain policy."""
    if receipt.get("ok") is False:
        return False
    if receipt.get("operation") not in {None, operation}:
        return False
    status = str(receipt.get("status", "")).lower()
    if status in _BAD_STATUSES or status.endswith("_error"):
        return False
    returncode = receipt.get("returncode")
    if returncode is not None and returncode != 0:
        return False
    if operation in {"harnessforge", "sft", "artifacts"} and not receipt.get("candidate_state_path"):
        return False
    if operation == "evaluate" and not (receipt.get("complete_path") or receipt.get("partial_path")):
        return False
    if operation in {"append_skills", "maintain_skills"} and (int(receipt.get("appended_count", 0))
                                        + len(receipt.get("reused_ids", []))) < 1:
        return False
    if operation == "compare_scores" and status != "compared":
        return False
    if operation == "snapshot_task_meta" and status != "snapshotted":
        return False
    return True


def _path_refs(value: Any, *, parent_key: str = "") -> list[str]:
    """Keep evidence references, never large receipt payloads, in the ledger."""
    refs: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            lower = key.lower()
            if isinstance(item, str) and (
                lower == "destination" or lower == "path" or lower.endswith("_path")
                or lower.endswith("_dir") or lower.endswith("_directory")
            ):
                refs.append(item)
            elif isinstance(item, (dict, list)):
                refs.extend(_path_refs(item, parent_key=lower))
    elif isinstance(value, list):
        for item in value:
            refs.extend(_path_refs(item, parent_key=parent_key))
    return list(dict.fromkeys(refs))


def _explicit_round(request: dict[str, Any], receipt: dict[str, Any] | None) -> int | None:
    for source in (request, receipt or {}):
        for key in ("round_number", "round"):
            value = source.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
    refs = _path_refs(request) + _path_refs(receipt or {})
    # Run names may contain ``a0`` even while later B-round artifacts live
    # below them. Prefer the most specific B/round path before the run-root
    # A0 marker.
    for ref in reversed(refs):
        match = _ROUND_RE.search(ref)
        if match:
            return int(match.group(1))
    if any(_A0_RE.search(ref) for ref in refs):
        return 0
    return None


def _milestone(operation: str | None, request: dict[str, Any]) -> str | None:
    if operation == "bootstrap":
        return "bootstrap"
    if operation == "run_parent":
        return "parent_rollout"
    if operation == "prepare_meta_evidence":
        return "meta_evidence"
    if operation in {"harnessforge", "sft", "artifacts"}:
        return "candidate_built"
    if operation == "run_candidate":
        return "candidate_rollout"
    if operation == "compare_scores":
        return "scores_compared"
    if operation in {"append_skills", "maintain_skills"}:
        return "skills_appended"
    if operation == "snapshot_task_meta":
        return "task_meta_snapshot"
    if operation == "prepare_validation_snapshot":
        return "validation_snapshot"
    if operation == "evaluate":
        return "independent_validation"
    if operation == "write_text":
        path = str(request.get("path", "")).replace("\\", "/").lower()
        if path.endswith("/decision.json"):
            return "decision_recorded"
        if path.endswith("/selection.json"):
            return "selection_recorded"
        if path.endswith("/meta/context.json") or path.endswith("/context.json"):
            return "context_updated"
    return None


def _stage_template(rounds: int) -> list[dict[str, Any]]:
    stages = [{
        "id": "A0",
        "round": 0,
        "purpose": "Freeze the initial Task/Meta state and establish the independent baseline.",
        "required_milestones": list(A0_MILESTONES),
        "completed_milestones": [],
        "remaining_milestones": list(A0_MILESTONES),
        "evidence": [],
        "status": "pending",
    }]
    for number in range(1, rounds + 1):
        stages.append({
            "id": f"B{number}",
            "round": number,
            "purpose": (
                f"Evolution round {number}: use fresh training tasks to diagnose the parent, "
                "test one Meta-chosen candidate per attempt, retry after rejection, then independently report the accepted Task."
            ),
            "required_milestones": list(ROUND_MILESTONES),
            "completed_milestones": [],
            "remaining_milestones": list(ROUND_MILESTONES),
            "evidence": [],
            "status": "pending",
        })
    return stages


def _earliest_incomplete_round(stages: list[dict[str, Any]]) -> int:
    for stage in stages:
        if stage["remaining_milestones"]:
            return int(stage["round"])
    return int(stages[-1]["round"])


def _refresh_stage(stage: dict[str, Any]) -> None:
    completed = set(stage["completed_milestones"])
    stage["remaining_milestones"] = [
        item for item in stage["required_milestones"] if item not in completed
    ]
    if stage.get("retry_pending"):
        # Report Meta's recorded rejection, never infer it from a numerical threshold.
        stage["remaining_milestones"] = (
            ([] if "skills_appended" in completed else ["skills_appended"])
            + ["component_reselection"])
    if not stage["remaining_milestones"]:
        stage["status"] = "complete"
    elif completed:
        stage["status"] = "in_progress"
    else:
        stage["status"] = "pending"


class ExperimentProgress:
    """Rebuild and render an advisory experiment-progress ledger."""

    def __init__(self, meta_session: Path, *, rounds: int = 3):
        if rounds < 1:
            raise ValueError("rounds must be at least one")
        self.meta_session = Path(meta_session).resolve()
        self.rounds = rounds

    @property
    def state_path(self) -> Path:
        return self.meta_session / "workflow_state.json"

    @property
    def events_path(self) -> Path:
        return self.meta_session / "workflow_events.jsonl"

    @property
    def status_path(self) -> Path:
        return self.meta_session / "RUN_STATUS.md"

    def rebuild(self) -> dict[str, Any]:
        policy = _read_object(self.meta_session / "workflow_policy.json") or {}
        self.rounds = int(policy.get("rounds", self.rounds))
        stages = _stage_template(self.rounds)
        final_only = policy.get('evaluation_schedule') == 'final_only'
        if final_only:
            for stage in stages:
                if stage['round'] != self.rounds:
                    stage['required_milestones'] = [m for m in stage['required_milestones']
                        if m not in {'validation_snapshot', 'independent_validation'}]
                stage['purpose'] = ('Save initial Task/Meta state; no A0 evaluation.' if stage['round'] == 0
                    else f"Evolution round {stage['round']}: evolve Task and maintain skills; save the selected complete Task. "
                    + ('Then independently evaluate that final Task.' if stage['round'] == self.rounds
                       else 'Continue to the next fresh batch without independent evaluation.'))
                _refresh_stage(stage)
        retry_from = int(policy.get("retry_rejected_from_round", 1))
        by_round = {stage["round"]: stage for stage in stages}
        events: list[dict[str, Any]] = []
        turns = self.meta_session / "turns"
        turn_dirs = sorted((path for path in turns.iterdir() if path.is_dir()),
                           key=lambda path: path.name) if turns.is_dir() else []

        for turn_dir in turn_dirs:
            command = _read_object(turn_dir / "command.json")
            if command is None:
                continue
            action = command.get("action")
            request = command.get("request") if isinstance(command.get("request"), dict) else {}
            operation = request.get("operation") if isinstance(request.get("operation"), str) else None
            receipt_path = turn_dir / "receipt.json"
            receipt = _read_object(receipt_path)
            # Long operations return a job id immediately, so the turn receipt is
            # written while the controller job is still "running". Resolve the
            # durable job result so completed work is not reported as pending.
            if isinstance(receipt, dict) and str(receipt.get("status", "")).lower() == "running":
                durable_path = receipt.get("result_path")
                if isinstance(durable_path, str) and durable_path:
                    durable = _read_object(Path(durable_path))
                    if isinstance(durable, dict) and durable.get("status") not in {None, "running"}:
                        receipt = durable
            round_number = _explicit_round(request, receipt)
            if round_number is None:
                round_number = _earliest_incomplete_round(stages)
            stage = by_round.get(round_number)
            milestone = _milestone(operation, request)
            succeeded = bool(receipt is not None and _successful(operation, receipt))
            event = {
                "turn": turn_dir.name,
                "action": action,
                "operation": operation,
                "stage": stage["id"] if stage else None,
                "round": round_number,
                "receipt_status": receipt.get("status") if receipt else None,
                "outcome": "succeeded" if succeeded else ("failed" if receipt else "pending"),
                "milestone": milestone,
                "evidence_paths": _path_refs(receipt or request),
                "command_path": str(turn_dir / "command.json"),
                "receipt_path": str(receipt_path) if receipt else None,
            }
            events.append(event)
            if not (stage and milestone and succeeded):
                continue
            if milestone not in stage["required_milestones"]:
                continue
            if stage["round"] >= retry_from and milestone == "decision_recorded":
                decision_path = str(request.get("path", ""))
                previous_path = stage.get("decision_path")
                if previous_path and (stage.get("retry_pending") or previous_path != decision_path):
                    # Preserve earlier evidence on disk and in events, but do not let attempt 1's
                    # rollout/score/snapshot receipts satisfy attempt 2's unfinished milestones.
                    keep = {"parent_rollout", "meta_evidence"}
                    stage["completed_milestones"] = [m for m in stage["completed_milestones"] if m in keep]
                    stage["evidence"] = [e for e in stage["evidence"] if e["milestone"] in keep]
                    stage["attempt"] = stage.get("attempt", 1) + 1
                stage.setdefault("attempt", 1)
                stage["decision_path"] = decision_path
                stage["retry_pending"] = False
            if stage["round"] >= retry_from and milestone == "selection_recorded":
                selection = None
                try:
                    selection = json.loads(request.get("content", ""))
                except (TypeError, ValueError):
                    if request.get("path"):
                        selection = _read_object(Path(request["path"]))
                if isinstance(selection, dict):
                    choice = selection.get("accept_or_retain")
                    if choice in {"retain_parent", "reject", "rejected"}:
                        stage["retry_pending"] = True
                        stage["last_selection"] = choice
                        # These belong to the previous accepted round, not this rejected attempt.
                        late = {"task_meta_snapshot", "validation_snapshot", "independent_validation"}
                        stage["completed_milestones"] = [m for m in stage["completed_milestones"] if m not in late]
                    elif choice in {"accept", "accepted", "accept_candidate"}:
                        stage["retry_pending"] = False
                        stage["last_selection"] = choice
            if milestone not in stage["completed_milestones"]:
                stage["completed_milestones"].append(milestone)
            if (milestone == "task_meta_snapshot" and stage["round"] > 0
                    and request.get("context_path")
                    and (request["context_path"] in (receipt.get("sources") or {}).values()
                         or (request["context_path"] == (receipt.get("context_reference") or {}).get("source")
                             and (receipt.get("context_reference") or {}).get("sha256")))
                    and "context_updated" not in stage["completed_milestones"]):
                # Native Codex edits do not issue write_text receipts; a successful snapshot
                # that copied OR recorded the handoff file is evidence of that artifact.
                stage["completed_milestones"].append("context_updated")
            evidence = {
                "milestone": milestone,
                "turn": turn_dir.name,
                "operation": operation,
                "receipt_status": receipt.get("status"),
                "paths": _path_refs(receipt),
            }
            stage["evidence"] = [item for item in stage["evidence"]
                                 if item["milestone"] != milestone]
            stage["evidence"].append(evidence)
            _refresh_stage(stage)

        for stage in stages:
            _refresh_stage(stage)
        current = next((stage for stage in stages if stage["status"] != "complete"), None)
        finish = current is None
        state = {
            "schema_version": SCHEMA_VERSION,
            "rounds": self.rounds,
            "retry_rejected_from_round": retry_from,
            "mainline_goal": (f"Save A0, complete B1-B{self.rounds} with a snapshot each round, then independently evaluate ONLY the final selected complete Task."
                              if final_only else MAINLINE_GOAL.format(rounds=self.rounds)),
            "current_stage": current["id"] if current else "COMPLETE",
            "current_stage_purpose": current["purpose"] if current else "All required stages are complete.",
            "next_milestone": current["remaining_milestones"][0] if current else None,
            "next_milestone_purpose": (
                MILESTONE_PURPOSES[current["remaining_milestones"][0]] if current else None
            ),
            "next_stage": self._next_stage_id(stages, current),
            "final_stage": f"B{self.rounds}",
            "finish_allowed": finish,
            "observed_turns": len(events),
            "stages": stages,
        }
        self.meta_session.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.state_path, state)
        event_text = "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events)
        _atomic_text(self.events_path, event_text)
        _atomic_text(self.status_path, self._render_markdown(state))
        return state

    def record_turn(self, turn_dir: Path | None = None) -> dict[str, Any]:
        """Refresh after a receipt. ``turn_dir`` is an optional integration hint."""
        del turn_dir
        return self.rebuild()

    def compact_status_block(self) -> str:
        state = self.rebuild()
        current = next((item for item in state["stages"]
                        if item["id"] == state["current_stage"]), None)
        completed = ", ".join(current["completed_milestones"]) if current else "all"
        completed = completed or "none"
        acquired: list[str] = []
        if current:
            for evidence in current["evidence"]:
                acquired.extend(evidence["paths"])
        acquired_text = ", ".join(list(dict.fromkeys(acquired))[-4:]) or "none yet"
        return (
            "<MAINLINE_PROGRESS>\n"
            f"Goal: {state['mainline_goal']}\n"
            f"Current stage: {state['current_stage']} — {state['current_stage_purpose']}\n"
            f"Attempt: {current.get('attempt', 1) if current else 'none'}; rejected attempt awaiting reselection: {bool(current and current.get('retry_pending'))}\n"
            f"Completed here: {completed}\n"
            f"Acquired evidence: {acquired_text}\n"
            f"Next required milestone: {state['next_milestone'] or 'none'}"
            + (f" — {state['next_milestone_purpose']}" if state["next_milestone_purpose"] else "")
            + f"\nFollowing stage: {state['next_stage'] or 'none'}; final stage: {state['final_stage']}\n"
            f"Finish allowed: {'yes' if state['finish_allowed'] else 'no'}\n"
            "This block reports progress only. Meta still chooses the component, candidate, and acceptance.\n"
            "</MAINLINE_PROGRESS>"
        )

    def finish_allowed(self) -> bool:
        return bool(self.rebuild()["finish_allowed"])

    @staticmethod
    def _next_stage_id(stages: list[dict[str, Any]], current: dict[str, Any] | None) -> str | None:
        if current is None:
            return None
        index = stages.index(current)
        return stages[index + 1]["id"] if index + 1 < len(stages) else None

    @staticmethod
    def _render_markdown(state: dict[str, Any]) -> str:
        lines = [
            "# Experiment run status",
            "",
            f"Mainline: {state['mainline_goal']}",
            "",
            f"Current stage: **{state['current_stage']}**",
            f"Next required milestone: **{state['next_milestone'] or 'none'}**",
            f"Final stage: **{state['final_stage']}**",
            f"Finish allowed: **{'yes' if state['finish_allowed'] else 'no'}**",
            "",
            "## Stages",
            "",
        ]
        for stage in state["stages"]:
            lines.extend([
                f"### {stage['id']} — {stage['status']}",
                "",
                stage["purpose"],
                "",
                "Completed: " + (", ".join(stage["completed_milestones"]) or "none"),
                "",
                "Remaining: " + (", ".join(stage["remaining_milestones"]) or "none"),
                "",
            ])
            refs = list(dict.fromkeys(path for evidence in stage["evidence"] for path in evidence["paths"]))
            if refs:
                lines.append("Evidence paths:")
                lines.append("")
                lines.extend(f"- `{path}`" for path in refs)
                lines.append("")
        lines.extend([
            "This is an advisory ledger reconstructed from command/receipt facts. It does not",
            "choose components or candidates, decide acceptance, or block controller tools.",
            "",
        ])
        return "\n".join(lines)


def rebuild_progress(meta_session: Path, rounds: int = 3) -> dict[str, Any]:
    return ExperimentProgress(meta_session, rounds=rounds).rebuild()


def compact_status_block(meta_session: Path, rounds: int = 3) -> str:
    return ExperimentProgress(meta_session, rounds=rounds).compact_status_block()


def finish_allowed(meta_session: Path, rounds: int = 3) -> bool:
    return ExperimentProgress(meta_session, rounds=rounds).finish_allowed()
