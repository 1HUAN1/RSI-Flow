"""Tests for the advisory, reconstructable experiment-progress ledger."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_progress import (ExperimentProgress, compact_status_block,
                                 finish_allowed, rebuild_progress)  # noqa: E402


def _turn(session: Path, number: int, operation: str, *, round_number: int | None = None,
          status: str = "completed", path: str | None = None, returncode: int | None = None):
    turn = session / "turns" / f"{number:04d}"
    turn.mkdir(parents=True)
    request = {"operation": operation}
    if round_number is not None:
        request["round_number"] = round_number
    if path is not None:
        request["path"] = path
    (turn / "command.json").write_text(json.dumps({"action": "tool", "request": request}))
    receipt = {"operation": operation, "status": status,
               "output_path": f"/run/evidence/{number}.json"}
    if returncode is not None:
        receipt["returncode"] = returncode
    if operation in {"harnessforge", "sft", "artifacts"}:
        receipt["candidate_state_path"] = f"/run/candidates/{number}/task_state.json"
    elif operation == "evaluate":
        receipt["complete_path"] = f"/run/evaluations/{number}/complete.json"
    elif operation == "append_skills":
        receipt["appended_count"] = 1
    elif operation == "compare_scores":
        if status == "completed":
            receipt["status"] = "compared"
    elif operation == "snapshot_task_meta":
        if status == "completed":
            receipt["status"] = "snapshotted"
    (turn / "receipt.json").write_text(json.dumps(receipt))
    return turn


def _complete_a0(session: Path, start: int = 0) -> int:
    for offset, operation in enumerate((
        "bootstrap", "snapshot_task_meta", "prepare_validation_snapshot", "evaluate"
    )):
        _turn(session, start + offset, operation, round_number=0,
              status="exited" if operation == "evaluate" else "completed",
              returncode=0 if operation == "evaluate" else None)
    return start + 4


def _complete_round(session: Path, number: int, start: int) -> int:
    operations = [
        ("run_parent", None),
        ("prepare_meta_evidence", None),
        ("write_text", f"/run/round_{number}/decision.json"),
        ("harnessforge", None),
        ("run_candidate", None),
        ("compare_scores", None),
        ("write_text", f"/run/round_{number}/selection.json"),
        ("append_skills", None),
        ("write_text", "/run/meta/context.json"),
        ("snapshot_task_meta", None),
        ("prepare_validation_snapshot", None),
        ("evaluate", None),
    ]
    for offset, (operation, path) in enumerate(operations):
        _turn(session, start + offset, operation, round_number=number, path=path,
              status="exited" if operation == "evaluate" else "completed",
              returncode=0 if operation == "evaluate" else None)
    return start + len(operations)


def test_empty_ledger_points_to_a0_without_allowing_finish(tmp_path):
    session = tmp_path / "meta_session"
    state = rebuild_progress(session)

    assert state["current_stage"] == "A0"
    assert state["next_milestone"] == "bootstrap"
    assert state["finish_allowed"] is False
    assert (session / "workflow_state.json").is_file()
    assert (session / "workflow_events.jsonl").read_text() == ""
    assert "Current stage: **A0**" in (session / "RUN_STATUS.md").read_text()


def _selection_content(turn, choice):
    path = turn / "command.json"
    command = json.loads(path.read_text())
    command["request"]["content"] = json.dumps({"accept_or_retain": choice})
    path.write_text(json.dumps(command))


def test_rejection_reselects_same_round_and_new_attempt_needs_new_results(tmp_path):
    session = tmp_path / "meta_session"
    start = _complete_a0(session)
    end = _complete_round(session, 1, start)
    # Even old snapshot/eval receipts cannot make a rejected round finish.
    _selection_content(session / "turns" / f"{start + 6:04d}", "retain_parent")
    state = rebuild_progress(session, rounds=1)
    assert state["current_stage"] == "B1"
    assert state["next_milestone"] == "component_reselection"
    assert not state["finish_allowed"]
    assert state["stages"][1]["retry_pending"]

    _turn(session, end, "write_text", round_number=1,
          path="/run/round_1/attempts/attempt_2/decision.json")
    state = rebuild_progress(session, rounds=1)
    stage = state["stages"][1]
    assert stage["attempt"] == 2
    assert stage["completed_milestones"] == ["parent_rollout", "meta_evidence", "decision_recorded"]
    assert state["next_milestone"] == "candidate_built"
    assert not stage["retry_pending"]
    assert (session / "turns" / f"{start + 6:04d}" / "command.json").exists()

    operations = [("sft", None), ("run_candidate", None), ("compare_scores", None),
                  ("write_text", "/run/round_1/attempts/attempt_2/selection.json"),
                  ("append_skills", None), ("write_text", "/run/meta/context.json"),
                  ("snapshot_task_meta", None), ("prepare_validation_snapshot", None), ("evaluate", None)]
    for offset, (op, path) in enumerate(operations, start=end + 1):
        turn = _turn(session, offset, op, round_number=1, path=path)
        if path and path.endswith("selection.json"):
            _selection_content(turn, "accept_candidate")
    assert rebuild_progress(session, rounds=1)["finish_allowed"]


def test_rejection_records_failure_experience_before_reselection(tmp_path):
    session = tmp_path / "meta_session"
    start = _complete_a0(session)
    turn = _turn(session, start, "write_text", round_number=1, path="/run/round_1/selection.json")
    _selection_content(turn, "retain_parent")
    assert rebuild_progress(session)["next_milestone"] == "skills_appended"
    _turn(session, start + 1, "append_skills", round_number=1)
    assert rebuild_progress(session)["next_milestone"] == "component_reselection"


def test_policy_does_not_reopen_previously_finished_rounds(tmp_path):
    session = tmp_path / "meta_session"
    start = _complete_a0(session)
    _complete_round(session, 1, start)
    _selection_content(session / "turns" / f"{start + 6:04d}", "retain_parent")
    (session / "workflow_policy.json").write_text(json.dumps({"retry_rejected_from_round": 2}))
    state = rebuild_progress(session)
    assert state["current_stage"] == "B2"
    assert state["stages"][1]["status"] == "complete"


def test_native_written_context_is_observed_in_successful_snapshot(tmp_path):
    session = tmp_path / "meta_session"
    turn = _turn(session, 0, "snapshot_task_meta", round_number=1)
    command = json.loads((turn / "command.json").read_text())
    command["request"]["context_path"] = "/run/meta/context.json"
    (turn / "command.json").write_text(json.dumps(command))
    receipt = json.loads((turn / "receipt.json").read_text())
    receipt["sources"] = {"meta/context/context.json": "/run/meta/context.json"}
    (turn / "receipt.json").write_text(json.dumps(receipt))
    assert "context_updated" in rebuild_progress(session)["stages"][1]["completed_milestones"]


def test_lightweight_snapshot_context_reference_advances_handoff(tmp_path):
    session = tmp_path / "meta_session"
    turn = _turn(session, 0, "snapshot_task_meta", round_number=1)
    command = json.loads((turn / "command.json").read_text())
    command["request"]["context_path"] = "/run/meta/context.json"
    (turn / "command.json").write_text(json.dumps(command))
    receipt = json.loads((turn / "receipt.json").read_text())
    receipt["context_reference"] = {"source": "/run/meta/context.json", "sha256": "content-hash"}
    (turn / "receipt.json").write_text(json.dumps(receipt))
    assert "context_updated" in rebuild_progress(session)["stages"][1]["completed_milestones"]


def test_rebuild_uses_only_successful_command_receipt_pairs(tmp_path):
    session = tmp_path / "meta_session"
    _turn(session, 0, "bootstrap", round_number=0)
    _turn(session, 1, "snapshot_task_meta", round_number=0, status="tool_error")
    pending = session / "turns/0002"
    pending.mkdir(parents=True)
    (pending / "command.json").write_text(json.dumps({
        "action": "tool", "request": {"operation": "snapshot_task_meta", "round": 0}
    }))

    progress = ExperimentProgress(session)
    state = progress.record_turn(pending)

    assert state["stages"][0]["completed_milestones"] == ["bootstrap"]
    assert state["next_milestone"] == "task_meta_snapshot"
    events = [json.loads(line) for line in progress.events_path.read_text().splitlines()]
    assert [event["outcome"] for event in events] == ["succeeded", "failed", "pending"]
    assert progress.finish_allowed() is False


def test_a0_reconstruction_exposes_purpose_evidence_and_next_stage(tmp_path):
    session = tmp_path / "meta_session"
    _complete_a0(session)

    state = rebuild_progress(session)
    block = compact_status_block(session)

    assert state["current_stage"] == "B1"
    assert state["next_milestone"] == "parent_rollout"
    assert state["next_stage"] == "B2"
    assert state["stages"][0]["status"] == "complete"
    assert state["stages"][0]["evidence"][0]["paths"] == ["/run/evidence/0.json"]
    assert "Goal:" in block
    assert "Current stage: B1" in block
    assert "Meta still chooses the component, candidate, and acceptance" in block


def test_all_required_facts_allow_finish_without_requiring_activation(tmp_path):
    session = tmp_path / "meta_session"
    next_turn = _complete_a0(session)
    for round_number in range(1, 4):
        next_turn = _complete_round(session, round_number, next_turn)

    state = rebuild_progress(session)

    assert state["current_stage"] == "COMPLETE"
    assert state["finish_allowed"] is True
    assert all(stage["status"] == "complete" for stage in state["stages"])
    assert finish_allowed(session) is True
    operations = {json.loads(line)["operation"]
                  for line in (session / "workflow_events.jsonl").read_text().splitlines()}
    assert "activate_task" not in operations


def test_completed_three_rounds_can_extend_without_replaying_receipts(tmp_path):
    session = tmp_path / "meta_session"
    next_turn = _complete_a0(session)
    for number in range(1, 4):
        next_turn = _complete_round(session, number, next_turn)
    progress = ExperimentProgress(session, rounds=3)
    assert progress.finish_allowed()
    count = len(list((session / "turns").iterdir()))
    (session / "workflow_policy.json").write_text(json.dumps({"rounds": 5}))
    state = progress.rebuild()
    assert state["current_stage"] == "B4"
    assert state["final_stage"] == "B5"
    assert not state["finish_allowed"]
    assert all(stage["status"] == "complete" for stage in state["stages"][:4])
    assert len(list((session / "turns").iterdir())) == count
    for number in (4, 5):
        next_turn = _complete_round(session, number, next_turn)
    assert progress.finish_allowed()


def test_failed_evaluation_and_unknown_tools_do_not_complete_round(tmp_path):
    session = tmp_path / "meta_session"
    next_turn = _complete_a0(session)
    next_turn = _complete_round(session, 1, next_turn)
    # Replace the successful evaluation receipt with a factual process failure.
    receipt = session / "turns" / f"{next_turn - 1:04d}" / "receipt.json"
    value = json.loads(receipt.read_text())
    value["returncode"] = 2
    receipt.write_text(json.dumps(value))
    _turn(session, next_turn, "invented_debug_tool", round_number=1,
          status="unknown_operation")

    state = rebuild_progress(session)

    assert state["current_stage"] == "B1"
    assert state["next_milestone"] == "independent_validation"
    assert "independent_validation" not in state["stages"][1]["completed_milestones"]
    assert state["finish_allowed"] is False


def test_nested_round_path_wins_over_a0_marker_in_run_root(tmp_path):
    session = tmp_path / "rsiflow_8b_codex_180_a0_v1" / "meta_session"
    _turn(
        session,
        0,
        "run_parent",
        path=str(session.parent / "round_1" / "parent" / "summary.json"),
    )

    state = rebuild_progress(session)

    assert state["stages"][0]["completed_milestones"] == []
    assert state["stages"][1]["completed_milestones"] == ["parent_rollout"]


def test_validation_failed_candidate_is_not_successful(tmp_path):
    session = tmp_path / "meta_session"
    _turn(session, 0, "harnessforge", round_number=1, status="validation_failed")

    state = rebuild_progress(session)

    assert "candidate_built" not in state["stages"][1]["completed_milestones"]
    events = [
        json.loads(line)
        for line in (session / "workflow_events.jsonl").read_text().splitlines()
    ]
    assert events[0]["outcome"] == "failed"
