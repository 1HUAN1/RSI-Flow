"""Paired task evidence used to assess skills, without changing any scoring."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller_tools import ControllerTools
from task_differences import compare_task_differences, failure_fingerprint


def row(identifier, success, domain="searchqa", **kwargs):
    return {"task_id": identifier, "rollout_id": 0, "domain": domain,
            "verification": {"task_success": success}, "task_source_hash": identifier,
            "error_type": None if success else "wrong_answer", "model_call_count": 2,
            "wall_time_seconds": 10, "final_answer": "yes", **kwargs}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_pairing_uses_ids_not_order_and_catches_net_zero_regression(tmp_path):
    before, after = tmp_path / "parent.jsonl", tmp_path / "candidate.jsonl"
    write_rows(before, [row("A", True), row("B", False)])
    write_rows(after, [row("B", True), row("A", False, final_answer="Yes this is a long explanatory answer",
                                                  wall_time_seconds=15)])
    summary = compare_task_differences(before, after, tmp_path / "differences")
    assert summary["counts"] == {"new_regression": 1, "new_success": 1}
    assert summary["same_id_set"] is True
    tasks = json.loads(Path(summary["paired_tasks_path"]).read_text())["tasks"]
    assert tasks[0]["before"]["evidence"]["line"] == 1
    assert tasks[0]["after"]["evidence"]["line"] == 2
    assert tasks[0]["cost_delta"]["wall_time_seconds"] == 5
    assert tasks[0]["after"]["answer_shape"] == "sentence"
    assert tasks[0]["before"]["evidence"]["content_hash"] == hashlib.sha256(before.read_bytes().splitlines(keepends=True)[0]).hexdigest()
    assert "accepted" not in summary


def test_missing_success_is_unknown_even_with_positive_partial_reward(tmp_path):
    before, after = tmp_path / "p.jsonl", tmp_path / "c.jsonl"
    write_rows(before, [{"task_id": "A", "terminal_reward": .5, "verification": {}, "domain": "tool_use"}])
    write_rows(after, [row("A", True, "tool_use")])
    summary = compare_task_differences(before, after, tmp_path / "d")
    assert summary["counts"] == {"unknown": 1}
    assert failure_fingerprint([{"domain": "tool_use", "terminal_reward": .5}])["verified_successes"] == 0


def test_reports_missing_ids_duplicates_sources_and_unknown_costs(tmp_path):
    before, after = tmp_path / "p.jsonl", tmp_path / "c.jsonl"
    write_rows(before, [row("A", False), row("A", True), row("B", False)])
    write_rows(after, [row("A", True, task_source_hash="changed", wall_time_seconds=None), row("C", False)])
    summary = compare_task_differences(before, after, tmp_path / "d")
    assert summary["paired_tasks"] == 1
    assert summary["same_id_set"] is False
    assert summary["duplicates"]["before"][0]["line"] == 2
    assert summary["missing_in_after"] == [["B", "0"]]
    assert summary["extra_in_after"] == [["C", "0"]]
    assert summary["source_mismatches"][0]["task_id"] == "A"
    tasks = json.loads(Path(summary["paired_tasks_path"]).read_text())["tasks"]
    assert tasks[0]["cost_delta"]["wall_time_seconds"] is None


def test_original_long_trajectories_are_referenced_and_never_copied(tmp_path):
    before, after = tmp_path / "p.jsonl", tmp_path / "c.jsonl"
    write_rows(before, [row("A", False, messages=["raw_trace_marker" * 10000])])
    write_rows(after, [row("A", True, messages=["raw_trace_marker" * 10000])])
    summary = compare_task_differences(before, after, tmp_path / "d")
    details = Path(summary["paired_tasks_path"]).read_text()
    assert "raw_trace_marker" not in details
    assert len(details) < 4000 and before.stat().st_size > 100000
    assert "messages" not in details


def test_fingerprint_records_observed_errors_and_verified_domains():
    rows = [row("A", False, "tool_use", error_type="nullable_tool_schema", model_call_count=0),
            row("B", True, success_verifier_version="v2"), row("C", False, "code")]
    fingerprint = failure_fingerprint(rows)
    assert fingerprint["task_count"] == 3
    assert fingerprint["verified_success_counts"] == {"searchqa": 1}
    assert fingerprint["failed_zero_model_call_count"] == 1
    assert fingerprint["error_types"]["nullable_tool_schema"] == 1
    assert fingerprint["success_verifier_versions"] == ["v2"]


def test_skill_prediction_remains_frozen_when_factual_outcome_arrives(tmp_path):
    tools = ControllerTools(tmp_path)
    tools.execute({"operation": "append_skills", "path": "skills.jsonl", "entries": [
        {"id": "skill.HARNESS.1", "kind": "case", "component": "HARNESS"}]})
    request = {"operation": "record_skill_use", "path": "skill_use.json", "skills_path": "skills.jsonl",
               "component": "HARNESS", "relevant_skill_ids": ["skill.HARNESS.1"],
               "prediction": {"expected_errors_changed": ["wrong_answer"]}}
    tools.execute(request)
    frozen = (tmp_path / "skill_use.json").read_bytes()
    write_rows(tmp_path / "p.jsonl", [row("A", False)])
    write_rows(tmp_path / "c.jsonl", [row("A", True)])
    result = tools.execute({"operation": "compare_task_differences", "before_trajectories": "p.jsonl",
                            "after_trajectories": "c.jsonl", "output_dir": "differences",
                            "skill_use_path": "skill_use.json"})
    assert result["prediction_comparison"]["used_skill_ids"] == ["skill.HARNESS.1"]
    assert result["prediction_comparison"]["observed_task_transitions"] == {"new_success": 1}
    assert (tmp_path / "skill_use.json").read_bytes() == frozen
