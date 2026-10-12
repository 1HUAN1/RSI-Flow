"""Meta selects maintenance; the ledger preserves history and resolves its view."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller_tools import ControllerTools
from experiment_progress import _milestone, _successful
from launch_meta import experiment_prompt
from skill_memory import SkillMemory


def procedure(identifier, **fields):
    return {"id": identifier, "component": "HARNESS", "when": ["unverified_memory"],
            "steps": ["inspect observation provenance", "repair fact status"],
            "preserve": ["short answer contract"], **fields}


def add(memory, identifier="skill.HARNESS.memory.v1", **fields):
    return memory.maintain([{"action": "add", "record": procedure(identifier, **fields),
                             "reason": "uncovered method", "evidence_refs": ["B1/task_differences/summary.json"]}])


def test_add_links_provenance_range_and_does_not_limit_library_size(tmp_path):
    memory = SkillMemory(tmp_path / "skills.jsonl")
    result = add(memory)
    assert result["appended_count"] == 1
    record = memory.view()[0]["skill.HARNESS.memory.v1"]
    assert record["kind"] == "procedure" and record["version"] == 1
    assert record["when"] == ["unverified_memory"]
    assert record["evidence_refs"] == ["B1/task_differences/summary.json"]
    assert record["status"] == "hypothesis"
    add(memory, "skill.HARNESS.other")
    # Similarity is advisory, not an automatic deletion or component decision.
    assert memory.view()[0]["skill.HARNESS.other"]["active"]
    assert "skill.HARNESS.memory.v1" in memory.retrieve({"dominant_errors": ["unverified_memory"]})["matches"]["HARNESS"]["skills"]


def test_supplement_preserves_method_adds_both_sides_and_retry_is_idempotent(tmp_path):
    memory = SkillMemory(tmp_path / "skills.jsonl")
    add(memory)
    before = memory.path.read_bytes()
    operation = {"action": "supplement", "target": "skill.HARNESS.memory.v1",
                 "support": ["skill.HARNESS.B2", "skill.HARNESS.B2"],
                 "counterevidence": ["skill.HARNESS.B3"], "evidence_refs": ["B2/paired.json"],
                 "reason": "same method, another measured response"}
    result = memory.maintain([operation])
    assert result["record_count"] == 1 and result["appended_count"] == 1
    assert memory.path.read_bytes().startswith(before)
    record = memory.view()[0]["skill.HARNESS.memory.v1"]
    assert record["support"] == ["skill.HARNESS.B2"]
    assert record["counterevidence"] == ["skill.HARNESS.B3"]
    assert record["version"] == 1 and record["steps"] == procedure("x")["steps"]
    assert record["status"] == "hypothesis"  # no automatic credibility promotion
    once = memory.path.read_bytes()
    retry = memory.maintain([operation])
    assert retry["appended_count"] == 0 and retry["reused_ids"]
    assert memory.path.read_bytes() == once


def test_revision_creates_linked_version_without_relabelling_old_response(tmp_path):
    memory = SkillMemory(tmp_path / "skills.jsonl")
    memory.append([{"id": "skill.HARNESS.old", "kind": "skill", "component": "HARNESS",
                    "when": ["unverified_memory"], "measured_outcome": {"delta": .1},
                    "confidence": "single_paired_support", "status": "single_paired_support",
                    "support": ["case.old_success"], "counterevidence": ["case.old_failure"]}])
    before = memory.path.read_bytes()
    operation = {"action": "revise", "target": "skill.HARNESS.old",
                 "record": {"id": "skill.HARNESS.v2", "when": ["searchqa", "unverified_memory"],
                            "steps": ["inspect evidence", "repair fact status", "preserve output contract"],
                            "preserve": ["short answers"]}, "reason": "missing output-preservation step"}
    result = memory.maintain([operation])
    assert result["appended_count"] == 2 and memory.path.read_bytes().startswith(before)
    records = memory.view()[0]
    assert records["skill.HARNESS.old"]["measured_outcome"] == {"delta": .1}
    assert not records["skill.HARNESS.old"]["active"]
    assert records["skill.HARNESS.old"]["replaced_by"] == "skill.HARNESS.v2"
    current = records["skill.HARNESS.v2"]
    assert current["kind"] == "procedure" and current["version"] == 2
    assert current["previous_version"] == "skill.HARNESS.old"
    assert current["confidence"] == "unvalidated_revision"
    assert "measured_outcome" not in current
    assert not current.get("support") and not current.get("counterevidence")
    assert current["prior_version_support"] == ["case.old_success"]
    assert current["prior_version_counterevidence"] == ["case.old_failure"]
    normal = memory.retrieve({"domains": ["searchqa"], "dominant_errors": ["unverified_memory"]})
    assert "skill.HARNESS.old" not in normal["records"]
    assert "skill.HARNESS.v2" in normal["records"]
    history = json.loads(memory.read_skill(["skill.HARNESS.old"])["content"])[0]
    assert history["measured_outcome"]["delta"] == .1
    frozen = memory.path.read_bytes()
    assert memory.maintain([operation])["reused_ids"]
    assert memory.path.read_bytes() == frozen


def test_merge_preserves_support_and_counterexamples_in_canonical_view(tmp_path):
    memory = SkillMemory(tmp_path / "skills.jsonl")
    add(memory, "skill.HARNESS.a", support=["case.1"], evidence_refs=["a.json"])
    add(memory, "skill.HARNESS.b", support=["case.1", "case.2"], counterevidence=["case.fail"], evidence_refs=["b.json"])
    before = memory.path.read_bytes()
    result = memory.maintain([{"action": "merge", "targets": ["skill.HARNESS.a", "skill.HARNESS.b"],
                              "record": procedure("skill.HARNESS.canonical"), "reason": "same conditions and method"}])
    assert result["active_record_count"] == 1
    assert memory.path.read_bytes().startswith(before)
    records = memory.view()[0]
    canonical = records["skill.HARNESS.canonical"]
    assert canonical["support"] == ["case.1", "case.2"]
    assert canonical["counterevidence"] == ["case.fail"]
    assert {"a.json", "b.json"} <= set(canonical["evidence_refs"])
    assert canonical["merged_from"] == ["skill.HARNESS.a", "skill.HARNESS.b"]
    assert records["skill.HARNESS.a"]["lifecycle_status"] == "merged"
    result = memory.retrieve({"dominant_errors": ["unverified_memory"]}, per_category=5)
    assert set(result["records"]) == {"skill.HARNESS.canonical"}
    history = memory.retrieve({"dominant_errors": ["unverified_memory"]}, per_category=5, include_inactive=True)
    assert set(history["records"]) == set(records)


def test_merge_into_existing_canonical_and_retire_keep_all_source_records(tmp_path):
    memory = SkillMemory(tmp_path / "skills.jsonl")
    add(memory, "skill.HARNESS.a", support=["case.a"])
    add(memory, "skill.HARNESS.b", counterevidence=["case.b"])
    result = memory.maintain([{"action": "merge", "targets": ["skill.HARNESS.b"],
                              "canonical_id": "skill.HARNESS.a", "reason": "duplicate method"}])
    assert result["record_count"] == 2 and result["active_record_count"] == 1
    assert memory.view()[0]["skill.HARNESS.a"]["counterevidence"] == ["case.b"]
    result = memory.maintain([{"action": "retire", "target": "skill.HARNESS.a",
                              "reason": "method no longer applicable", "evidence_refs": ["new_state.json"]}])
    assert result["active_record_count"] == 0 and result["record_count"] == 2
    assert not memory.retrieve({"dominant_errors": ["unverified_memory"]})["records"]
    assert len(json.loads(memory.read_skill(["skill.HARNESS.a", "skill.HARNESS.b"])["content"])) == 2


def test_restore_rebuilds_view_and_bad_target_does_not_abort_valid_operations(tmp_path):
    tools = ControllerTools(tmp_path)
    result = tools.execute({"operation": "maintain_skills", "path": "skills.jsonl", "operations": [
        {"action": "supplement", "target": "missing", "support": ["case.1"]},
        {"action": "add", "record": procedure("skill.HARNESS.a")},
        {"action": "revise", "target": "skill.HARNESS.a", "record": procedure("skill.HARNESS.b")}]})
    assert result["status"] == "maintained" and result["warnings"]
    assert result["active_record_count"] == 1
    assert _successful("maintain_skills", result)
    assert _milestone("maintain_skills", {}) == "skills_appended"
    result = tools.execute({"operation": "snapshot", "destination": "snapshot", "sources": {"skills.jsonl": "skills.jsonl"}})
    assert result["status"] == "copied"
    restored = SkillMemory(tmp_path / "snapshot/skills.jsonl")
    current = restored.retrieve({"dominant_errors": ["unverified_memory"]})
    assert set(current["records"]) == {"skill.HARNESS.b"}
    index = restored.rebuild_index()
    assert index["record_count"] == 2 and index["active_record_count"] == 1


def test_legacy_rule_updates_and_procedure_maintenance_coexist(tmp_path):
    memory = SkillMemory(tmp_path / "skills.jsonl")
    memory.append([{"id": "principle.1", "kind": "principle", "when": ["unverified_memory"],
                    "statement": "preserve output", "supports": "case.1"}])
    memory.maintain([{"action": "supplement", "target": "principle.1", "counterevidence": ["case.2"]}])
    memory.append([{"id": "update.1", "kind": "rule_update", "target": "principle.1", "operation": "contradict"}])
    record = memory.view()[0]["principle.1"]
    assert record["status"] == "contradicted" and record["counterevidence"] == ["case.2"]
    assert record["support"] == ["case.1"]
    assert "principle.1" in memory.retrieve({"dominant_errors": ["unverified_memory"]})["matches"]["HARNESS"]["failure"]


def test_maintenance_operations_are_available_to_meta_without_new_completion_gate(tmp_path):
    prompt = experiment_prompt(tmp_path / "config.json", {"rounds": 3}, tmp_path / "run")
    assert "maintain_skills" in prompt and "canonical_id" in prompt
    tools = ControllerTools(tmp_path)
    request = {"operation": "maintain_skills", "path": "skills.jsonl", "operations": [
        {"action": "add", "record": procedure("skill.HARNESS.a")}]}
    first = tools.execute(request)
    again = tools.execute(request)
    assert first["appended_count"] == 1 and again["reused_ids"]
    assert _successful("maintain_skills", again)
    retrieval = tools.execute({"operation": "retrieve_skills", "path": "skills.jsonl",
                               "fingerprint": {"dominant_errors": ["unverified_memory"]}, "max_chars": 24000})
    assert len(json.dumps(retrieval, ensure_ascii=False)) <= 24000
