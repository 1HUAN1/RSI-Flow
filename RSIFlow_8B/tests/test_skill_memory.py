"""Evidence retrieval, append-only revisions and resume behaviour."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller_tools import ControllerTools
from launch_meta import experiment_prompt, prepare_skills_path
from skill_memory import SkillMemory


def case(identifier, delta, component="HARNESS", **kwargs):
    return {"id": identifier, "kind": "case", "component": component,
            "context": {"domains": ["searchqa"], "dominant_errors": ["unverified_memory"]},
            "measured_outcome": {"delta": delta}, **kwargs}


def test_revisions_preserve_original_lines_and_effective_counterexample(tmp_path):
    memory = SkillMemory(tmp_path / "skills.jsonl")
    memory.append([case("skill.HARNESS.1", .1), {
        "id": "principle.1", "kind": "rule", "component": "HARNESS",
        "status": "single_paired_support", "when": ["unverified_memory"],
        "support": ["skill.HARNESS.1"], "recommend": "repair memory"}])
    old = memory.path.read_bytes()
    memory.append([{"id": "update.1", "kind": "rule_update", "target": "principle.1",
                    "operation": "contradict", "counterevidence": ["skill.HARNESS.2"],
                    "reason": "same symptom, harmful repair"}, case("skill.HARNESS.2", -.1)])
    assert memory.path.read_bytes().startswith(old)
    records, _, _ = memory.view()
    assert records["principle.1"]["status"] == "contradicted"
    assert records["principle.1"]["support"] == ["skill.HARNESS.1"]
    assert records["principle.1"]["counterevidence"] == ["skill.HARNESS.2"]
    result = memory.retrieve({"dominant_errors": ["unverified_memory"]}, per_category=2)
    assert "principle.1" in result["matches"]["HARNESS"]["failure"]
    assert "principle.1" not in result["matches"]["HARNESS"]["rules"]


def test_repeat_append_is_idempotent_and_does_not_replace_conflicting_id(tmp_path):
    memory = SkillMemory(tmp_path / "skills.jsonl")
    record = case("skill.HARNESS.1", .1)
    assert memory.append([record])["appended_count"] == 1
    before = memory.path.read_bytes()
    retry = memory.append([record])
    assert retry["appended_count"] == 0 and retry["reused_ids"] == [record["id"]]
    conflict = memory.append([case(record["id"], -.1)])
    assert conflict["warnings"] and conflict["appended_count"] == 0
    assert memory.path.read_bytes() == before


def test_no_total_entry_quota_and_support_failure_retrieved_separately(tmp_path):
    memory = SkillMemory(tmp_path / "skills.jsonl")
    entries = [case(f"skill.HARNESS.{i:02}", .1 if i < 12 else -.1) for i in range(20)]
    receipt = memory.append(entries)
    assert receipt["appended_count"] == receipt["record_count"] == 20
    result = memory.retrieve({"domains": ["searchqa"]}, per_category=1)
    assert len(result["matches"]["HARNESS"]["support"]) == 1
    assert len(result["matches"]["HARNESS"]["failure"]) == 1
    assert not result["matches"]["MODEL"]["support"]


def test_incidents_are_not_harness_support_and_null_fields_do_not_match_everything(tmp_path):
    memory = SkillMemory(tmp_path / "skills.jsonl")
    memory.append([case("skill.HARNESS.1", .1), {
        "id": "incident.schema", "kind": "incident", "status": "fixed",
        "signature": "nullable_tool_schema", "owner": "runtime_adapter"}])
    result = memory.retrieve({"error_types": {"nullable_tool_schema": 20}})
    assert result["incidents"] == ["incident.schema"]
    assert not result["matches"]["HARNESS"]["support"]
    assert "selected_component" not in result


def test_legacy_skill_principle_compatibility_and_inherited_triggers(tmp_path):
    memory = SkillMemory(tmp_path / "skills.jsonl")
    memory.append([{"id": "skill.MODEL.old", "kind": "skill", "component": "MODEL",
                    "context": {"dominant_errors": ["policy_behavior"]},
                    "measured_outcome": {"delta": .02}},
                   {"id": "principle.old", "kind": "principle", "supports": "skill.MODEL.old",
                    "statement": "may help"}])
    result = memory.retrieve({"dominant_errors": ["policy_behavior"]})
    rule = result["records"]["principle.old"]
    assert rule["status"] == "hypothesis"
    assert rule["confidence"] == "legacy_observation"
    assert "principle.old" in result["matches"]["MODEL"]["rules"]


def test_budget_and_paginated_read_do_not_remove_original_content(tmp_path):
    memory = SkillMemory(tmp_path / "skills.jsonl")
    memory.append([case("skill.HARNESS.long", .1, mechanism="轨迹引用" * 20000)])
    result = memory.retrieve({"domains": ["searchqa"]}, max_chars=2200)
    assert len(json.dumps(result, ensure_ascii=False)) <= 2200
    assert "skill.HARNESS.long" in result["omitted_detail_ids"]
    offset, chunks = 0, []
    while True:
        page = memory.read_skill(["skill.HARNESS.long"], offset_chars=offset, max_chars=2000)
        chunks.append(page["content"])
        if page["next_offset_chars"] is None:
            break
        offset = page["next_offset_chars"]
    assert json.loads("".join(chunks))[0]["mechanism"] == "轨迹引用" * 20000


def test_malformed_historical_line_is_reported_and_retrieval_continues(tmp_path):
    path = tmp_path / "skills.jsonl"
    path.write_text('{invalid}\n' + json.dumps(case("skill.HARNESS.1", .1)))
    memory = SkillMemory(path)
    result = memory.append([case("skill.HARNESS.2", -.1)])
    assert result["appended_count"] == 1 and result["warnings"]
    assert memory.retrieve({"domains": ["searchqa"]})["record_count"] == 2


def test_index_is_rebuildable_and_unknown_revision_does_not_abort(tmp_path):
    memory = SkillMemory(tmp_path / "skills.jsonl")
    memory.append([{"id": "u", "kind": "rule_update", "target": "missing",
                    "operation": "weaken"}, case("skill.HARNESS.1", .1)])
    memory.index_path.write_text("not JSON")
    result = memory.retrieve({"domains": ["searchqa"]})
    assert result["status"] == "retrieved" and result["warnings"]
    assert json.loads(memory.index_path.read_text())["event_count"] == 2


def test_seed_only_initializes_new_ledger_and_does_not_reapply_on_resume(tmp_path):
    seed = tmp_path / "seed.jsonl"
    seed.write_text(json.dumps(case("skill.HARNESS.seed", -.1)) + "\n")
    run = tmp_path / "run"
    path = prepare_skills_path(run, seed)
    assert len(path.read_text().splitlines()) == 1
    SkillMemory(path).append([case("skill.HARNESS.next", .1)])
    before = path.read_bytes()
    prepare_skills_path(run, tmp_path / "no-longer-available-seed.jsonl")
    assert path.read_bytes() == before
    assert json.loads((path.parent / "skills_initialization.json").read_text())["mode"] == "seeded"
    empty = prepare_skills_path(tmp_path / "fresh")
    assert empty.read_text() == ""


def test_missing_seed_can_be_corrected_without_silently_starting_empty(tmp_path):
    # A failed seed read must not leave a ledger that resume considers initialized.
    import pytest
    with pytest.raises(FileNotFoundError):
        prepare_skills_path(tmp_path / "run", tmp_path / "missing")
    seed = tmp_path / "seed.jsonl"
    seed.write_text(json.dumps(case("skill.HARNESS.seed", .1)) + "\n")
    path = prepare_skills_path(tmp_path / "run", seed)
    assert "skill.HARNESS.seed" in path.read_text()


def test_tool_retrieval_and_prospective_skill_use_are_durable(tmp_path):
    tools = ControllerTools(tmp_path)
    tools.execute({"operation": "append_skills", "path": "meta/skills.jsonl",
                   "entries": [case("skill.HARNESS.1", .1)]})
    fingerprint = tmp_path / "fingerprint.json"
    fingerprint.write_text(json.dumps({"domains": ["searchqa"]}))
    retrieval = tools.execute({"operation": "retrieve_skills", "path": "meta/skills.jsonl",
                               "fingerprint_path": "fingerprint.json", "max_chars": 2200})
    assert Path(retrieval["retrieval_path"]).is_file()
    assert len(json.dumps(retrieval, ensure_ascii=False)) <= 2200
    request = {"operation": "record_skill_use", "path": "attempt/skill_use.json",
               "skills_path": "meta/skills.jsonl", "component": "HARNESS",
               "relevant_skill_ids": ["skill.HARNESS.1"], "prediction": {"target_domains": ["searchqa"]},
               "retrieval_path": retrieval["retrieval_path"]}
    result = tools.execute(request)
    assert result["status"] == "recorded"
    before = (tmp_path / "attempt/skill_use.json").read_bytes()
    repeated = tools.execute({**request, "prediction": {"target_domains": ["code"]}})
    assert repeated["status"] == "already_recorded"
    assert (tmp_path / "attempt/skill_use.json").read_bytes() == before


def test_new_prompt_exposes_retrieval_without_hard_routing_or_eval_learning(tmp_path):
    prompt = experiment_prompt(tmp_path / "config.json", {"rounds": 3}, tmp_path / "run")
    for word in ("retrieve_skills", "record_skill_use", "compare_task_differences", "rule_update"):
        assert word in prompt
    assert "Report-only independent validation never enters" in " ".join(prompt.split())
    assert "advisory matches, not component rankings" in prompt
    assert "Use controller read_text for skills_path" not in prompt


def test_reviewed_seed_has_conditional_rules_and_no_eval_scores(tmp_path):
    seed = Path(__file__).resolve().parents[1] / "meta_skills/reviewed_seed.jsonl"
    path = prepare_skills_path(tmp_path / "run", seed)
    memory = SkillMemory(path)
    records, events, warnings = memory.view()
    assert len(records) == len(events) == 8 and not warnings
    assert {r["kind"] for r in records.values()} == {"case", "rule", "incident"}
    assert all(r.get("when") and r.get("avoid_when") for r in records.values() if r["kind"] == "rule")
    cases = {r["round"]: r["measured_outcome"] for r in records.values() if r["kind"] == "case"}
    assert [(cases[r]["new_success"], cases[r]["new_regression"]) for r in (1, 2, 3)] == [(1, 9), (6, 3), (11, 6)]
    assert "86" not in seed.read_text() and "82" not in seed.read_text()


def test_usage_report_counts_interventions_once_and_keeps_unknown_predictions(tmp_path):
    memory = SkillMemory(tmp_path / "skills.jsonl")
    evidence = {"parent_performance_path": "p.json", "candidate_performance_path": "c.json"}
    memory.append([
        case("skill.HARNESS.1", .1, evidence=evidence, attempt=1, used_skill_ids=["principle.old"],
             prediction_assessment={"met": True}),
        case("skill.HARNESS.summary", .1, evidence=evidence),
        case("skill.MODEL.2", -.1, component="MODEL", attempt=1, used_skill_ids=["principle.old"]),
        {"id": "principle.summary", "kind": "rule", "statement": "repeat is not new evidence"}])
    result = memory.usage_report()
    assert result["unique_interventions"] == 2
    assert result["first_candidate_positive_rate"] == .5
    assert result["prediction_assessed"] == 1
    assert len(result["skill_uses"]["principle.old"]) == 2
    assert result["attempts"][1]["prediction_met_declared_by_meta"] is None


def test_three_round_tool_path_inherits_cases_revisions_and_snapshot(tmp_path):
    from experiment_progress import _successful
    tools = ControllerTools(tmp_path)
    path = prepare_skills_path(tmp_path / "run")
    for round_number in (1, 2, 3):
        retrieval = tools.execute({"operation": "retrieve_skills", "path": str(path),
                                   "fingerprint": {"domains": ["searchqa"]}})
        assert retrieval["record_count"] == 0 if round_number == 1 else retrieval["record_count"] >= 2
        entries = [case(f"skill.HARNESS.B{round_number}", -.1 if round_number == 1 else .1,
                        round=round_number, attempt=1)]
        if round_number == 1:
            entries.append({"id": "principle.base", "kind": "rule", "component": "HARNESS",
                            "when": ["searchqa"], "recommend": "check final answer form"})
        if round_number == 2:
            entries.append({"id": "update.B2", "kind": "rule_update", "target": "principle.base",
                            "operation": "strengthen", "support": ["skill.HARNESS.B2"]})
        receipt = tools.execute({"operation": "append_skills", "path": str(path), "entries": entries})
        assert receipt["status"] == "appended"
        retry = tools.execute({"operation": "append_skills", "path": str(path), "entries": entries})
        assert retry["appended_count"] == 0 and _successful("append_skills", retry)
        snapshot = tools.execute({"operation": "snapshot", "destination": f"round_{round_number}",
                                  "sources": {"skills.jsonl": str(path)}})
        assert snapshot["status"] == "copied"
        # Loading a saved ledger reconstructs the same effective rules without its index.
        restored = SkillMemory(tmp_path / f"round_{round_number}/skills.jsonl")
        assert restored.retrieve({"domains": ["searchqa"]})["record_count"] == round_number + 1
    assert SkillMemory(path).view()[0]["principle.base"]["status"] == "single_paired_support"
