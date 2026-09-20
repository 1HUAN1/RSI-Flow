"""Executable G behavior with explicit mock Codex callbacks; no API/GPU calls."""
import copy
import hashlib
import json

import pytest
from pydantic import BaseModel, ConfigDict

from sia.task_meta.meta_harness.policies import OPERATIONS, default_policy, validate_policy
from sia.task_meta.meta_harness.runtime import (
    CandidateRejected,
    PolicyBudgetExceeded,
    build_evidence,
    build_experience_context,
    condition,
    execute,
)


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate: str


class MockBundle:
    """The immutable-bundle implementation is separately tested; this is explicit fixture input."""
    def __init__(self, policy=None, version=0, context=None):
        self.files = {"evolution.json": json.dumps(policy or default_policy()),
                      "context.json": json.dumps(context or {"max_evidence_chars": 64000, "selection": "head_and_tail"})}
        self.hash = hashlib.sha256(json.dumps(self.files, sort_keys=True).encode()).hexdigest()
        self.version = version

    def verify(self):
        assert hashlib.sha256(json.dumps(self.files, sort_keys=True).encode()).hexdigest() == self.hash
        return self

    def read_files(self):
        return copy.deepcopy(self.files)


def envelope():
    trajectories = []
    for index, domain in enumerate(["code", "code", "tool_use", "searchqa"]):
        trajectories.append({"source_id": f"t{index}", "task_id": f"task{index}", "domain": domain,
            "generation": index, "terminal_reward": 1 if index == 3 else 0, "split": "evolve_train",
            "error_type": None if index == 3 else "runtime_error", "messages": [
                {"role": "system", "content": "Task contract"},
                {"role": "user", "content": "Prior user context"},
                {"role": "assistant", "content": "attempt"},
                {"role": "tool", "name": "run_code", "content": "Error: actual failure observed"},
                {"role": "assistant", "content": "revised attempt"}]})
    experiences = [{"experience_id": f"e{index}", "generation": index, "chosen_action": action,
                    "observed_performance_delta": delta, "cost_after": {"tokens": index + 20},
                    "versions": {"model": "frozen"}, "requested_change": ["target"],
                    "actual_change": {"summary": "changed"}}
                   for index, (action, delta) in enumerate([("HARNESS", -.2), ("MODEL", .3), ("HARNESS", .1)])]
    return {"raw_trajectories": trajectories, "experiences": experiences,
            "trusted_facts": {"domains": {"code": {"denominator": 2, "correct": 0},
                "tool_use": {"denominator": 1, "correct": 0}, "searchqa": {"denominator": 1, "correct": 1}},
                "macro_success": 1 / 3, "available_actions": {"MODEL": {"available": False, "reason": "no GPU"},
                                                            "HARNESS": {"available": True}}},
            "task_state": {"generation": 3, "model_ref": "frozen"},
            "decision": {"action": "HARNESS", "requested_changes": [{"target": "prompts.action_step"}]},
            "latest_experience": experiences[-1], "current_files": {"seed.json": '{"prompt":"keep behavior"}',
                                                                   "contract.md": "retain tool permissions"}}


def test_evidence_filters_and_groups_change_ids_without_changing_truth():
    source = envelope()
    original = copy.deepcopy(source)
    first = default_policy()
    first["evidence"].update(max_items=2, group_by=[], sort=[{"field": "generation", "direction": "asc"}])
    plain = build_evidence(source, first, {"bundle_hash": "g0", "version": 0})
    grouped = copy.deepcopy(first)
    grouped["evidence"]["group_by"] = ["domain"]
    second = build_evidence(source, grouped, {"bundle_hash": "g1", "version": 1})
    assert [item["source_id"] for item in plain["selected"]] == ["t0", "t1"]
    assert [item["source_id"] for item in second["selected"]] == ["t0", "t2"]
    grouped["evidence"]["filter"] = {"path": "item.success", "op": "eq", "value": True}
    success_only = build_evidence(source, grouped, {"bundle_hash": "g2", "version": 2})
    assert [item["source_id"] for item in success_only["selected"]] == ["t3"]
    assert len(success_only["omitted"]) == 3
    assert plain["trusted_facts_hash"] == second["trusted_facts_hash"] == success_only["trusted_facts_hash"]
    assert source == original


def test_error_fragments_include_preceding_context_and_required_tool():
    source = envelope()
    policy = default_policy()
    policy["evidence"].update(max_items=1, group_by=[], sort=[{"field": "generation", "direction": "asc"}])
    fragment = policy["evidence"]["fragments"]
    fragment.update(before=1, after=0, max_messages=1, retain_tools=["run_code"], message_chars=32)
    source["raw_trajectories"][0]["messages"][3]["content"] += "x" * 100
    result = build_evidence(source, policy, {"bundle_hash": "g0"})["selected"][0]
    assert result["anchor_indices"] == [3]
    assert [item["message_index"] for item in result["fragments"]] == [2, 3]
    assert result["fragments"][1]["message"] == source["raw_trajectories"][0]["messages"][3]
    assert result["omitted_message_indices"] == [0, 1, 4]
    assert result["fragments"][1]["mandatory_tool_observation"]


def test_context_json_preference_actually_changes_kept_sources():
    source = envelope()
    policy = default_policy()
    policy["evidence"].update(group_by=[], sort=[{"field": "generation", "direction": "asc"}])
    baseline = build_evidence(source, policy, {"bundle_hash": "g"})
    item_size = len(json.dumps(baseline["selected"][0], ensure_ascii=False, separators=(",", ":")))
    limit = item_size + 30
    head = build_evidence(source, policy, {"bundle_hash": "g"}, context_policy={"max_evidence_chars": limit, "selection": "head"})
    tail = build_evidence(source, policy, {"bundle_hash": "g"}, context_policy={"max_evidence_chars": limit, "selection": "tail"})
    assert [row["source_id"] for row in head["selected"]] == ["t0"]
    assert [row["source_id"] for row in tail["selected"]] == ["t3"]
    assert all(row["reason"] == "context_character_budget" for row in tail["omitted"])


def test_required_observation_cannot_silently_disappear_under_budget():
    source = envelope()
    source["raw_trajectories"][0]["messages"][3]["content"] = "Error " + "x" * 5000
    policy = default_policy()
    policy["evidence"].update(group_by=[], sort=[], max_items=1, max_chars=1000)
    policy["evidence"]["fragments"]["retain_tools"] = ["*"]
    with pytest.raises(PolicyBudgetExceeded, match="required tool"):
        build_evidence(source, policy, {"bundle_hash": "g"})


def test_required_observations_are_not_dropped_for_cumulative_budget():
    policy = default_policy()
    policy["evidence"].update(group_by=[], sort=[], max_items=2)
    policy["evidence"]["fragments"]["retain_tools"] = ["*"]
    source = envelope()
    initial = build_evidence(source, policy, {"bundle_hash": "g"})
    policy["evidence"]["max_chars"] = initial["cost"]["selected_chars"] - 10
    with pytest.raises(PolicyBudgetExceeded, match="required tool"):
        build_evidence(source, policy, {"bundle_hash": "g"})


def test_analysis_can_cite_a_permitted_dependency_source(tmp_path):
    policy = default_policy()
    policy["workflows"]["harness_patch"].insert(3, {"id": "inspect", "kind": "analyze"})
    def invoke(stage_id, prompt, schema, **kwargs):
        if schema.__name__ == "AnalysisOutput":
            return schema.model_validate({"analysis": "mock dependency inspection", "source_ids": ["file:seed.json"]})
        return schema.model_validate({"candidate": "good"})
    assert execute(MockBundle(policy), "harness_patch", envelope(), Output, invoke, tmp_path).candidate == "good"


def test_experience_rules_projection_conditions_and_cache_versioning():
    source = envelope()
    original = copy.deepcopy(source)
    policy = default_policy()
    policy["experience"].update(group_by=[], max_items=1, fields=["generation", "observed_performance_delta"])
    policy["experience"]["score_rules"] = [{"id": "same_component", "when": {
        "path": "item.chosen_action", "op": "eq", "value_path": "decision.action"}, "weight": 10}]
    policy["experience"]["failure"] = [{"id": "negative_outcome", "when": {
        "path": "item.observed_performance_delta", "op": "lt", "value": 0}, "instruction": "Do not generalize this failure to the current decision."}]
    cache = {}
    first = build_experience_context(source, policy, {"bundle_hash": "g0"}, cache=cache)
    assert first["selected"][0]["source_id"] == "e2"
    assert first["selected"][0]["summary"] == {"generation": 2, "observed_performance_delta": .1}
    assert any(item["source_id"] == "e0" and item["reason"] == "applicability_or_failure_condition" for item in first["omitted"])
    assert build_experience_context(source, policy, {"bundle_hash": "g0"}, cache=cache) == first
    changed = copy.deepcopy(policy)
    changed["experience"]["filter"] = {"path": "item.chosen_action", "op": "eq", "value": "MODEL"}
    second = build_experience_context(source, changed, {"bundle_hash": "g1"}, cache=cache)
    assert second["selected"][0]["source_id"] == "e1"
    assert first["cache_key"] != second["cache_key"]
    source["experiences"][1]["observed_performance_delta"] = .4
    third = build_experience_context(source, changed, {"bundle_hash": "g1"}, cache=cache)
    assert third["cache_key"] != second["cache_key"]
    source["experiences"][1]["observed_performance_delta"] = .3
    assert source == original and len(cache) == 3


def test_composed_predicates_and_projection_truncation_are_attributable():
    source = envelope()
    policy = default_policy()
    policy["experience"]["require_applicable"] = True
    policy["experience"]["applicability"] = [{"id": "bounded_current", "when": {"all": [
        {"path": "item.generation", "op": "lt", "value_path": "task.generation"},
        {"not": {"path": "item.chosen_action", "op": "eq", "value": "MODEL"}}]}, "instruction": "Prior comparable intervention"}]
    policy["experience"]["fields"] = ["actual_change"]
    policy["experience"]["field_chars"] = 32
    source["experiences"][0]["actual_change"]["summary"] = "x" * 300
    package = build_experience_context(source, validate_policy(policy), {"bundle_hash": "g"})
    assert {row["source_id"] for row in package["selected"]} == {"e0", "e2"}
    row = next(row for row in package["selected"] if row["source_id"] == "e0")
    assert row["truncations"][0]["omitted_range"][0] == 32
    assert row["conditions"]["applicability"][0]["matched"] is True


def mock_invoke(events, *, bad_first=False):
    def invoke(stage_id, prompt, schema, **kwargs):
        events.append({"stage_id": stage_id, "prompt": prompt, "schema": schema.__name__, "source": "mock"})
        assert kwargs == {"evidence_files": {}, "allowed_paths": []}
        if schema.__name__ == "AnalysisOutput":
            return schema.model_validate({"analysis": "mock inspection hypothesis", "source_ids": []})
        return schema.model_validate({"candidate": "bad" if bad_first and stage_id == "propose" else "good"})
    invoke.decision_source = "test_override"
    return invoke


def test_membership_predicates_fail_closed_on_incompatible_types():
    assert condition({"path": "item", "op": "contains", "value": 1}, {"item": "one"}) is False
    assert condition({"path": "item", "op": "in", "value": "123"}, {"item": 1}) is False
    assert condition({"path": "item", "op": "contains", "value": True}, {"item": [1]}) is False
    assert condition({"path": "item", "op": "contains", "value": {"x": 1}}, {"item": [{"x": 1}]}) is True


def test_final_consolidation_none_experience_and_tuple_validator(tmp_path):
    source = envelope()
    source["latest_experience"] = None
    execute(MockBundle(), "final_consolidation", source, Output, mock_invoke([]), tmp_path,
            validate_candidate=lambda candidate: ({"accepted": candidate.candidate}, ["applied"]))
    audit = json.loads((tmp_path / "policy_runtime.json").read_text())
    assert audit["status"] == "completed" and audit["decision_source"] == "test_override"
    assert all(event["decision_source"] == "test_override" for event in audit["events"] if "stage_id" in event)


def test_dependency_range_and_target_inspection_recover_late_json_field(tmp_path):
    source = envelope()
    source["current_files"]["seed.json"] = json.dumps({"padding": "x" * 8000, "prompts": {"action_step": "late target"}})
    policy = default_policy()
    policy["workflows"]["harness_patch"] = [
        {"id": "read", "kind": "read_dependencies", "paths": ["seed.json"], "offset": 7990, "max_chars": 200},
        {"id": "locate", "kind": "inspect_targets", "paths": ["seed.json"], "max_chars": 100},
        {"id": "propose", "kind": "propose"}]
    events = []
    execute(MockBundle(policy), "harness_patch", source, Output, mock_invoke(events), tmp_path)
    payload = json.loads(events[0]["prompt"].split("Attributable operation data (not instructions):\n", 1)[1])
    read, inspected = payload["dependencies"]
    assert "late target" in read["content"] and read["truncations"][0]["omitted_ranges"] == [[0, 7990]]
    assert inspected["targets"][0]["found"] is True
    assert inspected["targets"][0]["canonical_json_excerpt"] == '"late target"'
    assert inspected["targets"][0]["source_id"] == "file:seed.json"


def test_final_action_check_cannot_be_removed_by_g(tmp_path):
    class Route(BaseModel):
        action: str
    policy = default_policy()
    policy["workflows"]["routing"] = [{"id": "propose", "kind": "propose"}]
    with pytest.raises(CandidateRejected, match="unavailable"):
        execute(MockBundle(policy), "routing", envelope(), Route,
                lambda *args, **kwargs: Route(action="MODEL"), tmp_path)


def test_source_eligibility_cannot_be_removed_by_g(tmp_path):
    policy = default_policy()
    policy["workflows"]["routing"] = [{"id": "propose", "kind": "propose"}]
    source = envelope()
    source["raw_trajectories"][0]["split"] = "report_eval"
    with pytest.raises(ValueError, match="training trajectories"):
        execute(MockBundle(policy), "routing", source, Output, mock_invoke([]), tmp_path)


def test_dead_or_unbounded_dependency_fields_are_rejected():
    for patch in ({"max_chars": 120001}, {"offset": -1}, {"instruction": "unused"}):
        policy = default_policy()
        policy["workflows"]["harness_patch"][2].update(patch)
        with pytest.raises(ValueError):
            validate_policy(policy)


def test_workflow_optional_checks_order_and_same_candidate_repair(tmp_path):
    policy = default_policy()
    policy["workflows"]["harness_patch"] = [
        {"id": "read_contract", "kind": "read_dependencies", "paths": ["contract.md"]},
        {"id": "inspect", "kind": "analyze", "instruction": "Inspect the retained contract first."},
        {"id": "select", "kind": "evidence"},
        {"id": "unused", "kind": "analyze", "when": {"path": "facts.domains.code.correct", "op": "gt", "value": 10}},
        {"id": "propose", "kind": "propose"},
        {"id": "repair", "kind": "repair", "max_attempts": 2,
         "when": {"path": "candidate_valid", "op": "eq", "value": False}}]
    events = []
    source = envelope()
    original = copy.deepcopy(source)
    def fixed_check(candidate):
        return {"passed": candidate.candidate == "good", "errors": ["Required interface missing"]}
    output = execute(MockBundle(policy), "harness_patch", source, Output, mock_invoke(events, bad_first=True), tmp_path,
                     validate_candidate=fixed_check)
    assert output.candidate == "good"
    assert [event["stage_id"] for event in events] == ["inspect", "propose", "repair_1"]
    assert "retain tool permissions" in events[0]["prompt"]
    assert '"candidate":{"candidate":"bad"}' in events[-1]["prompt"]
    audit = json.loads((tmp_path / "policy_runtime.json").read_text())
    assert audit["candidate_revision"] == 2 and audit["status"] == "completed"
    assert any(event["status"] == "condition_skipped" for event in audit["events"])
    assert source == original


def test_changed_self_update_workflow_is_used_by_next_g(tmp_path):
    original = default_policy()
    evolved = copy.deepcopy(original)
    evolved["self_update"]["instruction"] = "SELF_UPDATE_REVISION_1: inspect outcome attribution before proposing changes."
    evolved["self_update"]["rules"] = [{"id": "negative_signal", "when": {"path": "latest_experience.observed_performance_delta", "op": "gt", "value": 0},
                                      "instruction": "Observed gain alone is not isolated evidence."}]
    evolved["workflows"]["meta_self_update"].insert(0, {"id": "attribution", "kind": "analyze"})
    first, second = [], []
    execute(MockBundle(original), "meta_self_update", envelope(), Output, mock_invoke(first), tmp_path / "g0")
    execute(MockBundle(evolved, version=1), "meta_self_update", envelope(), Output, mock_invoke(second), tmp_path / "g1")
    assert [event["stage_id"] for event in first] == ["propose"]
    assert [event["stage_id"] for event in second] == ["attribution", "propose"]
    assert all("SELF_UPDATE_REVISION_1" in event["prompt"] and "Observed gain alone" in event["prompt"] for event in second)
    assert all("SELF_UPDATE_REVISION_1" not in event["prompt"] for event in first)
    # Engineer-specified G revisions and mock callbacks demonstrate load behavior,
    # not autonomous Meta improvement or a real paid Codex operation.


@pytest.mark.parametrize("operation", sorted(OPERATIONS))
def test_every_operation_uses_same_pinned_entrypoint(tmp_path, operation):
    bundle = MockBundle(version=7)
    events = []
    execute(bundle, operation, envelope(), Output, mock_invoke(events), tmp_path / operation)
    assert events and all(bundle.hash in event["prompt"] and '"version":7' in event["prompt"] for event in events)
    record = json.loads((tmp_path / operation / "policy_runtime.json").read_text())
    assert record["entrypoint"].endswith("runtime.execute") and record["g"]["version"] == 7


def test_deterministic_stage_prompt_excludes_costs(tmp_path):
    bundle, first, second = MockBundle(), [], []
    execute(bundle, "routing", envelope(), Output, mock_invoke(first), tmp_path / "a")
    execute(bundle, "routing", envelope(), Output, mock_invoke(second), tmp_path / "b")
    assert [item["prompt"] for item in first] == [item["prompt"] for item in second]
    assert all("processing_seconds" not in item["prompt"] for item in first)


def test_invalid_schema_repair_and_fixed_checks_cannot_be_skipped(tmp_path):
    events = []
    def invalid_first(stage_id, prompt, schema, **kwargs):
        events.append(stage_id)
        return schema.model_validate({"wrong": True} if stage_id == "propose" else {"candidate": "good"})
    assert execute(MockBundle(), "routing", envelope(), Output, invalid_first, tmp_path / "repair").candidate == "good"
    assert events == ["propose", "repair_1"]
    policy = default_policy()
    policy["workflows"]["routing"] = [{"id": "propose", "kind": "propose"}]
    with pytest.raises(CandidateRejected):
        execute(MockBundle(policy), "routing", envelope(), Output, mock_invoke([]), tmp_path / "reject",
                validate_candidate=lambda candidate: False)


def test_pending_callback_is_never_retried_as_candidate_repair(tmp_path):
    from sia.task_meta.types import UpdatePending
    calls = []
    def pending(*args, **kwargs):
        calls.append(args[0])
        raise UpdatePending("existing external operation must be reconciled")
    with pytest.raises(UpdatePending):
        execute(MockBundle(), "routing", envelope(), Output, pending, tmp_path)
    assert calls == ["propose"]


def test_invocation_budget_and_mutated_source_fail_closed(tmp_path):
    policy = default_policy()
    policy["workflows"]["routing"].insert(0, {"id": "inspect", "kind": "analyze"})
    events = []
    with pytest.raises(PolicyBudgetExceeded):
        execute(MockBundle(policy), "routing", envelope(), Output, mock_invoke(events), tmp_path / "budget",
                external_budget={"max_invocations": 1})
    assert [event["stage_id"] for event in events] == ["inspect"]
    source = envelope()
    def mutate(stage_id, prompt, schema, **kwargs):
        source["trusted_facts"]["macro_success"] = 1
        return schema.model_validate({"candidate": "good"})
    with pytest.raises(ValueError, match="mutated"):
        execute(MockBundle(), "routing", source, Output, mutate, tmp_path / "mutate")


@pytest.mark.parametrize("bad", ["loop", "exec", "shell", "jump", "recurse"])
def test_unbounded_or_code_workflow_primitives_are_rejected(bad):
    policy = default_policy()
    policy["workflows"]["routing"].insert(0, {"id": "bad", "kind": bad})
    with pytest.raises(ValueError):
        validate_policy(policy)


def test_forbidden_paths_unknown_fields_and_final_data_are_rejected(tmp_path):
    policy = default_policy()
    policy["workflows"]["routing"].insert(0, {"id": "read", "kind": "read_dependencies", "paths": ["../secret"]})
    with pytest.raises(ValueError, match="relative"):
        validate_policy(policy)
    policy = default_policy()
    policy["code"] = "print('not executable')"
    with pytest.raises(ValueError, match="fields"):
        validate_policy(policy)
    source = envelope()
    source["raw_trajectories"][0]["split"] = "report_eval"
    with pytest.raises(ValueError, match="training trajectories"):
        build_evidence(source, default_policy(), {"bundle_hash": "g"})
    policy = default_policy()
    policy["workflows"]["routing"].insert(0, {"id": "read", "kind": "read_dependencies", "paths": ["not_in_index"]})
    with pytest.raises(ValueError, match="permitted envelope"):
        execute(MockBundle(policy), "routing", envelope(), Output, mock_invoke([]), tmp_path)
