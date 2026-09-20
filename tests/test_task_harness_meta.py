"""Offline Meta-to-Task contracts; these fixtures make no real model calls."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from sia.task_meta.meta import MetaAgent, evolution_kwargs, experience_input, validate_route_candidate
from sia.task_meta.observations import build_observation, operation_input
from sia.task_meta.seed import SeedHarnessUpdater, load_seed
from sia.task_meta.storage import artifact_manifest, digest, save_json
from sia.task_meta.task_harness import (
    load_harness,
    migrate_seed,
    task_harness_capabilities,
    task_harness_sources,
    validate_harness,
)
from sia.task_meta.types import (
    ArtifactState,
    DecisionConstraintError,
    EvaluationResult,
    GenerationContext,
    MetaAgentState,
    MetaDecision,
    TaskAgentState,
)


def state(tmp_path):
    seed = Path(__file__).resolve().parents[1] / "seed_harness/seed.json"
    spec = migrate_seed(load_seed(seed))
    path = tmp_path / "gen_0/seed.json"
    save_json(path, spec)
    return TaskAgentState(0, "mock-current-checkpoint", str(path)), spec


def decision(targets):
    return MetaDecision(action="HARNESS", target_components=["HARNESS"], diagnosis="Mock hypothesis",
                        evidence=["mock rollout"], rationale="Exercise the declared mechanism",
                        proposed_change="Mock intervention", expected_effect="Unverified",
                        decision_id="mock_decision_0", decision_source="test_override",
                        requested_changes=[{"id": f"change_{i}", "component": "HARNESS",
                                            "operation": "replace_config", "target": target,
                                            "harness_part": target.split(".")[1], "instruction": "Modify this target"}
                                           for i, target in enumerate(targets)])


def observation(task):
    result = EvaluationResult({"success_rate": 0.0}, [{"question_id": "mock_q", "terminal_reward": 0,
                                                     "model_answer": "wrong", "mock": True}])
    meta = MetaAgentState("mock-frozen-meta", "unused")
    obs = build_observation(task, task, meta, result, [], [], [],
                            {"HARNESS": task_harness_capabilities(task.harness_path)}, retain_raw=True)
    return obs, result, meta


class PatchClient:
    supports_evolution = True

    def __init__(self, edits):
        self.edits, self.calls = edits, []

    def complete(self, prompt, schema, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        output = schema(edits=self.edits, summary="Mock multi-part change; benefit unverified")
        kwargs["validate_candidate"](output)
        return output


def test_cross_part_role_and_graph_changes_are_validated_as_one_transaction(tmp_path):
    task, spec = state(tmp_path)
    obs, result, meta = observation(task)
    graph = copy.deepcopy(spec["parts"]["control"]["graph"])
    graph["entry"] = "review"
    graph["nodes"].insert(0, {"id": "review", "kind": "role", "role": "critic", "next": "prepare"})
    roles = [{"name": "critic", "instruction": "Review visible evidence", "result": "review"}]
    graph_only = copy.deepcopy(spec)
    graph_only["parts"]["control"]["graph"] = graph
    with pytest.raises(ValueError, match="role is not declared"):
        validate_harness(graph_only)
    client = PatchClient([{"target": "parts.control.graph", "value": graph},
                          {"target": "parts.tools.roles", "value": roles}])
    chosen = decision([edit["target"] for edit in client.edits])
    before = digest(Path(task.harness_path))
    successor, update = SeedHarnessUpdater(client).apply(
        task, chosen, GenerationContext(1, tmp_path / "gen_1", obs, result, meta))
    actual = load_harness(successor.harness_path)
    assert actual["parts"]["control"]["graph"] == graph
    assert actual["parts"]["tools"]["roles"] == roles
    assert digest(Path(task.harness_path)) == before
    assert successor.model_ref == task.model_ref and successor.artifacts == task.artifacts
    assert actual["budget"] == spec["budget"] and actual["reference"] == spec["reference"]
    assert update.details["changed_parts"] == ["control", "tools"]
    assert {edit["harness_part"] for edit in update.applied_changes} == {"control", "tools"}
    assert all(edit["semantic_status"] == "unverified" for edit in update.applied_changes)
    assert update.details["harness_identity"]["semantic_hash"] != update.details["parent_harness_identity"]["semantic_hash"]
    assert len(client.calls) == 1 and not Path(successor.harness_path + ".tmp").exists()


@pytest.mark.parametrize("mutation", ["wrong_part", "missing_part", "protected_target", "duplicate_target"])
def test_bad_request_rejected_before_model_call_and_generation_creation(tmp_path, mutation):
    task, spec = state(tmp_path)
    obs, result, meta = observation(task)
    chosen = decision(["parts.input.action_system"])
    if mutation == "wrong_part":
        chosen.requested_changes[0].harness_part = "control"
    elif mutation == "missing_part":
        chosen.requested_changes[0].harness_part = None
    elif mutation == "protected_target":
        chosen.requested_changes[0].target = "budget.max_model_calls"
    else:
        duplicate = chosen.requested_changes[0].model_copy(deep=True)
        duplicate.id = "another_id"
        chosen.requested_changes.append(duplicate)
    client = PatchClient([])
    with pytest.raises(ValueError):
        validate_route_candidate(chosen, obs.available_actions, harness_spec=spec)
    with pytest.raises(DecisionConstraintError):
        SeedHarnessUpdater(client).apply(task, chosen, GenerationContext(1, tmp_path / "gen_1", obs, result, meta))
    assert not client.calls and not (tmp_path / "gen_1").exists()


@pytest.mark.parametrize("edits", [
    [{"target": "parts.tools.roles", "value": [{"name": "helper", "instruction": "Review", "result": "review",
                                                "model": "foreign-provider"}]}],
    [{"target": "parts.tools.roles", "value": []}],
    [{"target": "budget.max_model_calls", "value": 999999}],
])
def test_invalid_noop_or_undeclared_patch_does_not_publish_successor(tmp_path, edits):
    task, _ = state(tmp_path)
    obs, result, meta = observation(task)
    client = PatchClient(edits)
    before = digest(Path(task.harness_path))
    with pytest.raises(DecisionConstraintError):
        SeedHarnessUpdater(client).apply(task, decision(["parts.tools.roles"]),
                                         GenerationContext(1, tmp_path / "gen_1", obs, result, meta))
    assert digest(Path(task.harness_path)) == before and not (tmp_path / "gen_1").exists()


def test_route_and_patch_receive_identical_actual_harness_and_dependency_sources(tmp_path):
    task, spec = state(tmp_path)
    obs, result, meta = observation(task)
    chosen = decision(["parts.input.action_system"])
    class RouteClient:
        supports_evolution = True
        def complete(self, _prompt, _schema, **kwargs):
            self.call = kwargs
            kwargs["validate_candidate"](chosen)
            return chosen
    route_client = RouteClient()
    actual_decision = MetaAgent(route_client, {}).diagnose_and_route(meta, obs)
    patch_client = PatchClient([{"target": "parts.input.action_system", "value": "Use the unchanged task and visible tools."}])
    SeedHarnessUpdater(patch_client).apply(task, actual_decision,
                                          GenerationContext(1, tmp_path / "gen_1", obs, result, meta))
    route_files = route_client.call["operation_input"]["current_files"]
    patch_files = patch_client.calls[0]["operation_input"]["current_files"]
    expected = task_harness_sources(task.harness_path)
    assert route_files == patch_files == expected
    assert json.loads(route_files["seed.json"]) == spec
    assert "task_harness" not in route_files
    assert "runtime/sia/task_meta/task_harness/policy.py" in route_files
    assert route_client.call["operation_input"]["trusted_facts"]["available_actions"]["HARNESS"]["target_parts"]


def test_live_asset_content_and_unverified_origin_reach_meta_without_harness_collision(tmp_path):
    task, _ = state(tmp_path)
    original = tmp_path / "gen_0/input"
    output = tmp_path / "gen_0/output"
    original.mkdir()
    output.mkdir()
    (original / "expired.md").write_text("Old input only", encoding="utf-8")
    long_content = "current unverified asset " * 1000 + "FULL_ASSET_TAIL"
    (output / "seed.json").write_text(long_content, encoding="utf-8")
    task_in = replace(task, artifacts=ArtifactState(str(original), artifact_manifest(original)))
    task_out = replace(task, artifacts=ArtifactState(str(output), artifact_manifest(output)))
    provenance = [{"path": "seed.json", "production_method": "mock_rollout", "terminal_reward": 0,
                   "knowledge_verified": False, "source_trajectory": "mock_q/0"}]
    result = EvaluationResult({"success_rate": 0}, [], artifact_provenance=provenance)
    meta = MetaAgentState("mock-meta", "unused")
    obs = build_observation(task_in, task_out, meta, result, [], [], [], {}, retain_raw=True)
    assert obs.output_artifacts["entries"][0]["content_status"] == "truncated"
    envelope = operation_input(obs)
    assert envelope["current_files"]["assets/seed.json"] == long_content
    assert json.loads(envelope["current_files"]["seed.json"])["schema_version"] == 2
    assert "assets/expired.md" not in envelope["current_files"]
    assert envelope["trusted_facts"]["output_artifacts"]["provenance"] == provenance
    assert "content" not in envelope["trusted_facts"]["output_artifacts"]["entries"][0]
    assert envelope["trusted_facts"]["input_artifacts"]["entries"][0]["path"] == "expired.md"
    envelope["trusted_facts"]["output_artifacts"]["provenance"][0]["knowledge_verified"] = True
    assert obs.output_artifacts["provenance"][0]["knowledge_verified"] is False
    artifact_decision = SimpleNamespace(action=SimpleNamespace(value="ARTIFACTS"), model_dump=lambda **_: {})
    forwarded = evolution_kwargs(SimpleNamespace(supports_evolution=True), SimpleNamespace(observation=obs),
                                 task_out, artifact_decision, {"seed.json": long_content}, lambda _: True)
    assert forwarded["operation_input"]["current_files"]["assets/seed.json"] == long_content
    assert forwarded["operation_input"]["current_files"]["seed.json"] != long_content


def test_conflicting_fixed_dependency_cannot_replace_the_observation_snapshot(tmp_path):
    task, _ = state(tmp_path)
    obs, _, _ = observation(task)
    with pytest.raises(ValueError, match="Conflicting current evidence source"):
        operation_input(obs, current_files={"seed.json": "forged source"})


def test_g_dependency_steps_deliver_actual_task_parts_to_the_mock_stage(tmp_path):
    from sia.task_meta.meta_harness import MetaHarnessStore
    from sia.task_meta.meta_harness.runtime import execute

    task, spec = state(tmp_path)
    spec["parts"]["input"]["action_system"] = "MOCK_ACTUAL_TASK_H_PART_INPUT"
    save_json(Path(task.harness_path), spec)
    obs, _, _ = observation(task)
    project = Path(__file__).resolve().parents[1]
    store = MetaHarnessStore(tmp_path / "meta")
    initial = store.initialize(project / "meta_harness/seed")
    chosen = decision(["parts.input.action_system"])
    prompts = []
    def invoke(_stage, prompt, _schema, **_):
        prompts.append(prompt)
        return chosen
    invoke.decision_source = "test_override"
    execute(initial, "route", operation_input(obs), MetaDecision, invoke, tmp_path / "g0_audit",
            validate_candidate=lambda value: validate_route_candidate(value, obs.available_actions, harness_spec=spec))
    assert "MOCK_ACTUAL_TASK_H_PART_INPUT" in prompts[0]
    assert "file:seed.json" in prompts[0] and initial.hash in prompts[0]
    policy = initial.execution_spec()
    policy["workflows"]["routing"].insert(0, {"id": "task_sources", "kind": "read_dependencies",
                                               "paths": ["runtime/sia/task_meta/task_harness/policy.py"]})
    active = store.commit_update(initial.hash, file_updates={"evolution.json": json.dumps(policy)})
    actual = execute(active, "route", operation_input(obs), MetaDecision, invoke, tmp_path / "g_audit",
                     validate_candidate=lambda value: validate_route_candidate(value, obs.available_actions, harness_spec=spec))
    assert actual == chosen and len(prompts) == 2
    assert "MOCK_ACTUAL_TASK_H_PART_INPUT" in prompts[1]
    assert "def validate_task_harness_request" in prompts[1]
    assert active.hash in prompts[1]


def test_self_update_uses_evaluated_input_provenance_not_same_round_output_origin(tmp_path):
    task, _ = state(tmp_path)
    assets = tmp_path / "gen_0/artifacts"
    assets.mkdir()
    (assets / "note.md").write_text("Unverified input method", encoding="utf-8")
    task.artifacts = ArtifactState(str(assets), artifact_manifest(assets))
    input_origin = [{"path": "note.md", "terminal_reward": 0, "knowledge_verified": False}]
    output_origin = [{"path": "different.md", "terminal_reward": 1, "knowledge_verified": False}]
    save_json(tmp_path / "gen_0/input_artifact_provenance.json", input_origin)
    save_json(tmp_path / "gen_0/artifact_provenance.json", output_origin)
    save_json(tmp_path / "gen_0/agent_execution.json", [{"question_id": "mock_q", "terminal_reward": 0}])
    envelope = experience_input(None, [], task)
    assert envelope["trusted_facts"]["evaluated_input_artifacts"]["provenance"] == input_origin
    assert envelope["trusted_facts"]["current_harness_identity"]["schema_version"] == 2
    assert envelope["current_files"]["assets/note.md"] == "Unverified input method"
    assert "assets/different.md" not in envelope["current_files"]
