"""Offline wiring tests using engineer-written mock/test_override responses.

Real controller, G interpreter, updater validation and commit paths are exercised.
No Codex/API/model inference occurs. MODEL stops before the trainer subprocess;
the fixture checkpoint is arbitrary bytes and never counts as trained weights.
"""
import copy
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

from sia.task_meta.durable import DurableClient, DurableExecutor, DurableUpdater, StageJournal
from sia.task_meta.loop import _accept_meta_update, run_task_meta
from sia.task_meta.meta import MetaAgent, experience_input, validate_meta_candidate
from sia.task_meta.meta_harness import MetaHarnessBundle, MetaHarnessStore
from sia.task_meta.meta_harness.runtime import execute
from sia.task_meta.observations import build_observation, operation_input, trajectory_statistics
from sia.task_meta.seed import SeedHarnessUpdater
from sia.task_meta.storage import checkpoint_manifest, save_json
from sia.task_meta.types import (
    EvaluationResult,
    GenerationContext,
    ImprovementExperience,
    MetaAgentState,
    MetaDecision,
    MetaHarnessUpdate,
    TaskAgentState,
    TaskUpdateAction,
)
from sia.task_meta.updaters import ArtifactUpdater, ModelUpdater, updater_capabilities

ROOT = Path(__file__).resolve().parents[1]


def make_decision(action="HARNESS"):
    operation, target = {"HARNESS": ("replace_config", "prompts.planning_initial"),
                         "ARTIFACTS": ("write_asset", "strategy.md"),
                         "MODEL": ("sft", "current_checkpoint")}[action]
    return MetaDecision(action=action, diagnosis="Mock hypothesis", evidence=["mock recorded outcome"],
        rationale="test_override wiring check", proposed_change="One bounded intervention", expected_effect="Unverified",
        target_components=[action], decision_id="test_override_decision", decision_source="mock",
        requested_changes=[{"id": "change_1", "component": action, "operation": operation,
                            "target": target, "instruction": "test_override only"}])


class CaptureEvolutionClient:
    """Mock only the model callback; execute the actual versioned G program."""
    supports_evolution = True
    decision_source = "mock"

    def __init__(self, store, audit_dir):
        self.bundle_manager, self.audit_dir = store, Path(audit_dir)
        self.calls, self.stages = [], []

    def complete(self, prompt, schema, *, meta_state, operation, operation_input,
                 validate_candidate=None, experience_id=None, **kwargs):
        number = len(self.calls)
        path = Path(meta_state.bundle_path)
        bundle = MetaHarnessBundle(path, json.loads((path / "manifest.json").read_text())).verify()
        assert bundle.hash == meta_state.bundle_hash
        self.calls.append({"operation": operation, "bundle_hash": bundle.hash,
                           "input": copy.deepcopy(operation_input), "schema": schema.__name__,
                           "meta_version": meta_state.version})

        def invoke(stage_id, stage_prompt, stage_schema, **_):
            self.stages.append({"operation": operation, "stage_id": stage_id,
                                "bundle_hash": bundle.hash, "prompt": stage_prompt})
            if operation == "route":
                return make_decision()
            if operation == "harness_patch":
                seed = json.loads(operation_input["current_files"]["seed.json"])
                return stage_schema(edits=[{"target": "prompts.planning_initial",
                    "value": seed["prompts"]["planning_initial"] + f" test_override_patch_{number}"}],
                    summary="mock seed patch")
            if operation == "artifact_patch":
                return stage_schema(edits=[{"path": "strategy.md", "content": "Mock reusable strategy"}],
                                    summary="mock artifact patch")
            if operation == "model_request":
                return stage_schema(objective="Mock organization; no training executed", preserve_behaviors=["Fixed checkpoint"],
                                    rationale="test_override organization check")
            return stage_schema(harness=bundle.read_files()["instructions.md"], rationale="Mock NO_CHANGE",
                                changed_rules=[], status="NO_CHANGE", request_id=f"mock_request_{number}",
                                experience_id=experience_id)

        return execute(bundle, operation, operation_input, schema, invoke,
                       self.audit_dir / f"call_{number:03d}", validate_candidate=validate_candidate)


def states(tmp_path):
    (tmp_path / "gen_0").mkdir(parents=True)
    shutil.copy2(ROOT / "seed_harness/seed.json", tmp_path / "gen_0/seed.json")
    checkpoint = tmp_path / "mock_checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"test_override arbitrary bytes; never loaded as weights")
    task = TaskAgentState(0, "mock_task_model", str(tmp_path / "gen_0/seed.json"),
                          checkpoint_path=str(checkpoint), checkpoint_manifest=checkpoint_manifest(checkpoint))
    store = MetaHarnessStore(tmp_path / "meta")
    bundle = store.initialize(ROOT / "meta_harness/seed")
    mirror = tmp_path / "meta/harness_v0.md"
    mirror.write_text(bundle.read_files()["instructions.md"], encoding="utf-8")
    meta = MetaAgentState("mock_frozen_meta_model", str(mirror), bundle_hash=bundle.hash, bundle_path=str(bundle.path))
    return task, meta, store


def evaluation(task, *, long=False):
    text = "BEGIN_" + ("x" * 120000 if long else "mock evidence") + "_FULL_RAW_TAIL"
    rows = [{"question_id": index, "task_id": f"mock_{index}", "rollout_id": 0, "domain": "searchqa",
             "split": "evolve_train", "terminal_reward": reward, "model_answer": "A", "valid_answer": True,
             "messages": [{"role": "user", "content": "mock question"}, {"role": "assistant", "content": text}],
             "mock": True} for index, reward in enumerate((1, 0))]
    return EvaluationResult({"success_rate": .5, "denominator": 2}, rows,
                            {"wall_time_seconds": 1, "api_cost_usd": 0}, evaluated_state=task)


def observation(task, meta, result):
    capabilities = updater_capabilities(task, result, trainer_configured=True, sft_profile="multidomain")
    return build_observation(task, task, meta, result, [result.performance], [], [result.cost],
                             capabilities, retain_raw=True)


def transition(tmp_path, task, result):
    successor = copy.deepcopy(task)
    successor.generation = 1
    successor.harness_path = str(tmp_path / "gen_1/seed.json")
    Path(successor.harness_path).parent.mkdir(exist_ok=True)
    shutil.copy2(task.harness_path, successor.harness_path)
    for generation in (0, 1):
        save_json(tmp_path / f"gen_{generation}/agent_execution.json", result.trajectories)
    experience = ImprovementExperience(0, asdict(task), asdict(successor), make_decision().model_dump(mode="json"),
        {"summary": "mock intervention"}, result.performance, result.performance, 0.0,
        result.cost, result.cost, {}, str(tmp_path / "gen_0/agent_execution.json"),
        str(tmp_path / "gen_1/agent_execution.json"), experience_id="mock_experience_0_1")
    return successor, experience


def test_preview_truncation_never_limits_g_raw_evidence_or_trusted_denominator(tmp_path):
    task, meta, _ = states(tmp_path)
    result = evaluation(task, long=True)
    original = copy.deepcopy(result.trajectories)
    obs = observation(task, meta, result)
    assert obs.observation_coverage["truncated_fields"]
    assert "_FULL_RAW_TAIL" not in json.dumps(obs.trajectories)
    envelope = operation_input(obs)
    assert envelope["raw_trajectories"] == original
    assert "_FULL_RAW_TAIL" in envelope["raw_trajectories"][0]["messages"][-1]["content"]
    assert envelope["trusted_facts"]["performance"]["denominator"] == 2
    assert envelope["trusted_facts"]["trajectory_statistics"] == trajectory_statistics(original)
    envelope["raw_trajectories"][0]["terminal_reward"] = 999
    assert result.trajectories == original and obs.raw_trajectories == original
    assert obs.current_performance == result.performance


def test_all_controller_stages_load_the_same_new_g_and_model_stops_before_training(tmp_path, monkeypatch):
    task, meta, store = states(tmp_path)
    prior = store.active()
    policy = prior.execution_spec()
    policy["diagnosis"]["instruction"] = "MOCK_G1_DIAGNOSIS_RULE"
    policy["self_update"]["instruction"] = "MOCK_G1_SELF_UPDATE_RULE"
    new = store.commit_update(prior.hash, file_updates={"evolution.json": json.dumps(policy)})
    meta.bundle_hash, meta.bundle_path, meta.version = new.hash, str(new.path), new.version
    meta.harness_path = str(tmp_path / f"meta/harness_v{new.version}.md")
    Path(meta.harness_path).write_text(new.read_files()["instructions.md"], encoding="utf-8")
    client = CaptureEvolutionClient(store, tmp_path / "audit")
    agent = MetaAgent(client, {"sft_profile": "multidomain"})
    result = evaluation(task, long=True)
    obs = observation(task, meta, result)
    selected = agent.diagnose_and_route(meta, obs)
    def context(name):
        return GenerationContext(1, tmp_path / name, obs, result, meta)
    seed_state, _ = SeedHarnessUpdater(client).apply(task, selected, context("harness_successor"))
    artifact_state, _ = ArtifactUpdater(client).apply(task, make_decision("ARTIFACTS"), context("artifact_successor"))
    assert seed_state.model_ref == artifact_state.model_ref == task.model_ref

    class TrainerNotExecuted(RuntimeError):
        pass
    trainer_calls = []
    def stop_before_training(*args, **kwargs):
        trainer_calls.append((args, kwargs))
        assert client.calls[-1]["operation"] == "model_request"
        raise TrainerNotExecuted("mock fixture stop before training")
    monkeypatch.setattr("sia.task_meta.updaters.subprocess.run", stop_before_training)
    updater = ModelUpdater(client, None, None, trainer_command=["never-executed", "{request_dir}"], sft_profile="legacy_gpqa")
    with pytest.raises(TrainerNotExecuted, match="before training"):
        updater.apply(task, make_decision("MODEL"), context("model_successor"))
    request = json.loads((Path(task.harness_path).parent / "model_update/training_request.json").read_text())
    assert request["status"] == "prepared" and request["meta_harness_hash"] == new.hash
    assert request["meta_request_plan"]["objective"] == "Mock organization; no training executed"
    assert len(trainer_calls) == 1 and not (tmp_path / "model_successor").exists()
    assert checkpoint_manifest(task.checkpoint_path) == task.checkpoint_manifest

    current, experience = transition(tmp_path, task, result)
    agent.learn_from_experience(meta, experience, [experience], current)
    agent.final_consolidation(meta, [experience], current)
    assert [call["operation"] for call in client.calls] == [
        "route", "harness_patch", "artifact_patch", "model_request", "learn", "final_consolidation"]
    assert {call["bundle_hash"] for call in client.calls} == {new.hash}
    assert all(call["input"]["raw_trajectories"] for call in client.calls)
    assert all("_FULL_RAW_TAIL" in json.dumps(call["input"]["raw_trajectories"]) for call in client.calls)
    assert all(("MOCK_G1_SELF_UPDATE_RULE" if stage["operation"] in {"learn", "final_consolidation"}
                else "MOCK_G1_DIAGNOSIS_RULE") in stage["prompt"] for stage in client.stages)


def test_real_loop_nochange_counts_events_and_crash_replay_does_not_repeat_calls(tmp_path, monkeypatch):
    from test_task_meta_loop import FakeExecutor

    import sia.task_meta.loop as loop
    task, meta, store = states(tmp_path)
    capture = CaptureEvolutionClient(store, tmp_path / "audit")
    journal = StageJournal(tmp_path)
    client = DurableClient(capture, journal)
    agent = MetaAgent(client, {"sft_profile": "multidomain", "trainer_configured": False})
    execution_events = []
    executor = DurableExecutor(FakeExecutor(execution_events), journal)
    class UnusedUpdater:
        def apply(self, *_):
            raise AssertionError("Mock fixture selects only HARNESS")
    updaters = {action: DurableUpdater(SeedHarnessUpdater(client) if action == TaskUpdateAction.HARNESS
                                       else UnusedUpdater(), journal) for action in TaskUpdateAction}
    commits = []
    def accept(state, update):
        commits.append(update.request_id)
        bundle = store.commit_update(state.bundle_hash, instruction_text=update.harness,
            file_updates=update.bundle_files, request_id=update.request_id, experience_id=update.experience_id,
            phase="meta_self_update" if update.experience_id else "final_consolidation")
        state.bundle_hash, state.bundle_path = bundle.hash, str(bundle.path)
        return state
    original_save = loop.save_json
    crashed = [False]
    def crash_before_final_state(path, value):
        if Path(path).name == "final_state.json" and value["status"] == "completed" and not crashed[0]:
            crashed[0] = True
            raise RuntimeError("mock crash after final acceptance")
        original_save(path, value)
    monkeypatch.setattr(loop, "save_json", crash_before_final_state)
    with pytest.raises(RuntimeError, match="after final acceptance"):
        run_task_meta(tmp_path, task, meta, executor, agent, updaters, max_generations=2, meta_update_handler=accept)
    prior_counts = (len(capture.calls), len(commits), len(execution_events))
    final = run_task_meta(tmp_path, task, meta, executor, agent, updaters, max_generations=2,
                          meta_update_handler=accept, resume=True)
    assert (len(capture.calls), len(commits), len(execution_events)) == prior_counts
    assert final["meta_state"]["version"] == 0 and final["meta_state"]["bundle_hash"] == meta.bundle_hash
    assert final["meta_update_events"] == final["meta_no_change_events"] == 2
    assert final["meta_version_changes"] == 0 and final["experiences"] == 1
    assert final["decision_mode"] == "mock"
    assert len(list((tmp_path / "meta/harness_versions").iterdir())) == 1
    assert len(list((tmp_path / "meta/commit_events").iterdir())) == 2
    assert len((tmp_path / "meta/experiences.jsonl").read_text().splitlines()) == 1


def test_forged_self_update_experience_id_is_rejected_before_handler(tmp_path):
    task, meta, store = states(tmp_path)
    _current, experience = transition(tmp_path, task, evaluation(task))
    update = MetaHarnessUpdate(harness=store.active().read_files()["instructions.md"], rationale="mock", changed_rules=[],
                               status="NO_CHANGE", request_id="mock_forged_request", experience_id="forged_experience")
    called = []
    with pytest.raises(ValueError, match="actual re-evaluated experience"):
        _accept_meta_update(tmp_path, meta, update, [experience], tmp_path / "receipt.json", "learn_from_experience",
                            handler=lambda *_: called.append(True))
    assert not called and not (tmp_path / "receipt.json").exists()


@pytest.mark.parametrize("split", ["final", "final_test", "probe", "evolve_dev"])
def test_forbidden_final_and_probe_evidence_rejected_before_mock_model(tmp_path, split):
    task, meta, store = states(tmp_path)
    result = evaluation(task)
    result.trajectories[0]["split"] = split
    current, experience = transition(tmp_path, task, result)
    client = CaptureEvolutionClient(store, tmp_path / "audit")
    with pytest.raises(ValueError, match="training trajectories"):
        MetaAgent(client, {}).learn_from_experience(meta, experience, [experience], current)
    assert client.stages == [] and store.active().hash == meta.bundle_hash


@pytest.mark.parametrize("path", ["../runtime.py", "manifest.json", "/outside", "evaluator.py"])
def test_model_returned_g_paths_rejected_without_touching_store(tmp_path, path):
    _, meta, store = states(tmp_path)
    update = MetaHarnessUpdate(harness=store.active().read_files()["instructions.md"], rationale="mock", changed_rules=[],
                               bundle_files={path: "forbidden candidate"})
    with pytest.raises(ValueError, match="protected paths"):
        validate_meta_candidate(update, meta)
    assert store.active().hash == meta.bundle_hash


def test_experience_paths_are_not_followed_and_latest_must_match_ledger(tmp_path):
    task, _, _ = states(tmp_path)
    current, experience = transition(tmp_path, task, evaluation(task))
    outside = tmp_path / "final_benchmark_answers.json"
    outside.write_text('[{"secret_final_answer":"DO_NOT_LOAD"}]', encoding="utf-8")
    experience.trajectory_before = experience.trajectory_after = str(outside)
    envelope = experience_input(experience, [experience], current)
    assert "DO_NOT_LOAD" not in json.dumps(envelope["raw_trajectories"])
    assert {source["source"] for source in envelope["trusted_facts"]["sources"]} == {
        "gen_0/agent_execution.json", "gen_1/agent_execution.json"}
    forged = copy.deepcopy(experience)
    forged.experience_id = "forged_not_in_ledger"
    with pytest.raises(ValueError, match="latest actual ledger"):
        experience_input(forged, [experience], current)
