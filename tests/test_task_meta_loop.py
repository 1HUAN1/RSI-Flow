"""Deterministic order/invariant tests; no API, GPU, model or trainer required."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sia.task_meta.loop import BudgetManager, performance_delta, primary_metric, run_task_meta
from sia.task_meta.storage import artifact_manifest, clone_task
from sia.task_meta.types import (
    ArtifactState,
    DecisionConstraintError,
    EvaluationResult,
    MetaAgentState,
    MetaDecision,
    MetaHarnessUpdate,
    TaskAgentState,
    TaskUpdate,
    TaskUpdateAction,
    UpdatePending,
)

ACTIONS = list(TaskUpdateAction)


def decision(action):
    operation, target = {"HARNESS": ("replace_hook", "format_question"),
                         "ARTIFACTS": ("write_asset", "knowledge.md"),
                         "MODEL": ("sft", "current_checkpoint")}.get(str(action), ("test", "test"))
    return MetaDecision(action=action, diagnosis="observed bottleneck", evidence=["rewarded trajectory"],
                        rationale="one change", proposed_change="update chosen component", expected_effect="unknown",
                        target_components=[action], requested_changes=[{"id": "edit_1", "component": action,
                          "operation": operation, "target": target, "instruction": "modify selected component"}])


class FakeExecutor:
    def __init__(self, events):
        self.events = events

    def execute(self, state, directory):
        self.events.append(("execute", state.generation))
        if state.generation == 0:
            assert state.artifacts.directory is None
            assert state.artifacts.manifest == []
        artifacts = directory / "artifacts_generated"
        artifacts.mkdir(exist_ok=True)
        (artifacts / "strategy.md").write_text(f"method from generation {state.generation}", encoding="utf-8")
        return EvaluationResult({"success_rate": [0.30, 0.40, 0.52, 0.58, 0.60][state.generation]},
                                [{"question_id": 1, "rollout_id": i, "terminal_reward": int(i % 2 == 0),
                                  "messages": [{"role": "assistant", "content": "solution"}]} for i in range(8)],
                                {"wall_time_seconds": state.generation + 1, "api_cost_usd": None},
                                evaluated_state=state, output_artifacts=ArtifactState(str(artifacts), artifact_manifest(artifacts)))


class FakeMeta:
    def __init__(self, events):
        self.events = events
        self.observations = []

    def diagnose_and_route(self, state, observation, feedback=None):
        self.events.append(("route", observation.generation, state.version))
        assert f"version {state.version}" in Path(state.harness_path).read_text(encoding="utf-8")
        self.observations.append(observation)
        return decision(["HARNESS", "MODEL", "ARTIFACTS", "HARNESS"][observation.generation])

    def learn_from_experience(self, state, experience, history, task):
        self.events.append(("learn", experience.generation, state.version))
        assert len(history) == task.generation
        assert experience.state_after["generation"] == task.generation
        assert Path(experience.trajectory_before).is_file()
        assert Path(experience.trajectory_after).is_file()
        return MetaHarnessUpdate(harness=f"version {state.version + 1}", rationale="actual feedback", changed_rules=["route"])

    def final_consolidation(self, state, history, task):
        self.events.append(("final", task.generation, state.version))
        return MetaHarnessUpdate(harness=f"version {state.version + 1}", rationale="all experiences", changed_rules=["summary"])


class FakeUpdater:
    def __init__(self, action, events):
        self.action = action
        self.events = events

    def apply(self, state, selected, context):
        self.events.append(("update", state.generation, self.action.value))
        assert selected.action == self.action
        new = clone_task(state, context.generation, context.directory)
        if self.action == TaskUpdateAction.MODEL:
            new.model_ref = f"fake-test-checkpoint-{context.generation}"
        elif self.action == TaskUpdateAction.HARNESS:
            with Path(new.harness_path).open("a", encoding="utf-8") as f:
                f.write(f"\n# change {context.generation}")
        else:
            (Path(new.artifacts.directory) / "knowledge.md").write_text("reusable rule", encoding="utf-8")
        return new, TaskUpdate(self.action, "applied exactly one action")


def components(tmp_path):
    events = []
    (tmp_path / "gen_0").mkdir()
    (tmp_path / "meta").mkdir()
    harness = tmp_path / "gen_0/target_agent.py"
    harness.write_text("print('fake')", encoding="utf-8")
    meta_path = tmp_path / "meta/harness_v0.md"
    meta_path.write_text("version 0", encoding="utf-8")
    return (events, TaskAgentState(0, "fake-base", str(harness)), MetaAgentState("frozen-model", str(meta_path)),
            FakeExecutor(events), FakeMeta(events), {a: FakeUpdater(a, events) for a in ACTIONS})


def test_five_generation_integration(tmp_path):
    events, task, meta, executor, meta_agent, updaters = components(tmp_path)
    final = run_task_meta(tmp_path, task, meta, executor, meta_agent, updaters)
    assert events == [
        ("execute", 0), ("route", 0, 0), ("update", 0, "HARNESS"),
        ("execute", 1), ("learn", 0, 0), ("route", 1, 1), ("update", 1, "MODEL"),
        ("execute", 2), ("learn", 1, 1), ("route", 2, 2), ("update", 2, "ARTIFACTS"),
        ("execute", 3), ("learn", 2, 2), ("route", 3, 3), ("update", 3, "HARNESS"),
        ("execute", 4), ("learn", 3, 3), ("final", 4, 4),
    ]
    assert final["generations_executed"] == 5
    assert final["experiences"] == 4
    assert final["task_state"]["generation"] == 4
    assert final["meta_state"]["version"] == 5
    assert final["meta_state"]["model_ref"] == "frozen-model"
    assert not (tmp_path / "gen_5").exists()
    assert not (tmp_path / "gen_0/meta_self_update.json").exists()
    assert not (tmp_path / "gen_4/meta_decision.json").exists()
    history = [json.loads(line) for line in (tmp_path / "meta/experiences.jsonl").read_text().splitlines()]
    assert [e["performance_delta"] for e in history] == pytest.approx([.1, .12, .06, .02])
    assert len(list((tmp_path / "meta").glob("harness_v*.md"))) == 6
    initial = meta_agent.observations[0]
    assert initial.previous_action is None and initial.previous_performance_delta is None
    assert initial.improvement_history == []
    assert len(initial.trajectories) == 8
    assert (tmp_path / "gen_0/artifacts_generated/strategy.md").read_text() == "method from generation 0"
    assert all("gen_" not in entry["path"] for entry in final["task_state"]["artifacts"]["manifest"])
    assert json.loads((tmp_path / "gen_0/task_state.json").read_text())["artifacts"]["directory"] is None


def test_one_generation_only_consolidates(tmp_path):
    events, task, meta, executor, agent, updaters = components(tmp_path)
    run_task_meta(tmp_path, task, meta, executor, agent, updaters, max_generations=1)
    assert events == [("execute", 0), ("final", 0, 0)]
    assert not (tmp_path / "gen_1").exists()


def test_pending_model_never_fabricates_successor(tmp_path):
    _events, task, meta, executor, agent, updaters = components(tmp_path)
    class Pending:
        def apply(self, *args):
            raise UpdatePending("trainer not configured")
    updaters[TaskUpdateAction.MODEL] = Pending()
    final = run_task_meta(tmp_path, task, meta, executor, agent, updaters)
    assert final["status"] == "pending_model_update"
    assert final["task_state"]["generation"] == 1
    assert final["experiences"] == 1
    assert not (tmp_path / "gen_2").exists()


def test_cross_component_change_rejected(tmp_path):
    _events, task, meta, executor, agent, updaters = components(tmp_path)
    original = updaters[TaskUpdateAction.HARNESS].apply
    def corrupt(*args):
        new, update = original(*args)
        new.model_ref = "also-changed"
        return new, update
    updaters[TaskUpdateAction.HARNESS].apply = corrupt
    with pytest.raises(ValueError, match="outside the selected"):
        run_task_meta(tmp_path, task, meta, executor, agent, updaters)
    assert json.loads((tmp_path / "final_state.json").read_text())["status"] == "failed"
    assert (tmp_path / "failure.json").is_file()


def test_budget_boundary_still_finishes_learning(tmp_path):
    events, task, meta, executor, agent, updaters = components(tmp_path)
    final = run_task_meta(tmp_path, task, meta, executor, agent, updaters, max_wall_time=1e-9)
    assert final["generations_executed"] == 1
    assert events == [("execute", 0), ("final", 0, 0)]


def test_state_and_delta_meaning_and_final_pointers(tmp_path):
    _events, task, meta, executor, agent, updaters = components(tmp_path)
    final = run_task_meta(tmp_path, task, meta, executor, agent, updaters)
    experience = json.loads((tmp_path / "gen_1/improvement_experience.json").read_text())
    assert experience["evaluated_state_before"]["artifacts"]["manifest"] == []
    assert experience["intervention_base_state"]["artifacts"]["manifest"]
    assert experience["evaluated_state_after"]["artifacts"]["manifest"] == experience["intervention_base_state"]["artifacts"]["manifest"]
    assert experience["intervention_diff"]["artifacts"]["changed"] == []
    assert experience["observed_performance_delta"] == pytest.approx(.1)
    assert final["last_evaluated_task_input"]["artifacts"]["manifest"] != final["last_output_artifacts"]["manifest"]


def test_invalid_cross_component_request_retries_before_any_update(tmp_path):
    events, task, meta, executor, agent, updaters = components(tmp_path)
    route = agent.diagnose_and_route
    def retry(state, observation, feedback=None):
        result = route(state, observation, feedback)
        if observation.generation == 0 and not feedback:
            result.target_components = [TaskUpdateAction.HARNESS, TaskUpdateAction.ARTIFACTS]
        return result
    agent.diagnose_and_route = retry
    run_task_meta(tmp_path, task, meta, executor, agent, updaters, max_generations=2)
    assert len([e for e in events if e[0] == "update"]) == 1
    feedback = json.loads((tmp_path / "gen_0/constraint_feedback.json").read_text())
    assert len(feedback) == 1 and not feedback[0]["actual_task_modification_executed"]


def test_final_summary_failure_preserves_experience_and_last_meta(tmp_path):
    _events, task, meta, executor, agent, updaters = components(tmp_path)
    def fail(*args):
        raise RuntimeError("test final API failure")
    agent.final_consolidation = fail
    with pytest.raises(RuntimeError, match="final API failure"):
        run_task_meta(tmp_path, task, meta, executor, agent, updaters, max_generations=2)
    final = json.loads((tmp_path / "final_state.json").read_text())
    assert final["experiences"] == 1 and final["meta_state"]["version"] == 1
    assert final["final_consolidation_status"] == "failed"
    assert len((tmp_path / "meta/experiences.jsonl").read_text().splitlines()) == 1


def test_truncated_route_is_retried_without_task_modification(tmp_path):
    events, task, meta, executor, agent, updaters = components(tmp_path)
    route = agent.diagnose_and_route
    def retry(state, observation, feedback=None):
        if not feedback:
            raise DecisionConstraintError("Meta output truncated")
        return route(state, observation, feedback)
    agent.diagnose_and_route = retry
    final = run_task_meta(tmp_path, task, meta, executor, agent, updaters, max_generations=2)
    assert final["experiences"] == 1
    assert len([e for e in events if e[0] == "update"]) == 1


def test_test_override_is_explicit_and_real_meta_resumes_after_feedback(tmp_path):
    events, task, meta, executor, agent, updaters = components(tmp_path)
    def override(state, observation):
        return decision("MODEL") if observation.generation == 0 else None
    final = run_task_meta(tmp_path, task, meta, executor, agent, updaters, max_generations=3,
                          test_decision_override=override)
    exp = json.loads((tmp_path / "gen_1/improvement_experience.json").read_text())
    assert exp["decision"]["decision_source"] == "test_override"
    assert exp["chosen_action"] == "MODEL"
    assert events.index(("learn", 0, 0)) < events.index(("route", 1, 1))
    assert final["decision_mode"] == "test_override_enabled"


@pytest.mark.parametrize("value", [0, -1])
def test_invalid_generation_budget(value):
    with pytest.raises(ValueError):
        BudgetManager(value)


@pytest.mark.parametrize("action", ["both", ["HARNESS", "MODEL"], "weights"])
def test_invalid_decisions(action):
    with pytest.raises(ValidationError):
        decision(action)


def test_metric_directions_and_nested_metric():
    assert performance_delta({"loss": 3}, {"loss": 2}, "loss", "min") == 1
    assert performance_delta({"accuracy": .5}, {"accuracy": .4}, "accuracy", "max") == pytest.approx(-.1)
    assert primary_metric({"metrics": {"runtime": 2.5}}, "metrics.runtime") == 2.5
    for bad in [None, True, "40%", float("nan")]:
        with pytest.raises(ValueError):
            primary_metric({"metric": bad}, "metric")
