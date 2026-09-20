"""Adapter contracts, terminal-reward training input, and artifact boundaries."""

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_task_meta_loop import decision

from sia.profiles import load_meta_agent_profile, load_target_agent_profile
from sia.run_setup import TaskFiles
from sia.task_meta.execution import SIAExecutor, prepare_gpqa_task
from sia.task_meta.storage import artifact_manifest, digest
from sia.task_meta.types import (
    ArtifactState,
    DecisionConstraintError,
    EvaluationResult,
    GenerationContext,
    TaskAgentState,
)
from sia.task_meta.updaters import ArtifactChanges, ArtifactUpdater, ModelUpdater, ProgramUpdate, artifact_path


def test_new_profiles():
    assert load_meta_agent_profile("task-meta-glm").model == "z-ai/glm-5.2"
    assert load_target_agent_profile("task-meta-qwen").provider.api_key_env == "LOCAL_QWEN_API_KEY"


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "x/../../escape", "C:\\temp\\escape",
                                  "results.json", "agent_execution.json", "target_agent.py", ".hidden", ""])
def test_artifact_path_guard(tmp_path, name):
    with pytest.raises(ValueError):
        artifact_path(tmp_path, name)


def test_artifact_updates_only_assets(tmp_path):
    old = tmp_path / "gen_0"
    (old / "artifacts").mkdir(parents=True)
    (old / "artifacts/knowledge.md").write_text("old")
    (old / "target_agent.py").write_text("print('same')")
    state = TaskAgentState(0, "base", str(old / "target_agent.py"),
                           ArtifactState(str(old / "artifacts"), artifact_manifest(old / "artifacts")))
    client = SimpleNamespace(complete=lambda p, s, **kwargs: ArtifactChanges(edits=[{"path": "knowledge.md", "content": "new"}], summary="corrected rule"))
    context = GenerationContext(1, tmp_path / "gen_1", SimpleNamespace(success_examples=[], failure_examples=[]), EvaluationResult({}, []))
    selected = decision("ARTIFACTS")
    selected.requested_changes[0].operation = "write_asset"
    selected.requested_changes[0].target = "knowledge.md"
    new, _ = ArtifactUpdater(client).apply(state, selected, context)
    assert new.model_ref == state.model_ref
    assert digest(Path(new.harness_path)) == digest(Path(state.harness_path))
    assert (old / "artifacts/knowledge.md").read_text() == "old"
    assert (tmp_path / "gen_1/artifacts/knowledge.md").read_text() == "new"


def test_model_unavailable_without_trainer_creates_no_fake_update(tmp_path):
    gen = tmp_path / "gen_0"
    gen.mkdir()
    harness = gen / "target_agent.py"
    harness.write_text("pass")
    state = TaskAgentState(0, "base-model", str(harness))
    rows = [{"question_id": 1, "rollout_id": i, "terminal_reward": reward,
             "messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": f"solution {i}"}]} for i, reward in enumerate([0, 1, 0, 1])]
    client = SimpleNamespace(complete=lambda p, s: ProgramUpdate(code="raise RuntimeError('trainer required')", summary="external training"))
    updater = ModelUpdater(client, TaskFiles("desc", "seed", {}, "spec"), load_target_agent_profile("task-meta-qwen").provider)
    context = GenerationContext(1, tmp_path / "gen_1", None, EvaluationResult({}, rows))
    selected = decision("MODEL")
    selected.requested_changes[0].operation = "sft"
    selected.requested_changes[0].target = "current_checkpoint"
    with pytest.raises(DecisionConstraintError, match="no fixed trainer"):
        updater.apply(state, selected, context)
    assert not (gen / "model_update").exists()
    assert not (tmp_path / "gen_1").exists()
    assert asdict(state)["model_ref"] == "base-model"


def test_gpqa_evaluator_repetition_and_missing_denominator(tmp_path, monkeypatch):
    source = Path(__file__).parents[1] / "sia/tasks/gpqa"
    task = tmp_path / "task"
    manifest = prepare_gpqa_task(source, task, 2)
    assert len(manifest["task_ids"]) == 2
    gen = tmp_path / "gen_0"
    gen.mkdir()
    harness = gen / "target_agent.py"
    harness.write_text("pass")
    provider = load_target_agent_profile("task-meta-qwen").provider
    executor = SIAExecutor(task, sys.prefix, provider, rollouts_per_task=8)
    seeds, artifact_inputs = [], []
    def fake_target(cmd, **kwargs):
        work = Path(cmd[cmd.index("--working_dir") + 1])
        config = json.loads((work / "target_config.json").read_text())
        seeds.append(config["seed"])
        artifact_inputs.append(artifact_manifest(work / "artifacts_input"))
        (work / "agent_execution.json").write_text(json.dumps([{"question_id": manifest["task_ids"][0], "messages": [], "input_tokens": 1, "output_tokens": 1}]))
        return SimpleNamespace(returncode=0)
    def fake_evaluator(directory, *args, **kwargs):
        Path(directory, "results.json").write_text(json.dumps({"accuracy": 1.0, "details": [
            {"question_id": manifest["task_ids"][0], "is_correct": True},
            {"question_id": manifest["task_ids"][1], "is_correct": False}]}))
        return {"status": "success"}
    monkeypatch.setattr("sia.task_meta.execution.subprocess.run", fake_target)
    monkeypatch.setattr("sia.orchestrator.run_evaluation", fake_evaluator)
    result = executor.execute(TaskAgentState(0, "base", str(harness)), gen)
    assert result.performance["success_rate"] == .5  # Missing outputs count in B*R, unlike attempted-only accuracy.
    assert result.performance["attempts"] == 16
    assert len(result.trajectories) == 16
    assert seeds == list(range(42, 50))
    assert artifact_inputs == [[]] * 8
    assert result.output_artifacts.manifest == []
    assert result.evaluated_state.artifacts.manifest == []
    assert result.cost["api_cost_usd"] is None
