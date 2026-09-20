"""Regression checks for model identity and immutable replay inputs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_task_meta_loop import decision

from sia.profiles import load_target_agent_profile
from sia.run_setup import TaskFiles
from sia.task_meta.harness import validate_harness_update
from sia.task_meta.storage import artifact_manifest, checkpoint_manifest
from sia.task_meta.types import (
    ArtifactState,
    DecisionConstraintError,
    EvaluationResult,
    GenerationContext,
    MetaAgentState,
    RequestedChange,
    TaskAgentState,
)
from sia.task_meta.updaters import (
    ArtifactChanges,
    ArtifactUpdater,
    HarnessUpdater,
    ModelUpdater,
    ProgramUpdate,
    positive_sft_rows,
    updater_capabilities,
)

SEED = Path(__file__).parents[1] / "sia/task_meta/gpqa_target.py"


def test_valid_prompt_change_keeps_model_runtime():
    old = SEED.read_text(encoding="utf-8")
    new = old.replace("Solve the multiple-choice problem.", "Consider the options carefully, then answer.")
    validate_harness_update(old, new)


def test_harness_updater_reuses_sia_feedback_and_changes_only_prompt(tmp_path):
    old = SEED.read_text(encoding="utf-8")
    new_code = old.replace("Solve the multiple-choice problem.", "Compare the options carefully, then answer.")
    gen = tmp_path / "gen_0"
    (gen / "artifacts").mkdir(parents=True)
    (gen / "target_agent.py").write_text(old, encoding="utf-8")
    (gen / "artifacts/knowledge.md").write_text("fixed reusable knowledge", encoding="utf-8")
    state = TaskAgentState(0, "fixed-model", str(gen / "target_agent.py"), ArtifactState(str(gen / "artifacts")))
    prompts = []
    def complete(prompt, schema, **kwargs):
        prompts.append(prompt)
        assert schema is ProgramUpdate
        assert kwargs["meta_state"].version == 2
        assert kwargs["operation"] == "harness_patch"
        return ProgramUpdate(code=new_code, summary="revised comparison instruction")
    client = SimpleNamespace(complete=complete)
    updater = HarnessUpdater(client, TaskFiles("description", "seed", {}, "task specification"),
                             load_target_agent_profile("task-meta-qwen").provider)
    context = GenerationContext(1, tmp_path / "gen_1", SimpleNamespace(success_examples=[], failure_examples=[]),
                                EvaluationResult({"success_rate": 0.5}, []), MetaAgentState("meta", "harness-v2", 2))
    new, update = updater.apply(state, decision("HARNESS"), context)
    assert len(prompts) == 1 and "task specification" in prompts[0]
    assert new.model_ref == state.model_ref
    assert Path(new.harness_path).read_text(encoding="utf-8") == new_code
    assert Path(state.harness_path).read_text(encoding="utf-8") == old
    assert artifact_manifest(new.artifacts.directory) == artifact_manifest(state.artifacts.directory)
    assert update.requested_changes[0]["target"] == "format_question"
    assert update.applied_changes[0]["target"] == "format_question"
    assert update.unapplied_changes == []
    assert update.semantic_status == "unverified"


@pytest.mark.parametrize("change", ["runtime", "global", "file", "import", "input"])
def test_harness_cannot_redirect_model_or_access_global_state(change):
    old = SEED.read_text(encoding="utf-8")
    if change == "runtime":
        new = old.replace('model=config["model_ref"]', 'model="different-model"')
    else:
        injected = {"global": "    global config\n", "file": '    open("weights.bin", "w")\n',
                    "import": "    import os\n", "input": '    text["id"] = 99\n'}[change]
        new = old.replace("def parse_answer(text):\n", "def parse_answer(text):\n" + injected)
    with pytest.raises(ValueError):
        validate_harness_update(old, new)


@pytest.mark.parametrize("modified", [False, True])
def test_model_trainer_must_change_weights(tmp_path, monkeypatch, modified):
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base model bytes")
    gen = tmp_path / "gen_0"
    (gen / "artifacts").mkdir(parents=True)
    (gen / "artifacts/knowledge.md").write_text("fixed knowledge")
    (gen / "target_agent.py").write_text("print('fixed harness')")
    state = TaskAgentState(0, str(base), str(gen / "target_agent.py"), ArtifactState(str(gen / "artifacts")))
    def unexpected_llm(*args, **kwargs):
        pytest.fail("The fixed MODEL backend must not generate a training-code proposal")
    client = SimpleNamespace(complete=unexpected_llm)
    provider = load_target_agent_profile("task-meta-qwen").provider
    updater = ModelUpdater(client, TaskFiles("desc", "seed", {}, "task"), provider, trainer_command=["trainer"])
    rows = [{"question_id": 1, "rollout_id": 0, "terminal_reward": 1,
             "messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}]
    context = GenerationContext(1, tmp_path / "gen_1", None, EvaluationResult({}, rows))
    checkpoint = tmp_path / "new_checkpoint"
    def train(*args, **kwargs):
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(b"trained model bytes" if modified else b"base model bytes")
        (Path(kwargs["cwd"]) / "checkpoint.json").write_text(json.dumps({"model_ref": "new-serving-id", "checkpoint_path": str(checkpoint)}))
    monkeypatch.setattr("sia.task_meta.updaters.subprocess.run", train)
    monkeypatch.setenv(provider.api_key_env, "local-test")
    monkeypatch.setattr("sia.task_meta.updaters.served_checkpoint_binding",
                        lambda *args: {"checkpoint_path": str(checkpoint), "weights": checkpoint_manifest(checkpoint)})
    if not modified:
        with pytest.raises(ValueError, match="unchanged checkpoint"):
            updater.apply(state, decision("MODEL"), context)
        assert not context.directory.exists()
    else:
        new, update = updater.apply(state, decision("MODEL"), context)
        assert new.model_ref == "new-serving-id"
        assert new.checkpoint_path == str(checkpoint)
        assert len(new.checkpoint_manifest) == 1
        assert Path(new.harness_path).read_text() == Path(state.harness_path).read_text()
        assert artifact_manifest(new.artifacts.directory) == artifact_manifest(state.artifacts.directory)
        assert update.details["weights"] == new.checkpoint_manifest
        assert update.details["served_binding"]["checkpoint_path"] == str(checkpoint)
        request = json.loads((gen / "model_update/training_request.json").read_text())
        assert request["checkpoint_path"] == str(base)
        assert request["generated_training_code_required"] is False
        assert not (gen / "model_update/train.py").exists()
    assert (base / "model.safetensors").read_bytes() == b"base model bytes"


def test_nested_logs_not_artifacts(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/results.json").write_text("{}")
    with pytest.raises(ValueError, match="cannot be an artifact"):
        artifact_manifest(tmp_path)


def patch_state(tmp_path):
    directory = tmp_path / "gen_0"
    (directory / "artifacts").mkdir(parents=True)
    (directory / "artifacts/knowledge.md").write_text("old reusable knowledge", encoding="utf-8")
    (directory / "target_agent.py").write_text(SEED.read_text(encoding="utf-8"), encoding="utf-8")
    state = TaskAgentState(0, "base-model", str(directory / "target_agent.py"), ArtifactState(str(directory / "artifacts")))
    context = GenerationContext(1, tmp_path / "gen_1", SimpleNamespace(success_examples=[], failure_examples=[]),
                                EvaluationResult({}, []), MetaAgentState("meta", "meta-v3", 3))
    return state, context


@pytest.mark.parametrize("changed_hook", ["none", "parse_answer"])
def test_harness_missing_requested_change_is_not_silently_accepted(tmp_path, changed_hook):
    state, context = patch_state(tmp_path)
    code = Path(state.harness_path).read_text(encoding="utf-8")
    if changed_hook == "parse_answer":
        code = code.replace('answer = str(payload.get("answer", "")).strip().upper()',
                            'answer = str(payload.get("answer", "")).upper().strip()')
    client = SimpleNamespace(complete=lambda *args, **kwargs: ProgramUpdate(code=code, summary="claimed desired improvement"))
    updater = HarnessUpdater(client, TaskFiles("desc", "seed", {}, "task"), load_target_agent_profile("task-meta-qwen").provider)
    with pytest.raises(DecisionConstraintError, match="Requested hooks"):
        updater.apply(state, decision("HARNESS"), context)
    assert not context.directory.exists()
    assert (Path(state.artifacts.directory) / "knowledge.md").read_text() == "old reusable knowledge"


def test_cross_component_decision_fails_before_patch_generation(tmp_path):
    state, context = patch_state(tmp_path)
    selected = decision("HARNESS")
    selected.requested_changes.append(RequestedChange(id="restore", component="ARTIFACTS", operation="write_asset",
                                                      target="knowledge.md", instruction="restore knowledge contents"))
    client = SimpleNamespace(complete=lambda *args, **kwargs: pytest.fail("Invalid cross-component decision reached LLM"))
    updater = HarnessUpdater(client, TaskFiles("desc", "seed", {}, "task"), load_target_agent_profile("task-meta-qwen").provider)
    with pytest.raises(DecisionConstraintError, match="single selected"):
        updater.apply(state, selected, context)
    assert not context.directory.exists()


@pytest.mark.parametrize("edits", [
    [{"path": "extra.md", "content": "unrequested rule"}],
    [{"path": "knowledge.md", "content": "old reusable knowledge"}],
    [{"path": "knowledge.md", "content": None}],
    [{"path": "knowledge.md", "content": "new"}, {"path": "extra.md", "content": "extra"}],
])
def test_artifact_patch_cannot_ignore_or_expand_requested_operations(tmp_path, edits):
    state, context = patch_state(tmp_path)
    client = SimpleNamespace(complete=lambda *args, **kwargs: ArtifactChanges(edits=edits, summary="claimed change"))
    with pytest.raises(DecisionConstraintError):
        ArtifactUpdater(client).apply(state, decision("ARTIFACTS"), context)
    assert not context.directory.exists()
    assert (Path(state.artifacts.directory) / "knowledge.md").read_text() == "old reusable knowledge"


def test_artifact_patch_uses_decisions_current_meta_state_and_audits_hashes(tmp_path):
    state, context = patch_state(tmp_path)
    def complete(prompt, schema, **kwargs):
        assert kwargs["meta_state"] is context.meta_state
        assert kwargs["operation"] == "artifact_patch"
        return ArtifactChanges(edits=[{"path": "knowledge.md", "content": "revised method"}], summary="revised rule")
    _, update = ArtifactUpdater(SimpleNamespace(complete=complete)).apply(state, decision("ARTIFACTS"), context)
    assert update.files[0]["before_sha256"] != update.files[0]["after_sha256"]
    assert update.semantic_status == "unverified"


def test_model_unavailable_without_positives_has_no_side_effect(tmp_path):
    state, context = patch_state(tmp_path)
    updater = ModelUpdater(None, None, load_target_agent_profile("task-meta-qwen").provider, trainer_command=["trainer"])
    assert not updater_capabilities(state, context.evaluation)["MODEL"]["available"]
    with pytest.raises(DecisionConstraintError, match="no complete positive"):
        updater.apply(state, decision("MODEL"), context)
    assert not context.directory.exists()
    assert not (Path(state.harness_path).parent / "model_update").exists()


def test_model_rejects_served_identity_with_wrong_weights(tmp_path, monkeypatch):
    state, context = patch_state(tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base")
    state.checkpoint_path = str(base)
    context.evaluation.trajectories = [{"question_id": 1, "rollout_id": 0, "terminal_reward": 1,
                                      "messages": [{"role": "user", "content": "question"},
                                                   {"role": "assistant", "content": "answer"}]}]
    trained = tmp_path / "trained"
    def train(*args, **kwargs):
        trained.mkdir()
        (trained / "model.safetensors").write_bytes(b"trained")
        (Path(kwargs["cwd"]) / "checkpoint.json").write_text(json.dumps({"model_ref": "registered",
                                                                       "checkpoint_path": str(trained)}))
    monkeypatch.setattr("sia.task_meta.updaters.subprocess.run", train)
    monkeypatch.setattr("sia.task_meta.updaters.served_checkpoint_binding",
                        lambda *args: {"checkpoint_path": str(trained), "weights": checkpoint_manifest(base)})
    updater = ModelUpdater(None, None, load_target_agent_profile("task-meta-qwen").provider, trainer_command=["trainer"])
    with pytest.raises(ValueError, match="binding does not match"):
        updater.apply(state, decision("MODEL"), context)
    assert not context.directory.exists()


def test_consecutive_model_updates_train_from_current_checkpoint(tmp_path, monkeypatch):
    state, context = patch_state(tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base")
    state.checkpoint_path = str(base)
    context.evaluation.trajectories = [{"question_id": 1, "rollout_id": 0, "terminal_reward": 1,
                                      "messages": [{"role": "user", "content": "question"},
                                                   {"role": "assistant", "content": "answer"}]}]
    starts = []
    def train(*args, **kwargs):
        request_dir = Path(kwargs["cwd"])
        request = json.loads((request_dir / "training_request.json").read_text())
        starts.append(request["checkpoint_path"])
        checkpoint = request_dir / "checkpoint"
        checkpoint.mkdir()
        old = Path(request["checkpoint_path"]) / "model.safetensors"
        (checkpoint / "model.safetensors").write_bytes(old.read_bytes() + b"-updated")
        (request_dir / "checkpoint.json").write_text(json.dumps({"model_ref": str(checkpoint),
                                                                "checkpoint_path": str(checkpoint)}))
    monkeypatch.setattr("sia.task_meta.updaters.subprocess.run", train)
    monkeypatch.setattr("sia.task_meta.updaters.served_checkpoint_binding",
                        lambda provider, ref: {"checkpoint_path": ref, "weights": checkpoint_manifest(ref)})
    updater = ModelUpdater(None, None, load_target_agent_profile("task-meta-qwen").provider, trainer_command=["trainer"])
    first, _ = updater.apply(state, decision("MODEL"), context)
    second_context = GenerationContext(2, tmp_path / "gen_2", context.observation, context.evaluation, context.meta_state)
    second, _ = updater.apply(first, decision("MODEL"), second_context)
    assert starts == [str(base), first.checkpoint_path]
    assert second.checkpoint_path != first.checkpoint_path
    assert Path(second.checkpoint_path, "model.safetensors").read_bytes() == b"base-updated-updated"


def test_positive_sft_rows_preserve_rewards_and_require_recorded_dialogues():
    messages = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]
    rows = [{"question_id": 1, "rollout_id": index, "terminal_reward": reward, "messages": messages}
            for index, reward in enumerate([0, 1, -1, float("nan"), True])]
    rows.append({"question_id": 1, "rollout_id": 9, "terminal_reward": 1, "messages": messages[-1:]})
    assert [row["rollout_id"] for row in positive_sft_rows(rows)] == [1]
