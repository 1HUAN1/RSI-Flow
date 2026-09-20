"""Input/output asset separation, bounded Harness hooks, and model identity."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sia.task_meta import gpqa_target as target
from sia.task_meta.execution import collect_generated_artifacts, result_metrics, rollout_artifact_diff
from sia.task_meta.harness import validate_harness_update
from sia.task_meta.storage import artifact_manifest


def response(text, reason="stop", model="current"):
    return SimpleNamespace(model=model, choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=reason)],
                           usage=SimpleNamespace(prompt_tokens=20, completion_tokens=12))


def fake_client(outputs, requests):
    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        value = outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return value
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def question(qid="../unsafe/id"):
    return {"id": qid, "Question": "A public question", "options": {"A": "one", "B": "two", "C": "three", "D": "four"}}


def runtime(rollout=0):
    return {"model_ref": "current", "max_tokens": 512, "temperature": 0.7, "seed": 42,
            "rollout_id": rollout, "service_binding": {"checkpoint_path": "/current-checkpoint"}}


def test_notes_are_isolated_by_task_rollout_and_input_never_written(tmp_path):
    resources = [{"path": "old.md", "content": "old knowledge"}]
    expected = copy.deepcopy(resources)
    requests = []
    records = []
    for qid in ("first", "second"):
        for rollout in (0, 1):
            client = fake_client([response('{"answer":"A","reusable_note":"new general method"}')], requests)
            records.append(target.run_question(question(qid), resources, runtime(rollout), target.harness_config(), client, tmp_path))
    assert resources == expected
    assert len(artifact_manifest(tmp_path)) == 4
    assert all("old knowledge" in call["messages"][0]["content"] for call in requests)
    assert all("new general method" not in call["messages"][0]["content"] for call in requests)
    assert len({row["generated_artifacts"][0]["path"] for row in records}) == 4
    assert all(row["service_binding"]["checkpoint_path"] == "/current-checkpoint" for row in records)
    assert all(row["model_ref_requested"] == row["model_ref_response"] == "current" for row in records)


def test_parse_retry_retains_all_attempts_and_distinguishes_truncation(tmp_path):
    requests = []
    config = {**target.harness_config(), "max_attempts": 2}
    client = fake_client([response("incomplete reasoning", "length"), response('{"answer":"B"}')], requests)
    row = target.run_question(question(), [], runtime(), config, client, tmp_path)
    assert row["valid_answer"] is True
    assert row["parse_failure"] is False
    assert row["parse_failure_count"] == row["output_truncation_count"] == 1
    assert row["input_tokens"] == 40 and row["output_tokens"] == 24
    assert len(requests) == 2 and requests[1]["seed"] == 100042
    assert [message["role"] for message in row["messages"]] == ["user", "assistant", "user", "assistant"]


def test_api_failure_redacts_exception_body_and_has_separate_statistics(tmp_path):
    requests = []
    client = fake_client([RuntimeError("secret credential must not reach trajectory")], requests)
    row = target.run_question(question(), [], runtime(), target.harness_config(), client, tmp_path)
    assert row["api_error"] is True and row["parse_failure"] is False
    assert "secret credential" not in json.dumps(row)
    assert row["input_tokens"] is None
    metrics = result_metrics([{**row, "terminal_reward": 0}])
    assert metrics["api_errors"] == 1 and metrics["parse_failures"] == 0
    assert metrics["valid_answer_rate"] == 0


def test_model_identity_mismatch_fails_instead_of_scoring_wrong_model(tmp_path):
    client = fake_client([response('{"answer":"A"}', model="old-checkpoint")], [])
    with pytest.raises(ValueError, match="immutable requested model binding"):
        target.run_question(question(), [], runtime(), target.harness_config(), client, tmp_path)


def test_matching_model_name_with_wrong_checkpoint_weights_is_rejected(tmp_path):
    config = runtime()
    config["service_binding"]["weights"] = [{"path": "model.safetensors", "sha256": "current"}]
    reply = response('{"answer":"A"}')
    reply.local_checkpoint_binding = {"checkpoint_path": "/current-checkpoint",
                                      "weights": [{"path": "model.safetensors", "sha256": "old"}]}
    with pytest.raises(ValueError, match="checkpoint binding"):
        target.run_question(question(), [], config, target.harness_config(), fake_client([reply], []), tmp_path)


def test_config_can_disable_asset_use_and_generation(tmp_path):
    requests = []
    config = {**target.harness_config(), "use_artifacts": False, "generate_artifacts": False}
    client = fake_client([response('{"answer":"A","reusable_note":"discarded"}')], requests)
    row = target.run_question(question(), [{"path": "old.md", "content": "old contents"}], runtime(), config, client, tmp_path)
    assert "old contents" not in requests[0]["messages"][0]["content"]
    assert row["generated_artifacts"] == [] and artifact_manifest(tmp_path) == []


@pytest.mark.parametrize("changes", [{"max_attempts": 4}, {"max_attempts": True}, {"artifact_char_limit": 12001},
                                     {"use_artifacts": "yes"}, {"new_setting": True}])
def test_harness_configuration_bounds(changes, monkeypatch):
    config = {**target.harness_config(), **changes}
    monkeypatch.setattr(target, "harness_config", lambda: config)
    with pytest.raises(ValueError):
        target.validated_harness_config()


def test_five_editable_hooks_preserve_runtime_contract():
    code = Path(target.__file__).read_text(encoding="utf-8")
    validate_harness_update(code, code)
    changed = code.replace('"max_attempts": 1,', '"max_attempts": 2,').replace('"use_artifacts": True,', '"use_artifacts": False,')
    validate_harness_update(code, changed)
    illegal = code.replace('model=config["model_ref"]', 'model="different-checkpoint"')
    with pytest.raises(ValueError):
        validate_harness_update(code, illegal)
    with pytest.raises(ValueError, match="between 1 and 3"):
        validate_harness_update(code, code.replace('"max_attempts": 1,', '"max_attempts": 9,'))
    with pytest.raises(ValueError, match="literal settings"):
        validate_harness_update(code, code.replace('"max_attempts": 1,', '"max_attempts": min(1, 3),'))


def test_current_output_drops_unproduced_old_assets_without_overwrite():
    difference = rollout_artifact_diff([{"path": "old.md", "sha256": "old"}], [])
    assert difference["lifecycle_removed"] == ["old.md"]
    assert difference["accidental_overwrites"] == []


def test_generated_asset_provenance_is_required(tmp_path):
    work, output = tmp_path / "work", tmp_path / "output"
    produced = work / "artifacts_generated"
    produced.mkdir(parents=True)
    (produced / "undeclared.md").write_text("must not be silently included")
    with pytest.raises(ValueError, match="provenance declarations"):
        collect_generated_artifacts(work, output, [{"question_id": "q", "terminal_reward": 1}], 0)


def test_same_output_path_cannot_overwrite_previous_produced_asset(tmp_path):
    client = fake_client([response('{"answer":"A","reusable_note":"first"}')], [])
    target.run_question(question(), [], runtime(), target.harness_config(), client, tmp_path)
    client = fake_client([response('{"answer":"A","reusable_note":"second"}')], [])
    with pytest.raises(FileExistsError):
        target.run_question(question(), [], runtime(), target.harness_config(), client, tmp_path)
    only = next(tmp_path.rglob("*.md"))
    assert only.read_text() == "first\n"
