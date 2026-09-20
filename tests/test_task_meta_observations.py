"""Evidence coverage, current Meta versions, and truncation are engineering contracts."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from sia.task_meta.meta import StructuredClient
from sia.task_meta.observations import build_observation, trajectory_evidence, trajectory_statistics
from sia.task_meta.storage import artifact_manifest, digest
from sia.task_meta.types import (
    ArtifactState,
    DecisionConstraintError,
    EvaluationResult,
    MetaAgentState,
    MetaHarnessUpdate,
    TaskAgentState,
)


def _rows(long=False):
    return [{"question_id": q, "rollout_id": r, "terminal_reward": int(r % 2 == 0),
             "model_answer": "A", "messages": [{"role": "assistant", "content": "reason " * (2000 if long else 2)}]}
            for q in ["q1", "q2"] for r in range(8)]


def test_all_sixteen_trajectories_reach_meta_when_budget_allows():
    rows = _rows()
    selected, coverage = trajectory_evidence(rows)
    assert selected == rows
    assert len(coverage["included_trajectory_ids"]) == 16
    assert coverage["omitted_trajectory_ids"] == []
    assert coverage["truncated_fields"] == []
    assert coverage["statistics"]["rewarded"] == 8


def test_compression_covers_task_and_result_strata_and_records_omissions():
    rows = _rows(long=True)
    selected, coverage = trajectory_evidence(rows, max_chars=2000)
    assert {(r["question_id"], r["terminal_reward"]) for r in selected} == {
        ("q1", 0), ("q1", 1), ("q2", 0), ("q2", 1),
    }
    assert coverage["omitted_trajectory_ids"]
    assert coverage["truncated_fields"]
    assert len(coverage["included_trajectory_ids"]) + len(coverage["omitted_trajectory_ids"]) == 16
    assert coverage["statistics"]["count"] == 16
    assert trajectory_evidence(rows, max_chars=2000) == (selected, coverage)


def test_failure_statistics_distinguish_wrong_answer_truncation_parse_and_api():
    rows = [
        {"question_id": "q", "model_answer": "A", "terminal_reward": 0},
        {"question_id": "q", "model_answer": "", "terminal_reward": 0, "output_truncated": True},
        {"question_id": "q", "model_answer": "", "terminal_reward": 0, "parse_failure": True},
        {"question_id": "q", "model_answer": "", "terminal_reward": 0, "api_error": True},
    ]
    stats = trajectory_statistics(rows)
    assert stats["observed_outcomes"] == {"incorrect_answer": 1, "truncated_output": 1,
                                         "invalid_answer": 1, "model_api_failure": 1}
    assert stats["valid_answer_rate"] == 0.25
    assert stats["model_api_failures"] == 1


def test_observation_has_evaluated_input_output_contents_and_lifecycle_diff(tmp_path):
    task_harness = tmp_path / "task.py"
    task_harness.write_text("# task execution rules", encoding="utf-8")
    meta_harness = tmp_path / "harness_v2.md"
    meta_harness.write_text("Diagnose truncation before reasoning", encoding="utf-8")
    initial = tmp_path / "input"
    output = tmp_path / "output"
    initial.mkdir()
    output.mkdir()
    (initial / "old.md").write_text("input-only method", encoding="utf-8")
    (output / "new.md").write_text("current generated method", encoding="utf-8")
    task_in = TaskAgentState(2, "checkpoint-v2", str(task_harness), ArtifactState(str(initial), artifact_manifest(initial)))
    task_out = replace(task_in, artifacts=ArtifactState(str(output), artifact_manifest(output)))
    meta = MetaAgentState("frozen-meta", str(meta_harness), 2)
    capabilities = {"MODEL": {"available": False, "reason": "no positive training examples"},
                    "HARNESS": {"available": True}, "ARTIFACTS": {"available": True}}
    observation = build_observation(task_in, task_out, meta, EvaluationResult({"success_rate": 0.5}, _rows()),
                                    [], [], [], capabilities)
    assert observation.evaluated_state["artifacts"]["directory"] == str(initial)
    assert observation.intervention_base_state["artifacts"]["directory"] == str(output)
    assert observation.input_artifacts["entries"][0]["content"] == "input-only method"
    assert observation.output_artifacts["entries"][0]["content"] == "current generated method"
    assert observation.rollout_artifact_diff["removed"][0]["path"] == "old.md"
    assert observation.rollout_artifact_diff["added"][0]["path"] == "new.md"
    assert observation.available_actions["MODEL"]["available"] is False
    assert len(observation.trajectories) == 16


def _client(tmp_path, monkeypatch, finish_reason="stop"):
    seen = []
    output = MetaHarnessUpdate(harness="new working rules", rationale="feedback", changed_rules=["diagnosis"])

    class Agent:
        def __init__(self, model, **kwargs):
            seen.append({"model": model, **kwargs})

        def run_sync(self, prompt, **kwargs):
            seen[-1].update({"prompt": prompt, **kwargs})
            return SimpleNamespace(output=output, usage=SimpleNamespace(input_tokens=13, output_tokens=8, requests=1),
                                   all_messages=lambda: [SimpleNamespace(finish_reason=finish_reason)])

    monkeypatch.setitem(sys.modules, "pydantic_ai", SimpleNamespace(Agent=Agent))
    monkeypatch.setitem(sys.modules, "pydantic_ai.usage", SimpleNamespace(UsageLimits=lambda **kwargs: kwargs))
    client = StructuredClient.__new__(StructuredClient)
    client.model = "fake-frozen-model"
    client.model_name = "fake-frozen-model"
    client.journal = tmp_path / "calls"
    client.timeout, client.max_tokens, client.calls = 12, 32, []
    return client, seen


def test_routing_and_patch_wrapper_use_same_latest_harness_and_audit_ids(tmp_path, monkeypatch):
    client, seen = _client(tmp_path, monkeypatch)
    harness = tmp_path / "harness_v2.md"
    harness.write_text("working rules version TWO", encoding="utf-8")
    state = MetaAgentState("frozen", str(harness), 2)
    for operation in ["route", "harness_patch", "artifact_patch"]:
        client.complete("bounded task evidence", MetaHarnessUpdate, meta_state=state,
                        operation=operation, decision_id="generation_2_decision_0")
    assert all("working rules version TWO" in request["prompt"] for request in seen)
    assert all("immutable" in request["instructions"] for request in seen)
    assert all(request["usage_limits"]["request_limit"] == 2 for request in seen)
    for index, operation in enumerate(["route", "harness_patch", "artifact_patch"]):
        row = json.loads((client.journal / f"call_{index:03d}.json").read_text())
        assert row["operation"] == operation
        assert row["meta_harness_version"] == 2
        assert row["meta_harness_hash"] == digest(harness)
        assert row["decision_id"] == "generation_2_decision_0"


def test_truncated_valid_json_is_rejected_and_failure_journaled(tmp_path, monkeypatch):
    client, _seen = _client(tmp_path, monkeypatch, finish_reason="length")
    with pytest.raises(ValueError, match="truncated"):
        client.complete("evidence", MetaHarnessUpdate, operation="learn", experience_id="e0")
    record = json.loads((client.journal / "call_000.json").read_text())
    assert record["status"] == "failed"
    assert record["output"] is None
    assert record["experience_id"] == "e0"
    assert record["output_tokens"] == 8


def test_credentials_are_redacted_before_prompt_is_sent_or_saved(tmp_path, monkeypatch):
    client, seen = _client(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-secret-fixture-value")
    client.complete("accidental echo private-secret-fixture-value", MetaHarnessUpdate)
    assert "private-secret-fixture-value" not in seen[0]["prompt"]
    assert "private-secret-fixture-value" not in (client.journal / "call_000_prompt.txt").read_text()


def test_new_client_does_not_overwrite_existing_call_files(tmp_path, monkeypatch):
    client, _seen = _client(tmp_path, monkeypatch)
    client.complete("first", MetaHarnessUpdate)
    original = (client.journal / "call_000.json").read_text()
    client.calls = []
    client.complete("after restart", MetaHarnessUpdate)
    assert (client.journal / "call_000.json").read_text() == original
    assert (client.journal / "call_001.json").is_file()


@pytest.mark.parametrize("error_name", ["UnexpectedModelBehavior", "IncompleteToolCall"])
def test_malformed_output_is_a_retryable_uncommitted_constraint(tmp_path, monkeypatch, error_name):
    client, _seen = _client(tmp_path, monkeypatch)
    error = type(error_name, (RuntimeError,), {})

    def fail_output(*_args, **_kwargs):
        raise error("provider returned incomplete structured output")

    monkeypatch.setattr(sys.modules["pydantic_ai"].Agent, "run_sync", fail_output)
    with pytest.raises(DecisionConstraintError, match="incomplete structured Meta"):
        client.complete("evidence", MetaHarnessUpdate, operation="route", decision_id="generation_0_decision_0")
    record = json.loads((client.journal / "call_000.json").read_text())
    assert record["error_type"] == error_name
    assert record["output"] is None
