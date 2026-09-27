"""Regressions for pre-model schema failures and TACO false-negative labels."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from sia.task_meta.harnessforge_runtime import (
    _normalize_schema, _ToolRecorder, _environment_tools, _activated_candidate,
)
from sia.task_meta.harnessforge_manifest import load_manifest
from sia.task_meta.environments import _structured_match, _taco_function_match, TACOAdapter
from sia.task_meta.data import TaskRecord


@pytest.mark.parametrize("declared, normalized", [
    (["string", "null"], "string"), (["null", "integer"], "integer"),
    (["number", "string", "null"], "any"), (["null"], "null"),
    (["integer", "number"], "any"),
])
def test_tool_type_projection_keeps_original_validation(declared, normalized):
    schema = {"name": "inspect", "parameters": {"type": "object", "properties": {
        "value": {"type": declared}}, "required": ["value"]}}
    saved = copy.deepcopy(schema)
    _, _, inputs, validation = _normalize_schema(schema)
    assert inputs["value"]["type"] == normalized
    assert validation["properties"]["value"]["type"] == declared
    assert validation["required"] == ["value"]
    assert inputs["value"].get("nullable", False) == ("null" in declared)
    assert schema == saved


def test_real_harnessforge_tool_constructs_and_calls_nullable_union():
    class Environment:
        tools = [{"type": "function", "function": {"name": "inspect", "parameters": {
            "type": "object", "properties": {
                "value": {"type": ["integer", "null"]},
                "extra": {"type": ["string", "number"]},
            }, "required": ["value"], "additionalProperties": False}}}]

        def step(self, name, arguments):
            return {"name": name, **arguments}

    env = Environment()
    recorder = _ToolRecorder(env, max_tool_calls=20)
    manifest = load_manifest(ROOT / "seed_harness/harnessforge_base_manifest.json")
    with _activated_candidate(manifest):
        tool = _environment_tools(env, recorder)[0]
        assert tool(value=None)["value"] is None
        assert tool(value=2, extra="text")["value"] == 2
        assert tool(value=2, extra=3.5)["extra"] == 3.5
        for kwargs in ({}, {"value": "2"}, {"value": True},
                       {"value": 2, "extra": None}, {"value": 2, "unknown": 0}):
            with pytest.raises(ValueError, match="Invalid arguments"):
                tool(**kwargs)
    assert len([call for call in recorder.calls if call["status"] == "completed"]) == 3


@pytest.mark.parametrize("actual, expected, matches", [
    (True, [True], True), (False, [False], True), (True, [False], False),
    ([True], [True], True), ([1, 2], [[1, 2]], True),
    ([1, 2], [1, 2], True), ([1, 3], [[1, 2]], False),
    ([], [], True), (False, [], False), (True, [1], False),
    (3, [4], False), (True, [[True]], False),
])
def test_taco_call_based_wrapper_without_recursive_flatten(actual, expected, matches):
    assert _taco_function_match(actual, expected) is matches


def test_correct_function_is_not_excluded_from_success_sft():
    class Sandbox:
        def validate(self):
            pass

        def run(self, code, stdin, **kwargs):
            a, b = json.loads(stdin)["args"]
            value = sum(x * y for x, y in zip(a, b)) == 0
            return SimpleNamespace(stdout=json.dumps(value), returncode=0,
                                   timed_out=False, output_truncated=False, wall_seconds=0.01)

    pairs = [[[1, 0], [0, 1]], [[1, 1], [1, 1]]] * 5
    task = TaskRecord("orthogonal", "code", "deepcoder_taco", "evolve_train", "dot product", {
        "tests": {"fn_name": "is_orthogonal", "inputs": pairs,
                  "outputs": [[True], [False]] * 5}})
    adapter = TACOAdapter(Sandbox())
    adapter.reset(task, "0", 42)
    result = adapter.evaluate("def is_orthogonal(a, b):\n    return sum(x*y for x,y in zip(a,b)) == 0")
    assert result.verification["success"] is True
    assert result.verification["full_verifier"] is True
    assert result.verification["tests_completed"] == 10
    assert all(case["passed"] for case in result.details["test_outcomes"])
    assert _structured_match(True, [True]) is False  # Former, incorrect call-site comparison.


def test_real_isolated_harness_validation_ignores_host_directory_name(tmp_path):
    sys.path.insert(0, str(ROOT))
    from controller_tools import ControllerTools
    from task_adapter import TaskAdapter
    from sia.task_meta.durable import load_task
    parent = tmp_path / ".attempt.with.dots and spaces"
    candidate = parent / "complete_candidate"
    state = tmp_path / "parent.json"
    state.write_text(json.dumps({"generation": 0, "model_ref": "Qwen3-4B",
                                "harness_path": str(ROOT / "seed_harness/harnessforge_base_manifest.json")}))
    adapter = TaskAdapter(ROOT, tmp_path / "receipts")
    tools = ControllerTools(ROOT.parent, adapter=adapter)
    materialized = tools.execute({"operation": "materialize_harness", "state_path": str(state),
                                  "destination": str(candidate)})
    assert materialized["status"] == "materialized"
    assert all(Path(p).is_file() for p in materialized["production_templates"])
    Path(materialized["localization_report_path"]).write_text("Inspect successful and failed traces alongside parent source.")
    Path(materialized["improvement_direction_path"]).write_text("Keep baseline modules for this executable-contract regression.")
    reused = tools.execute({"operation": "materialize_harness", "state_path": str(state),
                            "destination": str(candidate)})
    assert reused["reused_directory"] is True
    receipt = tools.execute({"operation": "harnessforge", "state_path": str(state),
                             "candidate_dir": str(candidate), "output_dir": str(parent / "built")})
    assert receipt["status"] == "completed", receipt
    child = load_task(json.loads(Path(receipt["candidate_state_path"]).read_text()))
    assert child.generation == 1
    assert load_manifest(child.harness_path).files == load_manifest(ROOT / "seed_harness/harnessforge_base_manifest.json").files
    report = json.loads(Path(receipt["validation_path"]).read_text())
    assert report["verdict"] == "passed", report
    assert all(item["status"] == "passed" for item in report["checks"])
