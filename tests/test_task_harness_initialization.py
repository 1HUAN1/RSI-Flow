"""Five-part initialization equivalence with explicit mock Task responses.

The preserved v1 implementation is the oracle, not a second configuration of the
new interpreter. No API/GPU/model loading, training, or baseline execution occurs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from scripts.export_task_harness_initialization import export, verify_export
from sia.task_meta.seed import load_seed, run_seed
from sia.task_meta.task_harness import legacy_view, load_harness, migrate_seed, runtime_dependencies

ROOT = Path(__file__).resolve().parents[1]
OLD_SEED = ROOT / "seed_harness/seed.json"
NEW_SEED = ROOT / "seed_harness/v2/seed.json"
REGISTERED_OLD_SEED_HASH = "4b000e83f0be22a4c22cca3e2a45f0a0b2d913fd4c2d621c03a0524201bb2941"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def action(name, arguments):
    return json.dumps({"tools": [{"name": name, "arguments": arguments}]})


LOOKUP = action("lookup", {"query": "France"})
FINAL = action("final_answer", {"answer": " Paris "})


class ToyEnvironment:
    tools: ClassVar[list] = [{"name": "lookup", "description": "Find a toy fact.",
        "parameters": {"properties": {"query": {"type": "string"}}, "required": ["query"]}}]

    def __init__(self, terminal=False, infrastructure=False):
        self.calls = []
        self.terminal, self.infrastructure = terminal, infrastructure

    def step(self, name, arguments):
        self.calls.append({"name": name, "arguments": copy.deepcopy(arguments)})
        if self.infrastructure:
            raise RuntimeError("Explicit mock infrastructure failure")
        if "query" not in arguments:
            raise ValueError("Missing query in the unchanged toy environment")
        if self.terminal:
            return {"terminal": True, "final_answer": " Paris "}
        return " Paris "


class ToyModel:
    def __init__(self, actions=None, extracts=None, fail_call=None):
        self.actions = iter([LOOKUP, FINAL] if actions is None else actions)
        self.extracts = iter(extracts or [])
        self.requests = []
        self.fail_call = fail_call

    def __call__(self, messages, **kwargs):
        self.requests.append({"messages": copy.deepcopy(messages), **copy.deepcopy(kwargs)})
        if len(self.requests) == self.fail_call:
            raise RuntimeError("Explicit mock current-Qwen request failure")
        first = messages[0]["content"]
        if first.startswith("Create the shortest"):
            content = "Look up the requested fact and finalize."
        elif first.startswith("You are analyzing"):
            content = json.dumps(next(self.extracts, {"step_summary": "", "key_extracts": []}))
        elif first.startswith("You are managing"):
            content = "[2,1]"
        elif first.startswith("Summarize progress"):
            content = "The toy observations are available. Continue or finalize."
        elif first.startswith("You need to produce"):
            content = '{"answer":" Paris "}'
        else:
            content = next(self.actions)
        return {"message": {"role": "assistant", "content": content}, "usage": {"output_tokens": 5},
                "binding": {"model_ref": "test_override_current_Qwen", "checkpoint_hash": "fixture_only"}}


def behavior(result):
    """Compare all pre-existing behavioral fields, excluding duration/H identity."""
    model_fields = ("call_id", "operation", "messages", "seed", "max_tokens", "temperature", "status",
                    "assistant", "usage", "binding", "error_type")
    calls = [{key: row[key] for key in model_fields if key in row} for row in result["model_calls"]]
    fields = ("messages", "final_answer", "notes", "error_type", "error", "memory_errors",
              "steps", "artifact_chars_used", "artifact_chars_omitted", "final_context", "infrastructure_failure")
    conversations = [{key: row[key] for key in ("call_id", "operation", "messages", "sft_supervision")}
                     for row in result["sft_conversations"]]
    tools = [{key: row[key] for key in ("name", "arguments", "step", "observation", "status", "error_type") if key in row}
             for row in result["tool_calls"]]
    return {"model_calls": calls, "tool_calls": tools, "sft_conversations": conversations,
            **{key: result[key] for key in fields}}


CASES = [
    ("lookup_then_submit", {}, [LOOKUP, FINAL], {}, {}),
    ("immediate_submit", {}, [FINAL], {}, {}),
    ("terminal_environment", {}, [LOOKUP], {"terminal": True}, {}),
    ("summary_boundary", {}, [LOOKUP] * 9 + [FINAL], {}, {}),
    ("step_exhaustion", {"planning.max_steps": 1}, [LOOKUP], {}, {}),
    ("parse_repair", {}, ["not JSON", FINAL], {}, {}),
    ("parse_repair_exhaustion", {}, ["not JSON", "still not JSON"], {}, {}),
    ("tool_argument_error_then_fix", {}, [action("lookup", {}), LOOKUP, FINAL], {}, {}),
    ("memory_prune", {}, [LOOKUP, FINAL], {}, {"extracts": [{"step_summary": "Toy extracted facts.",
        "key_extracts": [f"Toy fact number {index} remains attributable." for index in range(11)]}]}),
    ("model_budget", {"budget.max_model_calls": 1}, [LOOKUP], {}, {}),
    ("tool_budget", {"budget.max_tool_calls": 1}, [LOOKUP, LOOKUP], {}, {}),
    ("model_infrastructure_failure", {}, [LOOKUP], {}, {"fail_call": 1}),
    ("tool_infrastructure_failure", {}, [LOOKUP], {"infrastructure": True}, {}),
]


@pytest.mark.parametrize("name,changes,actions,environment_options,model_options", CASES, ids=[case[0] for case in CASES])
def test_migrated_seed_preserves_actual_requests_tools_context_and_stop(name, changes, actions, environment_options, model_options):
    original = load_seed(OLD_SEED)
    for target, value in changes.items():
        section, key = target.split(".")
        original[section][key] = value
    migrated = migrate_seed(original)
    old_model, new_model = ToyModel(actions, **model_options), ToyModel(actions, **model_options)
    old_env, new_env = ToyEnvironment(**environment_options), ToyEnvironment(**environment_options)
    raw_task = "What is France's capital? Return only the requested answer."
    old = run_seed(original, old_model, old_env, raw_task, seed=123)
    new = run_seed(migrated, new_model, new_env, raw_task, seed=123)
    assert old_model.requests == new_model.requests, name
    assert old_env.calls == new_env.calls, name
    assert behavior(old) == behavior(new), name
    assert all(call.get("binding", {}).get("model_ref") == "test_override_current_Qwen"
               for call in new["model_calls"] if call["status"] == "completed")
    assert all(dialogue["messages"][-1] == call["assistant"] for dialogue, call in
               zip(new["sft_conversations"], [call for call in new["model_calls"] if call["status"] == "completed"]))


def test_shipped_v2_is_legacy_equivalent_and_has_no_extra_default_model_calls():
    original, current = load_seed(OLD_SEED), load_harness(NEW_SEED)
    assert sha(OLD_SEED) == REGISTERED_OLD_SEED_HASH
    assert legacy_view(current) == original
    old_model, new_model = ToyModel(), ToyModel()
    old = run_seed(original, old_model, ToyEnvironment(), "Question")
    new = run_seed(current, new_model, ToyEnvironment(), "Question")
    assert old_model.requests == new_model.requests
    assert behavior(old) == behavior(new)
    assert [call["operation"] for call in new["model_calls"]] == ["planning", "memory_extract", "action", "memory_extract", "action"]


def test_artifact_input_and_working_memory_are_equivalent_and_isolated():
    original = load_seed(OLD_SEED)
    original["context"]["artifact_char_limit"] = 12
    current = migrate_seed(original)
    text = "Allowed reusable note. Extra text exceeds the fixture budget."
    extracts = [{"step_summary": "Toy memory", "key_extracts": ["Task-local memory must not leak into the next rollout."]}]
    old = run_seed(original, ToyModel(extracts=extracts), ToyEnvironment(), "Question", text)
    new = run_seed(current, ToyModel(extracts=extracts), ToyEnvironment(), "Question", text)
    assert behavior(old) == behavior(new)
    assert new["artifact_chars_used"] == 12 and new["artifact_chars_omitted"] == len(text) - 12
    clean = run_seed(current, ToyModel(), ToyEnvironment(), "Another question", text)
    assert clean["notes"] == []
    assert "Task-local memory" not in json.dumps(clean["model_calls"])
    assert current == migrate_seed(original)


def fixture_config(tmp_path):
    model_dir = tmp_path / "Qwen3-4B"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"test_override fixture bytes; not actual model weights")
    (model_dir / "config.json").write_text('{"model_type":"qwen3","test_override":true}', encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text('{"chat_template":"test_override template"}', encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    (data / "tasks.sqlite").write_bytes(b"test_override manifest fixture, not benchmark data")
    (data / "search.sqlite").write_bytes(b"test_override corpus fixture, no hidden answers")
    (data / "search.sqlite.manifest.json").write_text(json.dumps({"sha256": sha(data / "search.sqlite"),
        "source": "test_override"}), encoding="utf-8")
    model = {"path": str(model_dir), "model_type": "qwen3", "status": "test_override_fingerprint_fixture_not_loaded",
        "weights": [{"path": "model.safetensors", "bytes": (model_dir / "model.safetensors").stat().st_size,
                     "sha256": sha(model_dir / "model.safetensors")}],
        "tokenizer_and_config": {name: sha(model_dir / name) for name in ("config.json", "tokenizer_config.json")},
        "chat_template_sha256": hashlib.sha256(b"test_override template").hexdigest()}
    (data / "model_identity.json").write_text(json.dumps(model), encoding="utf-8")
    return SimpleNamespace(seed_harness=str(NEW_SEED), data_dir=str(data), task_checkpoint=str(model_dir),
        seed=42, probe_per_domain={"code": 1, "tool_use": 1, "searchqa": 1},
        window_quotas={"code": 1, "tool_use": 1, "searchqa": 1}, rollouts_per_task=1,
        probe_rollouts=1, task_enable_thinking=False)


def test_export_is_complete_immutable_and_external_bytes_are_checked(tmp_path):
    config = fixture_config(tmp_path)
    destination = tmp_path / "export"
    original_hash = sha(OLD_SEED)
    record = export(config, destination)
    assert verify_export(destination, verify_external=True) == record
    assert record["state"] == "CODE_COMPLETE" and record["gpu_calls"] == record["real_api_calls"] == 0
    assert record["artifact_initial_manifest"] == [] and list((destination / "artifacts").iterdir()) == []
    assert record["runtime_dependencies"]
    assert set(record["runtime_dependencies"]) == set(runtime_dependencies())
    assert all((destination / "runtime" / key).is_file() for key in runtime_dependencies())
    assert sha(OLD_SEED) == original_hash
    with pytest.raises(FileExistsError):
        export(config, destination)
    (Path(config.task_checkpoint) / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checkpoint"):
        verify_export(destination, verify_external=True)


@pytest.mark.parametrize("target", ["seed.json", "runtime/sia/task_meta/seed.py", "extra.py"])
def test_export_verifier_rejects_changed_or_undeclared_executable_files(tmp_path, target):
    destination = tmp_path / "export"
    export(fixture_config(tmp_path), destination)
    path = destination / target
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("tampered source", encoding="utf-8")
    with pytest.raises(ValueError, match="Initialization"):
        verify_export(destination)


def test_exported_runtime_imports_and_executes_without_live_project_sources(tmp_path):
    destination = tmp_path / "export"
    export(fixture_config(tmp_path), destination)
    script = """
import json, sys
from pathlib import Path
snapshot = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(snapshot / 'runtime'))
import sia.task_meta.seed as seed_module
assert Path(seed_module.__file__).resolve().is_relative_to(snapshot / 'runtime')
from sia.task_meta.task_harness import load_harness
calls = []
def model(messages, **kwargs):
    calls.append(messages)
    first = messages[0]['content']
    if first.startswith('Create the shortest'):
        content = 'Finalize the toy task.'
    elif first.startswith('You are analyzing'):
        content = '{"step_summary":"","key_extracts":[]}'
    else:
        content = '{"tools":[{"name":"final_answer","arguments":{"answer":"fixture done"}}]}'
    return {'message': {'role': 'assistant', 'content': content}, 'binding': {'source': 'test_override'}}
class Environment:
    tools = []
    def step(self, *args):
        raise AssertionError('No environment call expected')
result = seed_module.run_seed(load_harness(snapshot / 'seed.json'), model, Environment(), 'Toy isolated import question')
assert result['final_answer'] == 'fixture done', result
assert result['error_type'] is None, result
print(json.dumps({'source':'test_override','calls':len(calls),'final_answer':result['final_answer']}))
"""
    child_env = dict(os.environ)
    # The child uses exported source plus installed packages, never source-tree
    # entries inherited through PYTHONPATH or its working directory.
    package_paths = [path for path in sys.path if path and (Path(path) / "pydantic").is_dir()]
    child_env["PYTHONPATH"] = os.pathsep.join(package_paths)
    completed = subprocess.run([sys.executable, "-B", "-c", script, str(destination)],
        cwd=tmp_path, env=child_env, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["source"] == "test_override"
    verify_export(destination)


def test_independent_copies_share_initial_identity_not_mutable_artifacts(tmp_path):
    first = tmp_path / "first"
    export(fixture_config(tmp_path), first)
    second = tmp_path / "second"
    shutil.copytree(first, second)
    assert verify_export(first)["initialization_hash"] == verify_export(second)["initialization_hash"]
    (second / "artifacts/toy.txt").write_text("new local artifact", encoding="utf-8")
    assert list((first / "artifacts").iterdir()) == []
    with pytest.raises(ValueError):
        verify_export(second)
    verify_export(first)
