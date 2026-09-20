#!/usr/bin/env python3
"""Export an actual three-generation controller run with engineer-written mocks.

Both Harness interpreters, immutable G commits, Task H updates, manifest windows,
fixed probes and artifact isolation execute normally. Only the Task/Meta model
responses and the tiny environment/verifier are fixtures. No network, Codex
process, checkpoint loading, training, GPU or pytest is launched by this script.
This is engineering evidence, never evidence of autonomous improvement.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sia.task_meta.data import DOMAINS, ManifestStore, content_hash, sha256_file  # noqa: E402
from sia.task_meta.durable import value_hash  # noqa: E402
from sia.task_meta.environments import AdapterResult  # noqa: E402
from sia.task_meta.loop import run_task_meta  # noqa: E402
from sia.task_meta.meta import MetaAgent  # noqa: E402
from sia.task_meta.meta_harness import MetaHarnessBundle, MetaHarnessStore  # noqa: E402
from sia.task_meta.meta_harness.bundle import reject_links  # noqa: E402
from sia.task_meta.meta_harness.policies import validate_policy  # noqa: E402
from sia.task_meta.meta_harness.runtime import AnalysisOutput, execute  # noqa: E402
from sia.task_meta.pipeline_execution import MultiDomainExecutor  # noqa: E402
from sia.task_meta.seed import SeedHarnessUpdater  # noqa: E402
from sia.task_meta.storage import checkpoint_manifest, save_json  # noqa: E402
from sia.task_meta.types import MetaAgentState, TaskAgentState, TaskUpdateAction  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def toy_manifest(directory):
    """Create an explicitly synthetic source index consumed by real ManifestStore."""
    directory.mkdir()
    database = directory / "tasks.sqlite"
    with sqlite3.connect(database) as conn:
        conn.executescript("""
            CREATE TABLE sources (path TEXT PRIMARY KEY, hash TEXT, metadata TEXT);
            CREATE TABLE tasks (task_id TEXT PRIMARY KEY, domain TEXT, source TEXT, split TEXT,
              content_hash TEXT, group_key TEXT, path TEXT, offset INTEGER, length INTEGER,
              order_key TEXT, seq INTEGER);
            CREATE TABLE probe (task_id TEXT PRIMARY KEY);
            CREATE TABLE omissions (source TEXT, task_id TEXT, reason TEXT);
        """)
        for domain in DOMAINS:
            source = directory / f"test_override_{domain}.jsonl"
            with source.open("wb") as stream:
                for index in range(4):
                    task_id = f"test_override_{domain}_{index}"
                    prompt = f"Synthetic {domain} fixture {index}: look up the toy fact and submit it."
                    payload = {{"tool_use": "task", "code": "problem", "searchqa": "question"}[domain]: prompt,
                               "fixture_origin": "test_override"}
                    line = (json.dumps(payload) + "\n").encode()
                    offset = stream.tell()
                    stream.write(line)
                    conn.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                        task_id, domain, "test_override_toy", "evolve_train" if index < 3 else "search_dev",
                        content_hash(prompt), task_id, str(source), offset, len(line), str(index), index))
                    if index == 3:
                        conn.execute("INSERT INTO probe VALUES (?)", (task_id,))
            conn.execute("INSERT INTO sources VALUES (?,?,?)", (
                str(source), sha256_file(source), json.dumps({"evidence_origin": "test_override"})))
    store = ManifestStore(database)
    store.validate_sources()
    return store


class ToyEnvironment:
    """A local in-memory fixture; does not represent actual domain benchmarks."""
    tools: ClassVar[list] = [{"name": "lookup", "description": "Return the synthetic fixture fact.",
              "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                             "required": ["query"], "additionalProperties": False}}]

    def reset(self, task, rollout_id, seed):
        self.task = task
        return {**task.public_payload(), "evidence_origin": "test_override"}

    def step(self, name, arguments):
        if name != "lookup" or arguments != {"query": "toy fact"}:
            raise ValueError("Synthetic tool fixture expects its declared lookup arguments")
        return {"fact": "toy answer", "source_id": "test_override_toy_fact", "mock": True}

    def evaluate(self, final_answer):
        success = final_answer == "toy answer"
        return AdapterResult(float(success), {"f1": float(success)},
            {"status": "complete", "success": success, "verifier_id": "test_override_toy_verifier",
             "passed": int(success), "total": 1},
            details={"evidence_origin": "test_override", "real_benchmark": False})

    def close(self):
        pass


class ToyTaskModel:
    """Scripted responses, bound to arbitrary test bytes, never an actual model."""
    enable_thinking = False

    def __init__(self, state):
        self.state, self.actions, self.calls = state, 0, 0

    def __call__(self, messages, **kwargs):
        self.calls += 1
        first, last = messages[0]["content"], messages[-1]["content"]
        if last.startswith("TEST_OVERRIDE_TASK_ROLE_"):
            content = f"Test context produced by the added H role in generation {self.state.generation}."
        elif first.startswith("Create the shortest"):
            content = "Read the toy fact, then submit it."
        elif first.startswith("You are analyzing"):
            content = json.dumps({"step_summary": "Synthetic memory extraction.", "key_extracts": [
                f"test_override natural rollout note generated by T_{self.state.generation}; unverified toy knowledge."]})
        elif first.startswith("You are managing"):
            content = "[1]"
        elif first.startswith("Summarize progress"):
            content = "The toy fact is available."
        elif first.startswith("You need to produce"):
            content = '{"answer":"toy answer"}'
        else:
            self.actions += 1
            name, arguments = (("lookup", {"query": "toy fact"}) if self.actions == 1
                               else ("final_answer", {"answer": "toy answer"}))
            content = json.dumps({"tools": [{"name": name, "arguments": arguments}]})
        return {"message": {"role": "assistant", "content": content},
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "finish_reason": "stop",
                "request_id": f"test_override_task_{self.state.generation}_{kwargs['seed']}_{self.calls}",
                "binding": {"model_ref": self.state.model_ref, "checkpoint_path": self.state.checkpoint_path,
                            "checkpoint_manifest_hash": value_hash(self.state.checkpoint_manifest),
                            "decision_source": "test_override", "weights_loaded": False}}


def mock_h_edits(seed, meta_version):
    """The engineering fixture authors these patches, never a real Meta model."""
    role = {"name": "toy_context", "instruction": f"TEST_OVERRIDE_TASK_ROLE_H{meta_version + 1}: inspect {{{{task}}}}.",
            "result": "context"}
    edits = [{"target": "parts.input.task_template", "value": f"TEST_OVERRIDE_TASK_INPUT_H{meta_version + 1}\n{{{{task}}}}"},
             {"target": "parts.tools.roles", "value": [role]}]
    if meta_version == 0:
        graph = copy.deepcopy(seed["parts"]["control"]["graph"])
        graph["nodes"].append({"id": "toy_context", "kind": "role", "role": "toy_context", "next": graph["entry"]})
        graph["entry"] = "toy_context"
        edits.append({"target": "parts.control.graph", "value": graph})
    return edits


def mock_g_policy(bundle):
    policy = bundle.execution_spec()
    policy["evidence"]["group_by"] = ["domain"]
    policy["experience"]["fields"] = ["generation", "chosen_action", "versions", "observed_performance_delta"]
    policy["diagnosis"]["instruction"] = "TEST_OVERRIDE_G1_DIAGNOSIS: inspect the current H before authoring a compatible candidate."
    policy["self_update"]["instruction"] = "TEST_OVERRIDE_G1_SELF_UPDATE: check the attributable transition before any further candidate."
    policy["self_update"]["rules"] = [{"id": "has_actual_transition", "when": {
        "path": "latest_experience.experience_id", "op": "exists"},
        "instruction": "TEST_OVERRIDE_G1_RULE_MATCHED: retain uncertainty; a toy score is not causal evidence."}]
    workflow = policy["workflows"]["meta_self_update"]
    position = next(index for index, step in enumerate(workflow) if step["kind"] == "propose")
    workflow[position:position] = [
        {"id": "g1_outcome_check", "kind": "check", "checks": ["experience_outcome", "source_integrity"]},
        {"id": "g1_attribution_analysis", "kind": "analyze",
         "when": {"path": "last_check_passed", "op": "eq", "value": True},
         "instruction": "TEST_OVERRIDE_G1_ANALYSIS: compare observed changes before deciding NO_CHANGE."}]
    return validate_policy(policy)


class MockEvolutionClient:
    """Calls actual G execute; all model decisions are engineering test fixtures."""
    supports_evolution = True
    decision_source = "test_override"

    def __init__(self, store, directory):
        self.bundle_manager, self.directory = store, directory
        self.calls, self.stages = [], []

    def complete(self, prompt, schema, *, meta_state, operation, operation_input,
                 validate_candidate=None, experience_id=None, **kwargs):
        number = len(self.calls)
        bundle = MetaHarnessBundle(Path(meta_state.bundle_path), read(Path(meta_state.bundle_path) / "manifest.json")).verify()
        if bundle.hash != meta_state.bundle_hash:
            raise AssertionError("The fixture must use the actual bound G")
        call_dir = self.directory / f"call_{number:03d}_{operation}"
        original = copy.deepcopy(operation_input)
        save_json(call_dir / "input_envelope.json", original)
        record = {"call": number, "operation": operation, "bundle_hash": bundle.hash, "meta_version": meta_state.version,
                  "experience_id": experience_id, "input_hash": value_hash(original),
                  "evidence_origin": "test_override", "directory": str(call_dir.relative_to(self.directory.parent))}
        self.calls.append(record)

        def invoke(stage_id, stage_prompt, stage_schema, **permissions):
            if permissions != {"evidence_files": {}, "allowed_paths": []}:
                raise AssertionError("Offline fixture must receive no external permissions")
            stage = {**record, "stage_id": stage_id, "prompt": stage_prompt,
                     "request_id": f"test_override_meta_{number}_{stage_id}"}
            self.stages.append(stage)
            if stage_schema is AnalysisOutput:
                response = stage_schema(analysis="Engineer-written offline attribution analysis.", hypotheses=[], source_ids=[])
            elif operation in {"route", "harness_patch"}:
                seed = json.loads(operation_input["current_files"]["seed.json"])
                edits = mock_h_edits(seed, meta_state.version)
                if operation == "harness_patch":
                    response = stage_schema(edits=edits, summary="test_override atomic multi-part H candidate")
                else:
                    response = stage_schema(action="HARNESS", diagnosis="Offline graph and prompt wiring hypothesis.",
                        evidence=["Synthetic recorded Task call sequence"], rationale="test_override engineering fixture",
                        proposed_change="Add or revise an executable context role and its input.",
                        expected_effect="Unverified; no improvement claim.", target_components=["HARNESS"],
                        requested_changes=[{"id": f"h_change_{index}", "component": "HARNESS", "operation": "replace_config",
                            "target": edit["target"], "harness_part": edit["target"].split(".")[1],
                            "instruction": "Apply the engineering fixture's bounded candidate."} for index, edit in enumerate(edits)],
                        decision_source="test_override")
            else:
                changed = operation == "learn" and meta_state.version == 0
                response = stage_schema(harness=bundle.read_files()["instructions.md"],
                    rationale="Engineer-written G1 policy fixture." if changed else "Engineer-written NO_CHANGE fixture.",
                    changed_rules=["evidence", "experience", "diagnosis", "self_update", "workflow"] if changed else [],
                    summary="Offline structural validation only; no autonomous learning or training.",
                    bundle_files={"evolution.json": json.dumps(mock_g_policy(bundle), indent=2)} if changed else {},
                    status="UPDATED" if changed else "NO_CHANGE", request_id=stage["request_id"], experience_id=experience_id)
            save_json(call_dir / f"{stage_id}_mock_response.json", response)
            if operation == "learn" and meta_state.version == 0 and stage_schema is not AnalysisOutput:
                save_json(self.directory.parent / "g1_candidate_patch.json", response)
            return response

        invoke.decision_source = "test_override"
        response = execute(bundle, operation, operation_input, schema, invoke, call_dir,
                           validate_candidate=validate_candidate)
        if original != operation_input:
            raise AssertionError("The real G runtime changed the trusted input")
        return response


class UnusedUpdater:
    def apply(self, *_):
        raise AssertionError("This offline fixture selects HARNESS; MODEL/ARTIFACTS are not invoked")


def run_demo(output):
    output = Path(output).absolute()
    reject_links(output)
    output.mkdir(parents=True, exist_ok=False)
    run = output / "run"
    (run / "gen_0").mkdir(parents=True)
    shutil.copy2(ROOT / "seed_harness/v2/seed.json", run / "gen_0/seed.json")
    checkpoint = output / "test_override_checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"test_override arbitrary fixture bytes; NOT real weights")
    weights = checkpoint_manifest(checkpoint)
    task = TaskAgentState(0, "test_override_task_checkpoint", str(run / "gen_0/seed.json"),
                          checkpoint_path=str(checkpoint), checkpoint_manifest=weights)
    meta_store = MetaHarnessStore(run / "meta")
    g0 = meta_store.initialize(ROOT / "meta_harness/seed")
    mirror = run / "meta/harness_v0.md"
    mirror.write_text(g0.read_files()["instructions.md"], encoding="utf-8")
    meta = MetaAgentState("test_override_frozen_meta_model", str(mirror), bundle_hash=g0.hash, bundle_path=str(g0.path))
    client = MockEvolutionClient(meta_store, output / "meta_operations")
    agent = MetaAgent(client, {"sft_profile": "multidomain", "trainer_configured": False})
    store = toy_manifest(output / "toy_data")
    source_hashes = {path.name: sha256_file(path) for path in (output / "toy_data").iterdir()}
    executor = MultiDomainExecutor(store, lambda domain: ToyEnvironment(), ToyTaskModel,
        quotas=dict.fromkeys(DOMAINS, 1), rollouts_per_task=2, probe_rollouts=1)
    updaters = {action: SeedHarnessUpdater(client) if action == TaskUpdateAction.HARNESS else UnusedUpdater()
                for action in TaskUpdateAction}

    def accept(state, update):
        bundle = meta_store.commit_update(state.bundle_hash, instruction_text=update.harness,
            file_updates=update.bundle_files, request_id=update.request_id, experience_id=update.experience_id,
            phase="meta_self_update" if update.experience_id else "final_consolidation")
        state.bundle_hash, state.bundle_path = bundle.hash, str(bundle.path)
        return state

    try:
        final = run_task_meta(run, task, meta, executor, agent, updaters,
            max_generations=3, primary_metric_name="macro_success", meta_update_handler=accept)
    finally:
        store.conn.close()
        save_json(output / "meta_calls.json", client.calls)
        save_json(output / "meta_stages.json", client.stages)
    experiences = rows(run / "meta/experiences.jsonl")
    g1 = meta_store.active()
    generations = []
    for generation in range(3):
        directory = run / f"gen_{generation}"
        train, probe = rows(directory / "train_trajectories.jsonl"), rows(directory / "probe_trajectories.jsonl")
        incoming = read(directory / "evaluated_state.json")
        outgoing = read(directory / "task_state_after_rollout.json")
        assert len(train) == 6 and len(probe) == 3
        assert all(row["artifact_input_manifest"] == incoming["artifacts"]["manifest"] for row in train + probe)
        assert all(not row["notes"] for row in probe)
        assert all(row["source"] == "test_override_toy" for row in train + probe)
        assert all(call["binding"]["decision_source"] == "test_override"
                   for row in train + probe for call in row["model_calls"])
        role_calls = [call for row in train + probe for call in row["model_calls"] if call.get("role") == "toy_context"]
        assert len(role_calls) == (0 if generation == 0 else 9)
        if generation:
            marker = f"TEST_OVERRIDE_TASK_ROLE_H{generation}"
            assert all(marker in call["messages"][-1]["content"] for call in role_calls)
            assert incoming["artifacts"]["manifest"] == generations[-1]["output_artifacts"]
        assert outgoing["artifacts"]["manifest"]
        generations.append({"generation": generation, "harness_sha256": sha256_file(directory / "seed.json"),
            "probe_ids": [row["task_id"] for row in probe], "train_ids": sorted({row["task_id"] for row in train}),
            "probe_identity": read(directory / "results.json")["probe_identity"],
            "probe_denominator": len(probe), "train_denominator": len(train), "role_calls": len(role_calls),
            "input_artifacts": incoming["artifacts"]["manifest"], "output_artifacts": outgoing["artifacts"]["manifest"],
            "task_model_calls": sum(len(row["model_calls"]) for row in train + probe),
            "task_model_operations": [call["operation"] for call in train[0]["model_calls"]]})
    assert final["status"] == "completed" and final["generations_executed"] == 3 and final["experiences"] == 2
    assert final["meta_version_changes"] == 1 and final["meta_no_change_events"] == 2
    assert final["meta_update_events"] == 3 and final["meta_state"]["version"] == 1
    assert not (run / "gen_3").exists()
    assert [item["versions"]["meta_bundle_hash_at_decision"] for item in experiences] == [g0.hash, g1.hash]
    assert [call["operation"] for call in client.calls] == ["route", "harness_patch", "learn", "route", "harness_patch", "learn", "final_consolidation"]
    assert [call["meta_version"] for call in client.calls] == [0, 0, 0, 1, 1, 1, 1]
    for stage in client.stages:
        if stage["meta_version"] == 1:
            marker = "TEST_OVERRIDE_G1_SELF_UPDATE" if stage["operation"] in {"learn", "final_consolidation"} else "TEST_OVERRIDE_G1_DIAGNOSIS"
            assert marker in stage["prompt"]
    later_learn = next(call for call in client.calls if call["operation"] == "learn" and call["meta_version"] == 1)
    audit = read(output / later_learn["directory"] / "policy_runtime.json")
    completed = [event["step_id"] for event in audit["events"] if event.get("status") == "completed" and "step_id" in event]
    assert completed.index("g1_outcome_check") < completed.index("g1_attribution_analysis") < completed.index("propose")
    assert all(value["probe_ids"] == generations[0]["probe_ids"] for value in generations)
    assert len({value["probe_identity"] for value in generations}) == 1
    assert len({value["harness_sha256"] for value in generations}) == 3
    assert checkpoint_manifest(checkpoint) == weights
    assert source_hashes == {path.name: sha256_file(path) for path in (output / "toy_data").iterdir()}
    updates = [read(run / f"gen_{index}/task_update.json") for index in (0, 1)]
    assert updates[0]["details"]["changed_parts"] == ["control", "input", "tools"]
    evidence = {"evidence_type": "offline_engineering_full_loop", "decision_source": "test_override",
        "autonomous_meta_evolution_verified": False, "real_api_calls": 0, "training_runs": 0,
        "real_weights_loaded": False, "gpu_operations": 0, "score_improvement_claim": False,
        "limitations": ["Task and Meta responses are authored by the engineer in this script.",
                        "Toy tools, toy scoring and synthetic data do not validate real benchmark adapters or model quality.",
                        "Does not establish managed Codex runtime readiness, real API operation or pilot verification."],
        "components_executed": ["run_task_meta", "MetaAgent", "MetaHarnessStore", "Meta v2 runtime.execute",
                                "SeedHarnessUpdater", "Task run_seed v2", "ManifestStore", "MultiDomainExecutor"],
        "status_scope": {"CODE_COMPLETE": "offline wiring demonstrated", "RUNTIME_READY": False,
                         "PILOT_VERIFIED": False, "REPORT_READY": False},
        "generations": generations, "experiences": len(experiences), "meta_version_changes": 1,
        "regular_no_change_events": 1, "final_consolidation_no_change_events": 1,
        "total_no_change_events": final["meta_no_change_events"], "g0_hash": g0.hash, "g1_hash": g1.hash,
        "experience_decision_g_hashes": [item["versions"]["meta_bundle_hash_at_decision"] for item in experiences],
        "h_changed_parts": [item["details"]["changed_parts"] for item in updates],
        "g1_later_self_update_completed_steps": completed,
        "fixed_probe_identity": generations[0]["probe_identity"], "raw_sources_unchanged": True,
        "same_generation_assets_isolated": True, "next_generation_h_reload_verified": True,
        "checkpoint_unchanged": True, "no_unevaluated_successor": True,
        "meta_operations": client.calls, "meta_model_stage_count": len(client.stages),
        "g1_candidate_patch": "g1_candidate_patch.json",
        "final_state": "run/final_state.json", "experience_ledger": "run/meta/experiences.jsonl",
        "meta_commit_events": "run/meta/commit_events", "source_hashes": source_hashes}
    save_json(output / "behavior_evidence.json", evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths are refused")
    args = parser.parse_args()
    evidence = run_demo(args.output)
    print(json.dumps({"output": str(args.output.absolute()), "evidence_origin": evidence["decision_source"],
                      "generations": len(evidence["generations"]), "experiences": evidence["experiences"],
                      "real_api_calls": 0, "training_runs": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
