"""Offline controller recovery, with explicit mock models and trusted fixtures.

The full-loop case runs real adapters, SQLite retrieval, candidate Python and
the EnvScaler worker on small fixture tasks. The local fixture subprocess is
not a security sandbox; production isolation has separate Linux tests.
"""

import copy
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from test_task_meta_loop import components, decision

from sia.task_meta.data import DOMAINS, ManifestStore, build_manifest
from sia.task_meta.durable import DurableClient, DurableExecutor, DurableUpdater, StageJournal
from sia.task_meta.environments import _ENV_WORKER, EnvScalerAdapter, SearchQAAdapter, TACOAdapter
from sia.task_meta.loop import BudgetManager, _accept_meta_update, run_task_meta
from sia.task_meta.meta import MetaAgent
from sia.task_meta.meta_harness.bundle import MetaHarnessStore
from sia.task_meta.pipeline_execution import MultiDomainExecutor
from sia.task_meta.retrieval import FrozenSearchIndex, build_search_index
from sia.task_meta.sandbox import SandboxResult
from sia.task_meta.seed import SeedHarnessUpdater, load_seed
from sia.task_meta.sft import select_positive_rows
from sia.task_meta.storage import save_json
from sia.task_meta.types import MetaAgentState, MetaHarnessUpdate, TaskAgentState, TaskUpdateAction

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = "print(sum(map(int, input().split())))"
ENV_CODE = """class Counter:
    def __init__(self, config): self.value = config.get('value', 0)
    def increment(self, amount):
        self.value += amount
        return self.value
"""
CHECK = "def check_func(final_state): return final_state['value'] == initial_state['value'] + 1"


def test_receipt_rejects_modified_generated_assets_and_evidence(tmp_path):
    _, task, _, executor, _, _ = components(tmp_path)
    wrapped = DurableExecutor(executor, StageJournal(tmp_path))
    wrapped.execute(task, tmp_path / "gen_0")
    (tmp_path / "gen_0/artifacts_generated/strategy.md").write_text("changed")
    with pytest.raises(ValueError, match="output artifacts"):
        wrapped.execute(task, tmp_path / "gen_0")


def test_receipt_rejects_altered_result(tmp_path):
    _, task, _, executor, _, _ = components(tmp_path)
    wrapped = DurableExecutor(executor, StageJournal(tmp_path))
    wrapped.execute(task, tmp_path / "gen_0")
    path = tmp_path / "gen_0/execution_receipt.json"
    saved = json.loads(path.read_text())
    saved["result"]["performance"]["success_rate"] = 1.0
    save_json(path, saved)
    with pytest.raises(ValueError, match="integrity"):
        wrapped.execute(task, tmp_path / "gen_0")


def test_budget_keeps_absolute_deadline_and_rejects_extension(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("sia.task_meta.loop.time.time", lambda: now[0])
    first = BudgetManager(3, 100)
    first.persist(tmp_path / "budget.json", resume=False)
    now[0] = 1110
    restored = BudgetManager(3, 100)
    restored.persist(tmp_path / "budget.json", resume=True)
    assert restored.elapsed() == 110 and restored.stop_after(1)
    with pytest.raises(ValueError, match="cannot change"):
        BudgetManager(3, 1000).persist(tmp_path / "budget.json", resume=True)


def test_pending_durable_update_never_consolidates_uncertain_state(tmp_path):
    events, task, meta, executor, agent, updaters = components(tmp_path)
    class Disconnected:
        def apply(self, *args):
            events.append(("external_started",))
            raise RuntimeError("disconnect after side effect")
    journal = StageJournal(tmp_path)
    updaters[TaskUpdateAction.HARNESS] = Disconnected()
    wrapped = {action: DurableUpdater(updater, journal) for action, updater in updaters.items()}
    executor = DurableExecutor(executor, journal)
    with pytest.raises(RuntimeError, match="disconnect"):
        run_task_meta(tmp_path, task, meta, executor, agent, wrapped, max_generations=2)
    final = run_task_meta(tmp_path, task, meta, executor, agent, wrapped, max_generations=2, resume=True)
    assert final["status"] == "pending_model_update"
    assert final["final_consolidation_status"] == "not_attempted"
    assert not any(event[0] == "final" for event in events)
    assert events.count(("external_started",)) == 1
    assert final["experiences"] == 0 and final["generations_executed"] == 1


def test_expired_resume_completes_committed_successor_and_feedback(tmp_path, monkeypatch):
    events, task, meta, executor, agent, updaters = components(tmp_path)
    now = [1000.0]
    monkeypatch.setattr("sia.task_meta.loop.time.time", lambda: now[0])
    original_execute = executor.execute
    failed = [False]
    def disconnect_before_successor(state, directory):
        if state.generation == 1 and not failed[0]:
            failed[0] = True
            raise RuntimeError("disconnect before committed successor evaluation")
        return original_execute(state, directory)
    executor.execute = disconnect_before_successor
    journal = StageJournal(tmp_path)
    executor = DurableExecutor(executor, journal)
    wrapped = {action: DurableUpdater(updater, journal) for action, updater in updaters.items()}
    with pytest.raises(RuntimeError, match="committed successor"):
        run_task_meta(tmp_path, task, meta, executor, agent, wrapped, max_generations=3, max_wall_time=100)
    now[0] = 1200.0
    final = run_task_meta(tmp_path, task, meta, executor, agent, wrapped,
                          max_generations=3, max_wall_time=100, resume=True)
    assert final["stop_reason"] == "max_wall_time"
    assert final["wall_time_seconds"] == 200
    assert final["generations_executed"] == 2 and final["experiences"] == 1
    assert events.count(("update", 0, "HARNESS")) == 1
    assert events.count(("execute", 0)) == 1 and events.count(("execute", 1)) == 1
    assert ("learn", 0, 0) in events and ("final", 1, 1) in events
    assert not (tmp_path / "gen_2").exists()


def test_meta_acceptance_binds_bundle_edits_and_harness_bytes(tmp_path):
    _, _, meta, _, _, _ = components(tmp_path)
    bundles = MetaHarnessStore(tmp_path / "bundles")
    bundle = bundles.initialize(ROOT / "meta_harness/seed")
    meta.bundle_hash, meta.bundle_path = bundle.hash, str(bundle.path)
    def accept(state, update):
        changed = bundles.commit_update(state.bundle_hash, instruction_text=update.harness, file_updates=update.bundle_files)
        state.bundle_hash, state.bundle_path = changed.hash, str(changed.path)
        return state
    path = tmp_path / "meta/acceptance.json"
    update = MetaHarnessUpdate(harness="version 1", rationale="fixture", changed_rules=[],
                               bundle_files={"context.json": '{"selection":"tail","max_evidence_chars":4000}'})
    accepted = _accept_meta_update(tmp_path, meta, update, [], path, "learn", handler=accept)
    assert _accept_meta_update(tmp_path, meta, update, [], path, "learn", resume=True) == accepted
    altered = update.model_copy(deep=True)
    altered.bundle_files["context.json"] = '{"selection":"head","max_evidence_chars":4000}'
    with pytest.raises(ValueError, match="acceptance receipt"):
        _accept_meta_update(tmp_path, meta, altered, [], path, "learn", resume=True)
    Path(accepted.harness_path).write_text("corrupted")
    with pytest.raises(ValueError, match="integrity"):
        _accept_meta_update(tmp_path, meta, update, [], path, "learn", resume=True)


class FixedFixtureWorker:
    """Only the fixed reviewed fixture is allowed into this test subprocess."""
    def __init__(self, code, extra_files):
        assert code == _ENV_WORKER
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        for name, value in extra_files.items():
            assert name == "env_util.py"
            (root / name).write_text(value, encoding="utf-8")
        self.process = subprocess.Popen([sys.executable, "-u", "-c", code], cwd=root,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")

    def request(self, value):
        if value["operation"] == "reset":
            assert value["env_code"] == ENV_CODE
        if value["operation"] == "evaluate":
            assert value["checklist"] == [{"check_func": CHECK}]
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise AssertionError(self.process.stderr.read())
        return json.loads(line)

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=5)
        self.process.stdout.close()
        self.process.stderr.close()
        self.directory.cleanup()


class FixedFixtureSubprocess:
    def validate(self):
        return {"decision_source": "test_override", "isolation_test": False}

    def run(self, code, stdin="", **kwargs):
        assert code == PROGRAM
        result = subprocess.run([sys.executable, "-c", code], input=stdin, capture_output=True,
                                text=True, timeout=5, check=False)
        return SandboxResult(result.stdout, result.stderr, result.returncode, False, 0,
                             backend="test_override_trusted_fixture")

    def session(self, code, *, extra_files, seed):
        return FixedFixtureWorker(code, extra_files)


class MockBundleClient:
    decision_source = "mock"

    def __init__(self, bundles):
        self.bundles, self.calls = bundles, []
        self.fail_before = None

    def complete(self, prompt, schema, *, meta_state, operation, **kwargs):
        bundle = self.bundles.active()
        assert bundle.hash == meta_state.bundle_hash
        rendered, evidence = bundle.render(prompt, json.dumps(schema.model_json_schema()))
        if self.fail_before == operation:
            self.fail_before = None
            raise RuntimeError("injected failure before " + operation)
        self.calls.append({"operation": operation, "version": meta_state.version,
                           "rendered": rendered, "evidence": evidence, "prompt": prompt})
        if operation == "route":
            selected = decision("HARNESS").model_dump(mode="json")
            selected["decision_source"] = "mock"
            selected["requested_changes"][0].update(operation="replace_config", target="prompts.action_step")
            return schema.model_validate(selected)
        if operation == "harness_patch":
            payload = json.loads(prompt[prompt.index('{"seed":'):])
            previous = payload["seed"]["prompts"]["action_step"]
            return schema.model_validate({"edits": [{"target": "prompts.action_step",
                "value": previous + f"\nFIXTURE_USE_EVIDENCE_{meta_state.version}"}], "summary": "mock seed patch"})
        return schema.model_validate({"harness": f"mock meta rules {meta_state.version + 1}",
            "experience_id": kwargs.get("experience_id"),
            "rationale": "fixture feedback", "changed_rules": ["fixture context"],
            "bundle_files": {"workflow.json": json.dumps({"section_order": ["instructions", "evidence", "response_contract"],
                        "checklist": [f"MOCK_NEXT_LOAD_{meta_state.version + 1}"]})}})


def _fixture_catalog(path):
    entries = []
    def add(name, domain, role, rows):
        target = path / f"source_{len(entries)}.jsonl"
        target.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        entries.append({"dataset_name": name, "domain": domain, "role": role, "local_path": str(target)})
    for role, env in [("train", "counter_train"), ("validation", "counter_dev")]:
        add("EnvScaler scenarios", "tool_use", role, [{"task_id": str(i), "env_id": env,
            "task": f"Increment counter once, fixture {role} {i}", "env_class_name": "Counter",
            "init_config": {"value": 0}, "checklist_with_func": [{"check_func": CHECK}]} for i in range(3)])
        add("DeepCoder TACO", "code", role, [{"problem": f"Add integers on stdin, fixture {role} {i}",
            "tests": {"inputs": ["1 2", "8 9"], "outputs": ["3", "17"]},
            "solutions": ["HIDDEN_REFERENCE_SOLUTION"]} for i in range(3)])
    add("NQ Open", "searchqa", "train", [{"question": f"Capital of France? fixture {i}", "answer": ["Paris"],
        "context": [["France", ["Paris is the capital of France."]]]} for i in range(40)])
    target = path / "sources.json"
    save_json(target, {"datasets": entries})
    return target


def _full_fixture(tmp_path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    store = ManifestStore(build_manifest(_fixture_catalog(fixtures), fixtures / "manifest",
                                         search_dev_fraction=.5, probe_per_domain=dict.fromkeys(DOMAINS, 1)))
    index = FrozenSearchIndex(build_search_index(list(store.iter_split("evolve_train", "searchqa")),
                              fixtures / "corpus.sqlite", data_manifest_hash="explicit_test_fixture"))
    envs = fixtures / "environments.jsonl"
    tool = {"type": "function", "function": {"name": "increment", "description": "Increment counter",
        "parameters": {"type": "object", "properties": {"amount": {"type": "integer"}}, "required": ["amount"]}}}
    envs.write_text("".join(json.dumps({"env_id": name, "env_class_code": ENV_CODE, "tools": [tool]}) + "\n"
                             for name in ["counter_train", "counter_dev"]), encoding="utf-8")
    utils = ROOT / "tests/fixtures/envscaler_env_util.txt"
    def adapter(domain):
        if domain == "searchqa":
            return SearchQAAdapter(index)
        if domain == "code":
            return TACOAdapter(FixedFixtureSubprocess())
        return EnvScalerAdapter([envs], utils, runtime_commit="0" * 40, sandbox=FixedFixtureSubprocess())
    model_calls = []
    def model_factory(state):
        actions = [0]
        def model(messages, **kwargs):
            model_calls.append({"generation": state.generation, "messages": copy.deepcopy(messages)})
            first = messages[0]["content"]
            if first.startswith("Create the shortest"):
                content = "Use the available tools and finalize."
            elif first.startswith("You are analyzing"):
                content = json.dumps({"step_summary": "", "key_extracts": ["Fixture reusable strategy."]})
            elif first.startswith("You are managing"):
                content = "[1]"
            else:
                joined = json.dumps(messages)
                public = next(json.loads(message["content"].split("\n", 1)[1]) for message in messages
                              if message["content"].startswith("New task:\n"))
                domain = public["domain"]
                if actions[0] == 0:
                    name, arguments = {"searchqa": ("search", {"query": "France capital"}),
                        "code": ("run_code", {"code": PROGRAM, "stdin": "1 2"}),
                        "tool_use": ("increment", {"amount": 1})}[domain]
                else:
                    answer = PROGRAM if domain == "code" else "done" if domain == "tool_use" else (
                        "Paris" if "FIXTURE_USE_EVIDENCE_0" in joined else "London")
                    name, arguments = "final_answer", {"answer": answer}
                actions[0] += 1
                content = json.dumps({"tools": [{"name": name, "arguments": arguments}]})
            return {"message": {"role": "assistant", "content": content},
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    "binding": {"model_ref": state.model_ref, "decision_source": "mock"}}
        return model
    run_dir = tmp_path / "run"
    (run_dir / "gen_0").mkdir(parents=True)
    (run_dir / "meta").mkdir()
    shutil.copy2(ROOT / "seed_harness/seed.json", run_dir / "gen_0/seed.json")
    task = TaskAgentState(0, "mock_fixed_task", str(run_dir / "gen_0/seed.json"))
    bundles = MetaHarnessStore(run_dir / "meta")
    bundle = bundles.initialize(ROOT / "meta_harness/seed")
    meta_path = run_dir / "meta/harness_v0.md"
    meta_path.write_text((bundle.path / "instructions.md").read_text(encoding="utf-8"), encoding="utf-8")
    meta = MetaAgentState("mock_frozen_meta", str(meta_path), bundle_hash=bundle.hash, bundle_path=str(bundle.path))
    journal = StageJournal(run_dir)
    backend = MockBundleClient(bundles)
    client = DurableClient(backend, journal)
    executor = DurableExecutor(MultiDomainExecutor(store, adapter, model_factory, quotas=dict.fromkeys(DOMAINS, 1)), journal)
    agent = MetaAgent(client, {"sft_profile": "multidomain", "trainer_configured": False})
    class Unused:
        def apply(self, *args):
            raise AssertionError("This explicit mock test selects only HARNESS")
    updaters = {action: DurableUpdater(SeedHarnessUpdater(client) if action == TaskUpdateAction.HARNESS else Unused(), journal)
                for action in TaskUpdateAction}
    def accept(state, update):
        committed = bundles.commit_update(state.bundle_hash, instruction_text=update.harness, file_updates=update.bundle_files)
        state.bundle_hash, state.bundle_path = committed.hash, str(committed.path)
        return state
    return run_dir, task, meta, executor, agent, updaters, accept, backend, model_calls, store, index


def test_three_domain_loop_reloads_task_and_meta_harness_and_recovers_final_commit(tmp_path, monkeypatch):
    run_dir, task, meta, executor, agent, updaters, accept, backend, model_calls, store, index = _full_fixture(tmp_path)
    import sia.task_meta.loop as loop
    original_save = loop.save_json
    injected = [False]
    def crash_after_final_acceptance(path, value):
        if Path(path).name == "final_state.json" and value["status"] == "completed" and not injected[0]:
            injected[0] = True
            raise RuntimeError("injected crash after final acceptance before final state")
        original_save(path, value)
    monkeypatch.setattr(loop, "save_json", crash_after_final_acceptance)
    try:
        with pytest.raises(RuntimeError, match="after final acceptance"):
            run_task_meta(run_dir, task, meta, executor, agent, updaters, max_generations=3,
                          primary_metric_name="macro_success", meta_update_handler=accept)
        calls_before = len(model_calls), len(backend.calls)
        final = run_task_meta(run_dir, task, meta, executor, agent, updaters, max_generations=3,
                             primary_metric_name="macro_success", meta_update_handler=accept, resume=True)
        assert (len(model_calls), len(backend.calls)) == calls_before
        assert final["generations_executed"] == 3 and final["experiences"] == 2
        assert final["meta_state"]["version"] == 3 and final["decision_mode"] == "mock"
        assert [row["macro_success"] for row in final["performance_history"]] == pytest.approx([2 / 3, 1, 1])
        assert [call["version"] for call in backend.calls if call["operation"] == "route"] == [0, 1]
        next_route = next(call for call in backend.calls if call["operation"] == "route" and call["version"] == 1)
        assert "MOCK_NEXT_LOAD_1" in next_route["rendered"]
        assert len((run_dir / "meta/experiences.jsonl").read_text().splitlines()) == 2
        assert len(list((run_dir / "meta/harness_versions").glob("v*"))) == 4
        assert load_seed(run_dir / "gen_1/seed.json")["prompts"]["action_step"].endswith("FIXTURE_USE_EVIDENCE_0")
        assert all("HIDDEN_REFERENCE_SOLUTION" not in json.dumps(call) and "check_func" not in json.dumps(call)
                   for call in model_calls)
        train_ids, probe_ids = set(), set()
        for generation in range(3):
            rows = json.loads((run_dir / f"gen_{generation}/agent_execution.json").read_text(encoding="utf-8"))
            assert len(rows) == 3 and {row["domain"] for row in rows} == set(DOMAINS)
            train_ids.update(row["task_id"] for row in rows)
            samples = select_positive_rows(rows, profile="multidomain")
            assert samples and all(row["sft_supervision"] == "final_assistant" for row in samples)
            assert all(row["chat_template_kwargs"] == {"enable_thinking": False} for row in samples)
            assert all(call.get("operation") and call.get("call_id") is not None for row in rows for call in row["model_calls"])
            probes = [json.loads(line) for line in (run_dir / f"gen_{generation}/probe_trajectories.jsonl").read_text(encoding="utf-8").splitlines()]
            probe_ids.update(row["task_id"] for row in probes)
            assert all(not row["notes"] for row in probes)
        assert len(train_ids) == 9 and len(probe_ids) == 3 and train_ids.isdisjoint(probe_ids)
        assert all(row["domains"][domain]["denominator"] == 1 for row in final["performance_history"] for domain in DOMAINS)
    finally:
        index.close()
        store.close()
