"""Offline backend seams: every injected model event is test_override, never API evidence."""
import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from sia.task_meta.meta_backends import BackendUnavailable, CodexOpenRouterBackend, MetaBackendConfig
from sia.task_meta.meta_backends.contracts import MetaBudget
from sia.task_meta.meta_backends.operation_budget import OperationBudget
from sia.task_meta.meta_harness import MetaHarnessStore
from sia.task_meta.types import MetaDecision, MetaHarnessUpdate


class Result(BaseModel):
    answer: str


class OfflineCodex(CodexOpenRouterBackend):
    decision_source = "test_override"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed = []
        self.answer = lambda p: {"answer": p.request.stage_id or "legacy"}
        self.tokens = 20

    def validate(self, prepared):
        assert self.config.run_mode == "dev", "Offline fixture must not enter a paid mode"
        assert prepared.bundle.hash == self.bundle_manager.active().hash

    def run(self, prepared):
        ledger = prepared.operation_budget
        used = ledger.output_limit(self.tokens) if ledger else self.tokens
        if ledger:
            ledger.reserve_request()
        self.executed.append(prepared)
        value = self.answer(prepared)
        if ledger:
            ledger.finish_request(used)
        envelope = {**prepared.request.identity(), "result": value}
        (prepared.directory / "workspace/.meta_response.json").write_text(json.dumps(envelope))
        (prepared.directory / "events.jsonl").write_text(json.dumps({"type": "turn.completed", "source": "test_override"}) + "\n")
        return {"transport_error": None, "transport": [{"returned_model": prepared.request.model,
                "completed": True, "usage": {"output_tokens": used}, "source": "test_override"}]}


@pytest.fixture
def backend(tmp_path):
    store = MetaHarnessStore(tmp_path / "meta")
    store.initialize(Path(__file__).resolve().parents[1] / "meta_harness/seed")
    value = OfflineCodex(MetaBackendConfig(), tmp_path / "meta", store)
    value.bind_context("offline_evolution", 0, "a" * 64)
    return value


def inject_workflow(monkeypatch, count=2):
    from sia.task_meta.meta_harness import runtime

    def execute(bundle, operation, operation_input, schema, invoke_stage, audit_dir, **kwargs):
        for index in range(count):
            output = invoke_stage(f"stage{index}", f"Current G stage {index}", schema)
        return output
    monkeypatch.setattr(runtime, "execute", execute)


def test_stages_pin_one_g_and_recover_without_duplicate_calls(backend, monkeypatch):
    inject_workflow(monkeypatch)
    first = backend.complete("minimal result contract", Result, operation="route", operation_input={"trusted_facts": {"total": 10}})
    assert first.answer == "stage1" and len(backend.executed) == 2
    second = backend.complete("minimal result contract", Result, operation="route", operation_input={"trusted_facts": {"total": 10}})
    assert second == first and len(backend.executed) == 2
    assert len({p.directory for p in backend.executed}) == 2
    assert len({p.request.meta_harness_hash for p in backend.executed}) == 1
    assert len({p.request.workflow_operation_id for p in backend.executed}) == 1
    for prepared in backend.executed:
        load = json.loads((prepared.directory / "bundle_load.json").read_text())
        assert load["decision_source"] == "test_override" and load["runtime_verified"] is False
        assert (prepared.directory / "codex_home/config.toml").is_file()
    budget = json.loads(next((backend.journal / "operations").glob("*/budget.json")).read_text())
    assert budget["requests"] == 2 and budget["output_tokens"] == 40


def test_pending_stage_does_not_repeat_unknown_api_effect(backend, monkeypatch):
    inject_workflow(monkeypatch, 1)
    def interrupt(prepared):
        raise RuntimeError("offline simulated interruption after reservation")
    backend.answer = interrupt
    with pytest.raises(RuntimeError, match="interruption"):
        backend.complete("", Result, operation="route")
    with pytest.raises(BackendUnavailable, match="META_OPERATION_PENDING"):
        backend.complete("", Result, operation="route")
    assert len(backend.executed) == 1


def test_malformed_json_uses_distinct_bounded_repair(backend, monkeypatch):
    from sia.task_meta.meta_harness import runtime
    def execute(bundle, operation, operation_input, schema, invoke_stage, audit_dir, **kwargs):
        try:
            return invoke_stage("proposal", "propose", schema)
        except ValidationError:
            return invoke_stage("repair", "return raw JSON", schema)
    monkeypatch.setattr(runtime, "execute", execute)
    original_run = backend.run
    def run(prepared):
        result = original_run(prepared)
        if prepared.request.stage_id == "proposal":
            path = prepared.directory / "workspace/.meta_response.json"
            path.write_text("```json\n" + path.read_text() + "\n```")
        return result
    backend.run = run
    assert backend.complete("", Result, operation="route").answer == "repair"
    assert backend.complete("", Result, operation="route").answer == "repair"
    assert len(backend.executed) == 2
    receipts = [json.loads(p.read_text()) for p in (backend.journal / "operations").glob("*/stage_receipts/*.json")]
    assert {r["state"] for r in receipts} == {"completed", "completed_invalid"}
    assert "unparsed_response" in next(r for r in receipts if r["state"] == "completed_invalid")["output"]


def test_g_stage_exposes_only_protected_selected_snapshot(backend):
    backend.complete("", Result, operation="route", operation_input={
        "trusted_facts": {"current_harness_identity": {"schema_version": 2}},
        "current_files": {"seed.json": '{"test_override":true}'}})
    prepared = backend.executed[0]
    snapshot = json.loads((prepared.directory / "workspace/meta_input/operation.json").read_text())
    assert snapshot["dependencies"][0]["content"] == '{"test_override":true}'
    prompt = (prepared.directory / "prompt.txt").read_text()
    assert "Attributable operation data" in prompt
    assert '{\\"test_override\\":true}' in prompt
    assert prepared.request.allowed_paths == []
    for name, content in prepared.bundle.read_files().items():
        assert (prepared.directory / "workspace/meta_input/G" / name).read_text() == content


def test_invalid_completed_stage_replayed_then_distinct_repair(backend, monkeypatch):
    from sia.task_meta.meta_harness import runtime
    def execute(bundle, operation, operation_input, schema, invoke_stage, audit_dir, **kwargs):
        try:
            return invoke_stage("proposal", "propose", schema)
        except ValidationError:
            return invoke_stage("repair_1", "repair same candidate", schema)
    monkeypatch.setattr(runtime, "execute", execute)
    backend.answer = lambda p: {"answer": 99} if p.request.stage_id == "proposal" else {"answer": "repaired"}
    assert backend.complete("", Result, operation="route").answer == "repaired"
    assert backend.complete("", Result, operation="route").answer == "repaired"
    assert len(backend.executed) == 2
    receipts = [json.loads(p.read_text()) for p in (backend.journal / "operations").glob("*/stage_receipts/*.json")]
    assert {r["state"] for r in receipts} == {"completed", "completed_invalid"}


def test_internal_stages_cannot_multiply_request_budget(backend, monkeypatch):
    inject_workflow(monkeypatch, 3)
    backend.config.budget.max_requests = 2
    with pytest.raises(BackendUnavailable, match="request limit"):
        backend.complete("", Result, operation="route")
    assert len(backend.executed) == 2


def test_internal_stages_share_output_token_ceiling(backend, monkeypatch):
    inject_workflow(monkeypatch, 3)
    backend.config.budget.max_total_output_tokens = 64
    backend.tokens = 40
    with pytest.raises(BackendUnavailable, match="output-token"):
        backend.complete("", Result, operation="route")
    budget = json.loads(next((backend.journal / "operations").glob("*/budget.json")).read_text())
    assert budget["output_tokens"] == 64 and len(backend.executed) == 2


def decision(action):
    return {"action": action, "diagnosis": "hypothesis", "evidence": [], "rationale": "test fixture",
            "proposed_change": "single component", "expected_effect": "unverified", "target_components": [action],
            "requested_changes": [{"id": "one", "component": action, "operation": "test", "target": "test", "instruction": "test"}]}


@pytest.mark.parametrize("action,operation", [("HARNESS", "harness_patch"), ("ARTIFACTS", "artifact_patch"), ("MODEL", "model_request")])
def test_route_and_all_patch_kinds_bind_identical_snapshot(backend, monkeypatch, action, operation):
    inject_workflow(monkeypatch, 1)
    backend.answer = lambda p: decision(action) if p.request.operation == "routing" else {"answer": "same G"}
    accepted = backend.complete("", MetaDecision, operation="route", decision_id="decision0")
    envelope = {"decision": accepted.model_dump(mode="json")}
    assert backend.complete("", Result, operation=operation, decision_id="decision0", operation_input=envelope).answer == "same G"
    old = backend.bundle_manager.active()
    backend.bundle_manager.commit_update(old.hash, instruction_text="New bounded instructions")
    with pytest.raises(ValueError, match="snapshot"):
        backend.complete("", Result, operation=operation, decision_id="decision0", operation_input=envelope)
    assert len(backend.executed) == 2


def test_unrouted_or_wrong_component_patch_is_rejected(backend, monkeypatch):
    inject_workflow(monkeypatch, 1)
    with pytest.raises(ValueError, match="routing receipt"):
        backend.complete("", Result, operation="harness_patch", decision_id="missing")
    backend.answer = lambda p: decision("HARNESS")
    backend.complete("", MetaDecision, operation="route", decision_id="decision0")
    with pytest.raises(ValueError, match="snapshot/action"):
        backend.complete("", Result, operation="artifact_patch", decision_id="decision0")


@pytest.mark.parametrize("operation", ["meta_self_update", "final_consolidation"])
def test_self_update_sources_are_written_by_trusted_backend(backend, monkeypatch, operation):
    inject_workflow(monkeypatch, 1)
    backend.answer = lambda p: {"harness": "new rules", "rationale": "fixture", "changed_rules": [],
                                "request_id": "invented", "experience_id": "invented"}
    output = backend.complete("", MetaHarnessUpdate, operation=operation, experience_id="real_exp0")
    assert output.request_id == backend.executed[0].request.request_id
    assert output.experience_id == "real_exp0"


def test_active_bundle_cannot_change_between_stages(backend, monkeypatch):
    from sia.task_meta.meta_harness import runtime
    def execute(bundle, operation, operation_input, schema, invoke_stage, audit_dir, **kwargs):
        invoke_stage("one", "first", schema)
        backend.bundle_manager.commit_update(bundle.hash, instruction_text="changed mid-operation")
        return invoke_stage("two", "second", schema)
    monkeypatch.setattr(runtime, "execute", execute)
    with pytest.raises(ValueError, match="inside an operation"):
        backend.complete("", Result, operation="route")
    assert len(backend.executed) == 1


def test_cumulative_wall_time_survives_recovery(tmp_path, monkeypatch):
    from sia.task_meta.meta_backends import operation_budget
    clock = [1000.0]
    monkeypatch.setattr(operation_budget.time, "time", lambda: clock[0])
    limits = MetaBudget(wall_time_seconds=10)
    OperationBudget(tmp_path / "budget.json", limits).reserve_request()
    clock[0] += 11
    recovered = OperationBudget(tmp_path / "budget.json", limits)
    with pytest.raises(BackendUnavailable, match="wall-time"):
        recovered.remaining_seconds()


def test_performance_budget_matches_worker_and_remains_cumulative(tmp_path, backend):
    from sia.task_meta.meta_backends.remote_worker import BUDGET_BOUNDS
    from pydantic import ValidationError
    values = dict(wall_time_seconds=3600, max_requests=64,
                  max_output_tokens_per_request=128000, max_total_output_tokens=1024000,
                  max_event_bytes=512000000)
    limits = MetaBudget(**values)
    backend.config.budget = limits
    backend.complete("test_override expanded budget", Result, operation="route")
    assert backend.executed[0].request.budget.max_output_tokens_per_request == 128000
    for key, value in values.items():
        assert BUDGET_BOUNDS[key][0] <= value <= BUDGET_BOUNDS[key][1]
        with pytest.raises(ValidationError):
            MetaBudget(**{**values, key: BUDGET_BOUNDS[key][1] + 1})
    ledger = OperationBudget(tmp_path / "performance_budget.json", limits)
    for _ in range(8):
        assert ledger.output_limit(128000) == 128000
        ledger.reserve_request()
        ledger.finish_request(128000)
    recovered = OperationBudget(ledger.path, limits)
    assert recovered.state["requests"] == 8
    with pytest.raises(BackendUnavailable, match="output-token"):
        recovered.output_limit(128000)


def test_cumulative_file_and_stream_limits(tmp_path):
    limits = MetaBudget(max_event_bytes=1024, max_workspace_bytes=1024)
    ledger = OperationBudget(tmp_path / "budget.json", limits)
    ledger.check_files("one", 700, 100)
    with pytest.raises(BackendUnavailable, match="workspace/event"):
        ledger.check_files("two", 700, 100)
    ledger.receive_bytes(700)
    with pytest.raises(BackendUnavailable, match="response-stream"):
        ledger.receive_bytes(700)


def test_collected_response_reconciles_crash_before_stage_receipt(backend, monkeypatch):
    inject_workflow(monkeypatch, 1)
    expected = backend.complete("", Result, operation="route")
    receipt_path = next((backend.journal / "operations").glob("*/stage_receipts/*.json"))
    receipt = json.loads(receipt_path.read_text())
    receipt["state"] = "pending"
    receipt.pop("output")
    receipt.pop("output_hash")
    receipt_path.write_text(json.dumps(receipt))
    assert backend.complete("", Result, operation="route") == expected
    assert len(backend.executed) == 1
    assert json.loads(receipt_path.read_text())["reconciled_from"] == "collected.json"


def test_real_project_runtime_uses_changed_self_update_workflow(backend):
    """Handwritten G update + injected model: executable integration, not autonomous Meta."""
    def response(prepared):
        if prepared.schema.__name__ == "AnalysisOutput":
            return {"analysis": "offline hypothesis", "hypotheses": [], "source_ids": []}
        return {"harness": "fixture rules", "rationale": "offline only", "changed_rules": [], "status": "NO_CHANGE"}
    backend.answer = response
    backend.complete("Return a bounded Meta update", MetaHarnessUpdate, operation="learn", experience_id="experience0",
                     operation_input={"trusted_facts": {"total": 5}, "latest_experience": {"experience_id": "experience0"}})
    first_g = backend.bundle_manager.active()
    policy = first_g.execution_spec()
    policy["self_update"]["instruction"] = "UPDATED_SELF_UPDATE_PROCESS_MARKER: compare expectation and execution before revising the mechanism"
    policy["workflows"]["meta_self_update"].insert(0, {"id": "inspect_expectation", "kind": "analyze",
        "when": {"path": "facts.needs_analysis", "op": "eq", "value": True}})
    updated = backend.bundle_manager.commit_update(first_g.hash, file_updates={"evolution.json": json.dumps(policy)})
    before = len(backend.executed)
    backend.complete("Return a bounded Meta update", MetaHarnessUpdate, operation="learn", experience_id="experience1",
                     operation_input={"trusted_facts": {"total": 5, "needs_analysis": True},
                                      "latest_experience": {"experience_id": "experience1"}})
    stages = backend.executed[before:]
    assert [p.request.stage_id for p in stages] == ["inspect_expectation", "propose"]
    assert all(p.request.meta_harness_hash == updated.hash != first_g.hash for p in stages)
    assert all("UPDATED_SELF_UPDATE_PROCESS_MARKER" in (p.directory / "prompt.txt").read_text() for p in stages)
    assert "Return a bounded Meta update" in (stages[0].directory / "prompt.txt").read_text()
    assert all(json.loads((p.directory / "bundle_load.json").read_text())["runtime_verified"] is False for p in stages)


def test_legacy_bundle_keeps_complete_legacy_prompt(tmp_path, monkeypatch):
    from sia.task_meta.meta_harness import runtime
    seed = tmp_path / "old_seed"
    seed.mkdir()
    original = Path(__file__).resolve().parents[1] / "meta_harness/seed"
    for name in ("instructions.md", "context.json", "workflow.json"):
        (seed / name).write_bytes((original / name).read_bytes())
    store = MetaHarnessStore(tmp_path / "meta")
    store.initialize(seed)
    backend = OfflineCodex(MetaBackendConfig(), tmp_path / "meta", store)
    backend.bind_context("legacy", 0, "a" * 64)
    assert backend.supports_evolution is False
    monkeypatch.setattr(runtime, "execute", lambda *args, **kwargs: pytest.fail("Legacy v1 must remain explicit"))
    backend.complete("FULL_LEGACY_EVIDENCE", Result, operation="route")
    assert "FULL_LEGACY_EVIDENCE" in (backend.executed[0].directory / "prompt.txt").read_text()


def test_preselected_trusted_facts_are_kept_or_explicitly_rejected(backend):
    backend.config.budget.max_workspace_bytes = 1024
    with pytest.raises(BackendUnavailable, match="not silently truncated"):
        backend.prepare("TRUSTED_FACT_" * 500, Result, operation="routing", workflow_operation_id="b" * 64,
                        stage_id="proposal", bundle_snapshot=backend.bundle_manager.active())
    assert not backend.executed


def test_each_transport_stage_sees_the_same_remaining_tokens(backend, tmp_path):
    from sia.task_meta.meta_backends.transport import ResponsesTransport
    backend.config.budget.max_total_output_tokens = 64
    ledger = OperationBudget(tmp_path / "budget.json", backend.config.budget)
    first = ResponsesTransport(backend.config, "", ledger)
    assert first.validate_body({"model": backend.config.model, "max_output_tokens": 50})["max_output_tokens"] == 50
    ledger.reserve_request()
    ledger.finish_request(50)
    second = ResponsesTransport(backend.config, "", ledger)
    assert second.validate_body({"model": backend.config.model, "max_output_tokens": 50})["max_output_tokens"] == 14


def test_native_tool_events_are_attributed_and_undeclared_tools_rejected(backend):
    prepared = backend.prepare("offline", Result, operation="routing")
    result = backend.run(prepared)
    events = [{"type": "item.completed", "item": {"id": "fixture", "type": "command_execution",
               "command": "fixture command; never executed", "status": "completed", "exit_code": 0}},
              {"type": "turn.completed"}]
    (prepared.directory / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    backend.collect(prepared, result)
    record = json.loads((prepared.directory / "tool_events.json").read_text())
    assert record["decision_source"] == "test_override"
    assert record["events"][0]["tool"] == "shell"
    assert record["events"][0]["request_id"] == prepared.request.request_id
    forbidden = backend.prepare("offline", Result, operation="routing")
    result = backend.run(forbidden)
    events[0]["item"]["type"] = "mcp_tool_call"
    (forbidden.directory / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    with pytest.raises(ValueError, match="outside the declared"):
        backend.collect(forbidden, result)


@pytest.mark.parametrize("field", ["requested_changes", "evidence", "decision_id"])
def test_patch_cannot_change_accepted_decision_contents(backend, monkeypatch, field):
    inject_workflow(monkeypatch, 1)
    backend.answer = lambda p: decision("HARNESS") | {"decision_id": "model_invented_id"}
    accepted = backend.complete("", MetaDecision, operation="route", decision_id="decision0")
    assert accepted.decision_id == "decision0"
    payload = accepted.model_dump(mode="json")
    if field == "requested_changes":
        payload[field][0]["instruction"] = "different proposed modification"
    elif field == "evidence":
        payload[field] = ["invented evidence"]
    else:
        payload[field] = "different decision"
    with pytest.raises(ValueError, match="changed the decision"):
        backend.complete("", Result, operation="harness_patch", decision_id="decision0", operation_input={"decision": payload})
    assert len(backend.executed) == 1


def test_completed_stage_cannot_change_verification_source(backend, monkeypatch):
    inject_workflow(monkeypatch, 1)
    backend.complete("", Result, operation="route")
    path = next((backend.journal / "operations").glob("*/stage_receipts/*.json"))
    receipt = json.loads(path.read_text())
    receipt["decision_source"] = "codex_openrouter"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt was modified"):
        backend.complete("", Result, operation="route")
    assert len(backend.executed) == 1
