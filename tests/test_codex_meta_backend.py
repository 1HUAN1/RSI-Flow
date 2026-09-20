import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from sia.task_meta.meta_backends import BackendUnavailable, CodexOpenRouterBackend, MetaBackendConfig
from sia.task_meta.meta_backends.codex_openrouter import render_codex_config
from sia.task_meta.meta_backends.transport import ResponsesTransport
from sia.task_meta.meta_harness import MetaHarnessStore


class Result(BaseModel):
    answer: str


@pytest.fixture
def backend(tmp_path):
    store = MetaHarnessStore(tmp_path / "meta")
    store.initialize(Path(__file__).resolve().parents[1] / "meta_harness/seed")
    value = CodexOpenRouterBackend(MetaBackendConfig(), tmp_path / "meta", store)
    value.decision_source = "test_override"
    value.bind_context("run_contract", 0, "a" * 64)
    return value


def simulated_codex_files(prepared):
    """Explicit offline contract fixture; never routed through the real run method."""
    response = {**prepared.request.identity(), "result": {"answer": "fixture"}}
    (prepared.directory / "workspace/.meta_response.json").write_text(json.dumps(response))
    (prepared.directory / "events.jsonl").write_text(json.dumps({"type": "turn.completed"}) + "\n")
    return {"transport_error": None, "transport": [{"returned_model": prepared.request.model, "completed": True}]}


def test_dev_never_runs_api_or_falls_back(backend, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-fixture-not-a-real-key")
    with pytest.raises(BackendUnavailable, match="DEV_META_API_DISABLED"):
        backend.complete("evidence", Result, operation="route")
    call = next((backend.journal / "calls").iterdir())
    assert json.loads((call / "status.json").read_text())["state"] == "DEV_META_API_DISABLED"
    assert not (call / "collected.json").exists()


def test_missing_key_keeps_pending_request(backend, monkeypatch):
    backend.config.run_mode = "api_smoke"
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(BackendUnavailable, match="BLOCKED_META_CREDENTIALS"):
        backend.complete("evidence", Result, operation="route")
    assert next((backend.journal / "calls").glob("*/request.json")).is_file()


def test_actual_codex_config_uses_responses_and_isolated_transport(backend):
    import tomllib
    config = tomllib.loads(render_codex_config(backend.config, isolated=True))
    provider = config["model_providers"]["openrouter"]
    assert provider["wire_api"] == "responses"
    assert "env_key" not in provider and "auth" not in provider
    assert provider["base_url"].endswith("/api/v1")
    assert config["features"]["multi_agent"] is False
    assert config["shell_environment_policy"]["inherit"] == "none"
    assert 'env_key = "OPENROUTER_API_KEY"' in render_codex_config(backend.config)


def test_prepare_binds_all_identity_and_redacts_secrets(backend, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-fixture-only-secret")
    prepared = backend.prepare("observation test-fixture-only-secret", Result, operation="learn")
    assert prepared.request.operation == "meta_self_update"
    assert prepared.request.run_id == "run_contract" and prepared.request.meta_harness_hash == backend.bundle_manager.active().hash
    assert "test-fixture-only-secret" not in (prepared.directory / "prompt.txt").read_text()
    assert (prepared.directory / "workspace/AGENTS.md").read_text() == (prepared.bundle.path / "instructions.md").read_text()


def test_collect_typed_output_and_duplicate_rejection(backend):
    prepared = backend.prepare("observation", Result, operation="route")
    result = simulated_codex_files(prepared)
    assert backend.collect(prepared, result).answer == "fixture"
    with pytest.raises(ValueError, match="Duplicate"):
        backend.collect(prepared, result)


@pytest.mark.parametrize("field,value", [("run_id", "different"), ("generation", 3), ("task_state_hash", "b" * 64), ("request_id", "f" * 32)])
def test_rejects_cross_run_stale_response(backend, field, value):
    prepared = backend.prepare("observation", Result, operation="route")
    result = simulated_codex_files(prepared)
    path = prepared.directory / "workspace/.meta_response.json"
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identity"):
        backend.collect(prepared, result)


def test_rejects_changed_active_bundle(backend):
    prepared = backend.prepare("observation", Result, operation="route")
    result = simulated_codex_files(prepared)
    backend.bundle_manager.commit_update(prepared.bundle.hash, instruction_text="updated G")
    with pytest.raises(ValueError, match="Stale Meta Harness"):
        backend.collect(prepared, result)


def test_rejects_undeclared_patch(backend):
    prepared = backend.prepare("observation", Result, operation="harness_patch")
    result = simulated_codex_files(prepared)
    (prepared.directory / "workspace/undeclared.py").write_text("illegal")
    with pytest.raises(ValueError, match="undeclared"):
        backend.collect(prepared, result)


def test_rejects_model_claim_without_transport_evidence(backend):
    prepared = backend.prepare("observation", Result, operation="route")
    result = simulated_codex_files(prepared)
    result["transport"] = []
    with pytest.raises(ValueError, match="API model"):
        backend.collect(prepared, result)


@pytest.mark.parametrize("body", [{"model": "other"}, {"model": "deepseek/deepseek-v4-flash-0731", "store": True},
                                 {"model": "deepseek/deepseek-v4-flash-0731", "previous_response_id": "r1"},
                                 {"model": "deepseek/deepseek-v4-flash-0731", "models": ["other"]}])
def test_transport_rejects_fallback_and_server_state(backend, body):
    with pytest.raises(ValueError):
        ResponsesTransport(backend.config, "").validate_body(body)


def test_transport_caps_budget_and_preserves_tools(backend):
    transport = ResponsesTransport(backend.config, "")
    tools = [{"type": "custom", "name": "apply_patch", "format": {"type": "grammar"}}]
    result = transport.validate_body({"model": backend.config.model, "tools": tools, "max_output_tokens": 99999})
    assert result["tools"] == tools and result["store"] is False
    assert result["max_output_tokens"] == backend.config.budget.max_output_tokens_per_request
    assert result["provider"]["allow_fallbacks"] is False
    transport.output_tokens = backend.config.budget.max_total_output_tokens
    with pytest.raises(ValueError, match="exhausted"):
        transport.validate_body({"model": backend.config.model})


def test_transport_rejects_actual_model_switch_and_truncation(backend):
    transport = ResponsesTransport(backend.config, "")
    with pytest.raises(ValueError, match="different model"):
        transport.observe({}, {"response": {"model": "different-model"}})
    with pytest.raises(ValueError, match="truncated"):
        transport.observe({}, {"type": "response.incomplete", "response": {"model": backend.config.model}})


@pytest.mark.parametrize("path", ["/root/hidden", "../hidden", "C:\\secret", "x/../hidden"])
def test_paths_cannot_escape_workspace(backend, path):
    with pytest.raises(ValueError):
        backend.prepare("observation", Result, operation="route", allowed_paths=[path])
