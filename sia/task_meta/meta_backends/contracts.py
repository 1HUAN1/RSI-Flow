"""Versioned request identities and immutable controller resource limits."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class BackendUnavailable(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(f"{status}: {message}")


class MetaBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    wall_time_seconds: int = Field(default=300, ge=1, le=3600)
    max_requests: int = Field(default=12, ge=1, le=64)
    max_output_tokens_per_request: int = Field(default=8192, ge=64, le=128000)
    max_total_output_tokens: int = Field(default=32768, ge=64, le=1024000)
    max_event_bytes: int = Field(default=16000000, ge=1024, le=512000000)
    max_workspace_bytes: int = Field(default=4000000, ge=1024, le=16000000)
    memory_bytes: int = Field(default=4294967296, ge=536870912, le=17179869184)
    max_processes: int = Field(default=64, ge=4, le=128)


class MetaBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["codex_openrouter"] = "codex_openrouter"
    provider: Literal["openrouter", "autodl"] = "openrouter"
    model: str = "deepseek/deepseek-v4-flash-0731"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: Literal["OPENROUTER_API_KEY", "AUTODL_API_KEY"] = "OPENROUTER_API_KEY"
    wire_api: Literal["responses"] = "responses"
    response_delivery: Literal["stream", "buffered_json"] = "stream"
    response_timeout_seconds: int = Field(default=300, ge=1, le=3600)
    g_output_delivery: Literal["final_message", "candidate_file", "candidate_file_all"] = "final_message"
    allow_model_fallback: Literal[False] = False
    run_mode: Literal["dev", "api_smoke", "pilot", "full"] = "dev"
    codex_source: str | None = None
    codex_commit: str | None = None
    codex_executable: str | None = None
    codex_binary_sha256: str | None = None
    provenance_file: str | None = None
    model_catalog_json: str | None = None
    model_catalog_sha256: str | None = None
    compatibility_report: str | None = None
    compatibility_mode: Literal["prior_report", "in_run"] = "prior_report"
    execution_location: Literal["local", "ssh_worker"] = "local"
    remote_worker_socket: str = "/tmp/rsi_meta_worker.sock"
    remote_relay_socket: str = "/tmp/rsi_meta_relay.sock"
    remote_worker_token_env: Literal["RSI_REMOTE_WORKER_TOKEN"] = "RSI_REMOTE_WORKER_TOKEN"
    remote_worker_sha256: str | None = None
    remote_bwrap_sha256: str | None = None
    provider_order: list[str] = Field(default_factory=list)
    allow_provider_fallback: bool = False
    harness_version: str = "seed"
    harness_root: str = "meta_harness"
    budget: MetaBudget = Field(default_factory=MetaBudget)

    @property
    def expected_response_model(self):
        # AutoDL advertises the unversioned request alias but returns this exact
        # versioned identity. No other aliases or model fallbacks are accepted.
        if (self.provider, self.model) == ("autodl", "DeepSeek-V4-Flash"):
            return "DeepSeek-V4-Flash-0731"
        return self.model

    @property
    def accepted_response_models(self):
        if (self.provider, self.model) == ("autodl", "DeepSeek-V4-Flash"):
            return {"DeepSeek-V4-Flash-0731", "deepseek-v4-flash-0731"}
        return {self.model}


class MetaOperationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["meta-operation-v1", "meta-operation-v2"] = "meta-operation-v1"
    request_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    operation: Literal["routing", "harness_patch", "artifact_patch", "model_request", "meta_self_update", "final_consolidation", "compatibility_smoke"]
    run_id: str = Field(min_length=1)
    generation: int = Field(ge=0)
    task_state_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    meta_harness_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    model: str
    expected_response_model: str | None = None
    provider: str
    allowed_tools: list[str]
    allowed_paths: list[str]
    budget: MetaBudget
    output_schema: dict
    decision_id: str | None = None
    experience_id: str | None = None
    workflow_operation_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    stage_id: str | None = Field(default=None, min_length=1, max_length=160)

    def identity(self):
        return self.model_dump(exclude_none=True, include={"schema_version", "request_id", "operation", "run_id", "generation", "task_state_hash", "meta_harness_hash", "input_hash", "workflow_operation_id", "stage_id"})


class MetaBackend(Protocol):
    def prepare(self, prompt, schema, **context): ...
    def validate(self, prepared): ...
    def run(self, prepared): ...
    def collect(self, prepared, result): ...


def allowed_relative(path: str):
    value = Path(path)
    if value.is_absolute() or path.startswith("/") or not path or "\\" in path or ":" in path or any(x in {"", ".", ".."} for x in path.split("/")):
        raise ValueError("Only explicit contained relative candidate paths are permitted")
    return value
