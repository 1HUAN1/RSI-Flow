"""One fresh pinned Codex exec process per bounded Meta operation.

No direct-completion or developer-chat fallback exists. The transport holds the
API secret outside a Linux namespace; tools see only declared evidence/candidates.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from sia.task_meta.meta_harness.bundle import atomic_json, canonical, sha256

from .contracts import BackendUnavailable, MetaBackendConfig, MetaOperationRequest, allowed_relative
from .transport import ResponsesTransport

COMPATIBILITY_CHECKS = {"provider_identity", "tool_roundtrip", "patch_and_test", "streaming", "schema", "bundle_reload", "error_boundaries", "model_metadata"}
OPERATIONS = {"route": "routing", "learn": "meta_self_update"}


def redact(text):
    for name, value in os.environ.items():
        if len(value) >= 8 and name.upper().endswith(("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")):
            text = text.replace(value, "[REDACTED_CREDENTIAL]")
    return text


def render_codex_config(config: MetaBackendConfig, *, isolated=False):
    """Real Codex TOML keys, checked additionally against the pinned source schema."""
    url = "http://127.0.0.1:18443/api/v1" if isolated else config.base_url
    lines = [f"model = {json.dumps(config.model)}", f"model_provider = {json.dumps(config.provider)}",
             'approval_policy = "never"', 'sandbox_mode = "danger-full-access"',
             'web_search = "disabled"']
    if isolated:
        lines += ['model_catalog_json = "/catalog.json"']
    lines += ['[agents]', 'enabled = false', '[features]', 'multi_agent = false', 'multi_agent_v2 = false', f'[model_providers.{config.provider}]', f'name = {json.dumps(config.provider)}',
              f"base_url = {json.dumps(url)}", 'wire_api = "responses"', 'requires_openai_auth = false',
              'supports_websockets = false', 'request_max_retries = 0', 'stream_max_retries = 0',
              f"stream_idle_timeout_ms = {min(config.response_timeout_seconds * 1000, config.budget.wall_time_seconds * 1000)}"]
    if not isolated:
        lines += [f'env_key = {json.dumps(config.api_key_env)}']
    # Importing a read-only validator must not create undeclared workspace .pyc
    # files. Codex filters explicit set values through include_only as well.
    lines += ['[shell_environment_policy]', 'inherit = "none"', 'include_only = ["PATH", "HOME", "LANG", "TMPDIR", "PYTHONDONTWRITEBYTECODE"]',
              '[shell_environment_policy.set]', 'PYTHONDONTWRITEBYTECODE = "1"']
    return "\n".join(lines) + "\n"


@dataclass
class PreparedOperation:
    request: MetaOperationRequest
    directory: Path
    schema: type
    bundle: object
    operation_budget: object | None = None


class CodexOpenRouterBackend:
    decision_source = "codex_openrouter"

    @property
    def supports_evolution(self):
        bundle = self.bundle_manager.active()
        return hasattr(bundle, "execution_spec") and bundle.execution_spec() is not None

    def __init__(self, config, journal: Path, bundle_manager):
        self.config = MetaBackendConfig.model_validate(config)
        self.journal = Path(journal)
        self.bundle_manager = bundle_manager
        self.model_name = self.config.model
        self.calls = []
        self.context = None

    def bind_context(self, run_id: str, generation: int, task_state_hash: str):
        self.context = {"run_id": run_id, "generation": generation, "task_state_hash": task_state_hash}

    def prepare(self, prompt, schema, *, meta_state=None, operation=None, decision_id=None,
                experience_id=None, allowed_paths=None, evidence_files=None, bundle_snapshot=None,
                workflow_operation_id=None, stage_id=None, operation_budget=None):
        if self.context is None:
            raise ValueError("Trusted controller must bind the current Task context before each operation")
        bundle = bundle_snapshot or self.bundle_manager.active()
        bundle.verify()
        if bundle.hash != self.bundle_manager.active().hash:
            raise ValueError("Pinned Meta Bundle is no longer active")
        if meta_state is not None and getattr(meta_state, "bundle_hash", None) not in {None, bundle.hash}:
            raise ValueError("Meta state refers to a stale Bundle")
        operation = OPERATIONS.get(operation, operation)
        paths = list(allowed_paths or [])
        for path in paths:
            allowed_relative(path)
            if path == "AGENTS.md" or any(part.startswith(".") for part in path.split("/")):
                raise ValueError("Codex instructions/configuration/output paths are protected")
        prompt = redact(prompt)
        evidence_files = evidence_files or {}
        for path, content in evidence_files.items():
            allowed_relative(path)
            if path == "AGENTS.md" or any(part.startswith(".") for part in path.split("/")):
                raise ValueError("Evidence cannot shadow Codex control files")
            if not isinstance(content, str):
                raise ValueError("Only explicit text evidence may enter Meta workspace")
        if operation in {"meta_self_update", "final_consolidation"}:
            editable = {name: (bundle.path / name).read_text(encoding="utf-8") for name in bundle.manifest["editable_files"]}
            prompt += ("\nCurrent bounded Meta Bundle files (return optional bundle_files mapping filename to full UTF-8 content; "
                       "harness is the complete instructions.md CONTENT, never a version name or path):\n" + json.dumps(editable, ensure_ascii=False)
                       + "\nOutput contract: put the complete intended instruction text in result.harness. "
                       "Do not return a label such as meta_harness_v1. Prefer omitting instructions.md from "
                       "result.bundle_files; if included, it must exactly equal result.harness. "
                       "When modifying only evolution.json, copy current instructions.md unchanged into result.harness. "
                       "This contract controls serialization only; choose substantive G changes from the actual experience."
                       + "\nOnly these declared files may change: " + ", ".join(sorted(editable))
                       + ". Context max_evidence_chars is 4000..120000; "
                       "selection is head, tail, or head_and_tail. Workflow section_order must contain instructions, evidence, "
                       "response_contract exactly once; checklist contains 1..12 strings of at most 1000 characters. "
                       "Runtime source edits/builds are implemented=false and unavailable.")
        request = MetaOperationRequest(request_id=uuid.uuid4().hex, operation=operation,
            schema_version="meta-operation-v2" if workflow_operation_id else "meta-operation-v1",
            **self.context, meta_harness_hash=bundle.hash,
            input_hash=sha256(canonical({"prompt": prompt, "files": evidence_files})),
            model=self.config.model, provider=self.config.provider,
            expected_response_model=self.config.expected_response_model,
            allowed_tools=["shell", "apply_patch"], allowed_paths=paths, budget=self.config.budget,
            output_schema=schema.model_json_schema(), decision_id=decision_id, experience_id=experience_id,
            workflow_operation_id=workflow_operation_id, stage_id=stage_id)
        directory = self.journal / "calls" / request.request_id
        directory.mkdir(parents=True, exist_ok=False)
        work = directory / "workspace"
        work.mkdir()
        home = directory / "codex_home"
        home.mkdir()
        for path, content in evidence_files.items():
            target = work / allowed_relative(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(redact(content), encoding="utf-8")
        identity = request.identity()
        output_schema = {"type": "object", "additionalProperties": False,
                         "properties": {k: {"const": v, "type": "integer" if isinstance(v, int) else "string"} for k, v in identity.items()},
                         "required": [*identity, "result"]}
        result_schema = dict(request.output_schema)
        if "$defs" in result_schema:
            output_schema["$defs"] = result_schema.pop("$defs")
        output_schema["properties"]["result"] = result_schema
        contract = ("Return ONLY one raw JSON object with exactly this identity and your typed result. "
                    "No Markdown fences, preface, explanation, or trailing text in the final response:\n"
                    + json.dumps({**identity, "result": "<schema result>"})
                    + "\nThe complete required JSON schema is mounted read-only at /schema.json. "
                    "Read it before the final response when any result field is uncertain. "
                    "Use tools for inspection; the final message must contain the JSON object only.")
        if self._candidate_file_operation(operation):
            # Keep the full schema and payload in a native tool-written artifact.
            # The final message is a small hash receipt, never a second transcription.
            candidate_schema_path = work / "meta_input/result_schema.json"
            candidate_schema_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(candidate_schema_path, output_schema)
            output_schema = {
                "type": "object", "additionalProperties": False,
                "properties": {"request_id": {"type": "string", "const": request.request_id},
                    "candidate_file": {"type": "string", "const": ".meta_candidate.json"},
                    "candidate_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"}},
                "required": ["request_id", "candidate_file", "candidate_sha256"]}
            contract = (
                "Use native Codex tools to write the COMPLETE candidate JSON envelope into "
                "/workspace/.meta_candidate.json using json.dump, with this exact identity and your typed result: "
                + json.dumps({**identity, "result": "<complete typed result>"})
                + "\nThe COMPLETE candidate schema is at meta_input/result_schema.json. "
                "Read it, build the actual object in Python, write all contents (no placeholders, labels or ellipses), "
                "and validate the written file. For G files, harness is complete instructions.md text; "
                "omit duplicate instructions.md from bundle_files; serialize any JSON file contents with json.dumps. "
                "Only .meta_candidate.json is additionally writable; do not change input snapshots. "
                "Compute SHA-256 from the actual candidate file bytes. Your FINAL message must contain ONLY "
                "the small JSON receipt matching /schema.json: request_id, candidate_file, candidate_sha256. "
                "Do NOT repeat or summarize the candidate in your final message. The controller will validate "
                "the file, hash, original full schema, G rules, and identity before committing anything.")
        if workflow_operation_id:
            # The executable G runtime already selected evidence with source IDs.
            # Never character-truncate this stage again, especially trusted facts.
            rendered, load = bundle.render(prompt, contract, preselected=True)
            load.update({"execution_entry": "sia.task_meta.meta_harness.runtime.execute",
                    "entry_files": sorted(bundle.manifest["editable_files"]),
                    "context_processing": "executable_runtime_before_stage; trusted_facts_preserved",
                    "workflow_operation_id": workflow_operation_id, "stage_id": stage_id})
            if len(rendered.encode()) > self.config.budget.max_workspace_bytes:
                raise BackendUnavailable("META_INPUT_BUDGET_EXHAUSTED", "Prepared G stage exceeds the external input/workspace ceiling; evidence was not silently truncated")
        else:
            rendered, load = bundle.render(prompt, contract)
        # AGENTS.md loads the versioned instructions through Codex's normal path.
        (work / "AGENTS.md").write_bytes((bundle.path / "instructions.md").read_bytes())
        load["instruction_sources"] = {
            "bundle_instructions_sha256": sha256((bundle.path / "instructions.md").read_bytes()),
            "inline_prompt_sha256": sha256(rendered.encode()),
            "native_agents_path": "/workspace/AGENTS.md", "native_agents_consumption_verified": False,
            "native_agents_caveat": "Pinned Codex project trust and project_doc_max_bytes affect native discovery; the full instructions are also inline"}
        (directory / "prompt.txt").write_text(rendered, encoding="utf-8")
        (home / "config.toml").write_text(render_codex_config(self.config, isolated=True), encoding="utf-8")
        atomic_json(directory / "request.json", request.model_dump())
        atomic_json(directory / "runtime.json", {**self.compatibility_identity(), "codex_source": self.config.codex_source,
                    "codex_executable": self.config.codex_executable, "configured_run_mode": self.config.run_mode,
                    "config_sha256": sha256((home / "config.toml").read_bytes()),
                    "prompt_sha256": sha256(rendered.encode()), "decision_source": "pending",
                    "status": "IMPLEMENTED_NOT_API_VALIDATED"})
        atomic_json(directory / "schema.json", output_schema)
        atomic_json(directory / "bundle_load.json", load)
        atomic_json(directory / "workspace_before.json", self._workspace_files(work))
        atomic_json(directory / "status.json", {"state": "PREPARED", "request_id": request.request_id})
        return PreparedOperation(request, directory, schema, bundle, operation_budget)

    @staticmethod
    def _workspace_files(work):
        files = {}
        for path in work.rglob("*"):
            if path.is_symlink():
                raise ValueError("Meta workspace symlinks are not permitted")
            if path.is_file():
                files[path.relative_to(work).as_posix()] = sha256(path.read_bytes())
        return files

    def validate(self, prepared):
        c = self.config
        if c.run_mode == "dev":
            raise BackendUnavailable("DEV_META_API_DISABLED", "dev never calls the API; mock backends are explicit tests only")
        if not os.environ.get(c.api_key_env):
            raise BackendUnavailable("BLOCKED_META_CREDENTIALS", f"Provide a rotated credential through {c.api_key_env}")
        endpoints = {
            "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", {
                "deepseek/deepseek-v4-flash-0731", "openai/gpt-5.6-sol:batch", "openai/gpt-5.6-sol"}),
            "autodl": ("https://www.autodl.art/api/v1", "AUTODL_API_KEY", {"DeepSeek-V4-Flash", "DeepSeek-V4-Flash-0731", "gpt-5.6-sol"}),
        }
        endpoint, credential_env, models = endpoints[c.provider]
        if c.base_url != endpoint or c.api_key_env != credential_env or c.model not in models:
            raise BackendUnavailable("BLOCKED_META_BACKEND_COMPATIBILITY", "This experiment protocol requires its exact registered endpoint/model")
        if not c.codex_commit or not re.fullmatch(r"[a-f0-9]{40}", c.codex_commit):
            raise BackendUnavailable("BLOCKED_META_BACKEND_COMPATIBILITY", "Codex source commit is not pinned")
        if not c.codex_executable or not Path(c.codex_executable).is_absolute() or not Path(c.codex_executable).is_file():
            raise BackendUnavailable("BLOCKED_META_BACKEND_COMPATIBILITY", "An absolute pinned Codex binary is required")
        if sha256(Path(c.codex_executable).read_bytes()) != c.codex_binary_sha256:
            raise BackendUnavailable("BLOCKED_META_BACKEND_COMPATIBILITY", "Codex binary fingerprint mismatch")
        if not c.provenance_file or not Path(c.provenance_file).is_file():
            raise BackendUnavailable("BLOCKED_META_BACKEND_COMPATIBILITY", "Source/build provenance is required")
        provenance = json.loads(Path(c.provenance_file).read_text())
        if provenance.get("commit") != c.codex_commit or provenance.get("binary_sha256") != c.codex_binary_sha256:
            raise BackendUnavailable("BLOCKED_META_BACKEND_COMPATIBILITY", "Pinned runtime provenance mismatch")
        if not c.codex_source or not (Path(c.codex_source) / "codex-rs/core/config.schema.json").is_file():
            raise BackendUnavailable("BLOCKED_META_BACKEND_COMPATIBILITY", "Pinned Codex configuration schema is unavailable")
        try:
            head = subprocess.run(["git", "-C", c.codex_source, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=15, check=True)
            dirty = subprocess.run(["git", "-C", c.codex_source, "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True, text=True, timeout=15, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise BackendUnavailable("BLOCKED_META_BACKEND_COMPATIBILITY", "Cannot verify the pinned source checkout") from exc
        if head.stdout.strip() != c.codex_commit or dirty.stdout.strip():
            raise BackendUnavailable("BLOCKED_META_BACKEND_COMPATIBILITY", "Codex source commit/worktree differs from the registered runtime")
        self._validate_native_config()
        if not c.model_catalog_json or not Path(c.model_catalog_json).is_file() or sha256(Path(c.model_catalog_json).read_bytes()) != c.model_catalog_sha256:
            raise BackendUnavailable("BLOCKED_META_BACKEND_COMPATIBILITY", "Exact model capability catalog must be inspected and fingerprinted")
        catalog = json.loads(Path(c.model_catalog_json).read_text())
        models = catalog.get("models", []) if isinstance(catalog, dict) else catalog
        if not any(m.get("slug") == c.model and isinstance(m.get("context_window"), int) and m["context_window"] > 0 for m in models):
            raise BackendUnavailable("BLOCKED_META_BACKEND_COMPATIBILITY", "No verified context/capability entry for the exact model")
        if c.run_mode in {"pilot", "full"} and c.compatibility_mode == "prior_report":
            if not c.compatibility_report or not Path(c.compatibility_report).is_file():
                raise BackendUnavailable("BLOCKED_META_BACKEND_COMPATIBILITY", "API compatibility smoke has not passed")
            report = json.loads(Path(c.compatibility_report).read_text())
            expected = self.compatibility_identity()
            if any(report.get(key) != value for key, value in expected.items()) or not all(report.get("checks", {}).get(k) is True for k in COMPATIBILITY_CHECKS):
                raise BackendUnavailable("BLOCKED_META_BACKEND_COMPATIBILITY", "Compatibility evidence is incomplete or for another runtime")
            for item in report.get("evidence", []):
                path = Path(item["path"])
                if not path.is_file() or sha256(path.read_bytes()) != item["sha256"]:
                    raise BackendUnavailable("BLOCKED_META_BACKEND_COMPATIBILITY", "Compatibility evidence changed")
            if not report.get("evidence"):
                raise BackendUnavailable("BLOCKED_META_BACKEND_COMPATIBILITY", "Compatibility report has no real-call evidence")
        if c.execution_location == "ssh_worker":
            from .remote_execution import validate_worker
            worker = validate_worker(c)
            atomic_json(prepared.directory / "execution_host.json", worker)
        else:
            self._bubblewrap()
        prepared.bundle.verify()
        if self.bundle_manager.active().hash != prepared.request.meta_harness_hash:
            raise ValueError("Active Meta Bundle changed during operation preparation")

    def compatibility_identity(self):
        c = self.config
        return {"backend": c.backend, "provider": c.provider, "base_url": c.base_url, "model": c.model, "codex_commit": c.codex_commit,
                "response_delivery": c.response_delivery, "response_timeout_seconds": c.response_timeout_seconds,
                "g_output_delivery": c.g_output_delivery,
                "expected_response_model": c.expected_response_model,
                "accepted_response_models": sorted(c.accepted_response_models),
                "binary_sha256": c.codex_binary_sha256, "catalog_sha256": c.model_catalog_sha256,
                "provider_order": c.provider_order, "allow_provider_fallback": c.allow_provider_fallback,
                "execution_location": c.execution_location, "compatibility_mode": c.compatibility_mode,
                "remote_worker_sha256": c.remote_worker_sha256, "remote_bwrap_sha256": c.remote_bwrap_sha256}

    def _validate_native_config(self):
        import tomllib
        try:
            import jsonschema
        except ImportError as exc:
            raise BackendUnavailable("BLOCKED_META_BACKEND_COMPATIBILITY", "jsonschema is needed to validate the pinned native config") from exc
        native = json.loads((Path(self.config.codex_source) / "codex-rs/core/config.schema.json").read_text())
        try:
            jsonschema.validate(tomllib.loads(render_codex_config(self.config, isolated=True)), native)
        except Exception as exc:
            raise BackendUnavailable("BLOCKED_META_BACKEND_COMPATIBILITY", "Provider config does not match the pinned Codex schema") from exc

    @staticmethod
    def _bubblewrap():
        binary = shutil.which("bwrap") if sys.platform == "linux" else None
        if not binary:
            raise BackendUnavailable("BLOCKED_META_SANDBOX", "Linux bubblewrap with working user/PID/network namespaces is required; no host-root fallback")
        probe = subprocess.run([binary, "--unshare-all", "--die-with-parent", "--ro-bind", "/", "/", "/bin/true"],
                               capture_output=True, timeout=10, env={"PATH": "/usr/bin:/bin"})
        if probe.returncode:
            raise BackendUnavailable("BLOCKED_META_SANDBOX", "Kernel/container policy blocks required namespaces")
        return binary

    def _command(self, prepared, socket_path):
        c = self.config
        command = [self._bubblewrap(), "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL"]
        for path in ("/usr", "/bin", "/lib", "/lib64", "/etc/ssl", "/etc/ld.so.cache"):
            if Path(path).exists():
                command += ["--ro-bind", path, path]
        command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/home", "--dir", "/home/meta",
                    "--bind", str(prepared.directory / "workspace"), "/workspace",
                    "--bind", str(prepared.directory / "codex_home"), "/codex_home",
                    "--ro-bind", str(prepared.directory / "codex_home/config.toml"), "/codex_home/config.toml",
                    "--ro-bind", str(prepared.directory / "workspace/AGENTS.md"), "/workspace/AGENTS.md",
                    "--ro-bind", str(prepared.directory / "schema.json"), "/schema.json",
                    "--ro-bind", c.model_catalog_json, "/catalog.json",
                    "--ro-bind", str(socket_path), "/transport.sock",
                    "--ro-bind", str(Path(__file__).with_name("bridge.py")), "/bridge.py",
                    "--ro-bind", c.codex_executable, "/codex", "--chdir", "/workspace",
                    "/usr/bin/python3", "/bridge.py", "/codex", "exec", "--json", "--strict-config", "--skip-git-repo-check",
                    "--ephemeral", "--output-schema", "/schema.json", "--output-last-message", "/workspace/.meta_response.json", "-"]
        return command

    def run(self, prepared):
        self.validate(prepared)
        from . import bridge  # Ensure packaged bridge module exists before spending anything.
        del bridge
        budget = self.config.budget
        atomic_json(prepared.directory / "status.json", {"state": "META_RUNNING", "request_id": prepared.request.request_id})
        transport = ResponsesTransport(self.config, os.environ[self.config.api_key_env], prepared.operation_budget,
                                       evidence_dir=prepared.directory / "provider_responses")
        started = time.monotonic()

        def limits():
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (budget.memory_bytes, budget.memory_bytes))
            resource.setrlimit(resource.RLIMIT_CPU, (budget.wall_time_seconds, budget.wall_time_seconds + 1))
            resource.setrlimit(resource.RLIMIT_FSIZE, (budget.max_event_bytes, budget.max_event_bytes))
            resource.setrlimit(resource.RLIMIT_NPROC, (budget.max_processes, budget.max_processes))

        try:
            if self.config.execution_location == "ssh_worker":
                from .remote_execution import run_remote
                return run_remote(self, prepared, transport)
            # Short Unix socket path avoids platform path-length limits.
            with tempfile.TemporaryDirectory(prefix="rsi_meta_") as temporary:
                socket_path = Path(temporary) / "relay.sock"
                transport.start(socket_path)
                with (prepared.directory / "prompt.txt").open("rb") as stdin, (prepared.directory / "events.jsonl").open("wb") as stdout, (prepared.directory / "stderr.txt").open("wb") as stderr:
                    process = subprocess.Popen(self._command(prepared, socket_path), stdin=stdin, stdout=stdout, stderr=stderr,
                        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, start_new_session=True, preexec_fn=limits)
                    try:
                        while process.poll() is None:
                            size = sum(p.stat().st_size for p in (prepared.directory / "workspace").rglob("*") if p.is_file())
                            if prepared.operation_budget:
                                event_size = sum(p.stat().st_size for p in (prepared.directory / "events.jsonl", prepared.directory / "stderr.txt") if p.exists())
                                prepared.operation_budget.check_files(prepared.request.stage_id, size, event_size)
                            if time.monotonic() - started > budget.wall_time_seconds or size > budget.max_workspace_bytes or transport.error:
                                raise BackendUnavailable("META_BUDGET_OR_TRANSPORT_FAILURE", "Bounded operation aborted; no result committed")
                            time.sleep(0.1)
                    finally:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=5)
                if process.returncode:
                    raise BackendUnavailable("META_RUNTIME_FAILURE", f"Pinned Codex exited with status {process.returncode}; inspect isolated call evidence")
            if prepared.operation_budget:
                size = sum(p.stat().st_size for p in (prepared.directory / "workspace").rglob("*") if p.is_file())
                event_size = sum(p.stat().st_size for p in (prepared.directory / "events.jsonl", prepared.directory / "stderr.txt") if p.exists())
                prepared.operation_budget.check_files(prepared.request.stage_id, size, event_size)
            return {"returncode": process.returncode, "wall_time_seconds": time.monotonic() - started,
                    "transport": transport.requests, "transport_error": transport.error}
        finally:
            transport.close()
            atomic_json(prepared.directory / "transport.json", {"requests": transport.requests, "error": transport.error,
                        "output_tokens": transport.output_tokens, "wall_time_seconds": time.monotonic() - started})

    def _candidate_file_operation(self, operation):
        return (self.config.g_output_delivery == "candidate_file_all" or
                self.config.g_output_delivery == "candidate_file" and operation in {"meta_self_update", "final_consolidation"})

    def collect(self, prepared, result):
        request = prepared.request
        if (prepared.directory / "collected.json").exists():
            raise ValueError("Duplicate Meta response collection/commit")
        if self.context != {"run_id": request.run_id, "generation": request.generation, "task_state_hash": request.task_state_hash}:
            raise ValueError("Stale or cross-run Meta response")
        if self.bundle_manager.active().hash != request.meta_harness_hash:
            raise ValueError("Stale Meta Harness response")
        prepared.bundle.verify()
        output_path = prepared.directory / "workspace/.meta_response.json"
        before = json.loads((prepared.directory / "workspace_before.json").read_text())
        after = self._workspace_files(prepared.directory / "workspace")
        output_files = {".meta_response.json"}
        if self._candidate_file_operation(request.operation):
            output_files.add(".meta_candidate.json")
        changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)} - output_files
        if not changed <= set(request.allowed_paths):
            raise ValueError("Codex modified undeclared candidate files")
        events = [json.loads(line) for line in (prepared.directory / "events.jsonl").read_text().splitlines() if line.strip()]
        if not any(e.get("type") == "turn.completed" for e in events) or any(e.get("type") in {"error", "turn.failed"} for e in events):
            raise ValueError("Codex event stream is incomplete or failed")
        if result.get("transport_error") or not result.get("transport") or any(r.get("returned_model") not in self.config.accepted_response_models or not r.get("completed") for r in result["transport"]):
            raise ValueError("Actual API model/completion evidence is missing")
        tool_events = []
        native_tools = {"command_execution": "shell", "file_change": "apply_patch"}
        for ordinal, event in enumerate(events):
            item = event.get("item", {})
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind in {"mcp_tool_call", "collab_tool_call", "web_search"}:
                raise ValueError("Codex used a tool outside the declared isolated operation capabilities")
            if kind in native_tools:
                tool = native_tools[kind]
                if tool not in request.allowed_tools:
                    raise ValueError("Codex tool is outside the operation allowlist")
                tool_events.append({"ordinal": ordinal, "event_type": event.get("type"),
                                    "native_item_type": kind, "tool": tool, "item": item,
                                    "request_id": request.request_id, "stage_id": request.stage_id,
                                    "workflow_operation_id": request.workflow_operation_id,
                                    "decision_source": self.decision_source,
                                    "event_sha256": sha256(canonical(event))})
        atomic_json(prepared.directory / "tool_events.json", {"source": "pinned_codex_exec_jsonl",
                    "decision_source": self.decision_source, "events": tool_events})
        raw_output = output_path.read_text(encoding="utf-8")
        if self._candidate_file_operation(request.operation):
            acknowledgement = json.loads(raw_output)
            candidate_path = prepared.directory / "workspace/.meta_candidate.json"
            if (not isinstance(acknowledgement, dict) or set(acknowledgement) != {
                    "request_id", "candidate_file", "candidate_sha256"}
                    or acknowledgement["request_id"] != request.request_id
                    or acknowledgement["candidate_file"] != ".meta_candidate.json"
                    or not isinstance(acknowledgement["candidate_sha256"], str)
                    or not re.fullmatch(r"[a-f0-9]{64}", acknowledgement["candidate_sha256"])
                    or not candidate_path.is_file() or candidate_path.is_symlink()
                    or sha256(candidate_path.read_bytes()) != acknowledgement["candidate_sha256"]):
                raise ValueError("Native G candidate file/receipt identity or hash mismatch")
            if not any(e["event_type"] == "item.completed" for e in tool_events):
                raise ValueError("G candidate file requires completed native Codex tool evidence")
            atomic_json(prepared.directory / "candidate_delivery.json", {
                **acknowledgement, "source": "native_codex_tool_written_artifact",
                "full_schema": "workspace/meta_input/result_schema.json"})
            raw_output = candidate_path.read_text(encoding="utf-8")
        try:
            envelope = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            # The native process, transport, tools and input boundaries above are
            # verified. A formatting failure is a completed invalid candidate,
            # eligible only for G's existing bounded repair stage, never replay.
            raise ValidationError.from_exception_data("MetaResponseEnvelope", [{
                "type": "json_invalid", "loc": (), "input": raw_output,
                "ctx": {"error": "Final response must be raw JSON without Markdown or prose: " + str(exc)},
            }]) from exc
        identity = request.identity()
        if not isinstance(envelope, dict) or set(envelope) != {*identity, "result"} or any(type(envelope.get(k)) is not type(v) or envelope.get(k) != v for k, v in identity.items()):
            raise ValueError("Meta response identity/schema mismatch")
        output = prepared.schema.model_validate(envelope["result"])
        if hasattr(output, "decision_source"):
            output.decision_source = self.decision_source
        if hasattr(output, "decision_id"):
            output.decision_id = request.decision_id or ""
        if hasattr(output, "request_id"):
            output.request_id = request.request_id
        if hasattr(output, "experience_id"):
            output.experience_id = request.experience_id
        load = json.loads((prepared.directory / "bundle_load.json").read_text())
        load.update({"runtime_verified": self.decision_source == "codex_openrouter",
                     "load_status": "codex_operation_completed" if self.decision_source == "codex_openrouter" else "offline_contract_completed",
                     "request_id": request.request_id, "decision_source": self.decision_source})
        atomic_json(prepared.directory / "bundle_load.json", load)
        record = {**request.identity(), "model": request.model, "provider": request.provider,
                  "runtime": self.compatibility_identity(),
                  "decision_source": self.decision_source, "changed_candidates": sorted(changed), "output": output.model_dump(mode="json"),
                  "api_cost_usd": None, **result}
        atomic_json(prepared.directory / "collected.json", record)
        if self.config.compatibility_mode == "in_run":
            # First production responses are the evidence. This is never a fabricated
            # prior smoke report, and mechanisms not observed remain uncovered.
            atomic_json(prepared.directory / "in_run_validation.json", {
                **request.identity(), "runtime": self.compatibility_identity(),
                "execution_location": self.config.execution_location,
                "checks": {"provider_identity": True, "native_event_stream": True,
                    "output_schema": True, "request_and_bundle_binding": True,
                    "declared_candidate_paths": True, "budget_accounting": True},
                "observed_native_tools": sorted({e["tool"] for e in tool_events}),
                "standalone_smoke_executed": False,
                "unobserved_mechanisms_are_not_verified": True})
        atomic_json(prepared.directory / "status.json", {"state": "META_COLLECTED", "request_id": request.request_id})
        self.calls.append(record)
        return output

    def _decision_binding_path(self, decision_id):
        if self.context is None or not decision_id:
            raise ValueError("A bound decision ID is required for Task intervention preparation")
        key = sha256(canonical({**self.context, "decision_id": decision_id}))
        return self.journal / "decision_bindings" / (key + ".json")

    def _check_decision_binding(self, operation, decision_id, bundle, operation_input):
        component = {"harness_patch": "HARNESS", "artifact_patch": "ARTIFACTS", "model_request": "MODEL"}.get(operation)
        if component is None:
            return
        path = self._decision_binding_path(decision_id)
        if not path.exists():
            raise ValueError("No completed routing receipt binds this Task modification")
        value = json.loads(path.read_text(encoding="utf-8"))
        if (value.get("meta_harness_hash") != bundle.hash or value.get("context") != self.context
                or value.get("decision_id") != decision_id or value.get("action") != component):
            raise ValueError("Task modification does not match its routing G snapshot/action/state")
        from sia.task_meta.types import MetaDecision
        decision = MetaDecision.model_validate((operation_input or {}).get("decision", {}))
        if decision.decision_id != decision_id or sha256(canonical(decision.model_dump(mode="json"))) != value.get("output_hash"):
            raise ValueError("Task modification changed the decision accepted by routing")

    def _complete_evolved(self, schema, *, meta_state, operation, decision_id, experience_id,
                          operation_input, validate_candidate, bundle):
        from sia.task_meta.file_lock import exclusive_lock
        from sia.task_meta.meta_harness.runtime import execute

        from .operation_budget import OperationBudget

        if self.context is None:
            raise ValueError("Trusted controller must bind Task context before invoking G")
        bundle.verify()
        if meta_state is not None and getattr(meta_state, "bundle_hash", None) not in {None, bundle.hash}:
            raise ValueError("Meta state refers to a stale Bundle")
        self._check_decision_binding(operation, decision_id, bundle, operation_input)
        operation_input = json.loads(redact(json.dumps(operation_input or {}, ensure_ascii=False)))
        validator_identity = None if validate_candidate is None else (
            getattr(validate_candidate, "__module__", "") + ":" + getattr(validate_candidate, "__qualname__", type(validate_candidate).__name__))
        binding = {"context": dict(self.context), "operation": operation, "meta_harness_hash": bundle.hash,
                   "input_hash": sha256(canonical(operation_input)), "schema": schema.model_json_schema(),
                   "decision_id": decision_id, "experience_id": experience_id,
                   "validator_identity": validator_identity, "runtime": self.compatibility_identity(),
                   "decision_source": self.decision_source, "run_mode": self.config.run_mode,
                   "external_budget": self.config.budget.model_dump()}
        operation_id = sha256(canonical(binding))
        # Short directory components also work on Windows hosts without long-path
        # support. The full identity is still checked in operation.json.
        directory = self.journal / "operations" / operation_id[:24]
        directory.mkdir(parents=True, exist_ok=True)
        self.last_operation_identity = {"workflow_operation_id": operation_id, "meta_harness_hash": bundle.hash,
                                        "decision_id": decision_id, "experience_id": experience_id,
                                        "operation": operation}
        with exclusive_lock(directory / ".lock"):
            request_path = directory / "operation.json"
            if request_path.exists():
                if json.loads(request_path.read_text(encoding="utf-8")) != binding:
                    raise ValueError("Meta operation binding changed during recovery")
            else:
                atomic_json(request_path, binding)
            ledger = OperationBudget(directory / "budget.json", self.config.budget)

            def invoke_stage(stage_id, stage_prompt, stage_schema, evidence_files=None, allowed_paths=None):
                bundle.verify()
                if self.context != binding["context"] or self.bundle_manager.active().hash != bundle.hash:
                    raise ValueError("Task state or active G changed inside an operation")
                stage_binding = {"stage_id": stage_id, "prompt": redact(stage_prompt),
                                 "schema": stage_schema.model_json_schema(),
                                 "files": json.loads(redact(json.dumps(evidence_files or {}, ensure_ascii=False))),
                                 "allowed_paths": list(allowed_paths or [])}
                stage_hash = sha256(canonical(stage_binding))
                receipt_path = directory / "stage_receipts" / (sha256(stage_id.encode())[:24] + ".json")
                receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else None
                if receipt:
                    if receipt.get("input_hash") != stage_hash:
                        raise ValueError("A recovered workflow stage has different inputs")
                    if receipt["state"] in {"completed", "completed_invalid"}:
                        if (sha256(canonical(receipt["output"])) != receipt.get("output_hash")
                                or receipt.get("decision_source") != self.decision_source):
                            raise ValueError("Workflow stage output receipt was modified")
                        return stage_schema.model_validate(receipt["output"])
                    if receipt["state"] != "prepared":
                        # The parent may have died after collect committed but before
                        # the stage receipt was published. Reconcile only that already
                        # accepted response; never resend a possibly issued API call.
                        committed_path = Path(receipt["directory"]) / "collected.json"
                        if committed_path.resolve().is_relative_to((self.journal / "calls").resolve()) and committed_path.exists():
                            record = json.loads(committed_path.read_text(encoding="utf-8"))
                            if (record.get("workflow_operation_id") != operation_id or record.get("stage_id") != stage_id
                                    or record.get("request_id") != receipt["request_id"] or record.get("meta_harness_hash") != bundle.hash
                                    or record.get("task_state_hash") != binding["context"]["task_state_hash"]
                                    or record.get("decision_source") != self.decision_source):
                                raise ValueError("Collected stage recovery identity differs")
                            output = stage_schema.model_validate(record["output"])
                            receipt.update(state="completed", output=output.model_dump(mode="json"),
                                           output_hash=sha256(canonical(output.model_dump(mode="json"))),
                                           reconciled_from="collected.json", decision_source=self.decision_source)
                            atomic_json(receipt_path, receipt)
                            return output
                        raise BackendUnavailable("META_OPERATION_PENDING", "A Codex stage may have called the API; reconcile its existing evidence before any retry")
                    call_dir = Path(receipt["directory"])
                    if not call_dir.resolve().is_relative_to((self.journal / "calls").resolve()):
                        raise ValueError("Workflow stage receipt has an external call directory")
                    request = MetaOperationRequest.model_validate_json((call_dir / "request.json").read_text())
                    if (request.workflow_operation_id != operation_id or request.stage_id != stage_id
                            or request.meta_harness_hash != bundle.hash or request.output_schema != stage_schema.model_json_schema()):
                        raise ValueError("Recovered stage request binding mismatch")
                    prepared = PreparedOperation(request, call_dir, stage_schema, bundle, ledger)
                else:
                    ledger.remaining_seconds()
                    prepared = self.prepare(stage_binding["prompt"], stage_schema, meta_state=meta_state,
                        operation=operation, decision_id=decision_id, experience_id=experience_id,
                        evidence_files=stage_binding["files"], allowed_paths=stage_binding["allowed_paths"],
                        bundle_snapshot=bundle, workflow_operation_id=operation_id, stage_id=stage_id,
                        operation_budget=ledger)
                    receipt = {"input_hash": stage_hash, "state": "prepared", "directory": str(prepared.directory.resolve()),
                               "request_id": prepared.request.request_id, "stage_id": stage_id,
                               "meta_harness_hash": bundle.hash, "workflow_operation_id": operation_id}
                    atomic_json(receipt_path, receipt)
                # Validate before marking possible side effects; missing credentials or
                # isolation retain the same prepared request for explicit resumption.
                try:
                    self.validate(prepared)
                except Exception as exc:
                    atomic_json(prepared.directory / "status.json", {"state": getattr(exc, "status", "META_VALIDATION_FAILED"),
                                "request_id": prepared.request.request_id, "error_type": type(exc).__name__})
                    raise
                receipt["state"] = "pending"
                atomic_json(receipt_path, receipt)
                collect_started = False
                try:
                    result = self.run(prepared)
                    collect_started = True
                    output = self.collect(prepared, result)
                except ValidationError:
                    # collect has already checked identity, transport and events before
                    # final schema validation. Keep invalid output for a distinct repair
                    # stage; replaying this stage must never repeat the paid call.
                    if not collect_started:
                        raise
                    raw_output = (prepared.directory / "workspace/.meta_response.json").read_text()
                    try:
                        invalid_output = json.loads(raw_output)["result"]
                    except json.JSONDecodeError:
                        invalid_output = {"unparsed_response": raw_output}
                    receipt.update(state="completed_invalid", output=invalid_output,
                                   output_hash=sha256(canonical(invalid_output)), decision_source=self.decision_source)
                    atomic_json(receipt_path, receipt)
                    raise
                except Exception as exc:
                    receipt.update(error_type=type(exc).__name__, error_status=getattr(exc, "status", "META_STAGE_FAILED"))
                    atomic_json(receipt_path, receipt)
                    atomic_json(prepared.directory / "status.json", {"state": receipt["error_status"],
                                "request_id": prepared.request.request_id, "recovery": "pending_no_automatic_api_retry"})
                    raise
                value = output.model_dump(mode="json")
                receipt.update(state="completed", output=value, output_hash=sha256(canonical(value)),
                               decision_source=self.decision_source, evidence={name: sha256((prepared.directory / name).read_bytes())
                               for name in ("request.json", "bundle_load.json", "events.jsonl", "tool_events.json", "collected.json")})
                atomic_json(receipt_path, receipt)
                return output

            invoke_stage.decision_source = self.decision_source
            atomic_json(directory / "status.json", {"state": "G_OPERATION_RUNNING", **self.last_operation_identity})
            try:
                output = execute(bundle, operation, operation_input, schema, invoke_stage, directory / "harness_runtime",
                                 external_budget={"max_invocations": self.config.budget.max_requests,
                                                  "wall_time_seconds": self.config.budget.wall_time_seconds},
                                 validate_candidate=validate_candidate)
                output = schema.model_validate(output)
                if self.context != binding["context"] or self.bundle_manager.active().hash != bundle.hash:
                    raise ValueError("Task state or active G changed before operation completion")
                if operation == "routing" and decision_id:
                    route_path = self._decision_binding_path(decision_id)
                    route = {"context": dict(self.context), "decision_id": decision_id, "meta_harness_hash": bundle.hash,
                             "action": getattr(output, "action", None), "workflow_operation_id": operation_id,
                             "output_hash": sha256(canonical(output.model_dump(mode="json")))}
                    if route_path.exists() and json.loads(route_path.read_text()) != route:
                        raise ValueError("Routing decision already binds different G/output")
                    atomic_json(route_path, route)
                atomic_json(directory / "result.json", {**self.last_operation_identity,
                            "output": output.model_dump(mode="json"), "decision_source": self.decision_source})
                atomic_json(directory / "status.json", {"state": "G_OPERATION_COMPLETED", **self.last_operation_identity})
                return output
            except Exception as exc:
                atomic_json(directory / "status.json", {"state": getattr(exc, "status", "G_OPERATION_FAILED"),
                            **self.last_operation_identity, "error_type": type(exc).__name__,
                            "recovery": "replay_completed_stages; reconcile_pending_stages"})
                raise

    def complete(self, prompt, schema, *, meta_state=None, operation=None, decision_id=None, experience_id=None,
                 operation_input=None, validate_candidate=None):
        bundle = self.bundle_manager.active()
        spec = bundle.execution_spec() if hasattr(bundle, "execution_spec") else None
        operation = OPERATIONS.get(operation, operation)
        if spec is not None:
            operation_input = dict(operation_input or {})
            operation_input.setdefault("instruction", prompt)
            return self._complete_evolved(schema, meta_state=meta_state, operation=operation,
                decision_id=decision_id, experience_id=experience_id, operation_input=operation_input,
                validate_candidate=validate_candidate, bundle=bundle)
        prepared = self.prepare(prompt, schema, meta_state=meta_state, operation=operation,
                                decision_id=decision_id, experience_id=experience_id, bundle_snapshot=bundle)
        try:
            return self.collect(prepared, self.run(prepared))
        except Exception as exc:
            state = getattr(exc, "status", "META_OPERATION_FAILED")
            atomic_json(prepared.directory / "status.json", {"state": state, "request_id": prepared.request.request_id,
                        "error_type": type(exc).__name__, "error": redact(str(exc))})
            raise
