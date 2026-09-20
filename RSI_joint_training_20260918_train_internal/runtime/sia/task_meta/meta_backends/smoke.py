"""Explicit paid compatibility gate; dev never enters this function."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from sia.task_meta.meta_harness.bundle import atomic_json, sha256

from .codex_openrouter import COMPATIBILITY_CHECKS
from .contracts import BackendUnavailable


class SmokeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: int
    test_passed: bool
    bundle_marker: str


def compatibility_smoke(backend, report_path: Path):
    """Real Codex tools must fix a sample, run its test, then reload G.

The report binds exact artifacts and runtime identity. A model's claims alone
never satisfy tool, patch, protocol, model, or reload evidence.
"""
    if backend.config.run_mode != "api_smoke":
        raise BackendUnavailable("API_SMOKE_NOT_ENABLED", "Select run_mode=api_smoke explicitly")
    backend.bind_context("compatibility_smoke", 0, sha256(b"non-sensitive compatibility fixture"))
    first = backend.prepare(
        "Read AGENTS.md and sample.py using Codex tools. Correct add(2,3) to return 5 by editing sample.py with native apply_patch. "
        "If apply_patch is not a listed tool, invoke apply_patch with a heredoc through exec_command; Codex intercepts it using the same native patch executor. "
        "The patch must be the ONLY command in that exec_command call: do not append a test or any other shell command. Run the test in a separate tool call so native patch events are emitted. "
        "Run python3 -B -c \"from sample import add; assert add(2,3)==5; print('RSI_COMPAT_TEST_PASS')\". "
        "Return answer=5, test_passed=true only if that test passed, and bundle_marker='seed'.",
        SmokeResult, operation="compatibility_smoke", allowed_paths=["sample.py"],
        evidence_files={"sample.py": "def add(a, b):\n    return a - b\n"})
    evidence = []
    checks = dict.fromkeys(COMPATIBILITY_CHECKS, False)
    try:
        backend.collect(first, backend.run(first))
        events = [json.loads(line) for line in (first.directory / "events.jsonl").read_text().splitlines() if line.strip()]
        items = [e.get("item", {}) for e in events if e.get("type") == "item.completed"]
        commands = [i for i in items if i.get("type") == "command_execution"]
        changes = [i for i in items if i.get("type") == "file_change"]
        transport = json.loads((first.directory / "transport.json").read_text())["requests"]
        checks["provider_identity"] = bool(transport) and all(r.get("returned_model") in backend.config.accepted_response_models for r in transport)
        checks["tool_roundtrip"] = len(transport) >= 2 and bool(commands)
        passed = any(i.get("exit_code") == 0 and "RSI_COMPAT_TEST_PASS" in i.get("aggregated_output", "") for i in commands)
        before = json.loads((first.directory / "workspace_before.json").read_text())["sample.py"]
        checks["patch_and_test"] = bool(changes) and passed and sha256((first.directory / "workspace/sample.py").read_bytes()) != before
        checks["streaming"] = bool(events) and all(r.get("completed") and not r.get("previous_response_id") and r.get("store") is False for r in transport)
        checks["schema"] = (first.directory / "collected.json").is_file()
        checks["model_metadata"] = bool(backend.config.model_catalog_sha256)
        from .transport import ResponsesTransport
        probe = ResponsesTransport(backend.config, "")
        rejected = 0
        for body in ({"model": "unregistered"}, {"model": backend.config.model, "store": True},
                     {"model": backend.config.model, "previous_response_id": "forbidden"}):
            try:
                probe.validate_body(body)
            except ValueError:
                rejected += 1
        checks["error_boundaries"] = rejected == 3
        original = backend.bundle_manager.active()
        marker = "reload_" + first.request.request_id
        changed = backend.bundle_manager.commit_update(original.hash,
            instruction_text=(original.path / "instructions.md").read_text() + "\nCompatibility reload marker: " + marker,
            file_updates={"context.json": '{"max_evidence_chars":63000,"selection":"tail"}'})
        second = backend.prepare("Read AGENTS.md using a Codex tool, extract its Compatibility reload marker, "
            "and return that marker, answer=5 and test_passed=true. Do not modify any files.",
            SmokeResult, operation="compatibility_smoke")
        reloaded = backend.collect(second, backend.run(second))
        checks["bundle_reload"] = reloaded.bundle_marker == marker and second.request.meta_harness_hash == changed.hash != first.request.meta_harness_hash
        for call in (first, second):
            for filename in ("request.json", "events.jsonl", "transport.json", "collected.json", "bundle_load.json"):
                path = call.directory / filename
                evidence.append({"path": str(path.resolve()), "sha256": sha256(path.read_bytes())})
        if not all(checks.values()):
            raise BackendUnavailable("BLOCKED_META_BACKEND_COMPATIBILITY", "One or more end-to-end checks did not pass")
        report = {**backend.compatibility_identity(), "status": "API_SMOKE_PASSED", "checks": checks, "evidence": evidence,
                  "error_probe_scope": "local deterministic rejection contracts; no induced paid rate-limit/timeout calls",
                  "tool_semantics": "native Codex shell/apply_patch; no freeform conversion",
                  "validated_bundle_hash": changed.hash}
        atomic_json(Path(report_path), report)
        return report
    except Exception as exc:
        atomic_json(Path(report_path), {**backend.compatibility_identity(), "status": getattr(exc, "status", "BLOCKED_META_BACKEND_COMPATIBILITY"),
                    "checks": checks, "evidence": evidence, "error_type": type(exc).__name__,
                    "pending_request": str(first.directory / "request.json")})
        raise
