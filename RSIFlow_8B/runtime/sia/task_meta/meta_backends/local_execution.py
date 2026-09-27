"""Per-call local Codex execution; no remote worker or mount namespace required."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import tempfile
import time

from sia.task_meta.file_lock import exclusive_lock
from sia.task_meta.meta_harness.bundle import atomic_json
from sia.task_meta.resource_limits import resource_snapshot
from .contracts import BackendUnavailable
from .isolation_runtime import BACKEND, IsolationRuntime, regular_digest
from .remote_worker import regular_bytes, workspace_inventory


def runtime_identity(config):
    directory = Path(__file__).parent
    return {"backend": BACKEND, "sources": {
        name: regular_digest(directory / name) for name in
        ("local_execution.py", "isolation_runtime.py", "isolation_launcher.py", "bridge.py", "input_budget.py")},
        "codex": config.codex_binary_sha256, "catalog": config.model_catalog_sha256}


def runtime(config):
    directory = Path(__file__).parent.resolve()
    key = hashlib.sha256(json.dumps(runtime_identity(config), sort_keys=True).encode()).hexdigest()[:16]
    # Lustre rejects isolated UIDs. Keep this bounded, transient runtime on the system disk.
    return IsolationRuntime("/tmp/rsiflow_meta_local_" + key,
        launcher=directory / "isolation_launcher.py", codex=config.codex_executable,
        catalog=config.model_catalog_json, bridge=directory / "bridge.py",
        setpriv="/usr/bin/setpriv", unshare="/usr/bin/unshare")


LOCK = "/tmp/rsiflow_meta_local.lock"


def validate_local(config):
    with exclusive_lock(Path(LOCK)):
        instance = runtime(config)
        probe = instance.probe()
        if not probe.get("ready"):
            raise BackendUnavailable("LOCAL_META_EXECUTION_UNAVAILABLE", json.dumps(probe))
        return {"ready": True, "execution_location": "local_chroot",
                "runtime": instance.identity(), "isolation_probe": probe,
                "provider_credentials_in_child": False}


def _stage_files(directory, input_budget=None, output_limit=16000000):
    from .input_budget import MetaInputBudget, inventory
    _, workspace_paths = inventory(directory / 'workspace', input_budget or MetaInputBudget(), output_limit)
    paths = [directory / "codex_home/config.toml", directory / "schema.json", *workspace_paths]
    result = {}
    for path in paths:
        if path.is_symlink():
            raise ValueError("Local Meta input cannot contain symlinks")
        if path.is_dir():
            continue
        # Paths are streamed into the jail; never assemble a GB-sized bytes map.
        if not path.is_file():
            raise ValueError(f'Nonregular staged input: {path}')
        result[path.relative_to(directory).as_posix()] = path
    return result


def _return_workspace(source, directory, maximum, input_budget=None):
    from .input_budget import inventory, category
    _, paths = inventory(source, input_budget, maximum) if input_budget else workspace_inventory(source, maximum)
    target = directory / "workspace.returned"
    target.mkdir()
    for path in paths:
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        original = directory / 'workspace' / path.relative_to(source)
        if input_budget and category(path.relative_to(source)) != 'output':
            file_limit = max(input_budget.evidence_file_bytes, input_budget.skill_bytes, input_budget.control_bytes)
            if not original.is_file() or original.is_symlink() or regular_digest(original, file_limit) != regular_digest(path, file_limit):
                raise ValueError(f'Read-only Meta input changed: {path.relative_to(source)}')
            os.link(original, destination)
        else:
            shutil.copyfile(path, destination)
    (directory / "workspace").rename(directory / "workspace.input")
    target.rename(directory / "workspace")


def cpu_time_limit(budget):
    # RLIMIT_CPU is accumulated across threads, not elapsed wall time.
    # Allow the four Tokio plus four Rayon workers to use their full budget.
    return math.ceil(budget.wall_time_seconds * 8)


def run_local(backend, prepared, transport):
    from sia.task_meta.storage_budget import check_system_disk, RUNTIME_RESERVE_BYTES
    check_system_disk(RUNTIME_RESERVE_BYTES)
    directory = prepared.directory
    budget = backend.config.budget
    started = time.monotonic()
    resources_before = resource_snapshot()
    termination_reason = None
    with exclusive_lock(Path(LOCK)):
        instance = runtime(backend.config)
        uid = instance.uid_for(prepared.request.request_id)
        relay_identity = None
        process = None
        staged = False
        with tempfile.TemporaryDirectory(prefix="rsi_meta_local_relay_", dir=instance.root.parent) as temporary:
            relay = Path(temporary) / "relay.sock"
            transport.start(relay)
            try:
                staged_files = _stage_files(directory, backend.config.input_budget, budget.max_workspace_bytes)
                instance.stage(staged_files, uid)
                staged = True
                relay_identity = instance.link_relay(relay, uid)
                inner = ["/usr/bin/python3", "/bridge.py", "/codex", "exec", "--json", "--strict-config",
                         "--skip-git-repo-check", "--ephemeral", "--output-schema", "/schema.json",
                         "--output-last-message", "/workspace/.meta_response.json", "-"]
                command = instance.command(uid, memory_bytes=budget.memory_bytes,
                    cpu_seconds=cpu_time_limit(budget), file_bytes=budget.max_event_bytes,
                    processes=budget.max_processes, command=inner)
                atomic_json(directory / "execution_host.json", {
                    "execution_location": "local_chroot", "runtime": instance.identity(),
                    "provider_credentials_in_child": False,
                    "limits": {"wall_seconds": budget.wall_time_seconds, "cpu_seconds_per_process": cpu_time_limit(budget),
                               "max_processes_and_threads": budget.max_processes, "virtual_memory_bytes_per_process": budget.memory_bytes},
                    "resources_before": resources_before})
                with (directory / "prompt.txt").open("rb") as stdin, \
                     (directory / "events.jsonl").open("wb") as stdout, \
                     (directory / "stderr.txt").open("wb") as stderr:
                    atomic_json(directory / "local_dispatch.json", {
                        "protocol": "local_dispatch_v1",
                        "request_id": prepared.request.request_id,
                        "started_at": time.time()})
                    process = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=stderr,
                        cwd="/tmp", env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, start_new_session=True)
                    try:
                        while process.poll() is None:
                            check_system_disk()
                            from .input_budget import inventory
                            sizes, _ = inventory(instance.root / 'workspace', backend.config.input_budget, budget.max_workspace_bytes)
                            size = sizes['output']
                            if prepared.operation_budget:
                                event_bytes = sum((directory / name).stat().st_size for name in ("events.jsonl", "stderr.txt"))
                                prepared.operation_budget.check_files(prepared.request.stage_id, size, event_bytes)
                            if transport.error:
                                termination_reason = "transport_failure"
                                raise BackendUnavailable("META_TRANSPORT_FAILURE", "Inspect transport receipts")
                            if time.monotonic() - started > budget.wall_time_seconds:
                                termination_reason = "wall_time_exhausted"
                                raise BackendUnavailable("META_WALL_TIME_EXHAUSTED", "Elapsed wall-time budget exhausted")
                            time.sleep(0.2)
                    finally:
                        if process.poll() is None:
                            termination_reason = termination_reason or "controller_exception"
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=10)
                        atomic_json(directory / "resource_usage.json", {
                            "returncode": process.returncode,
                            "termination_reason": termination_reason or ("completed" if process.returncode == 0 else "child_exit"),
                            "wall_seconds": time.monotonic() - started,
                            "before": resources_before, "after": resource_snapshot()})
                if process.returncode:
                    raise BackendUnavailable("META_RUNTIME_FAILURE", f"Local Codex exited {process.returncode}; inspect {directory}/stderr.txt")
                return {"returncode": process.returncode, "wall_time_seconds": time.monotonic() - started,
                        "transport": transport.requests, "transport_error": transport.error}
            finally:
                try:
                    if staged:
                        _return_workspace(instance.root / "workspace", directory, budget.max_workspace_bytes, backend.config.input_budget)
                finally:
                    if relay_identity is not None:
                        instance.unlink_relay(relay_identity)
                    if staged:
                        instance.cleanup()
