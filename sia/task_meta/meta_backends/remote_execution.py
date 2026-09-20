"""Single SSH-forwarded WSL execution boundary; server retains state and API transport."""
from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import socket
import stat
import threading
import time
from pathlib import Path

from sia.task_meta.meta_harness.bundle import atomic_json
from .contracts import BackendUnavailable, allowed_relative


def digest(data):
    return hashlib.sha256(data).hexdigest()


def request_hash(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


class WorkerConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout):
        super().__init__("localhost", timeout=timeout)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


def rpc(config, method, path, payload=None, timeout=15, maximum=1000000):
    secret = os.environ.get(config.remote_worker_token_env)
    if not secret:
        raise BackendUnavailable("BLOCKED_REMOTE_META_AUTH", "Remote worker credential absent from trusted controller environment")
    connection = WorkerConnection(config.remote_worker_socket, timeout)
    try:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        connection.request(method, path, body, {"Authorization": "Bearer " + secret, "Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read(maximum + 1)
        if len(data) > maximum:
            raise BackendUnavailable("REMOTE_META_OUTPUT_LIMIT", "Worker response exceeded bounded byte limit")
        value = json.loads(data)
        if response.status != 200:
            raise BackendUnavailable("REMOTE_META_REQUEST_REJECTED", f"Worker HTTP {response.status}; inspect durable worker request receipt")
        return value
    except (OSError, http.client.HTTPException, ValueError) as exc:
        raise BackendUnavailable("REMOTE_META_CONNECTION_UNAVAILABLE", type(exc).__name__) from exc
    finally:
        connection.close()


def validate_worker(config):
    value = rpc(config, "GET", "/health")
    runtime = value.get("runtime", value)
    expected = {"binary_sha256": config.codex_binary_sha256, "catalog_sha256": config.model_catalog_sha256,
                "bridge_sha256": digest(Path(__file__).with_name("bridge.py").read_bytes()),
                "bwrap_sha256": config.remote_bwrap_sha256, "worker_sha256": config.remote_worker_sha256}
    if any(not wanted or runtime.get(key) != wanted for key, wanted in expected.items()):
        raise BackendUnavailable("REMOTE_META_RUNTIME_IDENTITY", "WSL worker code/binary/catalog/bridge/bwrap differs from registered identity")
    if not value.get("isolation_verified") or value.get("platform") != "linux":
        raise BackendUnavailable("REMOTE_META_SANDBOX_UNAVAILABLE", "Actual WSL execution host did not pass strict isolation")
    if value.get("active_job"):
        raise BackendUnavailable("REMOTE_META_BUSY", "An earlier worker job is active; reconcile it before dispatch")
    return value


def _input_files(directory):
    paths = [directory / "prompt.txt", directory / "schema.json", directory / "request.json", directory / "codex_home/config.toml"]
    paths.extend((directory / "workspace").rglob("*"))
    files = {}
    for path in paths:
        if path.is_symlink():
            raise ValueError("Remote inputs cannot contain links")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Remote inputs must be regular files")
        files[path.relative_to(directory).as_posix()] = base64.b64encode(path.read_bytes()).decode()
    return files


def run_remote(backend, prepared, transport):
    config, budget = backend.config, backend.config.budget
    directory = prepared.directory
    if (directory / "remote_dispatch.json").exists():
        raise BackendUnavailable("META_OPERATION_PENDING", "Remote request already dispatched; audit its receipt, never automatically resubmit")
    health = validate_worker(config)
    atomic_json(directory / "remote_worker_identity.json", health)
    files = _input_files(directory)
    hashes = {name: digest(base64.b64decode(data)) for name, data in files.items()}
    remaining = min(budget.wall_time_seconds, prepared.operation_budget.remaining_seconds() if prepared.operation_budget else budget.wall_time_seconds)
    payload = {"request": prepared.request.model_dump(mode="json"), "files": files, "file_hashes": hashes,
               "runtime_identity": {"binary_sha256": config.codex_binary_sha256, "catalog_sha256": config.model_catalog_sha256,
                   "bridge_sha256": digest(Path(__file__).with_name("bridge.py").read_bytes())}, "remaining_seconds": remaining}
    identity = {"request_id": prepared.request.request_id, "request_hash": request_hash(payload["request"]),
                "input_file_hashes": hashes, "worker": health, "started_at": time.time(), "state": "dispatching"}
    socket_path = Path(config.remote_relay_socket)
    if not socket_path.is_absolute() or socket_path.parent != Path("/tmp") or not socket_path.name.startswith("rsi_meta_"):
        raise ValueError("Remote relay must use a dedicated absolute /tmp/rsi_meta_ socket")
    if socket_path.exists():
        raise BackendUnavailable("META_OPERATION_PENDING", "Earlier relay socket exists; verify inactive owner and receipts before cleanup")
    atomic_json(directory / "remote_dispatch.json", identity)
    maximum = 2 * (budget.max_workspace_bytes + 2 * budget.max_event_bytes) + 1000000
    holder = {}
    started = time.monotonic()

    def dispatch():
        try:
            holder["result"] = rpc(config, "POST", "/run", payload, timeout=remaining + 20, maximum=maximum)
        except Exception as exc:
            holder["error"] = exc

    transport.start(socket_path)
    try:
        thread = threading.Thread(target=dispatch, daemon=True)
        thread.start()
        while thread.is_alive():
            if transport.error or time.monotonic() - started >= remaining:
                try:
                    cancel = rpc(config, "POST", "/cancel/" + prepared.request.request_id, {}, timeout=5)
                    atomic_json(directory / "remote_cancel.json", cancel)
                except Exception:
                    pass
                thread.join(timeout=5)
                raise BackendUnavailable("META_BUDGET_OR_TRANSPORT_FAILURE", "Remote stage aborted; receipts retained and retry prohibited")
            thread.join(timeout=0.2)
        if "error" in holder:
            # Read back the existing result once; this GET never resubmits inference.
            try:
                existing = rpc(config, "GET", "/jobs/" + prepared.request.request_id, timeout=10, maximum=maximum)
                if existing.get("status") == "completed" and existing.get("result"):
                    holder["result"] = existing["result"]
                else:
                    raise holder["error"]
            except Exception as exc:
                raise BackendUnavailable("META_OPERATION_PENDING", "SSH dispatch outcome unresolved; inspect original worker receipt before recovery") from exc
        response = holder["result"]
        if response.get("request_id") != identity["request_id"] or response.get("request_hash") != identity["request_hash"] or response.get("input_file_hashes") != hashes:
            raise BackendUnavailable("REMOTE_META_RESULT_IDENTITY", "Remote output is stale or its request/input hashes differ")
        if any(response.get("runtime", {}).get(key) != value for key, value in payload["runtime_identity"].items()):
            raise BackendUnavailable("REMOTE_META_RUNTIME_IDENTITY", "Worker result runtime does not match request")
        returned = response.get("files", {})
        if response.get("error") and not returned:
            atomic_json(directory / "remote_result.json", response)
            identity.update(state="failed", ended_at=time.time())
            atomic_json(directory / "remote_dispatch.json", identity)
            raise BackendUnavailable("META_RUNTIME_FAILURE", "Worker rejected output; original controller input and remote failure receipt are preserved")
        decoded = {}
        for name, content in returned.items():
            rel = allowed_relative(name)
            if not (name.startswith("workspace/") or name in {"events.jsonl", "stderr.txt"}):
                raise ValueError("Remote result includes a protected controller path")
            data = base64.b64decode(content, validate=True)
            if digest(data) != response.get("file_hashes", {}).get(name):
                raise ValueError("Remote result file fingerprint differs")
            decoded[name] = data
        work_bytes = sum(len(data) for name, data in decoded.items() if name.startswith("workspace/"))
        event_bytes = sum(len(decoded.get(name, b"")) for name in ["events.jsonl", "stderr.txt"])
        if work_bytes > budget.max_workspace_bytes or event_bytes > budget.max_event_bytes:
            raise BackendUnavailable("REMOTE_META_OUTPUT_LIMIT", "Returned workspace/events exceed configured limit")
        if prepared.operation_budget:
            prepared.operation_budget.check_files(prepared.request.stage_id, work_bytes, event_bytes)
        # Only this stage's candidate workspace is synchronized. H/G commits stay in collect/updaters.
        for path in (directory / "workspace").rglob("*"):
            if path.is_symlink():
                raise ValueError("Controller candidate workspace contains a symlink")
            if path.is_file() and path.relative_to(directory).as_posix() not in decoded:
                path.unlink()
        for name, data in decoded.items():
            path = directory / allowed_relative(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        record = {k: v for k, v in response.items() if k != "files"}
        atomic_json(directory / "remote_result.json", record)
        identity.update(state="completed" if not response.get("error") and response.get("returncode") == 0 else "failed", ended_at=time.time())
        atomic_json(directory / "remote_dispatch.json", identity)
        if response.get("error") or response.get("returncode") != 0:
            raise BackendUnavailable("META_RUNTIME_FAILURE", "Remote isolated Codex failed; inspect native events and worker receipt")
        return {"returncode": 0, "wall_time_seconds": time.monotonic() - started, "execution_location": "ssh_worker",
                "transport": transport.requests, "transport_error": transport.error, "remote_identity": record}
    finally:
        transport.close()
        if socket_path.exists() and stat.S_ISSOCK(socket_path.stat().st_mode):
            socket_path.unlink()
