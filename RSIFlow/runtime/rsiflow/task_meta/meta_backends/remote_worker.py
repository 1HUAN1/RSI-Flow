"""Credential-free Linux Codex worker behind an authenticated SSH tunnel.

The original controller owns provider credentials, transport accounting and G/H
commits. This process executes only the pinned isolated command, once per ID.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import http.server
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import signal
import socket
import stat
import subprocess
import threading
import time
import tomllib


MODELS = {
    "openrouter": {"deepseek/deepseek-v4-flash-0731", "openai/gpt-5.6-sol:batch", "openai/gpt-5.6-sol"},
    "autodl": {"DeepSeek-V4-Flash", "DeepSeek-V4-Flash-0731", "gpt-5.6-sol", "gpt-6-astra"},
}
REQUEST_ID = re.compile(r"^[a-f0-9]{32}$")
MAX_HTTP_BYTES = 36 * 1024 * 1024
MAX_INPUT_BYTES = 24 * 1024 * 1024
CHILD_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
BUDGET_BOUNDS = {
    "wall_time_seconds": (1, 7200), "max_requests": (1, 64),
    "max_output_tokens_per_request": (64, 128000),
    "max_total_output_tokens": (64, 1024000),
    "max_event_bytes": (1024, 512000000),
    "max_workspace_bytes": (1024, 16000000),
    "memory_bytes": (536870912, 17179869184), "max_processes": (4, 128),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    def invalid(_value):
        raise ValueError("nonfinite_json")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def checked_path(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("symlink_path")
    return path


def regular_bytes(path, limit):
    path = checked_path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("nonregular_or_oversized_file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            value = stream.read(limit + 1)
        if len(value) > limit:
            raise ValueError("oversized_file")
        return value
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = checked_path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def relative_path(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        raise ValueError("invalid_relative_path")
    value = PurePosixPath(name)
    if value.is_absolute() or any(part in {"", ".", ".."} for part in name.split("/")):
        raise ValueError("invalid_relative_path")
    return value


def workspace_inventory(root, limit):
    size, paths = 0, []
    for parent, directories, filenames in os.walk(root, followlinks=False):
        for name in directories:
            path = Path(parent) / name
            if not stat.S_ISDIR(path.lstat().st_mode):
                raise ValueError("workspace_special_file")
        for name in filenames:
            path = Path(parent) / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("workspace_special_file")
            size += info.st_size
            if size > limit:
                raise ValueError("workspace_budget_exhausted")
            paths.append(path)
            if len(paths) > 10000:
                raise ValueError("workspace_file_count_exhausted")
    return size, paths


class Conflict(RuntimeError):
    pass


class Worker:
    def __init__(self, args):
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise ValueError("linux_x86_64_required")
        self.root = checked_path(args.root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root = checked_path(self.root)
        self.paths = {name: checked_path(getattr(args, name))
                      for name in ("codex_executable", "catalog", "bridge", "bwrap")}
        self.paths["worker"] = checked_path(__file__)
        self.relay = checked_path(args.relay_socket)
        self.lock = threading.Lock()
        self.active_job = None
        self.process = None
        self.cancel_requested = threading.Event()
        self.runtime = self.fingerprints()
        probe_args = [str(self.paths["bwrap"]), "--unshare-all", "--unshare-user",
                      "--die-with-parent", "--ro-bind", "/", "/", "/bin/true"]
        self.probe = self.probe_command(probe_args)
        version = self.probe_command([str(self.paths["codex_executable"]), "--version"])
        self.version = version
        self.ready = self.probe["returncode"] == 0 and version["returncode"] == 0 and version["stdout"].strip() == "codex-cli 0.153.4"

    def fingerprints(self):
        keys = {"codex_executable": "binary_sha256", "catalog": "catalog_sha256",
                "bridge": "bridge_sha256", "bwrap": "bwrap_sha256", "worker": "worker_sha256"}
        return {keys[name]: digest(regular_bytes(path, 256 * 1024 * 1024))
                for name, path in self.paths.items()}

    @staticmethod
    def probe_command(command):
        started = time.time()
        try:
            result = subprocess.run(command, env=CHILD_ENV, stdin=subprocess.DEVNULL,
                                    capture_output=True, timeout=15, check=False)
            return {"command": command, "started_at": started, "ended_at": time.time(),
                    "returncode": result.returncode,
                    "stdout": result.stdout[:2048].decode("utf-8", "replace"),
                    "stderr": result.stderr[:2048].decode("utf-8", "replace")}
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"command": command, "started_at": started, "ended_at": time.time(),
                    "returncode": None, "stdout": "", "stderr": "", "error": type(exc).__name__}

    def health(self):
        with self.lock:
            active = self.active_job
        return {"ready": self.ready, "hostname": socket.gethostname(),
                "platform": "linux", "platform_detail": platform.platform(), "uid": os.getuid(),
                "runtime": self.runtime, "codex_version": self.version,
                "isolation_probe": self.probe, "isolation_verified": self.ready, "active_job": active,
                "provider_credentials_in_worker": False}

    def validate_payload(self, payload):
        if not isinstance(payload, dict) or set(payload) != {
                "request", "files", "file_hashes", "runtime_identity", "remaining_seconds"}:
            raise ValueError("invalid_run_payload")
        request = payload["request"]
        if (not isinstance(request, dict) or not REQUEST_ID.fullmatch(str(request.get("request_id", "")))
                or request.get("model") not in MODELS.get(request.get("provider"), set())):
            raise ValueError("invalid_request_identity")
        budget = request.get("budget")
        if not isinstance(budget, dict) or set(budget) != set(BUDGET_BOUNDS):
            raise ValueError("invalid_request_budget")
        for key, (lower, upper) in BUDGET_BOUNDS.items():
            if type(budget[key]) is not int or not lower <= budget[key] <= upper:
                raise ValueError("invalid_request_budget")
        remaining = payload["remaining_seconds"]
        if type(remaining) not in {int, float} or not math.isfinite(remaining) or not 0 < remaining <= budget["wall_time_seconds"]:
            raise ValueError("invalid_remaining_seconds")
        identity = payload["runtime_identity"]
        if (not isinstance(identity, dict) or set(identity) != {"binary_sha256", "catalog_sha256", "bridge_sha256"}
                or any(value != self.runtime[key] for key, value in identity.items())):
            raise ValueError("runtime_identity_mismatch")
        if not self.ready or self.fingerprints() != self.runtime:
            raise ValueError("runtime_not_ready_or_changed")
        files, hashes = payload["files"], payload["file_hashes"]
        if not isinstance(files, dict) or not isinstance(hashes, dict) or set(files) != set(hashes) or len(files) > 10000:
            raise ValueError("invalid_input_manifest")
        decoded, total, workspace_bytes = {}, 0, 0
        for name, encoded in files.items():
            path = relative_path(name)
            if name not in {"codex_home/config.toml", "schema.json", "prompt.txt", "request.json"} and not (len(path.parts) > 1 and path.parts[0] == "workspace"):
                raise ValueError("undeclared_input_path")
            if not isinstance(encoded, str):
                raise ValueError("invalid_file_encoding")
            try:
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("invalid_file_encoding") from exc
            if digest(data) != hashes[name]:
                raise ValueError("input_file_hash_mismatch")
            total += len(data)
            if path.parts[0] == "workspace":
                workspace_bytes += len(data)
            if total > MAX_INPUT_BYTES or workspace_bytes > budget["max_workspace_bytes"]:
                raise ValueError("input_budget_exhausted")
            decoded[name] = data
        required = {"workspace/AGENTS.md", "codex_home/config.toml", "schema.json", "prompt.txt"}
        if not required <= set(decoded):
            raise ValueError("missing_input_files")
        if "request.json" in decoded and strict_json(decoded["request.json"]) != request:
            raise ValueError("request_file_identity_mismatch")
        config = tomllib.loads(decoded["codex_home/config.toml"].decode("utf-8"))
        provider = config.get("model_providers", {}).get(request["provider"], {})
        if (config.get("model") != request["model"] or config.get("model_provider") != request["provider"]
                or config.get("model_catalog_json") != "/catalog.json"
                or provider.get("base_url") != "http://127.0.0.1:18443/api/v1"
                or provider.get("wire_api") != "responses" or "env_key" in provider
                or provider.get("request_max_retries") != 0 or provider.get("stream_max_retries") != 0
                or config.get("shell_environment_policy", {}).get("inherit") != "none"):
            raise ValueError("native_config_boundary_mismatch")
        if not stat.S_ISSOCK(self.relay.lstat().st_mode):
            raise ValueError("relay_socket_unavailable")
        return request, decoded, float(remaining)

    def command(self, directory):
        command = [str(self.paths["bwrap"]), "--unshare-all", "--unshare-user",
                   "--die-with-parent", "--new-session", "--cap-drop", "ALL"]
        for path in ("/usr", "/bin", "/lib", "/lib64", "/etc/ssl", "/etc/ld.so.cache"):
            if Path(path).exists():
                command += ["--ro-bind", path, path]
        command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                    "--dir", "/home", "--dir", "/home/meta",
                    "--bind", str(directory / "workspace"), "/workspace",
                    "--bind", str(directory / "codex_home"), "/codex_home",
                    "--ro-bind", str(directory / "codex_home/config.toml"), "/codex_home/config.toml",
                    "--ro-bind", str(directory / "workspace/AGENTS.md"), "/workspace/AGENTS.md",
                    "--ro-bind", str(directory / "schema.json"), "/schema.json",
                    "--ro-bind", str(self.paths["catalog"]), "/catalog.json",
                    "--ro-bind", str(self.relay), "/transport.sock",
                    "--ro-bind", str(self.paths["bridge"]), "/bridge.py",
                    "--ro-bind", str(self.paths["codex_executable"]), "/codex",
                    "--chdir", "/workspace", "/usr/bin/python3", "/bridge.py", "/codex",
                    "exec", "--json", "--strict-config", "--skip-git-repo-check",
                    "--ephemeral", "--output-schema", "/schema.json",
                    "--output-last-message", "/workspace/.meta_response.json", "-"]
        return command

    def run(self, payload):
        request, files, remaining = self.validate_payload(payload)
        request_id = request["request_id"]
        directory = self.root / request_id
        request_hash = digest(canonical(request))
        with self.lock:
            if directory.exists() or self.active_job is not None:
                raise Conflict("request_id_exists" if directory.exists() else "worker_busy")
            directory.mkdir(mode=0o700)
            self.active_job = request_id
            self.cancel_requested.clear()
        started = time.monotonic()
        result = {"request_id": request_id, "request_hash": request_hash,
                  "input_file_hashes": payload["file_hashes"], "runtime": self.runtime,
                  "returncode": None, "wall_time_seconds": 0.0,
                  "files": {}, "file_hashes": {}, "error": None}
        status = {"state": "prepared", "request_id": request_id, "request_hash": request_hash,
                  "started_at": time.time(), "deadline_seconds": remaining}
        process = None
        try:
            atomic_json(directory / "status.json", status)
            atomic_json(directory / "input_manifest.json", {"request": request, "file_hashes": payload["file_hashes"],
                        "runtime_identity": payload["runtime_identity"], "remaining_seconds": remaining})
            for name, data in files.items():
                path = directory / relative_path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(data)
            if "request.json" not in files:
                atomic_json(directory / "request.json", request)
            budget = request["budget"]
            import resource

            def limits():
                cpu = max(1, math.ceil(remaining))
                resource.setrlimit(resource.RLIMIT_AS, (budget["memory_bytes"], budget["memory_bytes"]))
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
                resource.setrlimit(resource.RLIMIT_FSIZE, (budget["max_event_bytes"], budget["max_event_bytes"]))
                resource.setrlimit(resource.RLIMIT_NPROC, (budget["max_processes"], budget["max_processes"]))

            if self.cancel_requested.is_set():
                raise ValueError("cancelled_before_dispatch")
            with (directory / "prompt.txt").open("rb") as stdin, (directory / "events.jsonl").open("xb") as stdout, (directory / "stderr.txt").open("xb") as stderr:
                process = subprocess.Popen(self.command(directory), stdin=stdin, stdout=stdout, stderr=stderr,
                                           env=CHILD_ENV, start_new_session=True, preexec_fn=limits)
                with self.lock:
                    self.process = process
                status.update(state="running", pid=process.pid)
                atomic_json(directory / "status.json", status)
                while process.poll() is None:
                    if self.cancel_requested.is_set():
                        raise ValueError("cancelled")
                    if time.monotonic() - started >= remaining:
                        raise ValueError("wall_time_exhausted")
                    workspace_inventory(directory / "workspace", budget["max_workspace_bytes"])
                    if sum((directory / name).stat().st_size for name in ("events.jsonl", "stderr.txt")) > budget["max_event_bytes"]:
                        raise ValueError("event_budget_exhausted")
                    time.sleep(0.1)
                result["returncode"] = process.returncode
                if process.returncode:
                    result["error"] = "isolated_process_failed"
        except Exception as exc:
            allowed = {"cancelled", "cancelled_before_dispatch", "wall_time_exhausted",
                       "workspace_budget_exhausted", "workspace_special_file",
                       "workspace_file_count_exhausted", "event_budget_exhausted"}
            result["error"] = str(exc) if isinstance(exc, ValueError) and str(exc) in allowed else type(exc).__name__
        finally:
            if process is not None and process.poll() is None:
                self.kill(process)
            if process is not None:
                result["returncode"] = process.returncode
            try:
                _, output_paths = workspace_inventory(directory / "workspace", request["budget"]["max_workspace_bytes"])
                event_bytes = 0
                for name in ("events.jsonl", "stderr.txt"):
                    path = directory / name
                    if path.exists():
                        event_bytes += path.lstat().st_size
                        output_paths.append(path)
                if event_bytes > request["budget"]["max_event_bytes"]:
                    raise ValueError("event_budget_exhausted")
                for path in output_paths:
                    name = path.relative_to(directory).as_posix()
                    data = regular_bytes(path, max(request["budget"]["max_event_bytes"], request["budget"]["max_workspace_bytes"]))
                    result["files"][name] = base64.b64encode(data).decode("ascii")
                    result["file_hashes"][name] = digest(data)
            except Exception as exc:
                result["error"] = result["error"] or (str(exc) if isinstance(exc, ValueError) and str(exc) in {
                    "event_budget_exhausted", "workspace_budget_exhausted", "workspace_special_file",
                    "workspace_file_count_exhausted"} else type(exc).__name__)
                result["files"], result["file_hashes"] = {}, {}
            result["wall_time_seconds"] = time.monotonic() - started
            try:
                atomic_json(directory / "result.json", result)
                status.update(state="completed" if result["error"] is None else "failed", ended_at=time.time(),
                              returncode=result["returncode"], error=result["error"])
                atomic_json(directory / "status.json", status)
            finally:
                with self.lock:
                    self.process = None
                    self.active_job = None
        return result

    @staticmethod
    def kill(process):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)

    def job(self, request_id):
        if not REQUEST_ID.fullmatch(request_id):
            raise ValueError("invalid_request_id")
        directory = checked_path(self.root / request_id)
        if not directory.is_dir():
            raise FileNotFoundError("job_not_found")
        status_path, result_path = directory / "status.json", directory / "result.json"
        status = strict_json(regular_bytes(status_path, 65536)) if status_path.exists() else {"state": "reserved", "request_id": request_id}
        result = strict_json(regular_bytes(result_path, 120 * 1024 * 1024)) if result_path.exists() else None
        return {"status": "completed" if result is not None else status["state"],
                "receipt": status, "result": result}

    def cancel(self, request_id):
        self.job(request_id)
        with self.lock:
            if self.active_job == request_id:
                self.cancel_requested.set()
                process = self.process
                if process is not None and process.poll() is None:
                    self.kill(process)
        return self.job(request_id)


def make_handler(worker, token):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(BUDGET_BOUNDS["wall_time_seconds"][1] + 100)

        def reply(self, status, payload):
            data = canonical(payload)
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            self.close_connection = True

        def authorized(self):
            supplied = self.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
                self.reply(401, {"error": "unauthorized"})
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            try:
                if self.path == "/health":
                    self.reply(200, worker.health())
                elif self.path.startswith("/jobs/"):
                    self.reply(200, worker.job(self.path.removeprefix("/jobs/")))
                else:
                    self.reply(404, {"error": "unknown_endpoint"})
            except FileNotFoundError:
                self.reply(404, {"error": "job_not_found"})
            except Exception as exc:
                self.reply(400, {"error": type(exc).__name__})

        def do_POST(self):
            if not self.authorized():
                return
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("transfer_encoding_not_allowed")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= MAX_HTTP_BYTES:
                    raise ValueError("request_body_too_large")
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("incomplete_request_body")
                if self.path == "/run":
                    self.reply(200, worker.run(strict_json(body)))
                elif self.path.startswith("/cancel/") and length <= 1024:
                    self.reply(200, worker.cancel(self.path.removeprefix("/cancel/")))
                else:
                    self.reply(404, {"error": "unknown_endpoint"})
            except Conflict as exc:
                self.reply(409, {"error": str(exc)})
            except FileNotFoundError:
                self.reply(404, {"error": "required_path_not_found"})
            except Exception as exc:
                self.reply(400, {"error": type(exc).__name__})

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "codex-executable", "catalog", "bridge", "bwrap", "relay-socket"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--port", type=int, default=19071)
    args = parser.parse_args(argv)
    token = os.environ.pop("RSI_REMOTE_WORKER_TOKEN", "")
    if len(token) < 24:
        parser.error("RSI_REMOTE_WORKER_TOKEN must be provided securely")
    if not 1024 <= args.port <= 65535:
        parser.error("Port must be 1024..65535")
    for name in list(os.environ):
        if name.upper().endswith(("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")):
            os.environ.pop(name, None)
    worker = Worker(args)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(worker, token))
    server.daemon_threads = True
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        if worker.active_job:
            worker.cancel(worker.active_job)
        server.server_close()


if __name__ == "__main__":
    main()
