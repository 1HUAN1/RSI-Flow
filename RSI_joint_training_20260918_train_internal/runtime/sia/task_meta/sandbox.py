"""Fail-closed Linux candidate runner with Landlock, seccomp and credential drop.

No host evaluator, source data, checkpoint, credentials or home directory is
granted. Hidden expected outputs stay in the controller, never in this process.
This is an OS boundary, not an attempt to make Python ``exec`` safe.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import json
import math
import os
import platform
import select
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path


class SandboxUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxLimits:
    wall_seconds: float = 10.0
    cpu_seconds: int = 5
    memory_bytes: int = 1024 * 1024 * 1024
    output_bytes: int = 1024 * 1024
    file_bytes: int = 1024 * 1024
    processes: int = 1

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError("All sandbox limits must be finite and positive")
        if self.processes != 1:
            raise ValueError("Candidate sandbox supports exactly one process")


@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool
    wall_seconds: float
    output_truncated: bool = False
    infrastructure_error: str | None = None
    backend: str = "linux_landlock_seccomp"


class _Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def probe_isolation() -> dict:
    if sys.platform != "linux" or platform.machine() not in {"x86_64", "aarch64"}:
        return {"available": False, "reason": "Linux x86_64/aarch64 required"}
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(444, 0, 0, 1)
    seccomp = ctypes.util.find_library("seccomp")
    return {"available": abi >= 1 and bool(seccomp) and os.geteuid() == 0,
            "landlock_abi": abi, "libseccomp": seccomp,
            "credential_drop_available": os.geteuid() == 0,
            "reason": None if abi >= 1 and seccomp and os.geteuid() == 0 else "Landlock, libseccomp and root-to-isolated-UID credential drop are required"}


def _restrict_filesystem(read_paths: list[Path], scratch: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    # ABI 1 filesystem access rights; unsupported newer rights are not assumed.
    handled = (1 << 13) - 1
    rule_set = _Ruleset(handled)
    fd = libc.syscall(444, ctypes.byref(rule_set), ctypes.sizeof(rule_set), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    read_access = (1 << 0) | (1 << 2) | (1 << 3)
    try:
        for path, access in [(p, read_access) for p in read_paths] + [(scratch, handled)]:
            path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                if not path.is_dir():
                    access &= ~((1 << 3) | sum(1 << n for n in range(4, 13)))
                rule = _PathRule(access, path_fd)
                if libc.syscall(445, fd, 1, ctypes.byref(rule), 0) < 0:
                    raise OSError(ctypes.get_errno(), "landlock_add_rule")
            finally:
                os.close(path_fd)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
        if libc.syscall(446, fd, 0) < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(fd)


def _restrict_syscalls(library: str = "libseccomp.so.2") -> None:
    # Do not use find_library in the resource-limited child: it can fork ldconfig.
    seccomp = ctypes.CDLL(library, use_errno=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = seccomp.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not ctx:
        raise SandboxUnavailable("seccomp_init failed")
    # Landlock ABI 1 does not cover metadata changes or truncate; deny those too.
    denied = ["socket", "socketpair", "connect", "bind", "listen", "accept", "accept4", "ptrace", "process_vm_readv", "process_vm_writev", "mount", "umount2", "pivot_root", "chroot", "unshare", "setns", "bpf", "keyctl", "add_key", "request_key", "kexec_load", "kexec_file_load", "reboot", "swapon", "swapoff", "open_by_handle_at", "name_to_handle_at", "fanotify_init", "perf_event_open", "userfaultfd", "io_uring_setup", "io_uring_enter", "io_uring_register", "clone", "clone3", "fork", "vfork", "kill", "tkill", "tgkill", "pidfd_open", "pidfd_getfd", "truncate", "truncate64", "chmod", "fchmod", "fchmodat", "chown", "fchown", "lchown", "fchownat", "link", "linkat", "symlink", "symlinkat", "rename", "renameat", "renameat2", "setxattr", "lsetxattr", "fsetxattr", "removexattr", "lremovexattr", "fremovexattr", "utime", "utimes", "futimesat", "utimensat", "setuid", "setgid", "setreuid", "setregid", "setresuid", "setresgid", "setgroups", "capset", "quotactl", "acct", "lookup_dcookie", "syslog", "iopl", "ioperm", "modify_ldt", "memfd_create"]
    try:
        for name in denied:
            number = seccomp.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and seccomp.seccomp_rule_add(ctx, 0x00050000 | errno.EPERM, number, 0) != 0:
                raise SandboxUnavailable(f"Cannot deny syscall {name}")
        if seccomp.seccomp_load(ctx) != 0:
            raise SandboxUnavailable("seccomp_load failed")
    finally:
        seccomp.seccomp_release(ctx)


class LinuxSandbox:
    """Root controller launches isolated UIDs; never silently runs without guards.

    ``runtime_read_paths`` is a trusted deployment allowlist, not a Task/Harness
    option. Prefer a dedicated clean Python runtime; do not grant project/data roots.
    """

    def __init__(self, python_executable: str | None = None, *, runtime_read_paths: list[str] | None = None,
                 scratch_root: str | Path | None = None, limits: SandboxLimits | None = None):
        self.python = Path(python_executable or "/usr/bin/python3").resolve()
        self.limits = limits or SandboxLimits()
        self.scratch_root = Path(scratch_root) if scratch_root else None
        self.runtime_read_paths = [Path(p).resolve() for p in (runtime_read_paths or [])]
        if not runtime_read_paths:
            self.runtime_read_paths = [self.python]
            self.runtime_read_paths.extend(Path(p).resolve() for p in ("/lib", "/lib64", "/usr/lib", "/usr/lib64") if Path(p).exists())
        # Dynamic loader cache and harmless device reads are individual grants.
        self.runtime_read_paths.extend(Path(p) for p in ("/etc/ld.so.cache", "/dev/null", "/dev/urandom", "/dev/random") if Path(p).exists())
        forbidden = {Path(p).resolve() for p in ("/", "/root", "/root/data", "/home", "/tmp", "/proc", "/sys", "/dev")}
        if any(path in forbidden for path in self.runtime_read_paths):
            raise ValueError("Unsafe broad sandbox read grant")

    def validate(self) -> dict:
        evidence = probe_isolation()
        if not evidence["available"]:
            raise SandboxUnavailable(str(evidence))
        if not self.python.is_file() or any(not p.exists() for p in self.runtime_read_paths):
            raise SandboxUnavailable("Configured runtime path missing")
        return evidence

    def _isolation_hook(self, scratch: Path):
        def isolate():
            try:
                import resource
                os.setsid()
                uid = 200000 + os.getpid()
                os.chown(scratch, uid, uid)
                for child in scratch.iterdir():
                    os.chown(child, uid, uid)
                os.setgroups([])
                os.setgid(uid)
                os.setuid(uid)
                os.umask(0o077)
                resource.setrlimit(resource.RLIMIT_CPU, (self.limits.cpu_seconds, self.limits.cpu_seconds + 1))
                resource.setrlimit(resource.RLIMIT_FSIZE, (self.limits.file_bytes, self.limits.file_bytes))
                resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
                resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                _restrict_syscalls()
                _restrict_filesystem(self.runtime_read_paths, scratch)
                # The fork temporarily inherits the controller's virtual memory
                # (NumPy/Torch can reserve many GB). Load all guard libraries
                # before imposing the candidate limit; exec then replaces it
                # with the small clean Python runtime under this same hard cap.
                resource.setrlimit(resource.RLIMIT_AS, (self.limits.memory_bytes, self.limits.memory_bytes))
            except BaseException as exc:
                os.write(2, f"isolation_setup: {type(exc).__name__}: {exc}\n".encode())
                raise
        return isolate

    def _environment(self, scratch: Path, seed: int) -> dict:
        return {"PATH": str(self.python.parent), "HOME": str(scratch), "TMPDIR": str(scratch),
                "LANG": "C.UTF-8", "PYTHONHASHSEED": str(seed % 4294967296)}

    def session(self, code: str, *, extra_files: dict[str, str] | None = None, seed: int = 0) -> SandboxSession:
        return SandboxSession(self, code, extra_files=extra_files, seed=seed)

    def run(self, code: str, stdin: str = "", *, extra_files: dict[str, str] | None = None,
            seed: int = 0) -> SandboxResult:
        self.validate()
        if self.scratch_root:
            self.scratch_root.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="rsi-candidate-", dir=self.scratch_root) as directory:
            scratch = Path(directory)
            (scratch / "candidate.py").write_text(code, encoding="utf-8")
            for name, text in (extra_files or {}).items():
                if Path(name).name != name or name in {"candidate.py", "stdout", "stderr"}:
                    raise ValueError("Invalid sandbox extra file name")
                (scratch / name).write_text(text, encoding="utf-8")

            with (scratch / "stdout").open("w+b") as out, (scratch / "stderr").open("w+b") as err:
                try:
                    process = subprocess.Popen([str(self.python), "-s", "-B", "candidate.py"],
                        cwd=scratch, stdin=subprocess.PIPE, stdout=out, stderr=err, close_fds=True,
                        env=self._environment(scratch, seed), preexec_fn=self._isolation_hook(scratch))
                except (OSError, subprocess.SubprocessError) as exc:
                    err.seek(0)
                    diagnostic = err.read(4096).decode("utf-8", "replace")
                    raise SandboxUnavailable(f"Sandbox launch failed: {type(exc).__name__}: {exc}; {diagnostic}") from exc
                timed_out = False
                try:
                    process.communicate(stdin.encode(), timeout=self.limits.wall_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()
                out.seek(0)
                err.seek(0)
                stdout, stderr = out.read(self.limits.output_bytes + 1), err.read(self.limits.output_bytes + 1)
                truncated = len(stdout) > self.limits.output_bytes or len(stderr) > self.limits.output_bytes
                return SandboxResult(stdout[:self.limits.output_bytes].decode("utf-8", "replace"),
                    stderr[:self.limits.output_bytes].decode("utf-8", "replace"), process.returncode, timed_out,
                    time.monotonic() - start, truncated)


class SandboxSession:
    """One bounded JSON-lines process per environment rollout; no state replay."""
    def __init__(self, runner: LinuxSandbox, code: str, *, extra_files: dict[str, str] | None, seed: int):
        runner.validate()
        self.runner = runner
        self.process = None
        self.stderr = None
        if runner.scratch_root:
            runner.scratch_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="rsi-environment-", dir=runner.scratch_root)
        self.scratch = Path(self.temp.name)
        try:
            (self.scratch / "candidate.py").write_text(code, encoding="utf-8")
            for name, value in (extra_files or {}).items():
                if Path(name).name != name or name in {"candidate.py", "stderr"}:
                    raise ValueError("Invalid sandbox extra file name")
                (self.scratch / name).write_text(value, encoding="utf-8")
            self.stderr = (self.scratch / "stderr").open("w+b")
            self.process = subprocess.Popen([str(runner.python), "-s", "-B", "-u", "candidate.py"],
                cwd=self.scratch, env=runner._environment(self.scratch, seed), close_fds=True,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, bufsize=0,
                preexec_fn=runner._isolation_hook(self.scratch))
        except (OSError, subprocess.SubprocessError) as exc:
            diagnostic = ""
            if self.stderr:
                self.stderr.seek(0)
                diagnostic = self.stderr.read(4096).decode("utf-8", "replace")
            self.close()
            raise SandboxUnavailable(f"Environment sandbox launch failed: {exc}; {diagnostic}") from exc

    def request(self, payload: dict) -> dict:
        if not self.process or self.process.poll() is not None:
            raise SandboxUnavailable("Environment sandbox is not running")
        deadline = time.monotonic() + self.runner.limits.wall_seconds
        data = memoryview((json.dumps(payload, allow_nan=False) + "\n").encode())
        # Bound both input writes and output reads; a stuck worker cannot block the controller.
        try:
            while data:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [self.process.stdin], [], remaining)[1]:
                    raise TimeoutError("environment request write timeout")
                written = os.write(self.process.stdin.fileno(), data[:4096])
                data = data[written:]
            output = bytearray()
            while b"\n" not in output:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                    raise TimeoutError("environment response timeout")
                chunk = os.read(self.process.stdout.fileno(), 4096)
                if not chunk:
                    raise OSError("environment worker ended without response")
                output.extend(chunk)
                if len(output) > self.runner.limits.output_bytes:
                    raise ValueError("environment response exceeds output limit")
            line, extra = output.split(b"\n", 1)
            if extra:
                raise ValueError("unsolicited environment protocol output")
            response = json.loads(line)
            if not isinstance(response, dict):
                raise ValueError("environment response must be an object")
            return response
        except (OSError, TimeoutError, ValueError) as exc:
            diagnostic = ""
            if self.stderr:
                self.stderr.seek(0)
                diagnostic = self.stderr.read(4096).decode("utf-8", "replace")
            self.close()
            raise SandboxUnavailable(f"{exc}; worker_stderr={diagnostic}") from exc

    def close(self):
        if self.process:
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None
        if self.stderr:
            self.stderr.close()
            self.stderr = None
        self.temp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
