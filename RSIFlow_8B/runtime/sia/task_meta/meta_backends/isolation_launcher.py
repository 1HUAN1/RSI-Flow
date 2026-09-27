"""Trusted inner launcher for the mount-free Meta worker sandbox.

The outer worker starts this file as a unique, unprivileged host UID and then
creates fresh user/PID/network/IPC/UTS namespaces.  This launcher is the only
code which runs before the chroot and the irreversible Landlock/seccomp guard.
It never receives provider credentials.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import json
import os
from pathlib import Path
import resource
import socket
import stat
import struct
import subprocess
import sys


BACKEND = "chroot_user_pid_net_ipc_uts_landlock_seccomp_v1"
PR_CAPBSET_DROP = 24
PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECUREBITS = 28
SECBIT_NOROOT = 1 << 0
SECBIT_NOROOT_LOCKED = 1 << 1
SECBIT_NO_SETUID_FIXUP = 1 << 2
SECBIT_NO_SETUID_FIXUP_LOCKED = 1 << 3
SIOCGIFFLAGS = 0x8913
SIOCSIFFLAGS = 0x8914
IFF_UP = 0x1

LANDLOCK_CREATE_RULESET = 444
LANDLOCK_ADD_RULE = 445
LANDLOCK_RESTRICT_SELF = 446
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
LANDLOCK_ACCESS_FS_ALL_V1 = (1 << 13) - 1
LANDLOCK_ACCESS_FS_READ = (
    LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR
)


class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class CapabilityHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class CapabilityData(ctypes.Structure):
    _fields_ = [
        ("effective", ctypes.c_uint32),
        ("permitted", ctypes.c_uint32),
        ("inheritable", ctypes.c_uint32),
    ]


def _namespace_inodes() -> dict[str, int]:
    return {name: os.stat(f"/proc/self/ns/{name}").st_ino for name in ("user", "pid", "net", "ipc", "uts")}


def _enable_private_loopback() -> list[str]:
    names = sorted(name for _, name in socket.if_nameindex())
    if "lo" not in names:
        raise RuntimeError(f"missing_loopback:{names}")
    control = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for name in names:
            if name != "lo":
                flags = struct.unpack("16sh", fcntl.ioctl(
                    control, SIOCGIFFLAGS, struct.pack("16sh", name.encode(), 0)))[1]
                if flags & IFF_UP:
                    raise RuntimeError(f"unexpected_active_network_interface:{name}")
        request = struct.pack("16sh", b"lo", 0)
        flags = struct.unpack("16sh", fcntl.ioctl(control, SIOCGIFFLAGS, request))[1]
        fcntl.ioctl(control, SIOCSIFFLAGS, struct.pack("16sh", b"lo", flags | IFF_UP))
    finally:
        control.close()
    return names


def _drop_capabilities() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    securebits = SECBIT_NOROOT | SECBIT_NOROOT_LOCKED | SECBIT_NO_SETUID_FIXUP | SECBIT_NO_SETUID_FIXUP_LOCKED
    if libc.prctl(PR_SET_SECUREBITS, securebits, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_SECUREBITS")
    for capability in range(64):
        if libc.prctl(PR_CAPBSET_DROP, capability, 0, 0, 0) != 0 and ctypes.get_errno() != errno.EINVAL:
            raise OSError(ctypes.get_errno(), "PR_CAPBSET_DROP")
    header = CapabilityHeader(0x20080522, 0)
    data = (CapabilityData * 2)()
    if libc.capset(ctypes.byref(header), data) != 0:
        raise OSError(ctypes.get_errno(), "capset")
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")


def _capability_state() -> dict[str, list[int]]:
    libc = ctypes.CDLL(None, use_errno=True)
    header = CapabilityHeader(0x20080522, 0)
    data = (CapabilityData * 2)()
    if libc.capget(ctypes.byref(header), data) != 0:
        raise OSError(ctypes.get_errno(), "capget")
    return {
        "effective": [data[0].effective, data[1].effective],
        "permitted": [data[0].permitted, data[1].permitted],
        "inheritable": [data[0].inheritable, data[1].inheritable],
    }


def _access_for(path: Path, access: int) -> int:
    if not path.is_dir():
        access &= ~(
            LANDLOCK_ACCESS_FS_READ_DIR
            | LANDLOCK_ACCESS_FS_REMOVE_DIR
            | LANDLOCK_ACCESS_FS_REMOVE_FILE
            | LANDLOCK_ACCESS_FS_MAKE_CHAR
            | LANDLOCK_ACCESS_FS_MAKE_DIR
            | LANDLOCK_ACCESS_FS_MAKE_REG
            | LANDLOCK_ACCESS_FS_MAKE_SOCK
            | LANDLOCK_ACCESS_FS_MAKE_FIFO
            | LANDLOCK_ACCESS_FS_MAKE_BLOCK
            | LANDLOCK_ACCESS_FS_MAKE_SYM
        )
    return access


def _apply_landlock() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(LANDLOCK_CREATE_RULESET, 0, 0, LANDLOCK_CREATE_RULESET_VERSION)
    if abi < 1:
        raise OSError(ctypes.get_errno(), "landlock_abi")
    ruleset = RulesetAttr(LANDLOCK_ACCESS_FS_ALL_V1)
    descriptor = libc.syscall(LANDLOCK_CREATE_RULESET, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    readable = [
        "/usr", "/bin", "/lib", "/lib64", "/etc/ssl", "/etc/ld.so.cache",
        "/etc/passwd", "/etc/group", "/dev/null", "/dev/urandom", "/dev/random",
        "/bridge.py", "/codex", "/catalog.json", "/schema.json", "/isolation_launcher.py",
        "/transport.sock", "/proc",
    ]
    writable = ["/workspace", "/codex_home", "/tmp", "/home/meta", "/dev/null"]
    try:
        for raw_path, requested in [*((value, LANDLOCK_ACCESS_FS_READ) for value in readable),
                                    *((value, LANDLOCK_ACCESS_FS_ALL_V1) for value in writable)]:
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(raw_path)
            path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathBeneathAttr(_access_for(path, requested), path_fd)
                if libc.syscall(LANDLOCK_ADD_RULE, descriptor, LANDLOCK_RULE_PATH_BENEATH,
                                ctypes.byref(rule), 0) < 0:
                    raise OSError(ctypes.get_errno(), f"landlock_add_rule:{raw_path}")
            finally:
                os.close(path_fd)
        if libc.syscall(LANDLOCK_RESTRICT_SELF, descriptor, 0) < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(descriptor)
    return int(abi)


def _apply_seccomp() -> None:
    library = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libseccomp.so.2", use_errno=True)
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    context = library.seccomp_init(0x7FFF0000)
    if not context:
        raise RuntimeError("seccomp_init")
    denied = [
        "mount", "umount2", "pivot_root", "chroot", "unshare", "setns",
        "ptrace", "process_vm_readv", "process_vm_writev", "bpf", "keyctl",
        "add_key", "request_key", "kexec_load", "kexec_file_load", "reboot",
        "swapon", "swapoff", "open_by_handle_at", "name_to_handle_at",
        "fanotify_init", "perf_event_open", "userfaultfd", "io_uring_setup",
        "io_uring_enter", "io_uring_register", "init_module", "finit_module",
        "delete_module", "mknod", "mknodat", "acct", "quotactl", "lookup_dcookie",
        "syslog", "iopl", "ioperm", "modify_ldt", "setuid", "setgid",
        "setreuid", "setregid", "setresuid", "setresgid", "setgroups", "capset",
    ]
    try:
        for name in denied:
            number = library.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and library.seccomp_rule_add(
                    context, 0x00050000 | errno.EPERM, number, 0) != 0:
                raise RuntimeError(f"seccomp_rule:{name}")
        if library.seccomp_load(context) != 0:
            raise RuntimeError("seccomp_load")
    finally:
        library.seccomp_release(context)


def _set_limits(args: argparse.Namespace) -> None:
    cpu = max(1, args.cpu_seconds)
    resource.setrlimit(resource.RLIMIT_AS, (args.memory_bytes, args.memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (args.file_bytes, args.file_bytes))
    resource.setrlimit(resource.RLIMIT_NPROC, (args.processes, args.processes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))


def _negative_probe(namespaces: dict[str, int], interfaces: list[str], landlock_abi: int) -> dict:
    checks: dict[str, bool] = {}
    try:
        descriptor = os.open("/host_marker", os.O_RDONLY)
    except OSError:
        checks["host_root_hidden"] = True
    else:
        os.close(descriptor)
        checks["host_root_hidden"] = False
    probe_path = Path("/workspace/.isolation_probe")
    probe_path.write_text("ok", encoding="utf-8")
    checks["workspace_write"] = probe_path.read_text(encoding="utf-8") == "ok"
    probe_path.unlink()
    try:
        os.open("/codex", os.O_WRONLY)
    except OSError:
        checks["runtime_write_denied"] = True
    else:
        checks["runtime_write_denied"] = False
    outbound = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    outbound.settimeout(0.1)
    try:
        checks["external_network_denied"] = outbound.connect_ex(("198.51.100.1", 9)) != 0
    finally:
        outbound.close()
    relay = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    relay.settimeout(2)
    try:
        relay.connect("/transport.sock")
        relay.sendall(b"probe")
        checks["relay_reachable"] = relay.recv(16) == b"ok"
    except OSError:
        checks["relay_reachable"] = False
    finally:
        relay.close()
    libc = ctypes.CDLL(None, use_errno=True)
    ctypes.set_errno(0)
    checks["mount_denied"] = libc.mount(None, b"/", None, 0, None) != 0 and ctypes.get_errno() == errno.EPERM
    ctypes.set_errno(0)
    checks["setns_denied"] = libc.setns(-1, 0) != 0 and ctypes.get_errno() == errno.EPERM
    ctypes.set_errno(0)
    checks["ptrace_denied"] = libc.ptrace(0, 0, None, None) != 0 and ctypes.get_errno() == errno.EPERM
    version = subprocess.run(["/codex", "--version"], capture_output=True, text=True, timeout=10,
                             env={"PATH": "/usr/bin:/bin", "HOME": "/home/meta", "CODEX_HOME": "/codex_home"})
    checks["codex_exec"] = version.returncode == 0 and version.stdout.strip() == "codex-cli 0.153.4"
    caps = _capability_state()
    checks["capabilities_dropped"] = not any(value for values in caps.values() for value in values)
    return {
        "backend": BACKEND,
        "checks": checks,
        "namespaces": namespaces,
        "interfaces": interfaces,
        "landlock_abi": landlock_abi,
        "capabilities": caps,
        "uid": os.getuid(),
        "gid": os.getgid(),
        "ready": all(checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--host-namespaces", required=True)
    parser.add_argument("--memory-bytes", required=True, type=int)
    parser.add_argument("--cpu-seconds", required=True, type=int)
    parser.add_argument("--file-bytes", required=True, type=int)
    parser.add_argument("--processes", required=True, type=int)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if os.getuid() != 0 or os.getgid() != 0:
        raise RuntimeError("namespace_root_required")
    host_namespaces = json.loads(args.host_namespaces)
    namespaces = _namespace_inodes()
    if set(namespaces) != set(host_namespaces) or any(
            namespaces[name] == host_namespaces[name] for name in namespaces):
        raise RuntimeError("namespace_isolation_missing")
    interfaces = _enable_private_loopback()
    root = Path(args.root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise RuntimeError("invalid_chroot")
    # No descriptor other than stdio crosses the chroot/exec boundary.
    os.closerange(3, 65536)
    os.chroot(root)
    os.chdir("/workspace")
    _set_limits(args)
    _drop_capabilities()
    landlock_abi = _apply_landlock()
    _apply_seccomp()
    if args.probe:
        result = _negative_probe(namespaces, interfaces, landlock_abi)
        sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
        return 0 if result["ready"] else 1
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise RuntimeError("missing_command")
    os.execve(command[0], command, {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/meta",
        "CODEX_HOME": "/codex_home",
        "LANG": "C.UTF-8",
        "TMPDIR": "/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
