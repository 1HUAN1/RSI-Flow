"""Build and operate the minimal mount-free chroot used by the Meta worker."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import stat
import subprocess
import tempfile
import threading


BACKEND = "chroot_user_pid_net_ipc_uts_landlock_seccomp_v1"
MANIFEST = ".isolation-manifest.json"
MUTABLE_TOP_LEVEL = {"workspace", "codex_home", "tmp", "home", "schema.json", "transport.sock"}
COMMANDS = (
    "bash", "dash", "sh", "env", "python3", "python3.12", "cat", "chmod", "cp", "cut",
    "date", "dirname", "find", "grep", "head", "ls", "mkdir", "mv", "pwd", "readlink",
    "rm", "sed", "sha256sum", "sort", "stat", "tail", "test", "touch", "tr", "wc",
    "git", "perl", "awk", "basename", "cmp", "diff", "ln", "patch", "printf",
    "sleep", "tee", "which", "xargs",
)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_absolute(path: str | Path, *, must_exist: bool = True) -> Path:
    value = Path(path).absolute()
    if not value.is_absolute():
        raise ValueError("absolute_path_required")
    existing = value if value.exists() else value.parent
    for part in (existing, *existing.parents):
        if part.is_symlink():
            raise ValueError("symlink_path")
    if must_exist and not value.exists():
        raise FileNotFoundError(value)
    return value


def regular_digest(path: Path, limit: int = 512 * 1024 * 1024) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError(f"invalid_runtime_file:{path}")
        hasher = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(1024 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest()
    finally:
        os.close(descriptor)


class IsolationRuntime:
    def __init__(self, root: str | Path, *, launcher: str | Path, codex: str | Path,
                 catalog: str | Path, bridge: str | Path, setpriv: str | Path,
                 unshare: str | Path, python: str | Path = "/usr/bin/python3"):
        self.root = checked_absolute(root, must_exist=False)
        if self.root.parent != Path("/tmp") or not self.root.name.startswith("rsiflow_meta_"):
            raise ValueError("runtime_root_must_be_dedicated_tmp_path")
        self.launcher = checked_absolute(launcher)
        self.codex = checked_absolute(codex)
        self.catalog = checked_absolute(catalog)
        self.bridge = checked_absolute(bridge)
        self.setpriv = checked_absolute(setpriv)
        self.unshare = checked_absolute(unshare)
        self.python = Path(python).resolve(strict=True)
        for path in (self.launcher, self.codex, self.catalog, self.bridge,
                     self.setpriv, self.unshare, self.python):
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"runtime_component_must_be_regular:{path}")
        self.host_namespaces = {
            name: os.stat(f"/proc/self/ns/{name}").st_ino
            for name in ("user", "pid", "net", "ipc", "uts")
        }
        self.manifest_hash = self.prepare()

    @property
    def launcher_in_host(self) -> Path:
        return self.root / "isolation_launcher.py"

    def _destination(self, source: Path, base: Path) -> Path:
        return base / source.relative_to("/")

    def _copy_one(self, source: Path, base: Path, copied: set[Path]) -> None:
        source = source.absolute()
        if source in copied:
            return
        info = source.lstat()
        target = self._destination(source, base)
        target.parent.mkdir(parents=True, exist_ok=True)
        if stat.S_ISLNK(info.st_mode):
            link = os.readlink(source)
            target.symlink_to(link)
            copied.add(source)
            linked = Path(os.path.normpath(link if os.path.isabs(link) else str(source.parent / link)))
            allowed = (Path("/usr"), Path("/lib"), Path("/lib64"), Path("/etc/ssl"),
                       Path("/etc/alternatives"), Path("/etc/python3.12"))
            if not any(linked == root or linked.is_relative_to(root) for root in allowed):
                raise ValueError(f"runtime_symlink_outside_allowlist:{source}")
            try:
                linked_info = linked.lstat()
            except FileNotFoundError:
                return
            if stat.S_ISDIR(linked_info.st_mode):
                self._copy_tree(linked, base, copied)
            else:
                self._copy_one(linked, base, copied)
            return
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"unsupported_runtime_entry:{source}")
        shutil.copy2(source, target, follow_symlinks=False)
        os.chown(target, 0, 0, follow_symlinks=False)
        os.chmod(target, stat.S_IMODE(info.st_mode) & ~0o6022, follow_symlinks=False)
        copied.add(source)

    def _copy_tree(self, source: Path, base: Path, copied: set[Path]) -> list[Path]:
        binaries: list[Path] = []
        resolved_source = source.resolve(strict=True)
        if resolved_source in copied:
            return binaries
        copied.add(resolved_source)
        for parent, directories, filenames in os.walk(source, followlinks=False):
            parent_path = Path(parent)
            relative = parent_path.relative_to("/")
            destination_parent = base / relative
            destination_parent.mkdir(parents=True, exist_ok=True)
            os.chown(destination_parent, 0, 0)
            os.chmod(destination_parent, 0o755)
            for name in list(directories):
                path = parent_path / name
                if path.is_symlink():
                    self._copy_one(path, base, copied)
                    directories.remove(name)
            for name in filenames:
                path = parent_path / name
                self._copy_one(path, base, copied)
                try:
                    if not path.is_symlink() and path.stat().st_mode & 0o111:
                        binaries.append(path)
                except FileNotFoundError:
                    raise ValueError(f"runtime_tree_changed:{path}")
        return binaries

    @staticmethod
    def _ldd_paths(path: Path) -> set[Path]:
        result = subprocess.run(["/usr/bin/ldd", str(path)], stdin=subprocess.DEVNULL,
                                capture_output=True, timeout=20, check=False,
                                env={"PATH": "/usr/bin:/bin", "LANG": "C"})
        if result.returncode:
            return set()
        paths = set()
        for line in result.stdout.decode("utf-8", "replace").splitlines():
            match = re.search(r"=>\s+(/\S+)\s+\(", line) or re.match(r"\s*(/\S+)\s+\(", line)
            if match:
                paths.add(Path(match.group(1)))
        return paths

    def _populate(self, base: Path) -> None:
        copied: set[Path] = set()
        binaries: list[Path] = []
        for link, destination in (("bin", "usr/bin"), ("lib", "usr/lib"), ("lib64", "usr/lib64")):
            (base / destination).mkdir(parents=True, exist_ok=True)
            (base / link).symlink_to(destination)
        for name in COMMANDS:
            source = Path("/usr/bin") / name
            if source.exists() or source.is_symlink():
                self._copy_one(source, base, copied)
                resolved = source.resolve()
                if resolved.is_file():
                    binaries.append(resolved)
        (base / "usr/bin/python").symlink_to("python3")
        binaries.extend(self._copy_tree(Path("/usr/lib/python3.12"), base, copied))
        for optional in (
            Path("/usr/lib/git-core"), Path("/usr/share/git-core"),
            Path("/usr/lib/x86_64-linux-gnu/perl-base"), Path("/usr/share/perl"),
            Path("/usr/share/perl5"), Path("/usr/share/terminfo"),
            Path("/usr/lib/locale/C.utf8"), Path("/etc/ssl"),
        ):
            if optional.is_dir():
                binaries.extend(self._copy_tree(optional, base, copied))
        for path in (Path("/etc/ld.so.cache"), Path("/usr/lib/x86_64-linux-gnu/libseccomp.so.2"),
                     Path("/usr/lib/x86_64-linux-gnu/libnss_files.so.2"),
                     Path("/usr/lib/x86_64-linux-gnu/libnss_dns.so.2")):
            if path.exists() or path.is_symlink():
                self._copy_one(path, base, copied)
                resolved = path.resolve()
                if resolved.is_file():
                    binaries.append(resolved)
        pending = list(dict.fromkeys(binaries))
        inspected: set[Path] = set()
        while pending:
            binary = pending.pop()
            if binary in inspected or not binary.is_file():
                continue
            inspected.add(binary)
            for dependency in self._ldd_paths(binary):
                resolved = dependency.resolve()
                self._copy_one(dependency, base, copied)
                if resolved not in inspected:
                    pending.append(resolved)
        for source, name, mode in (
            (self.codex, "codex", 0o555), (self.catalog, "catalog.json", 0o444),
            (self.bridge, "bridge.py", 0o444), (self.launcher, "isolation_launcher.py", 0o444),
        ):
            target = base / name
            shutil.copyfile(source, target)
            os.chown(target, 0, 0)
            os.chmod(target, mode)
        # Codex resolves its own executable through this Linux convention.
        # This is a fixed alias inside the jail, not the host proc filesystem.
        (base / "proc/self").mkdir(parents=True)
        (base / "proc/self/exe").symlink_to("/codex")
        etc = base / "etc"
        etc.mkdir(exist_ok=True)
        (etc / "passwd").write_text("root:x:0:0:Meta sandbox:/home/meta:/bin/sh\n", encoding="utf-8")
        (etc / "group").write_text("root:x:0:\n", encoding="utf-8")
        for path in (etc / "passwd", etc / "group"):
            os.chown(path, 0, 0)
            os.chmod(path, 0o444)
        dev = base / "dev"
        dev.mkdir(mode=0o755)
        for name, source in (("null", Path("/dev/null")), ("urandom", Path("/dev/urandom")),
                             ("random", Path("/dev/random"))):
            info = source.stat()
            os.mknod(dev / name, stat.S_IFCHR | 0o666, info.st_rdev)
            os.chmod(dev / name, 0o666)
        for path in (base / "workspace", base / "codex_home", base / "tmp", base / "home/meta"):
            path.mkdir(parents=True, exist_ok=True)
        os.chmod(base, 0o711)

    @staticmethod
    def _manifest_entries(base: Path) -> list[dict]:
        entries: list[dict] = []
        for parent, directories, filenames in os.walk(base, followlinks=False):
            parent_path = Path(parent)
            relative_parent = parent_path.relative_to(base)
            if relative_parent.parts and relative_parent.parts[0] in MUTABLE_TOP_LEVEL:
                directories[:] = []
                continue
            directories[:] = [name for name in directories if not (
                (relative_parent / name).parts and (relative_parent / name).parts[0] in MUTABLE_TOP_LEVEL)]
            for name in sorted(directories + filenames):
                path = parent_path / name
                relative = path.relative_to(base).as_posix()
                if relative == MANIFEST or relative.split("/", 1)[0] in MUTABLE_TOP_LEVEL:
                    continue
                info = path.lstat()
                record = {"path": relative, "mode": stat.S_IMODE(info.st_mode), "uid": info.st_uid,
                          "gid": info.st_gid}
                if stat.S_ISDIR(info.st_mode):
                    record["type"] = "dir"
                elif stat.S_ISREG(info.st_mode):
                    record.update(type="file", size=info.st_size, sha256=regular_digest(path))
                elif stat.S_ISLNK(info.st_mode):
                    record.update(type="symlink", target=os.readlink(path))
                elif stat.S_ISCHR(info.st_mode):
                    record.update(type="char", rdev=info.st_rdev)
                else:
                    raise ValueError(f"invalid_runtime_entry:{relative}")
                entries.append(record)
        return sorted(entries, key=lambda item: item["path"])

    def prepare(self) -> str:
        if self.root.exists():
            return self.validate()
        self.root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=self.root.name + ".build-", dir=self.root.parent))
        try:
            self._populate(temporary)
            entries = self._manifest_entries(temporary)
            manifest_hash = digest(canonical(entries))
            manifest = {"version": 1, "backend": BACKEND, "entries": entries,
                        "runtime_manifest_sha256": manifest_hash}
            manifest_path = temporary / MANIFEST
            manifest_path.write_bytes(canonical(manifest))
            os.chown(manifest_path, 0, 0)
            os.chmod(manifest_path, 0o444)
            os.rename(temporary, self.root)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return self.validate()

    def validate(self) -> str:
        info = self.root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_gid != 0 or info.st_mode & 0o022:
            raise ValueError("unsafe_runtime_root")
        manifest_path = self.root / MANIFEST
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("runtime_manifest_missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = self._manifest_entries(self.root)
        manifest_hash = digest(canonical(entries))
        if (set(manifest) != {"version", "backend", "entries", "runtime_manifest_sha256"}
                or manifest["version"] != 1 or manifest["backend"] != BACKEND
                or manifest["entries"] != entries or manifest["runtime_manifest_sha256"] != manifest_hash):
            raise ValueError("runtime_manifest_mismatch")
        for source, name in ((self.codex, "codex"), (self.catalog, "catalog.json"),
                             (self.bridge, "bridge.py"), (self.launcher, "isolation_launcher.py")):
            if regular_digest(source) != regular_digest(self.root / name):
                raise ValueError(f"runtime_source_changed:{name}")
        return manifest_hash

    @staticmethod
    def uid_for(request_id: str) -> int:
        return 200000 + int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16) % 800000

    def _clear_mutable(self) -> None:
        for name in ("workspace", "codex_home", "tmp", "home"):
            path = self.root / name
            if path.exists() or path.is_symlink():
                if path.is_symlink() or not path.is_dir():
                    raise ValueError(f"unsafe_mutable_path:{name}")
                shutil.rmtree(path)
        for name in ("schema.json", "transport.sock"):
            path = self.root / name
            if path.exists() or path.is_symlink():
                if name == "transport.sock":
                    raise ValueError("stale_transport_socket")
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"unsafe_mutable_path:{name}")
                path.unlink()

    def stage(self, files: dict[str, bytes], uid: int) -> None:
        self.validate()
        self._clear_mutable()
        workspace = self.root / "workspace"
        codex_home = self.root / "codex_home"
        tmp = self.root / "tmp"
        home = self.root / "home"
        meta_home = home / "meta"
        for path, mode, owner in ((workspace, 0o1777, 0), (codex_home, 0o1777, 0),
                                  (tmp, 0o1777, 0), (home, 0o755, 0), (meta_home, 0o700, uid)):
            path.mkdir(parents=True, exist_ok=True)
            os.chown(path, owner, owner)
            os.chmod(path, mode)
        for name, data in files.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                raise ValueError("invalid_stage_path")
            if relative.parts[0] == "workspace":
                target = self.root.joinpath(*relative.parts)
            elif name == "codex_home/config.toml":
                target = self.root / "codex_home/config.toml"
            elif name == "schema.json":
                target = self.root / "schema.json"
            else:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            protected = name.startswith('workspace/meta_input/') or name in {'workspace/AGENTS.md', 'codex_home/config.toml', 'schema.json'}
            for parent in target.parents:
                if parent in {self.root, workspace, codex_home}:
                    break
                os.chown(parent, 0 if protected else uid, 0 if protected else uid)
                os.chmod(parent, 0o755 if protected else 0o700)
            with target.open('xb') as stream:
                if isinstance(data, Path):
                    source = checked_absolute(data)
                    with source.open('rb') as incoming:
                        shutil.copyfileobj(incoming, stream, length=1024**2)
                else:
                    stream.write(data)
            os.chown(target, 0 if protected else uid, 0 if protected else uid)
            os.chmod(target, 0o444 if protected else 0o600)
        for required in (workspace / "AGENTS.md", codex_home / "config.toml", self.root / "schema.json"):
            if not required.is_file() or required.is_symlink():
                raise ValueError("missing_staged_control_file")

    def cleanup(self) -> None:
        self._clear_mutable()

    def link_relay(self, relay: Path, uid: int) -> dict:
        relay = checked_absolute(relay)
        source = relay.lstat()
        if not stat.S_ISSOCK(source.st_mode):
            raise ValueError("relay_not_socket")
        target = self.root / "transport.sock"
        if target.exists() or target.is_symlink():
            raise ValueError("transport_target_exists")
        identity = {"dev": source.st_dev, "ino": source.st_ino, "uid": source.st_uid,
                    "gid": source.st_gid, "mode": stat.S_IMODE(source.st_mode), "path": str(relay)}
        os.link(relay, target, follow_symlinks=False)
        try:
            linked = target.lstat()
            if not stat.S_ISSOCK(linked.st_mode) or (linked.st_dev, linked.st_ino) != (source.st_dev, source.st_ino):
                raise ValueError("relay_link_identity_mismatch")
            os.chown(target, uid, uid, follow_symlinks=False)
            os.chmod(target, 0o600)
            return identity
        except BaseException:
            linked = target.lstat()
            if (linked.st_dev, linked.st_ino) == (source.st_dev, source.st_ino):
                os.chown(target, source.st_uid, source.st_gid, follow_symlinks=False)
                os.chmod(target, identity["mode"])
            target.unlink()
            raise

    def unlink_relay(self, identity: dict) -> None:
        target = self.root / "transport.sock"
        if target.exists() or target.is_symlink():
            info = target.lstat()
            if not stat.S_ISSOCK(info.st_mode) or (info.st_dev, info.st_ino) != (identity["dev"], identity["ino"]):
                raise ValueError("transport_identity_changed")
            os.chown(target, identity["uid"], identity["gid"], follow_symlinks=False)
            os.chmod(target, identity["mode"])
            target.unlink()
        source = Path(identity["path"])
        try:
            info = source.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(info.st_mode) or (info.st_dev, info.st_ino) != (identity["dev"], identity["ino"]):
            raise ValueError("relay_identity_changed")
        if (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) != (
                identity["uid"], identity["gid"], identity["mode"]):
            raise ValueError("relay_metadata_restore_failed")

    def command(self, uid: int, *, memory_bytes: int, cpu_seconds: int, file_bytes: int,
                processes: int, probe: bool = False, command: list[str] | None = None) -> list[str]:
        result = [str(self.setpriv), "--reuid", str(uid), "--regid", str(uid), "--clear-groups",
                  str(self.unshare), "--user", "--map-root-user", "--pid", "--fork",
                  "--kill-child=KILL", "--net", "--ipc", "--uts", str(self.python), "-I", "-B",
                  str(self.launcher_in_host), "--root", str(self.root), "--host-namespaces",
                  json.dumps(self.host_namespaces, sort_keys=True), "--memory-bytes", str(memory_bytes),
                  "--cpu-seconds", str(cpu_seconds), "--file-bytes", str(file_bytes),
                  "--processes", str(processes)]
        if probe:
            result.append("--probe")
        elif command:
            result += ["--", *command]
        else:
            raise ValueError("sandbox_command_missing")
        return result

    def probe(self) -> dict:
        uid = self.uid_for("isolation-health-probe")
        self.stage({"workspace/AGENTS.md": b"probe\n", "codex_home/config.toml": b"",
                    "schema.json": b"{}"}, uid)
        source_path = self.root.parent / f".{self.root.name}-probe-{os.getpid()}.sock"
        if source_path.exists() or source_path.is_symlink():
            raise ValueError("probe_socket_exists")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(source_path))
        server.listen(1)
        errors: list[str] = []

        def serve():
            try:
                connection, _ = server.accept()
                with connection:
                    if connection.recv(16) != b"probe":
                        raise ValueError("probe_relay_payload")
                    connection.sendall(b"ok")
            except BaseException as exc:
                errors.append(type(exc).__name__)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        identity = None
        try:
            identity = self.link_relay(source_path, uid)
            result = subprocess.run(self.command(uid, memory_bytes=1024 * 1024 * 1024,
                                    cpu_seconds=15, file_bytes=1024 * 1024, processes=16, probe=True),
                                    cwd="/tmp", env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                                    stdin=subprocess.DEVNULL, capture_output=True, timeout=20, check=False)
            thread.join(timeout=3)
            try:
                value = json.loads(result.stdout)
            except (ValueError, TypeError):
                value = {}
            value.update(returncode=result.returncode,
                         stderr=result.stderr[:2048].decode("utf-8", "replace"), relay_errors=errors)
            value["ready"] = bool(value.get("ready") and result.returncode == 0 and not errors)
            return value
        finally:
            server.close()
            if identity is not None:
                self.unlink_relay(identity)
            if source_path.exists() or source_path.is_symlink():
                source_path.unlink()
            self._clear_mutable()

    def identity(self) -> dict:
        manifest_hash = self.validate()
        components = {
            "backend": BACKEND,
            "launcher_sha256": regular_digest(self.launcher),
            "setpriv_sha256": regular_digest(self.setpriv),
            "unshare_sha256": regular_digest(self.unshare),
            "python_sha256": regular_digest(self.python),
            "runtime_manifest_sha256": manifest_hash,
        }
        components["isolation_sha256"] = digest(canonical(components))
        return components

