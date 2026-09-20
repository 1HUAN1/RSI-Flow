"""One disposable process per upstream HarnessForge validation attempt.

The controller never imports generated code. The child validates a private copy,
with only scratch writable and no provider credentials. Repairs remain in the
controller's existing production loop; this worker never generates a candidate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace


class ValidationResult(SimpleNamespace):
    def to_dict(self):
        return dict(vars(self))


def validate_candidate(candidate_dir: Path, *, timeout: float = 300) -> ValidationResult:
    from sia.task_meta.harnessforge_manifest import HarnessBundleManifest

    candidate_dir = Path(candidate_dir).resolve()
    manifest = HarnessBundleManifest.from_directory(candidate_dir)
    # Keep diagnostics with the workflow, not in the home directory. Each attempt
    # gets a fresh interpreter and copy, including after a failed-build repair.
    with tempfile.TemporaryDirectory(prefix=".validation-", dir=candidate_dir.parent) as raw:
        scratch = Path(raw)
        project = scratch / "project"
        copied = project / "generated_harnesses" / "rounds" / candidate_dir.parent.name / candidate_dir.name
        manifest.materialize(copied)
        for package in (copied.parent, copied.parent.parent, copied.parent.parent.parent):
            (package / "__init__.py").touch()
        report_path = scratch / "report.json"
        env = {
            "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
            "LANG": "C.UTF-8", "HOME": str(scratch), "TMPDIR": str(scratch),
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
        }
        with (scratch / "stdout.log").open("wb") as out, (scratch / "stderr.log").open("wb") as err:
            process = subprocess.Popen(
                [sys.executable, "-I", "-B", "-X", "faulthandler", str(Path(__file__).resolve()),
                 str(copied), str(project), str(report_path)],
                cwd=scratch, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                close_fds=True, start_new_session=True,
            )
            timed_out = False
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Also remove descendants if a candidate left background work.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        if timed_out:
            return ValidationResult(
                candidate_name=candidate_dir.name, candidate_path=str(candidate_dir),
                import_name="", verdict="failed_build", checks=[],
                errors=[f"Candidate validation exceeded {timeout:g} seconds"], traceback="",
            )
        if process.returncode != 0 or not report_path.is_file():
            with (scratch / "stderr.log").open("rb") as stream:
                diagnostic = stream.read(8192).decode("utf-8", "replace")
            if (scratch / "ready").is_file():
                return ValidationResult(
                    candidate_name=candidate_dir.name, candidate_path=str(candidate_dir),
                    import_name="", verdict="failed_build", checks=[],
                    errors=[f"Candidate validation worker exited without a report (exit={process.returncode})"],
                    traceback=diagnostic,
                )
            raise RuntimeError(
                f"HarnessForge validation worker failed (exit={process.returncode}): {diagnostic}"
            )
        if report_path.stat().st_size > 8 * 1024 * 1024:
            raise RuntimeError("HarnessForge validation report exceeds 8 MiB")
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if "worker_error" in payload:
            raise RuntimeError("HarnessForge validation isolation failed: " + payload["worker_error"])
        if payload.get("verdict") not in {
            "passed", "failed_static", "failed_import", "failed_build", "failed_environment"
        } or not isinstance(payload.get("errors"), list) or not isinstance(payload.get("checks"), list):
            raise RuntimeError("Malformed upstream validation report")
        # Reports must refer to the durable candidate, not a deleted scratch copy.
        encoded = json.dumps(payload, ensure_ascii=False).replace(str(copied), str(candidate_dir))
        return ValidationResult(**json.loads(encoded))


def _worker(candidate: Path, project: Path, report_path: Path) -> None:
    import resource

    runtime = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(runtime))
    from sia.task_meta.harnessforge_production import UPSTREAM_ROOT, upstream_validation_module
    from sia.task_meta.harnessforge_runtime import _install_json_repair_fallback
    from sia.task_meta.sandbox import _restrict_filesystem, _restrict_syscalls

    scratch = report_path.parent
    readable = [runtime, UPSTREAM_ROOT, Path(sys.prefix).resolve()]
    readable.extend(Path(p).resolve() for p in (
        "/lib", "/lib64", "/usr/lib", "/usr/lib64", "/etc/ld.so.cache",
        "/etc/localtime", "/dev/null", "/dev/urandom", "/dev/random",
        "/proc/self", "/proc/cpuinfo", "/proc/meminfo", "/sys/devices/system/cpu",
    ) if Path(p).exists())
    # Guards are installed before any candidate import. Thread creation is needed
    # by native dependencies; host mutation, network and process escape are denied.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    try:
        _restrict_filesystem(readable, scratch, write_paths=(Path("/dev/null"),))
        _restrict_syscalls(allow_threads=True)
    except Exception as exc:
        report_path.write_text(json.dumps({"worker_error": f"{type(exc).__name__}: {exc}"}))
        return
    for source in (UPSTREAM_ROOT, UPSTREAM_ROOT / "harness", UPSTREAM_ROOT / "harness_factory"):
        sys.path.insert(0, str(source))
    _install_json_repair_fallback()
    upstream = upstream_validation_module()
    (scratch / "ready").touch()
    report = upstream.validate_once(candidate, project, "generated_harnesses")
    report_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    _worker(*(Path(value).resolve() for value in sys.argv[1:]))
