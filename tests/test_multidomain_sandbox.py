import json
import sys

import pytest

from sia.task_meta.data import TaskRecord
from sia.task_meta.environments import TACOAdapter
from sia.task_meta.sandbox import LinuxSandbox, SandboxLimits, SandboxUnavailable, probe_isolation


def test_sandbox_fail_closed_when_platform_unavailable():
    if sys.platform == "linux" and probe_isolation()["available"]:
        pytest.skip("This platform supports the real sandbox")
    with pytest.raises(SandboxUnavailable):
        LinuxSandbox().run("print('must not run')")


def test_sandbox_refuses_broad_runtime_mount():
    with pytest.raises(ValueError, match="broad"):
        LinuxSandbox(runtime_read_paths=["/"])


@pytest.mark.skipif(sys.platform != "linux", reason="Real Linux Landlock/seccomp integration")
def test_real_candidate_isolation_and_grader_boundary(tmp_path):
    if not probe_isolation()["available"]:
        pytest.skip("Kernel Landlock/seccomp/root credential drop unavailable")
    secret = tmp_path / "host_secret.txt"
    secret.write_text("NEVER_READ_THIS")
    runner = LinuxSandbox()
    probe = f'''import json,os,socket
result={{"uid":os.getuid(),"env_secret":os.getenv("OPENROUTER_API_KEY")}}
for name,call in [("read_host",lambda:open({str(secret)!r}).read()),("network",lambda:socket.socket()),("fork",lambda:os.fork())]:
    try: call(); result[name]="allowed"
    except Exception: result[name]="denied"
print(json.dumps(result))
'''
    result = runner.run(probe)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["uid"] != 0
    assert observed["env_secret"] is None
    assert all(observed[key] == "denied" for key in ("read_host", "network", "fork"))
    adapter = TACOAdapter(runner)
    adapter.reset(TaskRecord("fixture", "code", "fixture", "evolve_train", "Add integers", {"tests": json.dumps({"inputs": ["1 2", "7 9"], "outputs": ["3", "16"]})}), "r0", 42)
    verdict = adapter.evaluate("print(sum(map(int, input().split())))")
    assert verdict.reward == 1.0 and verdict.verification["full_verifier"]
    assert LinuxSandbox(limits=SandboxLimits(wall_seconds=0.2)).run("while True: pass").timed_out


@pytest.mark.skipif(sys.platform != "linux", reason="Real Linux Landlock/seccomp integration")
def test_real_environment_session_preserves_generated_state_once():
    if not probe_isolation()["available"]:
        pytest.skip("Kernel Landlock/seccomp/root credential drop unavailable")
    worker = '''import json,sys,uuid
state = {"token":str(uuid.uuid4()), "count":0}
for line in sys.stdin:
    request=json.loads(line)
    if request["operation"]=="increment": state["count"]+=1
    print(json.dumps(state),flush=True)
'''
    runner = LinuxSandbox()
    with runner.session(worker) as first, runner.session(worker) as second:
        before = first.request({"operation": "get"})
        after = first.request({"operation": "increment"})
        independent = second.request({"operation": "get"})
        assert before["token"] == after["token"]
        assert before["count"] == 0 and after["count"] == 1
        assert independent["count"] == 0 and independent["token"] != before["token"]


@pytest.mark.skipif(sys.platform != "linux", reason="Real Linux memory/exec boundary")
def test_candidate_limit_does_not_limit_inherited_controller_reservations():
    if not probe_isolation()["available"]:
        pytest.skip("Kernel isolation unavailable")
    import mmap
    # Reserve address space, without physically allocating it as training would.
    with mmap.mmap(-1, 2 * 1024**3) as reservation:
        assert len(reservation) > SandboxLimits().memory_bytes
        result = LinuxSandbox().run("import resource; print(resource.getrlimit(resource.RLIMIT_AS)[0])")
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) == SandboxLimits().memory_bytes
