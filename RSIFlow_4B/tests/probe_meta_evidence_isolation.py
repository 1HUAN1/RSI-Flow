"""Real local sandbox check, without Codex/API/model invocation."""
import json
from pathlib import Path
import subprocess
import socket
import tempfile

from common import read
from launch import make_pipeline
from sia.task_meta.file_lock import exclusive_lock
from sia.task_meta.meta_backends.contracts import MetaBackendConfig
from sia.task_meta.meta_backends.local_execution import runtime, LOCK, _stage_files, _return_workspace


def main():
    root = Path(__file__).resolve().parents[1]
    config = MetaBackendConfig(**make_pipeline(read(root/'configs/train_180.json'), root/'runtime')['meta'])
    with tempfile.TemporaryDirectory(prefix='rsiflow_evidence_probe_', dir='/tmp') as tmp, exclusive_lock(Path(LOCK)):
        call = Path(tmp)
        work = call/'workspace'; (work/'meta_input/evidence').mkdir(parents=True)
        (call/'codex_home').mkdir()
        (call/'codex_home/config.toml').write_text('model="offline-fixture"')
        (call/'schema.json').write_text('{}')
        (work/'AGENTS.md').write_text('Offline sandbox test')
        (work/'meta_input/evidence/large.json').write_text(json.dumps({'context': 'x' * 17000000}))
        source = '''import json
from pathlib import Path
p=Path('/workspace/meta_input/evidence/large.json')
assert len(json.loads(p.read_text())['context']) == 17000000
denied=False
try:
    p.write_text('corruption')
except PermissionError:
    denied=True
assert denied, 'input was writable'
q=Path('/workspace/.meta_candidate.json')
q.write_text(json.dumps({'read_complete':True,'input_write_denied':denied}))
print(q.read_text())
'''
        instance = runtime(config)
        uid = instance.uid_for('offline-evidence-budget-probe')
        relay = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        relay_path = call/'offline-relay.sock'
        relay.bind(str(relay_path))
        relay_identity = None
        try:
            instance.stage(_stage_files(call, config.input_budget, 1024), uid)
            relay_identity = instance.link_relay(relay_path, uid)
            command = instance.command(uid, memory_bytes=1024**3, cpu_seconds=60,
                file_bytes=64*1024**2, processes=64,
                command=['/usr/bin/python3', '-c', source])
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise RuntimeError(result.stderr)
            _return_workspace(instance.root/'workspace', call, 1024, config.input_budget)
            assert (work/'meta_input/evidence/large.json').stat().st_size > 16000000
            print(json.dumps({'status':'NATIVE_ISOLATION_PASSED',
                'checks':json.loads(result.stdout), 'large_evidence_bytes':17000000,
                'output_budget_bytes':1024, 'api_calls':0}))
        finally:
            if relay_identity is not None:
                instance.unlink_relay(relay_identity)
            relay.close()
            instance.cleanup()


if __name__ == '__main__':
    main()
