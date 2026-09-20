"""Actual isolation setup under worker-pool descriptor pressure; no model calls."""
import json
import multiprocessing
import os
import unittest
from concurrent.futures import ProcessPoolExecutor
import test_contracts as native
from sia.task_meta.sandbox import LinuxSandbox


def isolated_check(index):
    descriptors = [os.open('/dev/null', os.O_RDONLY) for _ in range(64)]
    try:
        sandbox = LinuxSandbox()
        result = sandbox.run('''import json,os,resource,socket
denied=[]
for name, action in [('host',lambda:open('/root/.bashrc')),('network',lambda:socket.socket()),('fork',os.fork)]:
 try: action()
 except OSError: denied.append(name)
print(json.dumps({'limit':resource.getrlimit(resource.RLIMIT_NOFILE),'denied':denied}))
''')
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == dict(limit=[32,32],denied=['host','network','fork'])
        with sandbox.session('import json,sys\nfor line in sys.stdin: print(json.dumps({"ok":True}),flush=True)') as session:
            assert session.request({'probe':True}) == {'ok':True}
        return index
    finally:
        for fd in descriptors:os.close(fd)


class SandboxWorkers(unittest.TestCase):
    def test_32_real_isolated_runs_and_sessions_on_16_workers(self):
        with ProcessPoolExecutor(16, mp_context=multiprocessing.get_context('fork')) as pool:
            result = list(pool.map(isolated_check, range(32)))
        self.assertEqual(result, list(range(32)))


if __name__ == '__main__':unittest.main()
