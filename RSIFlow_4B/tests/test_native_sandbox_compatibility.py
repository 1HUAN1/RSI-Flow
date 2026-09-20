"""Exercise the official evaluator launcher against current shared guards."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from tool_sandbox import run_native


class NativeSandboxCompatibilityTests(unittest.TestCase):
    def test_native_launch_preserves_threads_and_denies_network(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            work = root / "work"
            output = root / "output"
            work.mkdir()
            output.mkdir()
            script = work / "probe.py"
            script.write_text('''import json, socket, threading
seen = []
thread = threading.Thread(target=lambda: seen.append("thread"))
thread.start()
thread.join()
try:
    socket.socket()
except PermissionError:
    seen.append("network_denied")
print(json.dumps(seen))
''')
            log_path = output / "probe.log"
            with log_path.open("wb") as log:
                run_native([sys.executable, "-B", str(script)], work, output, log, timeout=30)
            self.assertEqual(json.loads(log_path.read_text()), ["thread", "network_denied"])


if __name__ == "__main__":
    unittest.main()
