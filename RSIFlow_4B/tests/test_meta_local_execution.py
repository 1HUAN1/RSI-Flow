"""Local Meta input/output boundary tests, without model requests."""
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from sia.task_meta.meta_backends.local_execution import _stage_files, _return_workspace, runtime_identity
from sia.task_meta.meta_backends.codex_openrouter import local_stage_never_dispatched
from sia.task_meta.meta_backends.contracts import MetaBackendConfig
from sia.task_meta.meta_backends.isolation_runtime import IsolationRuntime


class LocalMetaExecution(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.call = self.root / "call"
        (self.call / "workspace").mkdir(parents=True)
        (self.call / "codex_home").mkdir()
        (self.call / "codex_home/config.toml").write_text("model='fixture'")
        (self.call / "schema.json").write_text("{}")
        (self.call / "workspace/AGENTS.md").write_text("fixture rules")
        (self.call / "workspace/sample.py").write_text("return a-b")

    def test_only_declared_workspace_and_control_files_enter_child(self):
        (self.call / "private.txt").write_text("not an input")
        files = _stage_files(self.call)
        self.assertEqual(set(files), {"codex_home/config.toml", "schema.json",
                                    "workspace/AGENTS.md", "workspace/sample.py"})
        (self.call / "workspace/linked.txt").symlink_to(self.call / "private.txt")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            _stage_files(self.call)

    def test_child_changes_return_and_original_input_is_preserved(self):
        child = self.root / "child"
        child.mkdir()
        (child / "sample.py").write_text("return a+b")
        (child / ".meta_response.json").write_text('{"answer":5}')
        _return_workspace(child, self.call, 10000)
        self.assertEqual((self.call / "workspace/sample.py").read_text(), "return a+b")
        self.assertEqual((self.call / "workspace.input/sample.py").read_text(), "return a-b")
        self.assertEqual(json.loads((self.call / "workspace/.meta_response.json").read_text()), {"answer": 5})

    def test_oversized_or_linked_child_output_is_not_published(self):
        child = self.root / "child"
        child.mkdir()
        output = child / "large"
        output.write_text("x" * 20)
        with self.assertRaises(ValueError):
            _return_workspace(child, self.call, 10)
        self.assertFalse((self.call / "workspace.input").exists())
        output.unlink()
        output.symlink_to(self.call / "schema.json")
        with self.assertRaises(ValueError):
            _return_workspace(child, self.call, 10000)

    def test_local_backend_is_explicit_and_binds_execution_sources(self):
        config = MetaBackendConfig(execution_location="local_chroot")
        identity = runtime_identity(config)
        self.assertEqual(set(identity["sources"]), {
            "local_execution.py", "isolation_runtime.py", "isolation_launcher.py", "bridge.py", "input_budget.py"})
        self.assertTrue(all(len(value) == 64 for value in identity["sources"].values()))

    def test_static_runtime_directories_are_traversable_with_restrictive_umask(self):
        jail = self.root / "jail"
        original_umask = os.umask(0o077)
        try:
            for name in ("usr/bin", "usr/lib/x86_64-linux-gnu", "etc/ssl", "dev", "proc/self",
                         "workspace", "codex_home", "tmp", "home/meta"):
                (jail / name).mkdir(parents=True, exist_ok=True)
        finally:
            os.umask(original_umask)
        (jail / "bin").symlink_to("usr/bin")
        self.assertEqual(stat.S_IMODE((jail / "usr").stat().st_mode), 0o700)
        IsolationRuntime._normalize_static_directory_modes(jail)
        for name in ("usr", "usr/bin", "usr/lib", "usr/lib/x86_64-linux-gnu",
                     "etc", "etc/ssl", "dev", "proc", "proc/self"):
            self.assertEqual(stat.S_IMODE((jail / name).stat().st_mode), 0o755, name)
        self.assertEqual(stat.S_IMODE((jail / "workspace").stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((jail / "home/meta").stat().st_mode), 0o700)
        self.assertTrue((jail / "bin").is_dir())

    def test_only_new_undispatched_local_stage_is_retryable(self):
        receipt = {"state": "pending", "dispatch_protocol": "local_dispatch_v1"}
        self.assertTrue(local_stage_never_dispatched(receipt, self.call, "local_chroot"))
        self.assertFalse(local_stage_never_dispatched(receipt, self.call, "ssh_worker"))
        self.assertFalse(local_stage_never_dispatched({"state": "pending"}, self.call, "local_chroot"))
        (self.call / "local_dispatch.json").write_text("{}")
        self.assertFalse(local_stage_never_dispatched(receipt, self.call, "local_chroot"))


if __name__ == "__main__":
    unittest.main()
