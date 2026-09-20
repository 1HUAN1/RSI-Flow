"""Local Meta input/output boundary tests, without model requests."""
import json
from pathlib import Path
import tempfile
import unittest

from sia.task_meta.meta_backends.local_execution import _stage_files, _return_workspace, runtime_identity
from sia.task_meta.meta_backends.contracts import MetaBackendConfig


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
            "local_execution.py", "isolation_runtime.py", "isolation_launcher.py", "bridge.py"})
        self.assertTrue(all(len(value) == 64 for value in identity["sources"].values()))


if __name__ == "__main__":
    unittest.main()
