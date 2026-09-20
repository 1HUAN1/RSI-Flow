"""Real subprocess coverage for candidate validation, without model/API calls."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from sia.task_meta.harnessforge_manifest import HarnessBundleManifest
from sia.task_meta.harnessforge_production import BASE_HARNESS_ROOT
from sia.task_meta.harnessforge_validation import validate_candidate


class ValidationIsolationTests(unittest.TestCase):
    def candidate(self, root, extra=""):
        manifest = HarnessBundleManifest.from_directory(BASE_HARNESS_ROOT, harness_name="fixture")
        files = dict(manifest.files)
        # __future__ imports must remain first. Run the probe before native imports.
        files["builder.py"] = files["builder.py"].replace(
            "from __future__ import annotations\n",
            "from __future__ import annotations\n" + extra + "\n", 1,
        )
        target = root / "rounds" / "fixture"
        HarnessBundleManifest("fixture", files).materialize(target)
        return target

    def test_native_baseline_and_candidate_side_effects_stay_in_child(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            protected = root / "parent.txt"
            protected.write_text("unchanged")
            extra = f'''
import os, sys
from pathlib import Path
assert "RSIFLOW_TEST_PROVIDER_KEY" not in os.environ
os.environ["RSIFLOW_VALIDATION_MUTATION"] = "child-only"
sys.path.insert(0, "/candidate/changed/import/path")
for operation in (
    lambda: Path({str(protected)!r}).write_text("corrupted"),
    lambda: os.truncate({str(protected)!r}, 0),
    lambda: os.chmod({str(protected)!r}, 0o777),
):
    try:
        operation()
    except PermissionError:
        pass
    else:
        raise AssertionError("Candidate could mutate protected host file")
Path("candidate-scratch.txt").write_text("scratch writes allowed")
'''
            candidate = self.candidate(root, extra)
            before = {p.relative_to(candidate): p.read_bytes() for p in candidate.rglob("*") if p.is_file()}
            paths = list(sys.path)
            modules = set(sys.modules)
            os.environ["RSIFLOW_TEST_PROVIDER_KEY"] = "test-only-not-a-secret"
            try:
                report = validate_candidate(candidate)
            finally:
                os.environ.pop("RSIFLOW_TEST_PROVIDER_KEY", None)
            self.assertEqual(report.verdict, "passed", report.to_dict())
            self.assertEqual(protected.read_text(), "unchanged")
            self.assertNotIn("RSIFLOW_VALIDATION_MUTATION", os.environ)
            self.assertEqual(sys.path, paths)
            leaked = set(sys.modules) - modules
            self.assertFalse(any(n.startswith(("generated_harnesses", "Agents", "module_action")) for n in leaked))
            self.assertEqual(before, {p.relative_to(candidate): p.read_bytes() for p in candidate.rglob("*") if p.is_file()})
            self.assertEqual(report.candidate_path, str(candidate))
            json.dumps(report.to_dict())

    def test_timeout_and_abrupt_exit_do_not_stop_controller(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate = self.candidate(Path(raw), "import time\ntime.sleep(60)")
            report = validate_candidate(candidate, timeout=1)
            self.assertEqual(report.verdict, "failed_build")
            self.assertIn("exceeded", report.errors[0])
        with tempfile.TemporaryDirectory() as raw:
            candidate = self.candidate(Path(raw), "import os\nos._exit(23)")
            report = validate_candidate(candidate)
            self.assertEqual(report.verdict, "failed_build")
            self.assertIn("exit=23", report.errors[0])

    def test_real_upstream_repair_loop_uses_fresh_validation(self):
        from sia.task_meta.harnessforge_production import _validate_and_repair, RepairResponse

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate = self.candidate(root, "raise RuntimeError('fixture wiring error')")
            fixed = (candidate / "builder.py").read_text().replace(
                "raise RuntimeError('fixture wiring error')", "# wiring repaired"
            )
            prompts = []
            def repair(prompt, schema):
                prompts.append(prompt)
                return RepairResponse(summary="Repair fixture wiring", files=[{
                    "path": "builder.py", "content": fixed,
                }])
            report = _validate_and_repair(candidate, root / "workflow", repair)
            self.assertEqual(report["final_verdict"], "fixed_after_retry")
            self.assertEqual(report["fixes_used"], 1)
            self.assertEqual([r["verdict"] for r in report["history"]], ["failed_import", "passed"])
            self.assertEqual(len(prompts), 1)
            self.assertIn("fixture wiring error", prompts[0])
            self.assertEqual((candidate / "builder.py").read_text(), fixed.rstrip() + "\n")

    def test_failed_candidate_does_not_poison_next_attempt(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate = self.candidate(Path(raw), "import os\nassert 'RSIFLOW_LEAK' not in os.environ\nos.environ['RSIFLOW_LEAK'] = 'bad'\nraise RuntimeError('candidate failure')")
            first = validate_candidate(candidate)
            self.assertEqual(first.verdict, "failed_import")
            builder = candidate / "builder.py"
            builder.write_text(builder.read_text().replace("raise RuntimeError('candidate failure')", "assert 'RSIFLOW_OLD' not in os.environ"))
            second = validate_candidate(candidate)
            self.assertEqual(second.verdict, "passed", second.to_dict())


if __name__ == "__main__":
    unittest.main()
