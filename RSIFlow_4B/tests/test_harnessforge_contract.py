import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sia.task_meta.harnessforge_production import _validate_and_repair, initialize_base_manifest
from sia.task_meta.meta_harness.bundle import MetaHarnessStore, validate_files
from sia.task_meta.sequential_loop import positive_gain
from sia.task_meta.types import DecisionConstraintError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
META_SEED = PROJECT_ROOT / "runtime/meta_harness/seed"


class HarnessForgeContractTests(unittest.TestCase):
    def test_meta_seed_routes_only_complete_bundles_and_retains_failures(self):
        files = {
            path.name: path.read_text(encoding="utf-8")
            for path in META_SEED.iterdir()
            if path.is_file()
        }
        self.assertEqual(validate_files(files), "meta-bundle-v3")
        policy = json.loads(files["evolution.json"])
        self.assertFalse(policy["experience"]["exclude_failed"])
        self.assertIn("candidate_attempts", policy["experience"]["fields"])
        self.assertIn("deployment_status", policy["experience"]["fields"])
        for operation in ("routing", "meta_self_update", "final_consolidation"):
            dependency = next(
                step for step in policy["workflows"][operation]
                if step["id"] == "task_harness_source"
            )
            self.assertEqual(dependency["paths"], ["harness_manifest.json"])
        routing = next(
            step["instruction"] for step in policy["workflows"]["routing"]
            if step["kind"] == "propose"
        )
        self.assertIn("operation=produce_harness", routing)
        self.assertIn("target=harness_bundle", routing)
        harness_stage = next(
            step["instruction"] for step in policy["workflows"]["harness_patch"]
            if step["kind"] == "propose"
        )
        self.assertIn("exact pinned upstream HarnessForge", harness_stage)
        joined = "\n".join(files.values())
        for retired in ("SeedPatch", "replace_config", "five-part"):
            self.assertNotIn(retired, joined)
        library = json.loads(files["principles.json"])
        self.assertEqual(len(library["records"]), 5)

    def test_project_local_meta_seed_initializes_without_external_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = MetaHarnessStore(Path(temporary)).initialize(
                META_SEED, source_commit="fixture", binary_sha256="fixture"
            )
            self.assertEqual(bundle.version, 0)
            self.assertEqual(bundle.schema_version, "meta-bundle-v3")
            bundle.verify()

    def test_harness_observation_accepts_only_a_complete_manifest(self):
        from sia.task_meta.observations import harness_evidence
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = initialize_base_manifest(root / "base.json")
            files, identity = harness_evidence(root / "base.json")
            self.assertEqual(json.loads(files["harness_manifest.json"])["bundle_sha256"], manifest.bundle_sha256)
            self.assertEqual(files["harness/builder.py"], manifest.files["builder.py"])
            self.assertEqual(identity["bundle_sha256"], manifest.bundle_sha256)
            legacy = root / "target_agent.py"
            legacy.write_text("def run(): pass\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                harness_evidence(legacy)

    def test_incomplete_or_missing_delta_never_deploys(self):
        self.assertFalse(positive_gain({"paired_evidence_complete": False, "success_delta": 1.0}))
        self.assertFalse(positive_gain({"paired_evidence_complete": True, "success_delta": None}))
        self.assertFalse(positive_gain({"paired_evidence_complete": True, "success_delta": 0.0}))
        self.assertTrue(positive_gain({"paired_evidence_complete": True, "success_delta": 0.01}))

    def test_candidate_validation_failure_is_constraint_but_environment_escapes(self):
        class Report:
            def __init__(self, verdict):
                self.verdict = verdict
                self.errors = [verdict]
            def to_dict(self):
                return {"verdict": self.verdict, "errors": self.errors}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            candidate.mkdir()
            for verdict, error in (
                ("failed_static", DecisionConstraintError),
                ("failed_environment", RuntimeError),
            ):
                fake = SimpleNamespace(
                    FIXABLE_STATUSES=set(),
                    validate_once=lambda *args, verdict=verdict: Report(verdict),
                )
                workflow = root / verdict
                with patch(
                    "sia.task_meta.harnessforge_production.upstream_validation_module",
                    return_value=fake,
                ), patch(
                    "sia.task_meta.harnessforge_validation.validate_candidate",
                    return_value=Report(verdict),
                ):
                    with self.assertRaises(error):
                        _validate_and_repair(candidate, workflow, lambda *_: None)


if __name__ == "__main__":
    unittest.main()
