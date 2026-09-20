"""CPU-only committed round snapshots; no model or trainer calls."""

import json
import tempfile
import unittest
from pathlib import Path

from sia.task_meta.harnessforge_production import initialize_base_manifest
from sia.task_meta.meta_harness.bundle import MetaHarnessStore
from sia.task_meta.round_checkpoint import commit_round_checkpoint, verify_round_checkpoint
from sia.task_meta.storage import artifact_manifest, save_json
from sia.task_meta.types import ArtifactState, MetaAgentState, TaskAgentState


ROOT = Path(__file__).resolve().parents[1]


class RoundCheckpoint(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.meta_store = MetaHarnessStore(self.root / "store")
        bundle = self.meta_store.initialize(ROOT / "runtime/meta_harness/seed", "pinned", "binary")
        harness = self.root / "base.json"
        initialize_base_manifest(harness)
        self.task = TaskAgentState(1, "model-weights", str(harness), checkpoint_path="model-weights",
                                   checkpoint_manifest=[{"path": "weights.safetensors", "sha256": "fixed"}])
        self.meta = MetaAgentState("meta-model", str(bundle.path / "instructions.md"),
                                   version=bundle.version, bundle_hash=bundle.hash,
                                   bundle_path=str(bundle.path))
        self.round_dir = self.root / "round_0"
        self.round_dir.mkdir()
        for name in ("complete.json", "experience.json", "deployment.json"):
            save_json(self.round_dir / name, {"round": 0, "name": name})

    def test_full_task_bundle_meta_bundle_and_skill_library_are_independently_saved(self):
        asset = self.root / "assets"
        asset.mkdir()
        (asset / "skill.txt").write_text("Task artifact", encoding="utf-8")
        self.task.artifacts = ArtifactState(str(asset), artifact_manifest(asset))
        first = commit_round_checkpoint(self.round_dir, 0, self.task, self.meta)
        self.assertEqual(first, commit_round_checkpoint(self.round_dir, 0, self.task, self.meta))
        checkpoint = self.round_dir / "checkpoint"
        self.assertTrue((checkpoint / "task/harness_bundle/builder.py").is_file())
        self.assertTrue((checkpoint / "task/harness_bundle/memory_module/provider.py").is_file())
        self.assertEqual((checkpoint / "task/artifacts/skill.txt").read_text(), "Task artifact")
        self.assertEqual(first["task"]["model_checkpoint_manifest"], self.task.checkpoint_manifest)
        self.assertEqual(first["meta"]["bundle"]["bundle_hash"], self.meta.bundle_hash)
        self.assertEqual(first["meta"]["skills"]["record_count"], len(json.loads(
            (checkpoint / "meta/skills/principles.json").read_text())["records"]))
        self.assertEqual((checkpoint / "meta/skills/principles.json").read_bytes(),
                         (checkpoint / "meta/bundle/principles.json").read_bytes())

    def test_resume_detects_changed_skill_or_round_evidence(self):
        commit_round_checkpoint(self.round_dir, 0, self.task, self.meta)
        skill = self.round_dir / "checkpoint/meta/skills/principles.json"
        skill.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "skill library changed"):
            verify_round_checkpoint(self.round_dir, 0, self.task, self.meta)

    def test_resume_rebuilds_missing_checkpoint_once_from_immutable_round_state(self):
        first = commit_round_checkpoint(self.round_dir, 0, self.task, self.meta)
        self.assertEqual(first["round_index"], 0)
        self.assertEqual(verify_round_checkpoint(self.round_dir, 0, self.task, self.meta), first)
        save_json(self.round_dir / "deployment.json", {"round": 0, "name": "tampered"})
        with self.assertRaisesRegex(ValueError, "committed round evidence"):
            verify_round_checkpoint(self.round_dir, 0, self.task, self.meta)


if __name__ == "__main__":
    unittest.main()
