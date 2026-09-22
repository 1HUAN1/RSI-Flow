"""Failed Meta proposals may retry only with proof no Task effect began."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sia.task_meta.durable import DurableUpdater, StageJournal
from sia.task_meta.storage import checkpoint_manifest
from sia.task_meta.types import (
    ArtifactState, TaskAgentState, TaskUpdate, TaskUpdateAction, UpdatePending)
from sia.task_meta.updaters import ArtifactUpdater, ModelUpdater
from sia.task_meta.harnessforge_production import HarnessForgeProductionUpdater, _candidate_name


class Decision:
    action = TaskUpdateAction.MODEL
    decision_id = "generation_0_decision_0"

    def model_dump(self, mode=None):
        return {"action": self.action.value, "decision_id": self.decision_id}


class FailBeforeEffect:
    def __init__(self, safe=True):
        self.safe = safe
        self.calls = 0

    def retry_safe_before_effect(self, task, decision, context):
        return self.safe

    def apply(self, task, decision, context):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("Meta candidate delivery failed before Task effect")
        successor = TaskAgentState(context.generation, task.model_ref, task.harness_path, task.artifacts)
        return successor, TaskUpdate(TaskUpdateAction.MODEL, "fixture committed")


class PreEffectRetryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.harness = self.root / "harness.json"
        self.harness.write_text("{}")
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        (self.checkpoint / "weight.safetensors").write_bytes(b"weight")
        self.task = TaskAgentState(0, "checkpoint", str(self.harness), ArtifactState(),
                                   str(self.checkpoint), checkpoint_manifest(self.checkpoint))
        self.context = SimpleNamespace(generation=1, directory=self.root / "gen_1", meta_state=None)
        self.decision = Decision()

    def test_started_receipt_is_archived_then_one_retry_commits(self):
        updater = FailBeforeEffect()
        durable = DurableUpdater(updater, StageJournal(self.root))
        with self.assertRaisesRegex(RuntimeError, "before Task effect"):
            durable.apply(self.task, self.decision, self.context)
        receipt = self.root / "gen_0/intervention_receipt.json"
        self.assertEqual(json.loads(receipt.read_text())["status"], "started")
        child, _ = durable.apply(self.task, self.decision, self.context)
        self.assertEqual(child.generation, 1)
        self.assertEqual(json.loads(receipt.read_text())["status"], "committed")
        self.assertEqual(len(list(receipt.parent.glob("intervention_receipt.pre_effect_retry_*.json"))), 1)
        durable.apply(self.task, self.decision, self.context)
        self.assertEqual(updater.calls, 2)

    def test_started_receipt_still_blocks_when_effect_is_uncertain(self):
        updater = FailBeforeEffect(safe=False)
        durable = DurableUpdater(updater, StageJournal(self.root))
        with self.assertRaises(RuntimeError):
            durable.apply(self.task, self.decision, self.context)
        with self.assertRaises(UpdatePending):
            durable.apply(self.task, self.decision, self.context)
        self.assertEqual(updater.calls, 1)
        self.assertEqual(json.loads((self.root / "gen_0/intervention_receipt.json").read_text())["status"], "started")

    def test_model_retry_requires_unchanged_weights_and_no_training_directory(self):
        updater = ModelUpdater(None, None, None)
        self.context.directory.mkdir()
        self.assertTrue(updater.retry_safe_before_effect(self.task, self.decision, self.context))
        (self.context.directory / "model_update").mkdir()
        self.assertFalse(updater.retry_safe_before_effect(self.task, self.decision, self.context))
        (self.context.directory / "model_update").rmdir()
        (self.checkpoint / "weight.safetensors").write_bytes(b"changed")
        self.assertFalse(updater.retry_safe_before_effect(self.task, self.decision, self.context))

    def test_artifact_and_harness_retry_refuse_materialized_child(self):
        self.context.directory.mkdir()
        artifact_decision = SimpleNamespace(action=TaskUpdateAction.ARTIFACTS)
        artifact = ArtifactUpdater(None)
        self.assertTrue(artifact.retry_safe_before_effect(self.task, artifact_decision, self.context))
        (self.context.directory / "artifacts").mkdir()
        self.assertFalse(artifact.retry_safe_before_effect(self.task, artifact_decision, self.context))
        (self.context.directory / "artifacts").rmdir()

        harness_decision = SimpleNamespace(action=TaskUpdateAction.HARNESS,
                                           decision_id="generation_0_decision_0")
        harness = HarnessForgeProductionUpdater(None)
        self.assertTrue(harness.retry_safe_before_effect(self.task, harness_decision, self.context))
        candidate = (self.context.directory / "harnessforge_production/validation_project"
                     / "generated_harnesses/rounds/round_rsi_0001"
                     / _candidate_name(1, harness_decision.decision_id))
        candidate.mkdir(parents=True)
        self.assertFalse(harness.retry_safe_before_effect(self.task, harness_decision, self.context))


if __name__ == "__main__":
    unittest.main()
