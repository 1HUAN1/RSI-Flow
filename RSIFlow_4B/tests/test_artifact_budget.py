"""ARTIFACTS updates use the Task runtime budget, not a legacy Harness field."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sia.task_meta.submissions import SubmissionUpdater
from sia.task_meta.types import GenerationContext, MetaDecision, TaskAgentState
from sia.task_meta.updaters import ArtifactChanges


class FakeMeta:
    supports_evolution = False

    def complete(self, prompt, schema, **kwargs):
        return ArtifactChanges(summary="repair answer", edits=[
            {"path": self.target, "content": json.dumps(
                {"final_answer": "corrected", "actions": []})}
        ])


class ArtifactBudgetTest(unittest.TestCase):
    def test_harnessforge_manifest_without_legacy_budget_can_be_updated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "seed.json"
            manifest.write_text(json.dumps({"kind": "harnessforge_candidate_bundle"}))
            parent = root / "candidates" / "ARTIFACTS" / "gen_0"
            parent.mkdir(parents=True)
            row = {
                "task_id": "search-1", "rollout_id": "r0", "domain": "searchqa",
                "purpose": "evolution_train", "split": "train", "seed": 42,
                "task_source_hash": "source", "final_answer": "wrong",
                "tool_calls": [], "verification": {"status": "completed", "success": False},
                "infrastructure_error": False,
            }
            (parent / "train_trajectories.jsonl").write_text(json.dumps(row) + "\n")
            name = hashlib.sha256(b"search-1:r0").hexdigest()
            target = f"submissions/train/{name}.json"
            client = FakeMeta()
            client.target = target
            decision = MetaDecision(
                action="ARTIFACTS", diagnosis="bad answer", evidence=[],
                rationale="repair submission", proposed_change="replace answer",
                expected_effect="one more correct submission", target_components=["ARTIFACTS"],
                requested_changes=[{"id": "edit-1", "component": "ARTIFACTS",
                                    "operation": "write_asset", "target": target,
                                    "instruction": "correct answer"}],
            )
            context = GenerationContext(1, parent.parent / "gen_1", None, None)
            child, update = SubmissionUpdater(client, max_tool_calls=128).apply(
                TaskAgentState(0, "model", str(manifest)), decision, context)
            self.assertEqual(update.action.value, "ARTIFACTS")
            self.assertEqual(child.generation, 1)
            self.assertEqual(json.loads((Path(child.artifacts.directory) / target).read_text())
                             ["payload"]["final_answer"], "corrected")


if __name__ == "__main__":
    unittest.main()
