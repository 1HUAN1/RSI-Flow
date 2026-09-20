import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sia.task_meta.harnessforge_manifest import REQUIRED_BUNDLE_FILES, HarnessBundleManifest, load_manifest, save_manifest
from sia.task_meta.harnessforge_production import (
    HarnessForgeProductionUpdater,
    MAX_FIX_ATTEMPTS,
    RepairResponse,
    StageResponse,
    _validate_and_repair,
    harnessforge_capabilities,
    initialize_base_manifest,
)
from sia.task_meta.types import MetaDecision, TaskAgentState


def decision(target="harness_bundle", operation="produce_harness"):
    return MetaDecision(
        action="HARNESS",
        diagnosis="Observed transferable harness failure.",
        evidence=["pre:fixture"],
        rationale="Run the complete upstream production workflow.",
        proposed_change="Produce one independent HarnessForge candidate.",
        expected_effect="Unverified until paired evaluation.",
        target_components=["HARNESS"],
        requested_changes=[{
            "id": "harness-bundle-1", "component": "HARNESS", "operation": operation,
            "target": target, "instruction": "Run all HarnessForge production stages.",
        }],
        decision_id="decision-fixture",
    )


def generated_files():
    return {
        "__init__.py": "from .builder import build_agent_from_context\n__all__ = ['build_agent_from_context']\n",
        "builder.py": "HARNESS_NAME='fixture'\nPLANNING_SYSTEM='p'\nACTION_SYSTEM='a'\nDEFAULT_MEMORY_SYSTEM='m'\ndef build_agent_from_context(context): return object()\n",
        "Description.md": "Complete fixture candidate.\n",
        "planning_module/provider.py": "PLANNING_SYSTEM='p'\nPlanningClass=object\n",
        "action_module/provider.py": "ACTION_SYSTEM='a'\ndef get_provider(): return object()\n",
        "memory_module/provider.py": "class MemoryProvider: pass\n",
        "action_module/prompts/fixture.yaml": "system_prompt: fixture\n",
    }


def stage3(files):
    return "\n\n".join(
        f"### FILE: {name}\n```text\n{content.rstrip()}\n```" for name, content in files.items()
    )


class FakeClient:
    supports_evolution = False

    def __init__(self, files):
        self.files = files
        self.prompts = []

    def complete(self, prompt, schema, **kwargs):
        self.prompts.append((prompt, schema, kwargs))
        if schema is not StageResponse:
            raise AssertionError(f"Unexpected schema: {schema}")
        if prompt.lstrip().startswith("# You are a Harness Failure Localization Agent"):
            return StageResponse(content="Stage 1 localized the failure to Action and Planning.")
        if prompt.lstrip().startswith("# You are a Harness Improvement Direction Agent"):
            return StageResponse(content="Stage 2 selected a transferable schema-aware repair.")
        return StageResponse(content=stage3(self.files))


class HarnessForgeProductionTests(unittest.TestCase):
    def test_base_initializer_is_pinned_complete_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "base.json"
            manifest = initialize_base_manifest(target)
            self.assertEqual(load_manifest(target), manifest)
            self.assertTrue(REQUIRED_BUNDLE_FILES <= manifest.files.keys())
            self.assertIn("action_module/prompts/toolcalling_agent.yaml", manifest.files)
            capability = harnessforge_capabilities(target)
            self.assertEqual(capability["operations"], [{"operation": "produce_harness", "target": "harness_bundle"}])

    def test_one_candidate_runs_three_upstream_stages_and_commits_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_path = root / "parent.json"
            initialize_base_manifest(parent_path)
            parent_bytes = parent_path.read_bytes()
            client = FakeClient(generated_files())
            state = TaskAgentState(0, "Qwen3-4B", str(parent_path), checkpoint_path="Qwen3-4B")
            observation = SimpleNamespace(
                current_performance={"macro_success": 0.25}, trajectory_summary={"failures": 3},
                observation_coverage={"tasks": 4}, raw_trajectories=[], trajectories=[],
                failure_examples=[{"id": "failure"}], success_examples=[{"id": "success"}],
                improvement_history=[{"component": "HARNESS", "outcome": "failure"}],
                experience_ledger=[],
            )
            context = SimpleNamespace(
                generation=1, directory=root / "child", observation=observation, meta_state=None,
            )
            validated = {"success": True, "final_verdict": "passed", "fixes_used": 0}
            with patch("sia.task_meta.harnessforge_production._archive_examples", return_value=("pool", "examples", "names")), patch(
                "sia.task_meta.harnessforge_production._validate_and_repair", return_value=validated
            ):
                child, update = HarnessForgeProductionUpdater(client).apply(state, decision(), context)
            self.assertEqual(parent_path.read_bytes(), parent_bytes)
            manifest = load_manifest(child.harness_path)
            self.assertEqual(dict(manifest.files), generated_files())
            self.assertEqual(len(client.prompts), 3)
            self.assertEqual(update.details["candidate_count"], 1)
            self.assertEqual(update.applied_changes[0]["target"], "harness_bundle")
            self.assertEqual(update.files[-1]["path"], "planning_module/provider.py")
            self.assertIn("Optional Meta experience", client.prompts[1][0])
            self.assertTrue((context.directory / "harnessforge_production/03_harness_generation.md").is_file())

    def test_file_preselection_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "parent.json"
            initialize_base_manifest(parent)
            state = TaskAgentState(0, "Qwen3-4B", str(parent))
            context = SimpleNamespace(generation=1, directory=Path(temporary) / "child")
            with self.assertRaisesRegex(ValueError, "without preselecting a module"):
                HarnessForgeProductionUpdater(FakeClient(generated_files())).apply(
                    state, decision("memory_module/provider.py", "replace_module"), context
                )

    def test_stage4_stops_after_three_small_repairs(self):
        class Report:
            verdict = "failed_static"
            errors = ["syntax"]
            def to_dict(self):
                return {"verdict": self.verdict, "errors": self.errors}

        calls = {"validate": 0, "repair": 0}

        def validate_once(*args):
            calls["validate"] += 1
            return Report()

        def file_hashes(path):
            return {"Description.md": hashlib.sha256((path / "Description.md").read_bytes()).hexdigest()}

        def request_fix(model, **kwargs):
            messages = [{"role": "user", "content": [{"type": "text", "text": "repair"}]}]
            response = json.loads(model(messages).content)
            return response

        def apply_fix(candidate, payload, dry_run=False):
            calls["repair"] += 1
            (candidate / "Description.md").write_text(payload["files"][0]["content"])
            return ["Description.md"]

        fake = SimpleNamespace(
            FIXABLE_STATUSES={"failed_static", "failed_import", "failed_build"},
            validate_once=validate_once, file_hashes=file_hashes, request_fix=request_fix,
            apply_fix_payload=apply_fix,
            changed_files=lambda before, after: [name for name in set(before) | set(after) if before.get(name) != after.get(name)],
        )
        with tempfile.TemporaryDirectory() as temporary:
            workflow = Path(temporary) / "workflow"
            candidate = workflow / "validation_project/generated_harnesses/rounds/round_1/candidate"
            candidate.mkdir(parents=True)
            (candidate / "Description.md").write_text("initial")

            def complete(prompt, schema):
                index = calls["repair"] + 1
                return RepairResponse(summary="small repair", files=[{
                    "path": "Description.md", "content": f"repair {index}",
                }])

            with patch("sia.task_meta.harnessforge_production.upstream_validation_module", return_value=fake), patch(
                "sia.task_meta.harnessforge_validation.validate_candidate", side_effect=validate_once
            ):
                with self.assertRaisesRegex(ValueError, "failed validation"):
                    _validate_and_repair(candidate, workflow, complete)
            self.assertEqual(calls["repair"], MAX_FIX_ATTEMPTS)
            self.assertEqual(calls["validate"], MAX_FIX_ATTEMPTS + 1)


if __name__ == "__main__":
    unittest.main()
