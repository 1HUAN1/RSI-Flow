"""Offline tests for the Codex-callable stage adapter."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from task_adapter import TaskAdapter


class TaskAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.adapter = TaskAdapter(ROOT, self.root / "receipts")

    def test_stage_error_is_a_fact_not_an_experiment_verdict(self):
        result = self.adapter.call("rollout", {"round_number": 1, "stage": "parent"})
        self.assertEqual(result["status"], "stage_error")
        self.assertEqual(result["stage"], "rollout")
        self.assertIn("error_type", result)
        self.assertNotIn("accepted", result)
        self.assertNotIn("next_component", result)

    def test_stage_dispatch_does_not_compare_or_accept(self):
        with patch.object(self.adapter, "rollout", return_value={
            "performance_path": "/tmp/example.json", "macro_success": 0.2,
            "tasks_observed": 180,
        }):
            receipt = self.adapter.call("rollout", {"stage": "candidate"})
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["macro_success"], 0.2)
        self.assertEqual(receipt["tasks_observed"], 180)
        self.assertNotIn("deploy", receipt)

    def test_bootstrap_reports_missing_checkpoint(self):
        result = self.adapter.call("bootstrap", {
            "checkpoint_path": str(self.root / "missing-Qwen3-4B"),
        })
        self.assertEqual(result["status"], "missing_checkpoint")
        self.assertFalse((self.root / "receipts/task_a0.json").exists())

    def test_artifacts_creates_candidate_state_without_scoring(self):
        harness = self.root / "harness.json"
        harness.write_text("{}", encoding="utf-8")
        state_path = self.root / "parent.json"
        state_path.write_text(json.dumps({
            "generation": 0, "model_ref": "Qwen3-4B",
            "harness_path": str(harness), "artifacts": {"directory": None, "manifest": []},
            "checkpoint_path": None, "checkpoint_manifest": [],
        }), encoding="utf-8")
        baseline = self.root / "parent_rollout"
        baseline.mkdir()
        row = {
            "task_id": "searchqa:1", "task_source_hash": "source", "rollout_id": 0,
            "split": "evolve_train", "purpose": "evolution_train", "seed": 42, "domain": "searchqa",
            "final_answer": "old", "tool_calls": [],
        }
        (baseline / "train_trajectories.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8")
        import hashlib
        name = "submissions/train/" + hashlib.sha256(b"searchqa:1:0").hexdigest() + ".json"
        result = self.adapter.call("artifacts", {
            "state_path": str(state_path), "baseline_dir": str(baseline),
            "edits": {name: {"final_answer": "new", "actions": []}},
            "output_dir": str(self.root / "artifact_candidate"),
        })
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["evaluation_mode"], "direct_submission")
        self.assertTrue(Path(result["candidate_state_path"]).is_file())
        self.assertNotIn("macro_success", result)
        saved = json.loads(Path(result["candidate_state_path"]).read_text())
        self.assertEqual(saved["generation"], 1)

    def test_experiment_config_normalizes_three_domain_180_round(self):
        config = self.adapter._config({"config_path": "configs/train_180_a0_v1.json"})
        self.assertEqual(config.window_quotas, {"tool_use": 60, "code": 60, "searchqa": 60})
        configured_rounds = json.loads((ROOT / "configs/train_180_a0_v1.json").read_text())["rounds"]
        self.assertEqual(config.max_generations, configured_rounds)
        self.assertEqual({item["gpu"] for item in config.task_replicas}, set(range(8)))
        self.assertTrue(config.task_checkpoint.endswith("Qwen3-4B"))

    def test_harnessforge_materializes_one_complete_candidate(self):
        from types import SimpleNamespace
        from sia.task_meta.harnessforge_manifest import load_manifest
        seed = ROOT / "seed_harness/harnessforge_base_manifest.json"
        harness = load_manifest(seed)
        candidate = self.root / "candidate"
        harness.materialize(candidate)
        production = self.root / "candidate_production"
        production.mkdir()
        (production / "module_localization_report.md").write_text("Parent trajectories implicate history selection.")
        (production / "improvement_direction_brief.md").write_text("Retain workflow; change history handling based on localization.")
        state_path = self.root / "parent_harness.json"
        state_path.write_text(json.dumps({
            "generation": 0, "model_ref": "Qwen3-4B",
            "harness_path": str(seed), "artifacts": {"directory": None, "manifest": []},
            "checkpoint_path": None, "checkpoint_manifest": [],
        }), encoding="utf-8")
        verdict = SimpleNamespace(to_dict=lambda: {"verdict": "passed", "checks": [], "errors": []})
        with patch("sia.task_meta.harnessforge_validation.validate_candidate", return_value=verdict):
            result = self.adapter.call("harnessforge", {
                "state_path": str(state_path), "candidate_dir": str(candidate),
                "output_dir": str(self.root / "harness_candidate"),
            })
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(len(result["bundle_sha256"]), 64)
        child = json.loads(Path(result["candidate_state_path"]).read_text())
        self.assertEqual(child["generation"], 1)
        self.assertEqual(child["harness_path"], result["candidate_manifest_path"])
        self.assertNotIn("accepted", result)
        self.assertEqual(result["validation_attempt"], 1)
        self.assertEqual(result["checks_remaining"], 2)
        self.assertTrue(Path(result["localization_report_path"]).is_file())
        self.assertTrue(Path(result["improvement_direction_path"]).is_file())

    def test_harnessforge_returns_missing_production_to_same_meta(self):
        arguments = {"state_path": str(self.root / "parent.json"),
                     "candidate_dir": str(self.root / "candidate"),
                     "output_dir": str(self.root / "built")}
        with patch("sia.task_meta.harnessforge_validation.validate_candidate") as validator:
            result = self.adapter.call("harnessforge", arguments)
        self.assertEqual(result["status"], "production_incomplete")
        self.assertEqual(len(result["missing"]), 2)
        validator.assert_not_called()
        self.assertNotIn("accepted", result)

    def test_harnessforge_limits_same_candidate_to_three_checks(self):
        from types import SimpleNamespace
        reports = self.root / "candidate_production"
        reports.mkdir()
        (reports / "module_localization_report.md").write_text("Source and rollout diagnosis")
        (reports / "improvement_direction_brief.md").write_text("Improvement from diagnosis and skills")
        arguments = {"state_path": str(self.root / "parent.json"),
                     "candidate_dir": str(self.root / "candidate"),
                     "output_dir": str(self.root / "built")}
        failed = SimpleNamespace(to_dict=lambda: {"verdict": "failed_import", "errors": ["missing module"]})
        with patch("sia.task_meta.harnessforge_validation.validate_candidate", return_value=failed) as validator:
            for attempt in (1, 2, 3):
                result = self.adapter.call("harnessforge", arguments)
                self.assertEqual(result["status"], "validation_failed")
                self.assertEqual(result["validation_attempt"], attempt)
            arguments["output_dir"] = str(self.root / "another_output")
            exhausted = self.adapter.call("harnessforge", arguments)
            self.assertEqual(exhausted["status"], "repair_limit_reached")
            self.assertEqual(validator.call_count, 3)
        self.assertEqual(len(json.loads((reports / "validation_attempts.json").read_text())), 3)
        self.assertFalse((self.root / "built/candidate_state.json").exists())

    def test_rollout_reports_executor_facts_without_accepting_candidate(self):
        from types import SimpleNamespace
        from sia.task_meta.types import TaskAgentState
        state = TaskAgentState(0, "Qwen3-4B", "/tmp/seed.json",
                               checkpoint_path="/tmp/Qwen3-4B")
        performance = {"macro_success": 0.25, "collection_stage": "child_post_update"}
        fake_executor = SimpleNamespace(execute=lambda task, directory:
            SimpleNamespace(performance=performance, cost={"model_calls": 5},
                            trajectories=[{"task_id": "a"}]))
        with patch.object(self.adapter, "_config", return_value=SimpleNamespace()), \
             patch.object(self.adapter, "_state", return_value=state), \
             patch.object(self.adapter, "_executor", return_value=fake_executor):
            result = self.adapter.call("rollout", {
                "config_path": "unused", "state_path": "unused",
                "round_number": 1, "stage": "candidate",
                "ensure_services": False,
                "output_dir": str(self.root / "child_rollout"),
            })
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(result["macro_success"], 0.25)
        self.assertEqual(result["tasks_observed"], 1)
        self.assertEqual(json.loads(Path(result["performance_path"]).read_text()),
                         performance)
        self.assertNotIn("accepted", result)

    def test_meta_evidence_keeps_all_180_and_selects_48_excerpts(self):
        baseline = self.root / "parent_evidence"
        baseline.mkdir()
        rows = []
        for domain in ("tool_use", "code", "searchqa"):
            for number in range(60):
                rows.append({
                    "task_id": f"{domain}:{number}", "question_id": f"{domain}:{number}",
                    "rollout_id": 0, "trajectory_id": f"{domain}-{number}",
                    "domain": domain, "source": domain, "split": "evolve_train",
                    "purpose": "evolution_train", "source_role": "train_evolution",
                    "collection_stage": "parent_pre_update",
                    "verification": {"status": "completed", "success": number % 2 == 0},
                    "terminal_reward": int(number % 2 == 0),
                    "error_type": "bad_call" if number % 3 else "parse_error",
                    "events": [{"step": 1, "message": "observed"}],
                })
        (baseline / "train_trajectories.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        result = self.adapter.call("prepare_meta_evidence", {
            "parent_rollout_dir": str(baseline),
            "output_dir": str(self.root / "meta_evidence"),
        })
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual((result["all_tasks"], result["selected_excerpts"]), (180, 48))
        sources = json.loads(Path(result["routing_sources_path"]).read_text())
        fingerprint = json.loads(Path(result["failure_fingerprint_path"]).read_text())
        self.assertEqual(fingerprint["task_count"], result["all_tasks"])
        selected = [row for row in sources
                    if row["evidence_detail"] == "bounded_representative_excerpt"]
        self.assertEqual(len(selected), 48)
        for domain in ("tool_use", "code", "searchqa"):
            group = [row for row in selected if row["domain"] == domain]
            self.assertEqual(len(group), 16)
            self.assertEqual(sum(row["verification"]["success"] is True for row in group), 8)

    def test_evaluation_uses_config_outside_snapshot_directory(self):
        from types import SimpleNamespace
        snapshot_dir = self.root / "validation" / "round_00"
        snapshot_dir.mkdir(parents=True)
        snapshot = snapshot_dir / "round_snapshot.json"
        snapshot.write_text("{}", encoding="utf-8")
        config = SimpleNamespace(model_dump=lambda: {"schema_version": 1},
                                 round_protocol={"skip_acebench": True},
                                 round_validation_config=str(self.root / "validation.json"))
        prepared = {"role": "independent_validation",
                    "manifest_path": str(self.root / "manifest.json")}
        fake_run = SimpleNamespace(returncode=0)
        with patch.object(self.adapter, "prepare_evaluation", return_value=prepared), \
             patch.object(self.adapter, "_config", return_value=config), \
             patch("task_adapter.subprocess.run", return_value=fake_run) as run:
            result = self.adapter.call("evaluate", {
                "config_path": "unused", "snapshot_path": str(snapshot),
                "output_dir": str(self.root / "eval_receipt"),
            })
        self.assertEqual(result["status"], "exited", result)
        command = run.call_args.args[0]
        effective = Path(command[command.index("--pipeline-config") + 1])
        self.assertFalse(snapshot_dir.is_relative_to(effective.parent))
        self.assertEqual(command[command.index("--manifest") + 1],
                         prepared["manifest_path"])
        self.assertIn("--skip-ace", command)

    def test_bootstrap_creates_a0_from_explicit_checkpoint_and_seed(self):
        checkpoint = self.root / "Qwen3-4B"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(b"offline-test-weights")
        result = self.adapter.call("bootstrap", {
            "checkpoint_path": str(checkpoint),
            "state_path": str(self.root / "a0.json"),
        })
        self.assertEqual(result["status"], "completed", result)
        state = json.loads(Path(result["state_path"]).read_text())
        self.assertEqual(state["generation"], 0)
        self.assertEqual(state["checkpoint_path"], str(checkpoint))
        self.assertEqual(len(state["checkpoint_manifest"]), 1)
        self.assertEqual(state["harness_path"],
                         str(ROOT / "seed_harness/harnessforge_base_manifest.json"))


    def test_real_round_scope_can_annotate_and_report_complete_pair(self):
        from types import SimpleNamespace
        from sia.task_meta.round_evolution import annotate_row, scoped_rollout
        from sia.task_meta.types import TaskAgentState

        config = self.adapter._config({"config_path": "configs/train_180_a0_v1.json"})
        seed = ROOT / "seed_harness/harnessforge_base_manifest.json"
        state = TaskAgentState(0, config.task_checkpoint, str(seed))
        output = self.root / "round_1" / "parent"
        with patch("sia.task_meta.pipeline.adapter_factory",
                   return_value=(lambda domain: None, None)):
            executor = self.adapter._executor(config, state, 1, "parent", output)
        task_id = next(iter(executor.store.members))
        scoped_rollout(executor, state, SimpleNamespace(task_id=task_id), 0,
                       output / "train_rollouts", {})
        row = {"task_id": task_id, "rollout_id": 0}
        annotate_row(executor, state, row)
        self.assertEqual(row["success_verifier_version"],
                         executor.round_protocol.execution_scope["protocol_hash"])
        rows = [{"task_id": task_id, "source": member["source"],
                 "verification": {"success": False}, "error_type": None}
                for task_id, member in executor.store.members.items()]
        result = SimpleNamespace(trajectories=rows, performance={})
        executor.round_protocol.observed(result, output)
        self.assertEqual(result.performance["evaluation_protocol"],
                         "same_round_full_B_parent_child_v2")
        self.assertEqual(sum(item["count"] for item in
                             result.performance["benchmarks"].values()), len(rows))
        with self.assertRaisesRegex(ValueError, "complete current B_r"):
            executor.round_protocol.observed(
                SimpleNamespace(trajectories=rows[:-1], performance={}), output)

    def test_independent_budget_includes_a0_initial_baseline(self):
        from evaluation_manifest import evaluation_budget

        frozen = {"manifest_hash": "frozen", "tasks": [
            {"source": "bfcl_v3"}, {"source": "livecodebench"}]}
        with patch("evaluation_manifest.manifest", return_value=frozen):
            budget = evaluation_budget(self.root / "manifest.json", "independent_validation")
        self.assertEqual(budget["checkpoints"], ["A0", "A1", "A2", "A3"])
        self.assertEqual(budget["tasks"], 2)

    def test_sft_refuses_candidate_rollout_as_training_source(self):
        from types import SimpleNamespace

        parent_dir = self.root / "wrong_sft_source"
        parent_dir.mkdir()
        (parent_dir / "train_trajectories.jsonl").write_text(
            json.dumps({"task_id": "one", "collection_stage": "child_post_update"}) + "\n",
            encoding="utf-8")
        (parent_dir / "performance.json").write_text(json.dumps({
            "total_rollouts": 1, "collection_stage": "child_post_update",
            "feedback_role": "train_evolution"}), encoding="utf-8")
        with (
            patch.object(self.adapter, "_config", return_value=SimpleNamespace(window_quotas={"tool_use": 1})),
            patch.object(self.adapter, "_state", return_value=SimpleNamespace()),
        ):
            result = self.adapter.call("sft", {
                "config_path": "unused", "state_path": "unused",
                "parent_rollout_dir": str(parent_dir),
                "output_dir": str(self.root / "sft_candidate"),
            })
        self.assertEqual(result["status"], "stage_error")
        self.assertIn("complete current parent", result["error"])

if __name__ == "__main__":
    unittest.main()
