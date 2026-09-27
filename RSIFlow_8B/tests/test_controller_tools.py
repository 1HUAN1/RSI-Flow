"""The Meta tool controller reports facts and never chooses a Task successor."""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller_tools import ControllerTools


class FakeTaskAdapter:
    def __init__(self):
        self.calls = []

    def call(self, operation, arguments):
        self.calls.append((operation, dict(arguments)))
        return {"status": "finished", "completed_tasks": 180, "result_path": "/run/results.json"}


def test_task_stages_delegate_without_acceptance(tmp_path):
    adapter = FakeTaskAdapter()
    tools = ControllerTools(tmp_path, adapter=adapter)
    parent = tools.execute({"operation": "run_parent", "round": 1})
    child = tools.execute({"operation": "run_candidate", "round": 1})
    sft = tools.execute({"operation": "sft", "positive_trajectories": "success.jsonl"})
    assert adapter.calls == [
        ("rollout", {"round": 1, "stage": "parent"}),
        ("rollout", {"round": 1, "stage": "candidate"}),
        ("sft", {"positive_trajectories": "success.jsonl"}),
    ]
    for value in (parent, child, sft):
        assert value["completed_tasks"] == 180
        assert "accepted" not in value
        assert "selected_component" not in value


def test_compare_scores_reports_delta_but_does_not_deploy(tmp_path):
    before = {"macro_success": 0.2, "total_rollouts": 180, "probe_identity": "same",
              "domains": {"code": {"correct": 12, "denominator": 60}}}
    after = {"macro_success": 0.25, "total_rollouts": 180, "probe_identity": "same",
             "domains": {"code": {"correct": 13, "denominator": 60}}}
    (tmp_path / "before.json").write_text(json.dumps(before), encoding="utf-8")
    (tmp_path / "after.json").write_text(json.dumps(after), encoding="utf-8")
    tools = ControllerTools(tmp_path)
    result = tools.execute({"operation": "compare_scores", "before": "before.json", "after": "after.json"})
    assert result["delta"] == 0.04999999999999999
    assert result["before_facts"]["total_rollouts"] == 180
    assert result["after_facts"]["probe_identity"] == "same"
    assert "accepted" not in result
    assert not (tmp_path / "active.json").exists()


def test_meta_explicitly_activates_task(tmp_path):
    (tmp_path / "parent.json").write_text('{"version":"T0"}', encoding="utf-8")
    (tmp_path / "candidate.json").write_text('{"version":"T1"}', encoding="utf-8")
    tools = ControllerTools(tmp_path)
    tools.execute({"operation": "activate_task", "source_state": "parent.json",
                   "active_state": "active.json", "selection_note": "retain T0"})
    assert json.loads((tmp_path / "active.json").read_text())["version"] == "T0"
    tools.execute({"operation": "deploy_candidate", "source_state": "candidate.json",
                   "active_state": "active.json", "selection_note": "use T1"})
    assert json.loads((tmp_path / "active.json").read_text())["version"] == "T1"


def test_skill_append_and_snapshot_keep_previous_entries(tmp_path):
    tools = ControllerTools(tmp_path)
    tools.execute({"operation": "append_skills", "path": "meta/skills.jsonl",
                   "entries": [{"id": "skill.MODEL.1", "outcome": "failure"}]})
    tools.execute({"operation": "append_skills", "path": "meta/skills.jsonl",
                   "entries": [{"id": "principle.2", "outcome": "success"}]})
    entries = [json.loads(line) for line in (tmp_path / "meta/skills.jsonl").read_text().splitlines()]
    assert [entry["id"] for entry in entries] == ["skill.MODEL.1", "principle.2"]
    (tmp_path / "active_task.json").write_text('{"version":"T1"}', encoding="utf-8")
    result = tools.execute({"operation": "snapshot", "destination": "round_1_snapshot",
                            "sources": {"task.json": "active_task.json",
                                        "skills.jsonl": "meta/skills.jsonl"}})
    assert result["status"] == "copied"
    assert (tmp_path / "round_1_snapshot/task.json").is_file()
    assert (tmp_path / "round_1_snapshot/skills.jsonl").read_text() == (tmp_path / "meta/skills.jsonl").read_text()


def test_command_writes_fact_receipt_without_controller_decision(tmp_path):
    tools = ControllerTools(tmp_path)
    result = tools.execute({"operation": "run_command", "argv": [sys.executable, "-c", "print('finished 180')"]})
    assert result["status"] == "exited"
    assert result["returncode"] == 0
    assert "finished 180" in Path(result["stdout_path"]).read_text()
    assert "accepted" not in result
    assert Path(result["receipt_path"]).is_file()


def test_cli_accepts_one_json_request(tmp_path):
    path = Path(__file__).resolve().parents[1] / "controller_tools.py"
    proc = subprocess.run([sys.executable, str(path), "--workspace", str(tmp_path)],
                          input='{"operation":"read_json","path":"fact.json"}',
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert json.loads(proc.stdout)["status"] == "tool_error"
    assert proc.returncode == 0



def test_cli_writes_atomic_result_file_instead_of_stdout(tmp_path):
    path = Path(__file__).resolve().parents[1] / "controller_tools.py"
    result_path = tmp_path / "transport/result.json"
    proc = subprocess.run(
        [sys.executable, str(path), "--workspace", str(tmp_path),
         "--result-file", str(result_path)],
        input='{"operation":"read_json","path":"missing.json"}',
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.returncode == 0
    assert proc.stdout == ""
    assert json.loads(result_path.read_text())["status"] == "tool_error"
    assert not list(result_path.parent.glob("*.tmp-*"))


def test_prepare_validation_snapshot_matches_evaluator_handoff(tmp_path):
    (tmp_path / "task.json").write_text('{"generation":1,"harness_path":"harness.json"}', encoding="utf-8")
    (tmp_path / "meta.json").write_text('{"version":2}', encoding="utf-8")
    (tmp_path / "round_complete.json").write_text('{"status":"complete"}', encoding="utf-8")
    tools = ControllerTools(tmp_path)
    receipt = tools.execute({
        "operation": "prepare_validation_snapshot", "task_state_path": "task.json",
        "meta_state_path": "meta.json", "source_round_path": "round_complete.json",
        "round": 1, "chosen_component": "HARNESS", "deployment_status": "deployed",
        "destination": "validation/round_01",
    })
    snapshot = json.loads(Path(receipt["snapshot_path"]).read_text())
    assert receipt["status"] == "prepared"
    assert snapshot["task_state"]["generation"] == 1
    assert snapshot["meta_state"]["version"] == 2
    assert snapshot["source_round"] == str(tmp_path / "round_complete.json")
    assert snapshot["chosen_component"] == "HARNESS"
    assert snapshot["purpose"] == "report_only"
    assert "accepted" not in receipt


def test_file_edits_are_exact_meta_requests(tmp_path):
    tools = ControllerTools(tmp_path)
    written = tools.execute({"operation": "write_text", "path": "candidate/provider.py",
                             "content": "VALUE = 1\n"})
    assert written["status"] == "written"
    assert (tmp_path / "candidate/provider.py").read_text() == "VALUE = 1\n"
    patch_text = (
        "--- a/candidate/provider.py\n"
        "+++ b/candidate/provider.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n"
    )
    patched = tools.execute({"operation": "apply_patch", "patch": patch_text})
    assert patched["returncode"] == 0
    assert patched["changed_files"][0]["sha256"]
    assert (tmp_path / "candidate/provider.py").read_text() == "VALUE = 2\n"
    assert "accepted" not in patched



def test_task_meta_snapshot_versions_only_changed_components(tmp_path):
    from component_versions import ComponentVersions
    tools = ControllerTools(tmp_path)
    (tmp_path / "h1.json").write_text('{"files":{"builder.py":"original"}}')
    (tmp_path / "h2.json").write_text('{"files":{"builder.py":"candidate"}}')
    assets = tmp_path / "artifacts"
    assets.mkdir()
    (assets / "workflow.md").write_text("workflow")
    state = {
        "generation": 0, "model_ref": "/models/qwen3-4b",
        "checkpoint_path": "/models/qwen3-4b",
        "checkpoint_manifest": [{"path": "weights.safetensors", "sha256": "abc"}],
        "harness_path": str(tmp_path / "h1.json"),
        "artifacts": {"directory": str(assets), "manifest": []},
    }
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps(state))
    skills = tmp_path / "skills.jsonl"
    skills.write_text('{"id":"skill.MODEL.1"}\n')
    (tmp_path / "context.json").write_text('{"round":1}')
    def save(name, selected, **kwargs):
        receipt = tools.execute({
            "operation": "snapshot_task_meta", "active_task_state": selected,
            "skills_path": str(skills), "destination": str(tmp_path / "snapshots" / name),
            "versions_root": str(tmp_path / "versions"), **kwargs})
        assert receipt["status"] == "snapshotted", receipt
        return receipt
    a0 = save("A0", str(parent))
    child = tmp_path / "candidate.json"
    child.write_text(json.dumps({**state, "generation": 1, "harness_path": str(tmp_path / "h2.json")}))
    with skills.open("a") as stream:
        stream.write('{"id":"skill.HARNESS.2"}\n')
    b1 = save("B1", str(child), parent_task_state=str(parent), candidate_task_state=str(child),
              context_path="context.json", meta_harness_path="persistent_meta.py")
    b2 = save("B2", str(child))
    index = json.loads((tmp_path / "versions/index.json").read_text())
    assert {k: len(v) for k, v in index.items()} == {"model": 1, "harness": 2, "artifacts": 1, "meta": 2}
    assert b1["combinations"]["parent"]["harness"] == "harness_001"
    assert b1["combinations"]["candidate"]["harness"] == "harness_002"
    assert b1["combinations"]["selected"]["model"] == "model_001"
    assert b1["context_reference"]["source"] == "context.json"
    assert b1["context_reference"]["sha256"]
    assert b2["meta"]["id"] == b1["meta"]["id"] == "meta_002"
    assert a0["meta"]["id"] == "meta_001"
    assert len(Path(a0["skills_snapshot_path"]).read_text().splitlines()) == 1
    assert len(Path(b1["skills_snapshot_path"]).read_text().splitlines()) == 2
    assert not (tmp_path / "snapshots/B1/meta/fixed_harness").exists()
    assert not (tmp_path / "snapshots/B1/task/artifacts").exists()
    assert not list((tmp_path / "versions").rglob("*.safetensors"))
    assert json.loads(Path(b1["checkpoint_reference_path"]).read_text())["weights_copied"] is False
    restored = json.loads(Path(b1["restorable_task_state_path"]).read_text())
    assert Path(restored["harness_path"]).read_text() == (tmp_path / "h2.json").read_text()
    assert (Path(restored["artifacts"]["directory"]) / "workflow.md").is_file()
    # A rejected candidate is still archived but does not become selected.
    rejected = save("rejected", str(parent), parent_task_state=str(parent), candidate_task_state=str(child))
    assert rejected["combinations"]["selected"]["harness"] == "harness_001"
    assert rejected["combinations"]["candidate"]["harness"] == "harness_002"
    registry = ComponentVersions(tmp_path / "versions", tmp_path)
    assert len(json.loads(registry.index_path.read_text())["harness"]) == 2


def test_compose_model_harness_for_ablation_without_deployment(tmp_path):
    from component_versions import ComponentVersions
    registry = ComponentVersions(tmp_path / "versions", tmp_path)
    tools = ControllerTools(tmp_path)
    states = []
    for n in (1, 2):
        harness = tmp_path / f"h{n}.json"
        harness.write_text(json.dumps({"code": str(n)}))
        state = tmp_path / f"task{n}.json"
        state.write_text(json.dumps({"generation": n, "model_ref": f"/models/M{n}",
                                    "checkpoint_path": f"/models/M{n}", "checkpoint_manifest": [],
                                    "harness_path": str(harness),
                                    "artifacts": {"directory": None, "manifest": []}}))
        registry.task(state)
        states.append(state)
    receipt = tools.execute({
        "operation": "compose_task", "versions_root": str(registry.root),
        "template_state_path": str(states[1]), "model_id": "model_001",
        "harness_id": "harness_002", "destination": str(tmp_path / "ablation.json")})
    assert receipt["status"] == "composed", receipt
    combined = json.loads(Path(receipt["state_path"]).read_text())
    assert combined["checkpoint_path"] == "/models/M1"
    assert json.loads(Path(combined["harness_path"]).read_text()) == {"code": "2"}
    assert receipt["activated"] is False
    assert json.loads(states[1].read_text())["checkpoint_path"] == "/models/M2"
    assert not (tmp_path / "active_task.json").exists()


def test_missing_snapshot_input_can_be_retried(tmp_path):
    tools = ControllerTools(tmp_path)
    result = tools.execute({"operation": "snapshot_task_meta", "active_task_state": "missing.json",
                            "skills_path": "skills.jsonl", "destination": "snapshots/B1"})
    assert result["status"] == "tool_error"
    assert not (tmp_path / "snapshots/B1").exists()



def test_read_text_pages_unicode_and_bounds_default(tmp_path):
    content = "前缀" + ("x" * 70000) + "终点"
    (tmp_path / "long.txt").write_text(content, encoding="utf-8")
    tools = ControllerTools(tmp_path)
    first = tools.execute({"operation": "read_text", "path": "long.txt"})
    assert first["status"] == "read"
    assert len(first["content"]) == 65536
    assert first["offset_chars"] == 0
    assert first["next_offset_chars"] == 65536
    assert first["total_chars"] == len(content)
    assert first["truncated"] is True
    second = tools.execute({"operation": "read_text", "path": "long.txt",
                            "offset_chars": first["next_offset_chars"],
                            "max_chars": 1000000})
    assert len(second["content"]) == len(content) - 65536
    assert second["next_offset_chars"] == len(content)
    assert second["truncated"] is False
    assert first["content"] + second["content"] == content



def test_compare_scores_reports_task_pairing_without_artifact_protocol_verdict(tmp_path):
    before_dir, after_dir = tmp_path / "before", tmp_path / "after"
    before_dir.mkdir()
    after_dir.mkdir()
    before = {"macro_success": 0.1, "total_rollouts": 2, "round_id": 1,
              "training_manifest": "B1", "evaluation_protocol": "same_round_full_B_parent_child_v2"}
    after = {"macro_success": 0.2, "total_rollouts": 2, "round_id": 1,
             "training_manifest": "B1", "evaluation_protocol": "direct_submission_same_tasks_v1"}
    (before_dir / "performance.json").write_text(json.dumps(before), encoding="utf-8")
    (after_dir / "performance.json").write_text(json.dumps(after), encoding="utf-8")
    parent_rows = [
        {"task_id": "A", "rollout_id": 0, "task_source_hash": "hash-A"},
        {"task_id": "B", "rollout_id": 0, "task_source_hash": "hash-B"},
    ]
    child_rows = [
        {"task_id": "A", "rollout_id": 0, "task_source_hash": "hash-A"},
        {"task_id": "C", "rollout_id": 0, "task_source_hash": "hash-C"},
    ]
    (before_dir / "train_trajectories.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in parent_rows), encoding="utf-8")
    (after_dir / "train_trajectories.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in child_rows), encoding="utf-8")
    result = ControllerTools(tmp_path).execute({
        "operation": "compare_scores", "before": "before/performance.json",
        "after": "after/performance.json",
    })
    assert result["status"] == "compared"
    assert result["matching_facts"]["training_manifest"] is True
    assert result["trajectory_pairing"]["same_id_set"] is False
    assert result["trajectory_pairing"]["missing_in_after"][0]["task_id"] == "B"
    assert result["trajectory_pairing"]["extra_in_after"][0]["task_id"] == "C"
    assert result["before_facts"]["evaluation_protocol"] != result["after_facts"]["evaluation_protocol"]
    assert "accepted" not in result
