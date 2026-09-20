"""Rollout output must not overwrite the artifact input needed for replay."""

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from sia.profiles import load_target_agent_profile
from sia.task_meta.execution import SIAExecutor, prepare_gpqa_task
from sia.task_meta.storage import artifact_manifest, clone_task, save_json
from sia.task_meta.types import ArtifactState, TaskAgentState


def test_eight_rollouts_preserve_generation_and_historical_artifact_inputs(tmp_path, monkeypatch):
    source = Path(__file__).parents[1] / "sia/tasks/gpqa"
    task_dir = tmp_path / "task"
    batch = prepare_gpqa_task(source, task_dir, 2)
    previous_dir = tmp_path / "gen_0"
    (previous_dir / "artifacts").mkdir(parents=True)
    (previous_dir / "artifacts/strategy.md").write_text("old reusable strategy", encoding="utf-8")
    (previous_dir / "target_agent.py").write_text("pass", encoding="utf-8")
    original_manifest = artifact_manifest(previous_dir / "artifacts")
    previous = TaskAgentState(
        0, "base", str(previous_dir / "target_agent.py"),
        ArtifactState(str(previous_dir / "artifacts"), original_manifest),
    )
    generation_dir = tmp_path / "gen_1"
    state = clone_task(previous, 1, generation_dir)
    original_state = asdict(state)
    save_json(generation_dir / "task_state.json", state)
    provider = load_target_agent_profile("task-meta-qwen").provider
    executor = SIAExecutor(task_dir, sys.prefix, provider, rollouts_per_task=8)
    observed_inputs = []

    def fake_target(command, **kwargs):
        work = Path(command[command.index("--working_dir") + 1])
        observed_inputs.append(artifact_manifest(work / "artifacts_input"))
        repetition = len(observed_inputs) - 1
        from sia.task_meta.gpqa_target import safe_task_id
        rows = []
        for question_id in batch["task_ids"]:
            relative = f"{safe_task_id(question_id)}/rollout_{repetition}/strategy.md"
            path = work / "artifacts_generated" / relative
            path.parent.mkdir(parents=True)
            path.write_text(f"new strategy {question_id} {repetition}", encoding="utf-8")
            rows.append({
                "question_id": question_id, "model_answer": "A", "valid_answer": True,
                "messages": [{"role": "assistant", "content": "solution"}],
                "generated_artifacts": [{"path": relative, "production_method": "task_response.reusable_note"}],
            })
        save_json(work / "agent_execution.json", rows)
        return SimpleNamespace(returncode=0)

    def fake_evaluator(directory, *args, **kwargs):
        save_json(Path(directory) / "results.json", {
            "details": [{"question_id": qid, "is_correct": True} for qid in batch["task_ids"]],
        })
        return {"status": "success"}

    monkeypatch.setattr("sia.task_meta.execution.subprocess.run", fake_target)
    monkeypatch.setattr("sia.orchestrator.run_evaluation", fake_evaluator)
    result = executor.execute(state, generation_dir)

    assert observed_inputs == [original_manifest] * 8
    assert result.performance["attempts"] == 16
    saved_input = json.loads((generation_dir / "evaluated_task_state.json").read_text(encoding="utf-8"))
    input_path = Path(saved_input["artifacts"]["directory"])
    assert input_path == generation_dir / "artifacts_input"
    assert saved_input["artifacts"]["manifest"] == original_manifest
    assert artifact_manifest(input_path) == original_manifest
    assert artifact_manifest(previous_dir / "artifacts") == original_manifest
    assert len(result.output_artifacts.manifest) == 16
    assert not (Path(result.output_artifacts.directory) / "strategy.md").exists()
    assert len(result.artifact_provenance) == 16
    assert all(p["terminal_reward"] == 1 and p["knowledge_verified"] is False for p in result.artifact_provenance)
    assert result.rollout_artifact_diff["lifecycle_removed"] == ["strategy.md"]
    assert asdict(state) == original_state
    assert result.evaluated_state.artifacts.directory == str(input_path)
    assert input_path != Path(result.output_artifacts.directory)
