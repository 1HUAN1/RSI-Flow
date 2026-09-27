"""Offline checks for the persistent Codex launcher and fresh tool process."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from launch_meta import (SubprocessToolExecutor, experiment_prompt, main, prepare_codex_home,
                         prepare_context_path, prepare_meta_storage, prepare_skills_path, wait_for_idle_gpus)  # noqa: E402


def test_gpu_wait_does_not_start_on_busy_or_only_one_idle_sample(tmp_path):
    idle = [{"gpu": i, "memory_used_mib": 10, "utilization_percent": 0} for i in range(4)]
    busy = [{**row, "memory_used_mib": 17000} for row in idle]
    samples = iter([busy, idle, busy, idle, idle])
    pauses = []
    result = wait_for_idle_gpus(tmp_path, probe=lambda: next(samples), pause=pauses.append)
    assert result["status"] == "gpus_idle_starting"
    assert pauses == [60, 60, 60, 60]
    assert json.loads((tmp_path / "gpu_wait.json").read_text())["consecutive_idle_checks"] == 2


def test_gpu_probe_error_or_missing_card_keeps_waiting_not_aborting(tmp_path):
    idle = [{"gpu": i, "memory_used_mib": 0, "utilization_percent": 0} for i in range(4)]
    samples = iter([RuntimeError("NVML temporarily unavailable"), idle[:3], idle, idle])
    states = []
    def probe():
        value = next(samples)
        if isinstance(value, Exception):
            raise value
        return value
    def pause(seconds):
        states.append(json.loads((tmp_path / "gpu_wait.json").read_text()))
    assert wait_for_idle_gpus(tmp_path, probe=probe, pause=pause)["status"] == "gpus_idle_starting"
    assert "probe_error" in states[0]
    assert states[1]["consecutive_idle_checks"] == 0


def test_new_run_writes_meta_records_into_central_directory(tmp_path):
    run = tmp_path / "runs/new_run"
    central = prepare_meta_storage(run, tmp_path / "Meta_logs")
    assert (run / "meta_session").is_symlink()
    assert (run / "meta").is_symlink()
    assert not (central / "meta_session").is_symlink()
    (run / "meta_session/state.json").write_text('{"status":"running"}')
    prepare_skills_path(run).write_text('{"id":"skill.HARNESS.test"}\n')
    assert (central / "meta_session/state.json").read_text() == '{"status":"running"}'
    assert (central / "meta/skills.jsonl").read_text().startswith('{"id"')
    assert prepare_meta_storage(run, tmp_path / "Meta_logs") == central


def test_live_legacy_records_keep_same_directory_and_open_file(tmp_path):
    run = tmp_path / "runs/active_run"
    session = run / "meta_session"
    session.mkdir(parents=True)
    (run / "meta").mkdir()
    before = session.stat().st_ino
    with (session / "events.jsonl").open("w") as stream:
        stream.write('first\n')
        stream.flush()
        central = prepare_meta_storage(run, tmp_path / "Meta_logs")
        stream.write('second\n')
        stream.flush()
        assert (central / "meta_session/events.jsonl").read_text() == 'first\nsecond\n'
    assert session.stat().st_ino == before
    assert not session.is_symlink()
    assert (central / "meta_session").is_symlink()
    assert prepare_meta_storage(run, tmp_path / "Meta_logs") == central


def test_central_paths_work_with_monitor_and_progress(tmp_path):
    from monitor_meta import read_json
    from experiment_progress import ExperimentProgress
    run = tmp_path / "runs/new_run"
    central = prepare_meta_storage(run, tmp_path / "Meta_logs")
    (run / "meta_session/state.json").write_text('{"status":"running"}')
    assert read_json(central / "meta_session/state.json")["status"] == "running"
    state = ExperimentProgress(run / "meta_session").rebuild()
    assert state["current_stage"] == "A0"
    assert (central / "meta_session/workflow_state.json").exists()


def test_skills_and_context_are_initialized_once(tmp_path):
    run = tmp_path / "run"
    skills = prepare_skills_path(run)
    context = prepare_context_path(run)
    assert skills.read_text() == ""
    assert context.read_text() == "{}\n"
    skills.write_text('{"id":"skill.HARNESS.1"}\n')
    context.write_text('{"round":1}\n')
    assert prepare_skills_path(run).read_text().startswith('{"id"')
    assert json.loads(prepare_context_path(run).read_text())["round"] == 1


def test_codex_home_is_persistent_and_resolves_model_catalog(tmp_path):
    home = prepare_codex_home(tmp_path / "run")
    config = (home / "config.toml").read_text()
    assert "__MODEL_CATALOG_JSON__" not in config
    assert "DeepSeek-V4.1-Flash" in config
    assert "AUTODL_API_KEY" in config
    assert "api_key =" not in config
    assert prepare_codex_home(tmp_path / "run") == home


def test_tool_executor_starts_new_controller_process_each_call(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    calls = []

    def fake_run(argv, *, input, text, stdout, stderr, cwd, check, env):
        assert "AUTODL_API_KEY" not in env
        calls.append((argv, json.loads(input)))
        stdout.write("rollout log that is not JSON\n")
        stderr.write("diagnostic log\n")
        result_path = Path(argv[argv.index("--result-file") + 1])
        result_path.write_text('{"status":"fact"}', encoding="utf-8")
        return SimpleNamespace(returncode=0)

    adapter = SubprocessToolExecutor(project_root=root, run_dir=tmp_path / "run")
    with patch("launch_meta.subprocess.run", side_effect=fake_run):
        assert adapter.execute({"operation": "run_parent"}) == {"status": "fact"}
        assert adapter.execute({"operation": "compare_scores"}) == {"status": "fact"}
    assert len(calls) == 2
    assert calls[0][0][1] == str(root / "controller_tools.py")
    assert "--result-file" in calls[0][0]
    assert calls[1][1]["operation"] == "compare_scores"
    call_dirs = list((tmp_path / "run/controller_transport").iterdir())
    assert len(call_dirs) == 2
    assert all((path / "stdout.log").read_text().startswith("rollout log") for path in call_dirs)


def test_tool_executor_error_returns_transport_log_paths(tmp_path):
    root = tmp_path / "project"
    root.mkdir()

    def fake_run(argv, *, input, text, stdout, stderr, cwd, check, env):
        stdout.write("noisy partial output\n")
        stderr.write("controller crashed\n")
        return SimpleNamespace(returncode=9)

    adapter = SubprocessToolExecutor(project_root=root, run_dir=tmp_path / "run")
    with patch("launch_meta.subprocess.run", side_effect=fake_run):
        receipt = adapter.execute({"operation": "run_parent"})
    assert receipt["status"] == "tool_process_error"
    assert receipt["returncode"] == 9
    assert Path(receipt["stdout_path"]).read_text() == "noisy partial output\n"
    assert Path(receipt["stderr_path"]).read_text() == "controller crashed\n"
    assert receipt["result_path"].endswith("result.json")


def test_experiment_prompt_assigns_acceptance_to_meta(tmp_path):
    config = {"rounds": 3, "training_tasks_per_pass": 180,
              "train_quotas_per_round": {"a": 60, "b": 60, "c": 60},
              "evaluate_initial_system": True, "skip_acebench": True}
    prompt = experiment_prompt(tmp_path / "config.json", config, tmp_path / "run")
    assert "48 representative trajectory excerpts" in prompt
    assert "one candidate" in prompt
    assert "strictly improve" in prompt
    assert "you must decide" in prompt
    assert '"rounds": 3' in prompt
    assert '"training_tasks_per_round": 180' in prompt
    assert "skill.HARNESS.<id>" in prompt
    assert "skill.MODEL.<id>" in prompt
    assert "skill.ARTIFACTS.<id>" in prompt
    assert "principle.<id>" in prompt
    assert "original rollout path, task ID and content hash" in prompt
    assert "Never copy a full trajectory" in prompt
    assert "reads relevant entries" in prompt
    assert "next_offset_chars" in prompt
    assert "Mainline stages:" in prompt
    assert "A0: freeze the initial Task/Meta state" in prompt
    assert "B1-B3: in each round" in prompt
    assert "workflow_state.json" in prompt
    assert "does not choose the component, candidate, or acceptance outcome" in prompt
    assert '"workspace_root"' in prompt
    assert '"project_root"' in prompt
    assert "Native shell and apply_patch are available" in prompt
    assert "tool implementations are fixed and may not be modified" not in prompt
    assert "Never pick a second candidate" not in prompt
    assert "Rejection returns to component selection" in prompt
    assert "do not go to validation or the next round" in prompt
    assert "attempts/attempt_K" in prompt
    assert "Calling the validator alone is NOT HarnessForge production" in prompt
    assert "BEFORE candidate edits" in prompt
    assert "at most 3 times" in prompt



def test_production_loop_receives_live_progress_callbacks(tmp_path, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "rounds": 3,
        "output_root": str(tmp_path),
        "run_name": "unused",
    }))
    run_dir = tmp_path / "run"
    captured = {"rebuilds": 0, "recorded_turns": []}

    class FakeProgress:
        def __init__(self, session, *, rounds):
            captured["session"] = session
            captured["rounds"] = rounds

        def rebuild(self):
            captured["rebuilds"] += 1
            return {
                "finish_allowed": False,
                "current_stage": "B1",
                "stages": [{"id": "B1", "remaining_milestones": ["parent_rollout"]}],
            }

        def compact_status_block(self):
            return "Current stage: B1; next: parent_rollout"

        def record_turn(self, turn_dir):
            captured["recorded_turns"].append(turn_dir)
            return self.rebuild()

    class FakeLoop:
        def __init__(self, **kwargs):
            captured["loop_kwargs"] = kwargs

        def run_experiment(self, prompt):
            captured["prompt"] = prompt
            return {"status": "complete", "thread_id": "thread-1", "summary": "done"}

    with (patch("launch_meta._autodl_key", return_value="secret"),
          patch("launch_meta.prepare_meta_storage"),
          patch("launch_meta.prepare_skills_path"),
          patch("launch_meta.prepare_context_path"),
          patch("launch_meta.prepare_codex_home", return_value=tmp_path / "codex-home"),
          patch("launch_meta.SubprocessToolExecutor", return_value=object()),
          patch("launch_meta.PersistentMeta", FakeLoop)):
        assert main(["--config", str(config_path), "--run-dir", str(run_dir)]) == 0

    loop_kwargs = captured["loop_kwargs"]
    assert loop_kwargs["run"] == run_dir
    assert loop_kwargs["rounds"] == 3
    assert loop_kwargs["codex_home"] == tmp_path / "codex-home"
    assert "native" in captured["prompt"].lower()
    assert json.loads(capsys.readouterr().out)["thread_id"] == "thread-1"
