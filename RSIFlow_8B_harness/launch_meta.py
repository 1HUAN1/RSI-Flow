"""Start or resume one Codex-led Task/Meta experiment.

The Codex app-server stays alive throughout the experiment and calls native
experiment tools. Python transports tool results and preserves durable jobs.

Offline check: ``python launch_meta.py --check``
Actual run: ``python launch_meta.py --run-dir /root/data/RSI_iclr2027/rsiH/Rollout_logs/runs/<name>``
Use --detach for a server-side launcher independent of the invoking terminal.
Re-run the same command to resume the native thread and existing jobs.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from meta_loop import MAINLINE_REMINDER
from persistent_meta import PERSISTENT_META_INSTRUCTIONS, PersistentMeta
from skill_memory import DEFAULT_CONTEXT_CHARS, DEFAULT_PER_CATEGORY, SKILL_GUIDANCE, SkillMemory, write_json


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs/train_180_a0_v1.json"
KEY_LABEL = "autodl.art"


def gpu_sample():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True,
                            check=True, timeout=20)
    return [{"gpu": int(index), "memory_used_mib": int(memory), "utilization_percent": int(utilization)}
            for line in result.stdout.splitlines() if line.strip()
            for index, memory, utilization in [line.split(",")]]


def process_still_running(record):
    try:
        fields = (Path("/proc") / str(record["pid"]) / "stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] != "Z" and fields[19] == str(record["start_ticks"])
    except FileNotFoundError:
        return False


def wait_for_idle_gpus(session_dir: Path, *, probe=None, pause=None, poll_seconds=60, gpu_ids=None, wait_processes=None):
    """Wait before starting Meta, never preempt another GPU experiment."""
    probe, pause = probe or gpu_sample, pause or time.sleep
    gpu_ids = list(gpu_ids if gpu_ids is not None else range(4))
    wait_processes = wait_processes or []
    stable = 0
    while True:
        state = {"status": "waiting_for_gpus", "launcher_pid": os.getpid(),
                 "updated_at": time.time(), "poll_seconds": poll_seconds,
                 "gpu_ids": gpu_ids}
        try:
            samples = {row["gpu"]: row for row in probe()}
            idle = all(gpu in samples and samples[gpu]["memory_used_mib"] <= 1024
                       and samples[gpu]["utilization_percent"] <= 5 for gpu in gpu_ids)
            alive = [p for p in wait_processes if process_still_running(p)]
            idle = idle and not alive
            state["waiting_processes"] = alive
            stable = stable + 1 if idle else 0
            state.update(gpus=list(samples.values()), consecutive_idle_checks=stable)
        except Exception as exc:
            stable = 0
            state["probe_error"] = f"{type(exc).__name__}: {exc}"
        if stable >= 2:
            state["status"] = "gpus_idle_starting"
        write_json(session_dir / "gpu_wait.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        if stable >= 2:
            return state
        pause(poll_seconds)


def prepare_meta_storage(run_dir: Path, meta_logs_root: Path | None = None) -> Path:
    """Central Meta records; preserve live legacy directories without moving open files."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    root = Path(meta_logs_root or PROJECT_ROOT.parent / "Meta_logs").resolve()
    central = root / run_dir.name
    central.mkdir(parents=True, exist_ok=True)
    for name in ("meta_session", "meta", "meta_harness"):
        legacy, destination = run_dir / name, central / name
        if legacy.exists():
            if not destination.exists() and not destination.is_symlink():
                destination.symlink_to(legacy.resolve(), target_is_directory=True)
            elif not os.path.samefile(legacy, destination):
                raise FileExistsError(f"Meta records already belong to a different run: {destination}")
        else:
            destination.mkdir(exist_ok=True)
            legacy.symlink_to(destination, target_is_directory=True)
    # These transports also hold Task job stdout. Keep their data in Rollout_logs,
    # with convenient links here rather than duplicating potentially large logs.
    for name in ("controller_receipts", "controller_transport"):
        target = run_dir / name
        target.mkdir(exist_ok=True)
        link = central / name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(target, target_is_directory=True)
    return central


def _autodl_key() -> str:
    """Prefer the environment; optionally reuse the local 4B secret file."""
    value = os.environ.get("AUTODL_API_KEY", "").strip()
    if value:
        return value
    for path in (PROJECT_ROOT / "API_key.md", PROJECT_ROOT.parent / "RSIFlow_4B/API_key.md"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            for marker in (f"{KEY_LABEL}：", f"{KEY_LABEL}:"):
                if line.startswith(marker):
                    value = line[len(marker):].strip()
                    if value:
                        return value
    raise RuntimeError("AUTODL_API_KEY is unset and no local autodl.art key was found")


def prepare_codex_home(run_dir: Path) -> Path:
    """Put Codex config and resumable thread state on the data disk."""
    codex_home = run_dir / "meta_session/codex_home"
    codex_home.mkdir(parents=True, exist_ok=True)
    template = (PROJECT_ROOT / "configs/codex_host.toml").read_text(encoding="utf-8")
    catalog = (PROJECT_ROOT / "configs/codex-deepseek-v4.1-flash.json").resolve()
    config = template.replace("__MODEL_CATALOG_JSON__", str(catalog))
    config_path = codex_home / "config.toml"
    if not config_path.exists():
        config_path.write_text(config, encoding="utf-8")
    return codex_home


def prepare_skills_path(run_dir: Path, seed_path: Path | None = None) -> Path:
    """Create one append-only Meta skill ledger; preserve it on resume."""
    path = run_dir / "meta/skills.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    created = not path.exists()
    entries = []
    if created and seed_path:
        with Path(seed_path).open(encoding="utf-8") as stream:
            entries = [json.loads(line) for line in stream if line.strip()]
        if not all(isinstance(entry, dict) for entry in entries):
            raise ValueError("Skill seed must contain one JSON object per line")
    path.touch(exist_ok=True)
    memory = SkillMemory(path)
    if created and seed_path:
        memory.append(entries)
    else:
        memory.rebuild_index()
    if created:
        (path.parent / "skills_initialization.json").write_text(json.dumps({
            "seed_path": str(seed_path) if seed_path else None,
            "mode": "seeded" if seed_path else "empty",
            "note": "A seed changes initial Meta knowledge; record it when comparing experiments."},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def prepare_context_path(run_dir: Path) -> Path:
    """Keep a concise cross-round handoff separate from the Codex conversation."""
    path = run_dir / "meta/context.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("{}\n", encoding="utf-8")
    return path


class SubprocessToolExecutor:
    """Run a short fixed tool in a separate process, with a dedicated result file."""

    def __init__(self, *, project_root: Path, run_dir: Path, python: str = sys.executable,
                 workspace: Path | None = None):
        self.project_root = Path(project_root)
        self.run_dir = Path(run_dir)
        self.python = python
        self.workspace = Path(workspace) if workspace is not None else self.project_root

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        transport_dir = self.run_dir / "controller_transport" / uuid.uuid4().hex
        transport_dir.mkdir(parents=True, exist_ok=False)
        result_path = transport_dir / "result.json"
        stdout_path = transport_dir / "stdout.log"
        stderr_path = transport_dir / "stderr.log"
        argv = [self.python, str(self.project_root / "controller_tools.py"),
                "--workspace", str(self.workspace), "--receipts",
                str(self.run_dir / "controller_receipts"),
                "--result-file", str(result_path)]
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout, \
                    stderr_path.open("w", encoding="utf-8") as stderr:
                completed = subprocess.run(
                    argv, input=json.dumps(request, ensure_ascii=False), text=True,
                    stdout=stdout, stderr=stderr, cwd=self.project_root, check=False,
                    env={k: v for k, v in os.environ.items() if k != "AUTODL_API_KEY"},
                )
        except OSError as exc:
            return {"operation": request.get("operation"), "status": "tool_process_error",
                    "error": str(exc), "returncode": None,
                    "result_path": str(result_path),
                    "stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}
        if completed.returncode:
            return {"operation": request.get("operation"), "status": "tool_process_error",
                    "returncode": completed.returncode, "result_path": str(result_path),
                    "stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}
        if not result_path.is_file():
            return {"operation": request.get("operation"), "status": "invalid_tool_receipt",
                    "error": "controller process did not write its result file",
                    "result_path": str(result_path),
                    "stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}
        try:
            receipt = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(receipt, dict):
                raise TypeError("controller result must be a JSON object")
            return receipt
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            return {"operation": request.get("operation"), "status": "invalid_tool_receipt",
                    "error": str(exc), "result_path": str(result_path),
                    "stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}


def experiment_prompt(config_path: Path, config: dict[str, Any], run_dir: Path) -> str:
    quotas = config.get("train_quotas_per_round", {})
    rounds = config.get("rounds", 5)
    protocol = {
        "config_path": str(config_path), "run_dir": str(run_dir),
        "workspace_root": str(PROJECT_ROOT.parent), "project_root": str(PROJECT_ROOT),
        "rounds": rounds, "training_tasks_per_round": config.get("training_tasks_per_pass", sum(quotas.values())),
        "gpu_ids": list(range(len(config.get("ports", [0,1,2,3])))),
        "train_quotas_per_round": quotas, "training_schedule": config.get("training_schedule"),
        "initial_independent_evaluation": config.get("evaluate_initial_system"),
        "validation_limits": config.get("validation_limits"), "skip_acebench": config.get("skip_acebench"),
        "active_task_state": str(run_dir / "active_task.json"),
        "skills_path": str(run_dir / "meta/skills.jsonl"),
        "context_path": str(run_dir / "meta/context.json"),
        "snapshots_root": str(run_dir / "snapshots"),
        "meta_skill_context_chars": config.get("meta_skill_context_chars", DEFAULT_CONTEXT_CHARS),
        "meta_skill_per_category": config.get("meta_skill_per_category", DEFAULT_PER_CATEGORY),
        "meta_harness_mode": config.get("meta_harness_mode", "evolving"),
        "meta_fixed_skills": config.get("meta_fixed_skills", False),
        "meta_harness_state": str(run_dir / "meta_harness/state.json"),
        "meta_harness_calls": str(run_dir / "meta_harness/calls.jsonl"),
        "workflow_state": str(run_dir / "meta_session/workflow_state.json"),
    }
    return (PERSISTENT_META_INSTRUCTIONS + "\n" + SKILL_GUIDANCE + "\n" + MAINLINE_REMINDER
            + "\nExperiment paths and facts:\n" + json.dumps(protocol, ensure_ascii=False, indent=2)
            + "\nMainline stages:\n"
            "A0: freeze the initial Task/Meta state and complete independent baseline validation.\n"
            f"B1-B{rounds}: in each round run parent → inspect evidence/skills/handoff → choose one component "
            "→ one candidate → same-batch retest → paired selection → maintain skills → self-review "
            "→ snapshot → report-only validation → next round. "
            "The ledger does not choose the component, candidate, or acceptance outcome.\n"
            f"Use round_number=1..{rounds} for B1..B{rounds}; A0 uses round=0. Include round_number "
            "on evolution helper calls. Keep data and model/API configuration unchanged.\n"
            "Helper usage: experiment(request={operation,...arguments}); helper arguments are documented "
            "in controller_tools.py and task_adapter.py. Native shell and apply_patch are available.\n"
            "NEW run only: bootstrap(config_path,state_path=active_task_state). Resume existing state/jobs "
            "instead of overwriting them. For A0 snapshot_task_meta(active_task_state,skills_path,context_path,"
            "destination=snapshots_root/A0,round_number=0), prepare_validation_snapshot(task_state_path="
            "restorable_task_state_path,source_round_path=receipt_path FILE,destination,round=0), evaluate"
            "(config_path,snapshot_path). Reuse the SAME frozen initial Task Harness in new experiments.\n"
            "Each round: run_parent(config_path,state_path,round_number,output_dir), prepare_meta_evidence"
            "(parent_rollout_dir). Read all statistics; inspect 48 representative trajectory excerpts "
            "(three domains ×16, success/failure 8/8 with shortage redistribution, error-type round-robin "
            "and stable hash success order). Full trajectories stay in Rollout_logs, indexed and readable "
            "on demand. read_text pages use next_offset_chars until selected evidence is read. "
            "retrieve_skills(path=skills_path,fingerprint_path=failure_fingerprint_path,max_chars="
            "meta_skill_context_chars,per_category=meta_skill_per_category); read_skill for full relevant "
            "records, read_json(context_path). Follow this round's Workflow/Planning hooks.\n"
            "Write attempts/attempt_K/decision.json (component, reasons, relevant_skill_ids, prediction, "
            "evidence refs); record_skill_use before candidate construction. HARNESS: materialize_harness"
            "(state_path,destination), read the three production_templates, write localization_report_path "
            "then improvement_direction_path, generate full bundle, harnessforge(state_path,candidate_dir,"
            "localization_report_path,improvement_direction_path). MODEL: sft(config_path,state_path,"
            "parent_rollout_dir,output_dir). ARTIFACTS: artifacts(state_path,baseline_dir,edits). "
            "run_candidate(config_path,state_path,round_number,output_dir), with baseline_dir and "
            "submission_targets for ARTIFACTS. Preserve all attempts and full paired trajectories.\n"
            "compare_task_differences(before_trajectories,after_trajectories,output_dir,skill_use_path) "
            "and compare_scores(before,after) report facts; you must decide whether complete paired "
            "scores strictly improve. Record attempt selection.json; only your acceptance invokes "
            "activate_task(source_state,active_state). Otherwise retain parent and retry on the same batch.\n"
            "append_skills(path=skills_path,entries) stores cases; maintain_skills(path,operations) applies "
            "add/supplement/revise/merge/retire while preserving history. Do not import report-only "
            "validation outcomes into skills. After feedback call prepare_meta_review(round_number,"
            "selection_paths,evidence_refs), inspect your decision/tool history and retrieved skills. "
            "If no concrete program gap, update_meta_harness(decision=keep,round_number,reason). Otherwise "
            "materialize_meta_harness(destination,round_number), produce ALL THREE files workflow.py, "
            "planning.py, memory.py, check_meta_harness(candidate_dir,round_number), "
            "update_meta_harness(decision=replace,candidate_dir,round_number,reason,gaps,evidence_refs). "
            "New package loads next round, not midway through the current attempt. A failed check returns "
            "to you; it does not end the experiment. Final-round candidates are preserved for later study.\n"
            "Write a concise evidence-linked context_path; snapshot_task_meta(active_task_state,skills_path,"
            "context_path,parent_task_state,candidate_task_state if available,round_number,destination,"
            "versions_root=run_dir/versions). Keep before-state before activation. Snapshot preserves "
            "Task combinations, immutable skill/handoff snapshots, used Meta package and selected-next "
            "package. Then prepare_validation_snapshot(source_round_path=accepted selection.json FILE,"
            "task_state_path=restorable_task_state_path,destination,round), evaluate(config_path,snapshot_path). "
            "ACEBench remains skipped; independent validation is report-only. "
            "fixed Meta mode keeps the supplied package; if meta_fixed_skills is true its skill seed "
            "must also remain unchanged. Resume by loading saved state and attach to existing jobs.\n"
            "At major phase changes the bridge executes the round-pinned program and supplies its outputs. "
            "Small reads/polls do not repeat the whole mainline. Hook traces show actual execution, not "
            "proof of better decisions. Finish via finish_experiment only after necessary round products.\n")


def prepare_meta_program(run_dir, config):
    from meta_harness_runtime import MetaHarnessRuntime
    seed = Path(config.get("meta_harness_seed_path") or PROJECT_ROOT / "meta_harness/G000")
    if not seed.is_absolute():
        seed = PROJECT_ROOT / seed
    settings = {"seed_path": str(seed.resolve()), "mode": config.get("meta_harness_mode", "evolving"),
                "fixed_skills": config.get("meta_fixed_skills", False)}
    path = run_dir / "meta/harness_config.json"
    if not path.exists():
        write_json(path, settings)
    return MetaHarnessRuntime(PROJECT_ROOT, run_dir).initialize()


def prepare_workflow_policy(run_dir, rounds):
    """The requested config controls the target, while retry settings survive resume."""
    path = run_dir / "meta_session/workflow_policy.json"
    policy = json.loads(path.read_text()) if path.exists() else {}
    policy['rounds'] = int(rounds)
    write_json(path, policy)
    return policy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--skill-seed", type=Path, help="optional JSONL seed for a NEW Meta ledger; resume preserves existing skills")
    parser.add_argument("--wait-for-gpus", action="store_true", help="start only after GPUs 0-3 are idle for two 60s checks")
    parser.add_argument("--detach", action="store_true", help="start a terminal-independent server process")
    parser.add_argument("--check", action="store_true", help="inspect paths without API calls")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.skill_seed:
        config["meta_skill_seed_path"] = str(args.skill_seed.resolve())
    seed = config.get("meta_skill_seed_path")
    seed_path = Path(seed) if seed else None
    if seed_path and not seed_path.is_absolute():
        seed_path = (PROJECT_ROOT / seed_path).resolve()
    run_dir = (args.run_dir or Path(config["output_root"]) / "runs" / config["run_name"]).resolve()
    if args.check:
        base_path = PROJECT_ROOT / config.get("runtime_source", "runtime") / config.get("base_config", "configs/base.json")
        base = json.loads(base_path.read_text(encoding="utf-8")) if base_path.is_file() else {}
        checkpoint = base.get("task_checkpoint")
        frozen_data = config.get("frozen_data_dir")
        facts = {"config": str(config_path), "run_dir": str(run_dir),
                 "project_root": str(PROJECT_ROOT),
                 "controller_exists": (PROJECT_ROOT / "controller_tools.py").is_file(),
                 "model_config_exists": base_path.is_file(),
                 "checkpoint_path": checkpoint,
                 "checkpoint_exists": bool(checkpoint and Path(checkpoint).is_dir()),
                 "frozen_data_exists": bool(frozen_data and Path(frozen_data).is_dir()),
                 "catalog_exists": (PROJECT_ROOT / "configs/codex-deepseek-v4.1-flash.json").is_file(),
                 "rounds": config.get("rounds", 5),
                 "training_tasks_per_round": config.get("training_tasks_per_pass"),
                 "meta_skill_seed_path": str(seed_path) if seed_path else None,
                 "meta_skill_seed_exists": seed_path.is_file() if seed_path else None,
                 "meta_skill_context_chars": config.get("meta_skill_context_chars", DEFAULT_CONTEXT_CHARS)}
        facts["wait_for_gpus"] = args.wait_for_gpus
        facts["gpu_ids"] = list(range(len(config["ports"])))
        facts["wait_for_processes"] = config.get("wait_for_processes", [])
        seed = Path(config.get('meta_harness_seed_path', 'meta_harness/G000'))
        seed = seed if seed.is_absolute() else PROJECT_ROOT / seed
        from meta_harness_runtime import wiring_probe
        try:
            facts['meta_harness_check'] = wiring_probe(seed)
        except Exception as exc:
            facts['meta_harness_check'] = {'status': 'validation_failed', 'error': str(exc)}
        facts['meta_harness_mode'] = config.get('meta_harness_mode', 'evolving')
        print(json.dumps(facts, ensure_ascii=False, indent=2))
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    prepare_meta_storage(run_dir, config.get("meta_logs_root"))
    session_dir = run_dir / "meta_session"
    session_dir.mkdir(exist_ok=True)
    if args.detach:
        log_path = session_dir / "launcher.log"
        argv = [sys.executable, "-u", str(PROJECT_ROOT / "launch_meta.py"),
                "--config", str(config_path), "--run-dir", str(run_dir)]
        if args.skill_seed:
            argv.extend(["--skill-seed", str(args.skill_seed.resolve())])
        if args.wait_for_gpus:
            argv.append("--wait-for-gpus")
        with log_path.open("a", encoding="utf-8") as log:
            child = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                cwd=PROJECT_ROOT, start_new_session=True,
            )
        print(json.dumps({"status": "launch_requested", "pid": child.pid, "log_path": str(log_path)}))
        return 0
    # One owner receives native tool replies. A reconnect never duplicates an active run.
    owner_lock = (session_dir / "launcher.lock").open("a")
    try:
        fcntl.flock(owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({"status": "already_running", "run_dir": str(run_dir)}))
        return 0
    if args.wait_for_gpus:
        wait_for_idle_gpus(session_dir, gpu_ids=range(len(config["ports"])),
                           wait_processes=config.get("wait_for_processes", []))
    os.environ["AUTODL_API_KEY"] = _autodl_key()
    prepare_skills_path(run_dir, seed_path)
    prepare_context_path(run_dir)
    prepare_meta_program(run_dir, config)
    prepare_workflow_policy(run_dir, config.get('rounds', 5))
    codex_home = prepare_codex_home(run_dir)
    tools = SubprocessToolExecutor(project_root=PROJECT_ROOT, run_dir=run_dir)
    loop = PersistentMeta(project=PROJECT_ROOT, run=run_dir, codex_home=codex_home,
                          executor=tools, rounds=int(config.get("rounds", 5)))
    while True:
        try:
            final = loop.run_experiment(experiment_prompt(config_path, config, run_dir))
            break
        except Exception as exc:
            print(f"Meta connection interrupted: {type(exc).__name__}: {exc}. Reconnecting in 30s.", flush=True)
            time.sleep(30)
    print(json.dumps({"status": final["status"], "run_dir": str(run_dir),
                      "thread_id": final["thread_id"], "summary": final.get("summary")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
