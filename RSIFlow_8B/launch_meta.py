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


def wait_for_idle_gpus(session_dir: Path, *, probe=None, pause=None, poll_seconds=60):
    """Wait before starting Meta, never preempt another GPU experiment."""
    probe, pause = probe or gpu_sample, pause or time.sleep
    stable = 0
    while True:
        state = {"status": "waiting_for_gpus", "launcher_pid": os.getpid(),
                 "updated_at": time.time(), "poll_seconds": poll_seconds,
                 "gpu_ids": [0, 1, 2, 3]}
        try:
            samples = {row["gpu"]: row for row in probe()}
            idle = all(gpu in samples and samples[gpu]["memory_used_mib"] <= 1024
                       and samples[gpu]["utilization_percent"] <= 5 for gpu in range(4))
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
    for name in ("meta_session", "meta"):
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
    total = config.get("training_tasks_per_pass", sum(quotas.values()))
    rounds = config.get("rounds", 3)
    base_path = (PROJECT_ROOT / config.get("runtime_source", "runtime")
                 / config.get("base_config", "configs/base.json")).resolve()
    base = json.loads(base_path.read_text(encoding="utf-8")) if base_path.is_file() else {}
    policy_path = run_dir / "meta_session/workflow_policy.json"
    policy = json.loads(policy_path.read_text()) if policy_path.is_file() else {"retry_rejected_from_round": 1}
    protocol = {
        "config_path": str(config_path), "run_dir": str(run_dir),
        "workspace_root": str(PROJECT_ROOT.parent), "project_root": str(PROJECT_ROOT),
        "rounds": rounds, "training_tasks_per_round": total,
        "train_quotas_per_round": quotas,
        "training_schedule": config.get("training_schedule"),
        "initial_independent_evaluation": config.get("evaluate_initial_system"),
        "validation_limits": config.get("validation_limits"),
        "skip_acebench": config.get("skip_acebench"),
        "model_config_path": str(base_path),
        "task_checkpoint": base.get("task_checkpoint"),
        "active_task_state": str(run_dir / "active_task.json"),
        "skills_path": str(run_dir / "meta/skills.jsonl"),
        "skills_index_path": str(run_dir / "meta/skills_index.json"),
        "meta_skill_context_chars": config.get("meta_skill_context_chars", DEFAULT_CONTEXT_CHARS),
        "meta_skill_per_category": config.get("meta_skill_per_category", DEFAULT_PER_CATEGORY),
        "meta_skill_seed_path": config.get("meta_skill_seed_path"),
        "context_path": str(run_dir / "meta/context.json"),
        "snapshots_root": str(run_dir / "snapshots"),
        "controller_receipts": str(run_dir / "controller_receipts"),
        "workflow_state": str(run_dir / "meta_session/workflow_state.json"),
        "workflow_events": str(run_dir / "meta_session/workflow_events.jsonl"),
        "run_status": str(run_dir / "meta_session/RUN_STATUS.md"),
        "workflow_policy": policy,
    }
    return (PERSISTENT_META_INSTRUCTIONS + "\n" + SKILL_GUIDANCE + "\n" + MAINLINE_REMINDER
            + "\nExperiment facts and required sequence:\n"
            + json.dumps(protocol, ensure_ascii=False, indent=2)
            + "\nMainline stages:\n"
              "A0: freeze the initial Task/Meta state and complete independent baseline validation.\n"
              f"B1-B{rounds}: in each round inspect the parent evidence, choose and test one candidate per attempt, "
              "record accept-or-retain and append experience. Rejection returns to component selection "
              "on the same batch; acceptance leads to skills/context/snapshots, then report-only "
              "independent validation. The per-turn progress block repeats the current stage, "
              "acquired evidence, next milestone, and final stage; it is advisory and does not "
              "choose the component, candidate, or acceptance outcome.\n"
              "Use one persistent Codex session for all rounds. Begin with A0 independent "
              "evaluation if configured (A0 round=0). Evolution rounds use "
              f"round_number=1..{rounds} for B1..B{rounds}; include round_number in every B-round "
              "controller request. For each round run the active parent on the "
              "configured fresh, round-disjoint training tasks using four GPUs and "
              "16 rollout workers. Save all task trajectories and score facts. Give Meta "
              "all outcome statistics plus 48 representative trajectory excerpts: "
              "16 per domain, 8 successes and 8 failures where available, with "
              "failure error-type round-robin and stable-hash success ordering. "
              "Use retrieve_skills with path=skills_path, fingerprint_path=failure_fingerprint_path and the configured "
              "meta_skill_context_chars/per_category budget, then read_skill for selected full records; "
              "read_json for context_path before routing. Read original evidence on demand. "
              "Consult those files plus current trajectories, never a hardcoded route. "
              "Use a new run_dir/round_N/attempts/attempt_K/ directory per attempt. "
              "Use write_text to record its decision.json with your chosen "
              "component, trajectory refs, relevant skill IDs, rationale and prediction before editing. "
              "Freeze these with record_skill_use before candidate construction. "
              "Choose only one of HARNESS, MODEL, ARTIFACTS and produce only one "
              "candidate per attempt. HARNESS uses HarnessForge localization, plan, full package, "
              "validation/limited repair; MODEL SFT uses verified successful parent "
              "trajectories only, one epoch on four GPUs; ARTIFACTS modifies reusable "
              "submissions and re-scores as presently implemented. Re-evaluate the "
              "candidate on exactly the same round tasks. Use compare_scores for measured "
              "facts; you must decide whether full paired scores strictly improve. "
              "Use activate_task only for your explicit acceptance choice. Append "
              "component-specific cases for each attempt; append conditional principles or revisions only "
              "when supported by evidence. After rejection "
              "keep the parent, read the failure evidence and choose again; reuse the unchanged parent "
              "rollout and the same round_number, without running independent validation. "
              "After each attempt snapshot_task_meta with versions_root=run_dir/versions, "
              "parent_task_state, candidate_task_state (if built), active_task_state and skills_path; "
              "keep a separate before-state file before activation. Save numbered, deduplicated "
              "model/Harness references and each Meta skill version, not full repeated code trees. "
              "After acceptance run the configured independent validation for reporting only, "
              "then continue with the next round's fresh tasks. Do not use validation "
              "for acceptance, SFT, or Meta learning. If ACEBench is configured skipped, "
              "record it as skipped, not scored.\n"
              "Native tool protocol: call experiment with request={operation,...arguments}. "
              "Do not return JSON command envelopes as final answers. Long calls return job_id; "
              "use wait_job and job_status to supervise them without ending the experiment. "
              "For a NEW run only, first request bootstrap with config_path and state_path=active_task_state. "
              "On migration or resume, reuse the active state, completed receipts and running jobs; "
              "workflow_policy.retry_rejected_from_round marks when retry-on-rejection begins; "
              "preserve earlier completed rounds under their original protocol, do not reopen them. "
              "never bootstrap over an existing evolved Task or repeat a running rollout. "
              "For A0, first request snapshot_task_meta with active_task_state, "
              "skills_path, context_path and a destination under snapshots_root. "
              "Then prepare_validation_snapshot with task_state_path from its "
              "restorable_task_state_path, source_round_path=its receipt_path FILE, "
              "destination under snapshots_root and round=0, then evaluate. "
              "For each round request run_parent with config_path, state_path, "
              "round_number and output_dir; inspect performance_path and "
              "trajectories_path from its receipt. Then call prepare_meta_evidence "
              "with parent_rollout_dir. Read its statistics_path via read_json and "
              "routing_sources_path via read_text in pages of at most 65536 chars, "
              "using next_offset_chars as the next offset_chars until all 48 selected "
              "excerpts are inspected. Full trajectories remain indexed on demand. "
              "For HARNESS call materialize_harness(state_path, destination) and read its three "
              "production_templates. Write localization and improvement reports in that order BEFORE "
              "candidate edits; then generate the full bundle. Only AFTER those stages call harnessforge "
              "(state_path, candidate_dir, localization_report_path, improvement_direction_path) "
              "for executable validation and Task-state materialization, not production. Check at most 3 times, "
              "repairing the same bundle after a failed check. Other branches use sft (config_path, state_path, "
              "parent_rollout_dir), or artifacts (state_path, baseline_dir, edits). "
              "Run run_candidate on the same round_number; for ARTIFACTS pass "
              "baseline_dir and submission_targets for direct re-scoring. "
              "Then compare_task_differences with both trajectory paths, output_dir and skill_use_path, "
              "and compare_scores "
              "with before and after paths. Use write_text to record "
              "the attempt's selection.json with accept-or-retain and paired facts. "
              "Only if you accept request activate_task with source_state and "
              "active_state; otherwise keep the current parent state. Use append_skills "
              "with path=skills_path for the factual cases; use maintain_skills with path=skills_path "
              "and operations for uncovered methods, supplementation, linked revision, merge or retirement. "
              "Existing conditional rules/rule_update remain compatible; "
              "the index rebuilds automatically. Update context_path via write_text with a concise "
              "structured handoff for the next round, preserving evidence paths. "
              "Then request snapshot_task_meta with active_task_state, skills_path, "
              "parent_task_state, candidate_task_state if available, versions_root=run_dir/versions, "
              "context_path and a unique destination under snapshots_root; it records numbered "
              "component references. Only after acceptance request prepare_validation_snapshot with "
              "task_state_path=restorable_task_state_path from snapshot_task_meta, "
              "source_round_path=the accepted attempt's selection.json FILE (not a directory), "
              "destination, round and optional "
              "meta_state_path; use its snapshot_path in evaluate. The evaluator "
              "takes config_path and snapshot_path, with optional validation config "
              "and manifest overrides; prepare_evaluation can inspect the budget. "
              "Read project_root/controller_tools.py and task_adapter.py for argument fields. "
              "Native shell and apply_patch are available in workspace_root; you may repair or "
              "extend these helpers, or write and run your own scoped diagnostic scripts. "
              "If a tool fails, read its receipt and repair the issue as Meta; avoid "
              "ending the session prematurely. Call finish_experiment only after all "
              "configured rounds and available independent validations are complete.\n")


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
                 "rounds": config.get("rounds"),
                 "training_tasks_per_round": config.get("training_tasks_per_pass"),
                 "meta_skill_seed_path": str(seed_path) if seed_path else None,
                 "meta_skill_seed_exists": seed_path.is_file() if seed_path else None,
                 "meta_skill_context_chars": config.get("meta_skill_context_chars", DEFAULT_CONTEXT_CHARS)}
        facts["wait_for_gpus"] = args.wait_for_gpus
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
        wait_for_idle_gpus(session_dir)
    os.environ["AUTODL_API_KEY"] = _autodl_key()
    prepare_skills_path(run_dir, seed_path)
    prepare_context_path(run_dir)
    codex_home = prepare_codex_home(run_dir)
    tools = SubprocessToolExecutor(project_root=PROJECT_ROOT, run_dir=run_dir)
    loop = PersistentMeta(project=PROJECT_ROOT, run=run_dir, codex_home=codex_home,
                          executor=tools, rounds=int(config.get("rounds", 3)))
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
