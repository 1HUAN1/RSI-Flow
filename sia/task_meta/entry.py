"""SIA CLI bridge; the original evolution mode keeps its existing control flow."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, Field

from sia.agent_reference import resolve_agent_reference
from sia.layout import RunLayout, TaskLayout, resolve_task_dir
from sia.profiles import load_meta_agent_profile, load_target_agent_profile
from sia.run_setup import load_task_files
from sia.task_meta.execution import SIAExecutor, prepare_gpqa_task
from sia.task_meta.loop import run_task_meta
from sia.task_meta.meta import MetaAgent, StructuredClient
from sia.task_meta.prompts import INITIAL_META_HARNESS
from sia.task_meta.storage import checkpoint_manifest, digest, save_json
from sia.task_meta.types import MetaAgentState, TaskAgentState, TaskUpdateAction
from sia.task_meta.updaters import ArtifactUpdater, HarnessUpdater, ModelUpdater


class TaskMetaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: int = Field(default=2, ge=1)
    rollouts_per_task: int = Field(default=8, ge=1)
    temperature: float = Field(default=0.7, ge=0, allow_inf_nan=False)
    max_tokens: int = Field(default=512, ge=1)
    seed: int = 42
    primary_metric: str = "success_rate"
    primary_metric_mode: Literal["min", "max"] = "max"
    max_wall_time: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    meta_timeout: float = Field(default=180, gt=0, allow_inf_nan=False)
    meta_max_tokens: int = Field(default=8192, ge=1)
    trainer_command: list[str] | None = None
    training_method: Literal["external", "positive_sft_lora"] = "external"
    require_cuda_inference: bool = False
    trainer_timeout: float = Field(default=3600, gt=0, allow_inf_nan=False)
    # Keep training infrastructure external, as in SIA's weight mode.
    training_sandbox: Literal["modal", "sandboxfusion"] = "sandboxfusion"


def run_from_args(args, *, test_decision_override=None):
    if args.max_gen < 1:
        raise ValueError("--max_gen must be at least 1")
    if args.sandbox != "none":
        raise ValueError("Task-meta local GPQA execution currently supports --sandbox none")
    if args.focus != "harness":
        raise ValueError("Task-meta routes MODEL/HARNESS/ARTIFACTS itself; omit --focus weights")
    cfg = TaskMetaConfig.model_validate_json(Path(args.task_meta_config).read_text(encoding="utf-8")) if args.task_meta_config else TaskMetaConfig()
    meta_profile = load_meta_agent_profile(args.meta_agent_profile)
    target_profile = load_target_agent_profile(args.target_agent_profile)
    if meta_profile.agent_impl != "pydantic-ai":
        raise ValueError("Task-meta structured Meta outputs use a pydantic-ai profile")
    if target_profile.provider.client_kind != "openai":
        raise ValueError("The GPQA Task adapter needs an OpenAI-compatible target provider")
    for provider in (meta_profile.provider, target_profile.provider):
        if not os.environ.get(provider.api_key_env):
            raise ValueError(f"Set {provider.api_key_env} before running")
    inference_health = None
    if cfg.require_cuda_inference:
        health_url = target_profile.provider.base_url.rstrip("/").removesuffix("/v1") + "/health"
        with urlopen(health_url, timeout=10) as response:
            inference_health = json.load(response)
        if not inference_health.get("ready") or not str(inference_health.get("device", "")).startswith("cuda"):
            raise ValueError("This GPU configuration requires a ready CUDA inference endpoint")
    task_dir, shared = resolve_task_dir(args.task, args.task_dir)
    reference = resolve_agent_reference(target_profile.agent_reference, TaskLayout(task_dir, shared))
    if reference.ref_dir is not None:
        raise ValueError("Task-meta MVP accepts a single-file harness seed; directory references need an adapter")
    task_files = load_task_files(task_dir, shared, reference)
    run_dir = Path(RunLayout.for_run_id(args.run_id).run_dir).resolve()
    if run_dir.exists():
        raise ValueError(f"Run exists: {run_dir}; use a new --run_id")
    run_dir.mkdir(parents=True)
    save_json(run_dir / "inference_health.json", inference_health)
    # Save the exact source and environment used by this run even in a dirty checkout.
    package_dir = Path(__file__).resolve().parent.parent
    source_files = [*package_dir.glob("*.py"), *package_dir.joinpath("task_meta").glob("*.py"),
                    *package_dir.joinpath("agent_impls").glob("*.py"),
                    *package_dir.parent.joinpath("scripts").glob("*.py"),
                    *package_dir.parent.joinpath("scripts").glob("*.sh")]
    sources = {}
    for source in source_files:
        relative = source.relative_to(package_dir.parent)
        destination = run_dir / "implementation" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        sources[relative.as_posix()] = digest(source)
    save_json(run_dir / "implementation/source_manifest.json", sources)
    packages = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"], capture_output=True, text=True, check=True)
    save_json(run_dir / "implementation/packages.json", json.loads(packages.stdout))
    gen0 = run_dir / "gen_0"
    gen0.mkdir()
    meta_dir = run_dir / "meta"
    meta_dir.mkdir()
    harness = gen0 / "target_agent.py"
    if target_profile.agent_reference.kind == "default":
        shutil.copy2(Path(__file__).with_name("gpqa_target.py"), harness)
    else:
        harness.write_text(reference.inline_seed, encoding="utf-8")
    # Preserve the selected public/private batch and official evaluator once for all generations.
    prepare_gpqa_task(Path(task_dir), run_dir / "task", cfg.batch_size)
    (meta_dir / "harness_v0.md").write_text(INITIAL_META_HARNESS, encoding="utf-8")
    save_json(run_dir / "profiles.json", {"meta": asdict(meta_profile), "target": asdict(target_profile)})
    save_json(run_dir / "task_meta_config.json", {**cfg.model_dump(), "max_generations": args.max_gen,
                                               "python_environment": sys.prefix, "evolution_mode": "task-meta"})
    client = StructuredClient(meta_profile, meta_dir / "calls", cfg.meta_timeout, cfg.meta_max_tokens)
    capabilities = {"trainer_configured": bool(cfg.trainer_command)}
    capabilities["execution"] = {"target_max_tokens": cfg.max_tokens, "temperature": cfg.temperature,
                                  "rollouts_per_task": cfg.rollouts_per_task, "batch_size": cfg.batch_size}
    meta = MetaAgent(client, capabilities)
    executor = SIAExecutor(run_dir / "task", sys.prefix, target_profile.provider, cfg.rollouts_per_task,
                           cfg.temperature, cfg.max_tokens, cfg.seed)
    updaters = {TaskUpdateAction.HARNESS: HarnessUpdater(client, task_files, target_profile.provider),
                TaskUpdateAction.ARTIFACTS: ArtifactUpdater(client),
                TaskUpdateAction.MODEL: ModelUpdater(client, task_files, target_profile.provider, cfg.trainer_command,
                                                     cfg.training_sandbox, cfg.trainer_timeout)}
    final = run_task_meta(
        run_dir, TaskAgentState(0, target_profile.model, str(harness),
                               checkpoint_path=target_profile.model if Path(target_profile.model).is_dir() else None,
                               checkpoint_manifest=checkpoint_manifest(target_profile.model) if Path(target_profile.model).is_dir() else []),
        MetaAgentState(meta_profile.model, str(meta_dir / "harness_v0.md")), executor, meta, updaters,
        max_generations=args.max_gen, primary_metric_name=cfg.primary_metric,
        primary_metric_mode=cfg.primary_metric_mode, max_wall_time=cfg.max_wall_time,
        test_decision_override=test_decision_override,
    )
    print(json.dumps({"run_directory": str(run_dir), "status": final["status"],
                      "generations_executed": final["generations_executed"]}, ensure_ascii=False), flush=True)
    if final["status"] != "completed":
        raise SystemExit(2)
    return final
