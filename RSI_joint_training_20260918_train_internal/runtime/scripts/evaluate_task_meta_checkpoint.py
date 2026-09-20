"""Replay a generation with an alternate checkpoint and the recorded input assets."""

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from urllib.request import urlopen

from sia.profiles import load_target_agent_profile
from sia.task_meta.execution import SIAExecutor
from sia.task_meta.storage import artifact_manifest, checkpoint_manifest, digest, save_json
from sia.task_meta.types import ArtifactState, TaskAgentState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-generation", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_generation.resolve()
    run = source.parent
    original = json.loads((source / "task_state.json").read_text())
    cfg = json.loads((run / "task_meta_config.json").read_text())
    incoming = original["artifacts"]
    # The state saved before rollout is authoritative. Never use outgoing artifacts.
    if artifact_manifest(incoming["directory"]) != incoming["manifest"]:
        raise ValueError("Recorded incoming artifact snapshot changed")
    model_weights = checkpoint_manifest(args.model)
    if args.model == original["model_ref"] and model_weights != original["checkpoint_manifest"]:
        raise ValueError("Base weights changed since the original generation")
    with urlopen(args.base_url.removesuffix("/v1") + "/health", timeout=10) as response:
        health = json.load(response)
    if not health.get("ready") or not health.get("device", "").startswith("cuda") or args.model not in health["models"]:
        raise ValueError("Selected checkpoint must be served on CUDA")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    harness = output / "target_agent.py"
    shutil.copy2(original["harness_path"], harness)
    assets = ArtifactState()
    if incoming["directory"]:
        shutil.copytree(incoming["directory"], output / "artifacts_input")
        assets = ArtifactState(str(output / "artifacts_input"), incoming["manifest"])
    state = TaskAgentState(0, args.model, str(harness), assets,
                           checkpoint_path=args.model, checkpoint_manifest=model_weights)
    save_json(output / "task_state.json", state)
    save_json(output / "input_spec.json", {
        "source_generation": str(source), "harness_sha256": digest(harness),
        "artifact_manifest": assets.manifest, "model_ref": args.model,
        "inference_health": health, "configuration": cfg,
        "source_task_sha256": digest(run / "task/batch_manifest.json"),
        "script_sha256": digest(Path(__file__)),
    })
    shutil.copy2(__file__, output / "replay_source.py")
    provider = replace(load_target_agent_profile("task-meta-qwen").provider, base_url=args.base_url)
    executor = SIAExecutor(run / "task", sys.prefix, provider, cfg["rollouts_per_task"],
                           cfg["temperature"], cfg["max_tokens"], cfg["seed"])
    result = executor.execute(state, output)
    save_json(output / "results.json", result.performance)
    save_json(output / "agent_execution.json", result.trajectories)
    save_json(output / "cost.json", result.cost)
    print(json.dumps(result.performance), flush=True)


if __name__ == "__main__":
    main()
