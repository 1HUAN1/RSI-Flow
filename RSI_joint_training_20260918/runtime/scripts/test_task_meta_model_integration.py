"""Explicit real-GPU integration test; never used by the autonomous CLI.

Inject one MODEL decision at the first eligible generation, then return routing
to the real Meta agent. This is branch wiring evidence, not autonomous selection.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sia.cli import parse_args as sia_args
from sia.config import Config
from sia.task_meta.entry import run_from_args
from sia.task_meta.types import MetaDecision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--config", default="configs/task-meta-gpu.json")
    args = parser.parse_args()
    injected = []

    def override(meta, observation):
        available = observation.available_actions.get("MODEL", {})
        if injected or observation.generation > 1 or not available.get("available"):
            return None
        injected.append(observation.generation)
        return MetaDecision(
            action="MODEL", target_components=["MODEL"],
            diagnosis="Test-only verification of the complete real SFT feedback path",
            evidence=[f"{available['positive_samples']} actual positive rollout samples available"],
            rationale="Exercise the MODEL updater in the same main loop with a declared test override",
            proposed_change="Fixed positive-reward SFT starting from the current Task checkpoint",
            expected_effect="Unknown; this checks execution and feedback, not a performance gain",
            requested_changes=[{"id": "sft_wiring_test", "component": "MODEL", "operation": "sft",
                                "target": "current_checkpoint", "instruction": "Run the configured bounded real SFT backend"}],
            decision_source="test_override",
        )

    cli = sia_args(Config(), ["run", "--evolution-mode", "task-meta", "--task", "gpqa",
                              "--meta-agent-profile", "task-meta-glm", "--target-agent-profile", "task-meta-qwen",
                              "--task-meta-config", args.config, "--max_gen", "4", "--run_id", str(args.run_id), "--no-web"])
    final = run_from_args(cli, test_decision_override=override)
    if not injected:
        raise RuntimeError("No eligible positive-sample generation; MODEL integration was not exercised")
    if final["experiences"] < injected[0] + 2:
        raise RuntimeError("MODEL evaluation, feedback and subsequent routing did not complete")


if __name__ == "__main__":
    main()
