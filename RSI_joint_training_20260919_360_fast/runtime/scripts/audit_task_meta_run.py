"""Read saved evidence for a completed Task-Meta run; no model calls or reruns."""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run):
    final = read(run / "final_state.json")
    count = final["generations_executed"]
    experiences = [json.loads(line) for line in (run / "meta/experiences.jsonl").read_text().splitlines()]
    calls = read(run / "meta/calls/usage.json")
    assert final["status"] == "completed" and len(experiences) == count - 1
    assert final["meta_state"]["version"] == count
    assert not (run / f"gen_{count}").exists()
    assert len([c for c in calls if c["operation"] == "learn" and c["status"] == "success"]) == count - 1
    assert len([c for c in calls if c["operation"] == "final_consolidation" and c["status"] == "success"]) == 1
    for call in calls:
        assert call["meta_harness_hash"] == digest(run / f"meta/harness_v{call['meta_harness_version']}.md")
    rows, model_chains = [], []
    for generation in range(count):
        directory = run / f"gen_{generation}"
        task_in = read(directory / "evaluated_state.json")
        task_out = read(directory / "task_state_after_rollout.json")
        performance = read(directory / "results.json")
        trajectories = read(directory / "agent_execution.json")
        binding = read(directory / "service_binding.json")
        assert binding["checkpoint_path"] == task_in["checkpoint_path"]
        assert binding["weights"] == task_in["checkpoint_manifest"]
        assert len(trajectories) == performance["attempts"]
        assert sum(t["terminal_reward"] for t in trajectories) == performance["successes"]
        if generation == 0:
            assert task_in["artifacts"]["manifest"] == []
        for trajectory in trajectories:
            assert trajectory["model_ref_requested"] == task_in["model_ref"]
            for call in trajectory["call_attempts"]:
                if call.get("api_error"):
                    continue
                assert call["model_ref_response"] == task_in["model_ref"]
                assert call["response_checkpoint_binding"]["weights"] == binding["weights"]
        provenance = read(directory / "artifact_provenance.json")
        assert {a["path"] for a in provenance} == {a["path"] for a in task_out["artifacts"]["manifest"]}
        assert all(a["knowledge_verified"] is False for a in provenance)
        decision = read(directory / "meta_decision.json") if generation < count - 1 else None
        meta_before = read(directory / "meta_state_before.json")
        meta_after = read(directory / "meta_state_after.json")
        rows.append({"generation": generation, "checkpoint": task_in["checkpoint_path"],
                     "task_harness_sha256": digest(Path(task_in["harness_path"])),
                     "meta_at_execution": meta_before["version"], "meta_after_learning": meta_after["version"],
                     "successes": performance["successes"], "attempts": performance["attempts"],
                     "valid_answer_rate": performance["valid_answer_rate"], "parse_failures": performance["parse_failures"],
                     "output_truncations": performance["output_truncations"], "api_errors": performance["api_errors"],
                     "actual_model_calls": sum(len(t["call_attempts"]) for t in trajectories),
                     "parse_failure_events": performance["parse_failure_events"],
                     "output_truncation_events": performance["output_truncation_events"],
                     "input_assets": len(task_in["artifacts"]["manifest"]), "output_assets": len(provenance),
                     "action": decision["action"] if decision else None,
                     "decision_source": decision["decision_source"] if decision else None})
        if decision is None:
            continue
        update = read(directory / "task_update.json")
        experience = experiences[generation]
        assert update["action"] == decision["action"] == experience["chosen_action"]
        assert decision["target_components"] == [decision["action"]]
        assert update["unapplied_changes"] == []
        assert experience["evaluated_state_before"] == task_in
        assert experience["intervention_base_state"] == task_out
        assert experience["evaluated_state_after"] == read(run / f"gen_{generation + 1}/evaluated_state.json")
        assert abs(experience["observed_performance_delta"] -
                   (experience["performance_after"]["success_rate"] - performance["success_rate"])) < 1e-12
        if decision["decision_source"] == "model":
            route = [c for c in calls if c["operation"] == "route" and c["decision_id"] == decision["decision_id"] and c["status"] == "success"]
            assert len(route) == 1 and route[0]["meta_harness_version"] == meta_after["version"]
        if decision["action"] in {"HARNESS", "ARTIFACTS"}:
            operation = "harness_patch" if decision["action"] == "HARNESS" else "artifact_patch"
            patch = [c for c in calls if c["operation"] == operation and c["decision_id"] == decision["decision_id"] and c["status"] == "success"]
            assert len(patch) == 1 and patch[0]["meta_harness_version"] == meta_after["version"]
        else:
            request = read(directory / "model_update/training_request.json")
            metrics = read(directory / "model_update/training_metrics.json")
            trained = read(directory / "model_update/checkpoint_verified.json")
            next_state = experience["evaluated_state_after"]
            assert request["checkpoint_path"] == task_in["checkpoint_path"] == metrics["base_model"]
            assert trained["checkpoint_path"] == next_state["checkpoint_path"] == next_state["model_ref"]
            assert metrics["optimizer_steps"] > 0 and metrics["changed_lora_tensors"] > 0
            assert metrics["merged_probe_changed_elements"] > 0 and metrics["sampled_supervised_tokens"] > 0
            learn = [c for c in calls if c["operation"] == "learn" and c["experience_id"] == experience["experience_id"] and c["status"] == "success"]
            assert len(learn) == 1
            following = run / f"gen_{generation + 1}/meta_decision.json"
            next_route = read(following) if following.exists() else None
            model_chains.append({"generation": generation, "decision_source": decision["decision_source"],
                                 "training_request": str(directory / "model_update/training_request.json"),
                                 "training_metrics": metrics, "experience_id": experience["experience_id"],
                                 "next_routing_after_real_meta_learning": next_route,
                                 "complete_through_next_route": bool(next_route)})
    return {"status": "evidence_consistent", "run": str(run), "generations": rows,
            "experience_count": len(experiences), "final_meta_version": final["meta_state"]["version"],
            "model_chains": model_chains, "interpretation": "Engineering evidence only; no claim of RSI effectiveness"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
