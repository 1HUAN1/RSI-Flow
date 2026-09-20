"""Integrated Task-H / Meta-G evidence, using explicit offline model fixtures."""
from __future__ import annotations

import json
import socket
import subprocess

import pytest

from scripts.verify_task_meta_closed_loop import read, rows, run_demo


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    output = tmp_path_factory.mktemp("closed_loop_parent") / "evidence"
    # A future accidental subprocess or network connection must fail this test.
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline verification must not launch a process or network request")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(subprocess, "Popen", forbidden)
        patch.setattr(socket, "create_connection", forbidden)
        patch.setattr(socket.socket, "connect", forbidden)
        evidence = run_demo(output)
    return output, evidence


def test_three_generations_pair_two_experiences_then_consolidate_without_training(demo):
    output, evidence = demo
    final = read(output / "run/final_state.json")
    ledger = rows(output / evidence["experience_ledger"])
    assert final["generations_executed"] == 3 and len(ledger) == 2
    assert final["decision_mode"] == "test_override_enabled"
    assert final["meta_version_changes"] == 1 and final["meta_update_events"] == 3
    assert evidence["regular_no_change_events"] == evidence["final_consolidation_no_change_events"] == 1
    assert not (output / "run/gen_3").exists()
    assert not list(output.rglob("training_request.json"))
    assert evidence["real_api_calls"] == evidence["training_runs"] == evidence["gpu_operations"] == 0
    assert evidence["autonomous_meta_evolution_verified"] is False
    assert evidence["score_improvement_claim"] is False
    assert all(item["performance_delta"] == 0 for item in ledger)
    versions = [read(path / "manifest.json") for path in (output / "run/meta/harness_versions").iterdir()]
    assert sorted(item["version"] for item in versions) == [0, 1]
    assert len(list((output / "run/meta/commit_events").glob("*.json"))) == 3


def test_multi_part_h_changes_real_next_generation_calls_while_using_current_weights(demo):
    output, evidence = demo
    assert evidence["h_changed_parts"] == [["control", "input", "tools"], ["input", "tools"]]
    reference = read(output / "run/gen_0/evaluated_state.json")
    for generation in range(3):
        directory = output / f"run/gen_{generation}"
        state = read(directory / "evaluated_state.json")
        assert state["checkpoint_manifest"] == reference["checkpoint_manifest"]
        assert state["model_ref"] == "test_override_task_checkpoint"
        actual = rows(directory / "train_trajectories.jsonl")
        for row in actual:
            calls = row["model_calls"]
            context_roles = [call for call in calls if call.get("role") == "toy_context"]
            assert len(context_roles) == (0 if generation == 0 else 1)
            if generation:
                assert context_roles[0]["node"] == "toy_context"
                assert f"TEST_OVERRIDE_TASK_ROLE_H{generation}" in context_roles[0]["messages"][-1]["content"]
                action = next(call for call in calls if call["operation"] == "action")
                assert f"TEST_OVERRIDE_TASK_INPUT_H{generation}" in json.dumps(action["messages"])
                assert f"Test context produced by the added H role in generation {generation}" in json.dumps(action["messages"])
            assert all(call["binding"]["model_ref"] == state["model_ref"] for call in calls)
            assert all(call["binding"]["weights_loaded"] is False for call in calls)
        if generation < 2:
            intervention = read(directory / "intervention_diff.json")
            assert not any(intervention["artifacts"][key] for key in ("added", "removed", "changed"))
            assert intervention["model"]["before"] == intervention["model"]["after"]


def test_committed_g1_organizes_both_later_h_patch_and_later_self_update(demo):
    output, evidence = demo
    calls, stages = read(output / "meta_calls.json"), read(output / "meta_stages.json")
    ledger = rows(output / evidence["experience_ledger"])
    assert [item["versions"]["meta_bundle_hash_at_decision"] for item in ledger] == [evidence["g0_hash"], evidence["g1_hash"]]
    later = [call for call in calls if call["meta_version"] == 1]
    assert [call["operation"] for call in later] == ["route", "harness_patch", "learn", "final_consolidation"]
    assert {call["bundle_hash"] for call in later} == {evidence["g1_hash"]}
    second_learn = next(call for call in later if call["operation"] == "learn")
    own_stages = [stage for stage in stages if stage["call"] == second_learn["call"]]
    assert any("g1_attribution_analysis" in stage["stage_id"] for stage in own_stages)
    assert all("TEST_OVERRIDE_G1_RULE_MATCHED" in stage["prompt"] for stage in own_stages)
    steps = evidence["g1_later_self_update_completed_steps"]
    assert steps.index("g1_outcome_check") < steps.index("g1_attribution_analysis") < steps.index("propose")
    final_input = read(output / calls[-1]["directory"] / "input_envelope.json")
    assert final_input["latest_experience"] is None and len(final_input["experiences"]) == 2
    assert read(output / evidence["g1_candidate_patch"])["status"] == "UPDATED"
    for call in calls:
        audit = read(output / call["directory"] / "policy_runtime.json")
        assert audit["raw_source_mutated"] is False
        assert audit["g"]["bundle_hash"] == call["bundle_hash"]
        assert audit["decision_source"] == "test_override"
        envelope = read(output / call["directory"] / "input_envelope.json")
        assert all(row["split"] == "evolve_train" for row in envelope["raw_trajectories"])


def test_fixed_probe_and_all_rollouts_use_frozen_input_assets_only(demo):
    output, evidence = demo
    baseline_ids = evidence["generations"][0]["probe_ids"]
    train_ids = []
    for generation in range(3):
        directory = output / f"run/gen_{generation}"
        train, probe = rows(directory / "train_trajectories.jsonl"), rows(directory / "probe_trajectories.jsonl")
        expected = evidence["generations"][generation]["input_artifacts"]
        assert [row["task_id"] for row in probe] == baseline_ids
        assert len(train) == 6 and len(probe) == 3
        assert all(row["artifact_input_manifest"] == expected for row in train + probe)
        for row in train + probe:
            # Distinguish frozen reusable assets from this rollout's fresh memory.
            messages = [message for call in row["model_calls"] for message in call["messages"]]
            assets = [message["content"] for message in messages
                      if message["content"].startswith("Current unverified reusable artifacts:")]
            assert bool(assets) == bool(generation)
            assert all(f"generated by T_{generation};" not in text for text in assets)
            assert all(f"generated by T_{generation - 1};" in text for text in assets)
        assert all(not row["notes"] for row in probe)
        train_ids.extend(sorted({row["task_id"] for row in train}))
    assert len(train_ids) == len(set(train_ids)) == 9
    assert not set(train_ids) & set(baseline_ids)
    assert read(output / "run/coverage.json")["full_coverage"] is True


def test_export_refuses_existing_output_without_overwriting(demo):
    output, _ = demo
    before = (output / "behavior_evidence.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_demo(output)
    assert (output / "behavior_evidence.json").read_bytes() == before
