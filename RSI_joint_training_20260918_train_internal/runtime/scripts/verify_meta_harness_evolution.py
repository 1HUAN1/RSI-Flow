#!/usr/bin/env python3
"""Export offline executable-G evidence using explicit engineer-written mocks.

Uses the actual immutable bundle store, policy interpreter and candidate checker.
It never launches Codex, an API, a model, training, pytest or a GPU operation.
This verifies behavior and wiring, not autonomous Meta learning or improvement.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Direct script execution needs the repository root before project imports.
from sia.task_meta.meta import validate_meta_candidate  # noqa: E402
from sia.task_meta.meta_harness import MetaHarnessStore  # noqa: E402
from sia.task_meta.meta_harness.bundle import reject_links  # noqa: E402
from sia.task_meta.meta_harness.policies import validate_policy  # noqa: E402
from sia.task_meta.meta_harness.runtime import AnalysisOutput, execute  # noqa: E402
from sia.task_meta.types import MetaAgentState, MetaDecision, MetaHarnessUpdate  # noqa: E402


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def toy_envelope():
    rows = []
    for index, (domain, reward) in enumerate((("code", 0), ("tool_use", 0), ("searchqa", 1))):
        rows.append({"source_id": f"toy_train_{index}", "task_id": f"toy_task_{index}",
            "domain": domain, "generation": index, "split": "evolve_train", "terminal_reward": reward,
            "error_type": "runtime_error" if not reward else None, "evidence_origin": "test_override",
            "messages": [{"role": "user", "content": f"Toy {domain} task; no real model execution."},
                         {"role": "assistant", "content": "Engineer-written toy action."},
                         {"role": "tool", "name": "toy_checker", "content": "OK" if reward else "Error: toy failure."}]})
    history = [{"experience_id": f"toy_experience_{index}", "generation": index,
        "chosen_action": action, "requested_change": "Toy bounded adjustment", "actual_change": {"summary": "Toy recorded intervention"},
        "observed_performance_delta": delta, "cost_before": {"model_calls": 1}, "cost_after": {"model_calls": 1},
        "versions": {"before": index, "after": index + 1}, "evidence_origin": "test_override"}
        for index, (action, delta) in enumerate((("HARNESS", -.2), ("MODEL", .1), ("HARNESS", .2)))]
    facts = {"denominator": len(rows), "correct": sum(row["terminal_reward"] > 0 for row in rows),
        "domains": {row["domain"]: {"denominator": 1, "correct": int(row["terminal_reward"] > 0)} for row in rows},
        "source_scope": "complete fixed toy training rows and toy experience ledger",
        "evidence_origin": "test_override", "available_actions": {"HARNESS": {"available": True},
            "MODEL": {"available": False, "reason": "Offline fixture never trains"}, "ARTIFACTS": {"available": True}}}
    return {"raw_trajectories": rows, "experiences": history, "trusted_facts": facts,
        "task_state": {"generation": 3, "model_ref": "test_override_no_model"},
        "latest_experience": history[-1], "decision": {"action": "HARNESS", "requested_changes": []},
        "current_files": {}, "instruction": "Export an explicit offline behavior demonstration, not an autonomous improvement claim."}


def mock_evolved_policy(g0):
    """Engineer-written proposal returned by the mock; no model inferred it."""
    policy = g0.execution_spec()
    policy["evidence"]["filter"] = {"path": "item.success", "op": "eq", "value": False}
    policy["evidence"]["group_by"] = ["domain"]
    policy["evidence"]["fragments"].update(before=2, after=0, retain_tools=["toy_checker"])
    policy["experience"]["fields"] = ["generation", "chosen_action", "observed_performance_delta"]
    policy["experience"]["require_applicable"] = True
    policy["experience"]["applicability"] = [{"id": "same_component", "when": {
        "path": "item.chosen_action", "op": "eq", "value_path": "decision.action"},
        "instruction": "Treat matching components as candidates for comparison, not proof of transfer."}]
    policy["experience"]["failure"] = [{"id": "negative_toy_delta", "when": {
        "path": "item.observed_performance_delta", "op": "lt", "value": 0},
        "instruction": "Exclude this toy negative-outcome example from the derived guidance."}]
    policy["diagnosis"]["instruction"] = "MOCK_G1_DIAGNOSIS: inspect retained failures and separate toy observations from causal claims."
    policy["diagnosis"]["rules"] = [{"id": "toy_failure_rule", "when": {
        "path": "facts.correct", "op": "lt", "value_path": "facts.denominator"},
        "instruction": "MOCK_G1_RULE_MATCHED: require an executable component and an explicit uncertainty statement."}]
    policy["self_update"]["instruction"] = "MOCK_G1_SELF_UPDATE: compare recorded outcomes and retained evidence before proposing further changes."
    policy["self_update"]["rules"] = [{"id": "toy_gain_not_causality", "when": {
        "path": "latest_experience.observed_performance_delta", "op": "gt", "value": 0},
        "instruction": "MOCK_G1_SELF_RULE_MATCHED: an observed gain does not establish that a component caused it."}]
    policy["workflows"]["meta_self_update"] = [
        {"id": "select", "kind": "evidence"}, {"id": "retrieve", "kind": "experience"},
        {"id": "verify_outcome", "kind": "check", "checks": ["experience_outcome", "source_integrity"]},
        {"id": "assess_attribution", "kind": "analyze", "when": {"path": "last_check_passed", "op": "eq", "value": True},
         "instruction": "Analyze the checked observations before returning the one candidate."},
        {"id": "propose", "kind": "propose"},
        {"id": "repair", "kind": "repair", "max_attempts": 1,
         "when": {"path": "candidate_valid", "op": "eq", "value": False}}]
    policy["workflows"]["harness_patch"].insert(2, {"id": "locate", "kind": "inspect_targets", "max_chars": 32000})
    return validate_policy(policy)


def bundle_state(bundle):
    return MetaAgentState("test_override_no_model", str(bundle.path / "instructions.md"), version=bundle.version,
                          bundle_hash=bundle.hash, bundle_path=str(bundle.path))


def run_demo(output):
    output = Path(output).absolute()
    reject_links(output)
    # Never overwrite a previous export or any user-owned directory.
    output.mkdir(parents=True, exist_ok=False)
    source = toy_envelope()
    original = copy.deepcopy(source)
    source_hash = digest(source)
    write(output / "input_envelope.json", source)
    store = MetaHarnessStore(output / "mock_meta_store")
    g0 = store.initialize(ROOT / "meta_harness/seed")
    proposed = mock_evolved_policy(g0)
    candidate = MetaHarnessUpdate(harness=g0.read_files()["instructions.md"],
        rationale="Engineer-written test_override proposal to demonstrate all five editable mechanisms.",
        changed_rules=["evidence", "experience", "diagnosis", "modification", "self_update"],
        summary="Mock executable-policy change; not learned by a live Meta model.",
        bundle_files={"evolution.json": json.dumps(proposed, ensure_ascii=False, sort_keys=True, indent=2) + "\n"},
        status="UPDATED", request_id="test_override_g0_self_update", experience_id="toy_experience_2")
    write(output / "g1_candidate_patch.json", candidate.model_dump(mode="json"))
    stages = []

    def invoke_for(bundle, operation, response):
        def invoke(stage_id, prompt, schema, **permissions):
            if permissions != {"evidence_files": {}, "allowed_paths": []}:
                raise AssertionError("The offline callback must not receive external permissions")
            stage = {"operation": operation, "stage_id": stage_id, "schema": schema.__name__,
                     "bundle_hash": bundle.hash, "bundle_version": bundle.version,
                     "decision_source": "test_override", "prompt_hash": digest(prompt),
                     "runtime_verified": False, "codex_invoked": False, "api_invoked": False}
            stages.append(stage)
            prompt_dir = output / "mock_stage_prompts"
            prompt_dir.mkdir(exist_ok=True)
            prompt_file = f"{len(stages):02d}_{operation}_{stage_id}.txt"
            (prompt_dir / prompt_file).write_text(prompt, encoding="utf-8")
            stage["prompt_file"] = "mock_stage_prompts/" + prompt_file
            if schema is AnalysisOutput:
                return schema(analysis="Engineer-written mock analysis: the retained toy observations do not establish causality.",
                              hypotheses=["Toy hypothesis, not a model finding"], source_ids=["toy_train_0", "toy_experience_2"])
            return schema.model_validate(response)
        invoke.decision_source = "test_override"
        return invoke

    first = execute(g0, "meta_self_update", source, MetaHarnessUpdate,
        invoke_for(g0, "g0_self_update", candidate), output / "g0_self_update",
        validate_candidate=lambda value: validate_meta_candidate(value, bundle_state(g0)))
    g1 = store.commit_update(g0.hash, instruction_text=first.harness, file_updates=first.bundle_files,
        request_id=first.request_id, experience_id=first.experience_id, phase="meta_self_update",
        changed_mechanisms=first.changed_rules)
    route = MetaDecision(action="HARNESS", diagnosis="Explicit mock diagnosis; behavior is demonstrated, effect is untested.",
        evidence=["toy_train_0", "toy_train_1"], rationale="test_override only; no autonomous routing claim.",
        proposed_change="No Task update is executed by this export.", expected_effect="Unknown; no real evaluation.",
        target_components=["HARNESS"], requested_changes=[], decision_id="test_override_g1_routing", decision_source="test_override")
    execute(g1, "routing", source, MetaDecision, invoke_for(g1, "g1_routing", route), output / "g1_routing")
    no_change = MetaHarnessUpdate(harness=g1.read_files()["instructions.md"], rationale="Engineer-written mock NO_CHANGE.",
        changed_rules=[], status="NO_CHANGE", request_id="test_override_g1_self_update", experience_id="toy_experience_2")
    second = execute(g1, "meta_self_update", source, MetaHarnessUpdate,
        invoke_for(g1, "g1_self_update", no_change), output / "g1_self_update",
        validate_candidate=lambda value: validate_meta_candidate(value, bundle_state(g1)))
    write(output / "no_change_candidate.json", second.model_dump(mode="json"))
    after = store.commit_update(g1.hash, instruction_text=second.harness, file_updates=second.bundle_files,
        request_id=second.request_id, experience_id=second.experience_id, phase="meta_self_update")

    def read(relative):
        return json.loads((output / relative).read_text(encoding="utf-8"))

    summaries = {}
    for name in ("g0_self_update", "g1_self_update"):
        evidence = read(name + "/select_evidence.json")
        experience = read(name + "/retrieve_experiences.json")
        runtime = read(name + "/policy_runtime.json")
        summaries[name] = {"g": runtime["g"], "source_hash": runtime["source_hash"],
            "evidence_ids": [row["source_id"] for row in evidence["selected"]],
            "experience_ids": [row["source_id"] for row in experience["selected"]],
            "experience_summaries": [row["summary"] for row in experience["selected"]],
            "evidence_package_hash": evidence["package_hash"], "evidence_cache_key": evidence["cache_key"],
            "experience_package_hash": experience["package_hash"], "experience_cache_key": experience["cache_key"],
            "trusted_facts_hash": evidence["trusted_facts_hash"],
            "executed_step_ids": [event["step_id"] for event in runtime["events"] if event["status"] == "completed"],
            "model_stage_ids": [event["stage_id"] for event in runtime["events"] if "stage_id" in event],
            "model_invocations": runtime["model_invocations"], "audit_file": name + "/policy_runtime.json"}
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((store.root / "commit_events").glob("*.json"))]
    version_changes = sum(row["status"] == "CHANGED" for row in receipts)
    no_change_events = sum(row["status"] == "NO_CHANGE" for row in receipts)
    routing_prompt = (output / next(row["prompt_file"] for row in stages if row["operation"] == "g1_routing")).read_text(encoding="utf-8")
    invariant_checks = {
        "raw_envelope_unchanged": source == original and digest(source) == source_hash,
        "same_operation_source_hash": summaries["g0_self_update"]["source_hash"] == summaries["g1_self_update"]["source_hash"],
        "same_denominators_and_facts": summaries["g0_self_update"]["trusted_facts_hash"] == summaries["g1_self_update"]["trusted_facts_hash"],
        "evidence_selection_changed": summaries["g0_self_update"]["evidence_ids"] != summaries["g1_self_update"]["evidence_ids"],
        "experience_context_changed": summaries["g0_self_update"]["experience_ids"] != summaries["g1_self_update"]["experience_ids"],
        "g1_diagnosis_and_rule_loaded": "MOCK_G1_DIAGNOSIS" in routing_prompt and "MOCK_G1_RULE_MATCHED" in routing_prompt,
        "g1_self_update_analysis_before_propose": summaries["g1_self_update"]["model_stage_ids"] == ["assess_attribution", "propose"],
        "one_substantive_version_change": g0.version == 0 and g1.version == after.version == 1 and g1.hash == after.hash and version_changes == 1,
        "one_no_change_commit": no_change_events == 1}
    if not all(invariant_checks.values()):
        raise AssertionError("Offline evidence invariant failed: " + encoded(invariant_checks))
    report = {"schema_version": "meta-harness-offline-behavior-evidence-v1", "status": "passed",
        "decision_source": "test_override", "evidence_origin": "engineer_written_mock_callbacks_and_toy_data",
        "interpretation": "Executable project interpreter and store behavior only; no autonomous Meta learning or quality improvement demonstrated.",
        "codex_invocations": 0, "api_invocations": 0, "gpu_invocations": 0, "training_runs": 0,
        "runtime_verified": False, "project_interpreter_executed": True,
        "source_envelope_hash": source_hash, "trusted_facts": source["trusted_facts"],
        "meta_version_changes": version_changes, "no_change_events": no_change_events,
        "mock_model_callbacks": len(stages), "g0": {"hash": g0.hash, "version": g0.version},
        "candidate_patch": "g1_candidate_patch.json", "candidate_hash": digest(first.model_dump(mode="json")),
        "g1": {"hash": g1.hash, "version": g1.version, "parent_hash": g1.manifest["parent_hash"]},
        "comparison": summaries, "stages": stages, "commit_receipts": receipts, "checks": invariant_checks,
        "source_files": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (Path(__file__).resolve(), ROOT / "sia/task_meta/meta_harness/runtime.py", ROOT / "sia/task_meta/meta_harness/policies.py",
             ROOT / "sia/task_meta/meta_harness/bundle.py", ROOT / "sia/task_meta/meta.py")}}
    write(output / "behavior_evidence.json", report)
    (output / "README.md").write_text(
        "# Offline executable Meta Harness behavior\n\n"
        "All responses and input observations are engineer-written `test_override` fixtures. "
        "The project interpreter and immutable bundle store actually execute; Codex/API/GPU/training do not. "
        "This demonstrates mechanism wiring, not autonomous evolution or performance improvement.\n\n"
        f"- Same raw input: `{source_hash}`; denominator remains 3 (one per domain), with 1 toy success.\n"
        f"- G0 selected evidence: {summaries['g0_self_update']['evidence_ids']}. G1: {summaries['g1_self_update']['evidence_ids']}.\n"
        f"- G0 selected experiences: {summaries['g0_self_update']['experience_ids']}. G1: {summaries['g1_self_update']['experience_ids']}.\n"
        "- The mock G0 self-update returns the validated `g1_candidate_patch.json`; the store commits one G1.\n"
        "- G1 routing consumes its changed diagnosis and conditional instruction.\n"
        "- G1 self-update checks recorded outcome/source integrity, executes `assess_attribution`, then `propose`; "
        "its mock NO_CHANGE response creates a commit event without creating G2.\n"
        "- Counts: meta_version_changes=1; no_change_events=1; real API/GPU/training calls=0.\n\n"
        "See `behavior_evidence.json`, each operation's `policy_runtime.json`, the saved derived packages and stage prompts for exact provenance.\n",
        encoding="utf-8")
    return output / "behavior_evidence.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="A new output directory; existing paths are rejected")
    arguments = parser.parse_args()
    try:
        result = run_demo(arguments.output)
    except FileExistsError:
        parser.error("--output must be a new directory; existing paths are never overwritten")
    print(result)


if __name__ == "__main__":
    main()
