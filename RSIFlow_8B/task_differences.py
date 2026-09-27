"""Small paired-outcome references and observed routing fingerprints.

Read original trajectories one row at a time; do not copy dialogues. No causal
diagnosis or candidate acceptance is inferred here.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from skill_memory import write_json


def verified_success(row):
    verification = row.get("verification") or {}
    for field in ("task_success", "success"):
        if isinstance(verification.get(field), bool):
            return verification[field]
    for field in ("success", "correct"):
        if isinstance(row.get(field), bool):
            return row[field]
    # Positive partial rewards are not sufficient evidence of task success.
    return None


def failure_fingerprint(rows):
    domains, errors, positive = Counter(), Counter(), Counter()
    known_success, unknown, zero_calls = 0, 0, 0
    verifiers = set()
    for row in rows:
        if row.get("success_verifier_version"):
            verifiers.add(str(row["success_verifier_version"]))
        domain = str(row.get("domain", "unknown"))
        domains[domain] += 1
        success = verified_success(row)
        if success is True:
            known_success += 1
            positive[domain] += 1
        elif success is None:
            unknown += 1
        if success is not True:
            errors.update(set(str(row[k]) for k in ("error_type", "execution_error_type") if row.get(k)))
            zero_calls += row.get("model_call_count") == 0
    return {"task_count": sum(domains.values()), "domains": dict(domains),
            "error_types": dict(errors), "verified_success_counts": dict(positive),
            "verified_successes": known_success, "unknown_success_status": unknown,
            "failed_zero_model_call_count": zero_calls,
            "success_verifier_versions": sorted(verifiers),
            "training_eligibility_note": "These are verified task counts; SFT sample eligibility comes from the SFT tool.",
            "diagnosis_note": "Error labels/zero calls are observations; Meta determines cause using the original evidence."}


def _shape(answer):
    text = str(answer or "").strip()
    if not text:
        return "empty"
    if "def " in text or "```" in text:
        return "code"
    if "\n" in text:
        return "multiline"
    return "short_text" if len(text.split()) <= 5 else "sentence"


def _collect(path):
    rows, duplicates = {}, []
    with Path(path).open("rb") as stream:
        for line, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            task_id = row.get("task_id") or row.get("question_id")
            key = (str(task_id), str(row.get("rollout_id", 0)))
            if key in rows:
                duplicates.append({"task_id": key[0], "rollout_id": key[1], "line": line})
                continue
            rows[key] = {"success": verified_success(row), "domain": row.get("domain"),
                         "error_type": row.get("error_type"),
                         "execution_error_type": row.get("execution_error_type"),
                         "answer_shape": _shape(row.get("final_answer", row.get("model_answer"))),
                         "model_call_count": row.get("model_call_count"),
                         "tool_call_count": len(row.get("tool_calls") or []),
                         "costs": {field: row.get(field) for field in
                                   ("input_tokens", "output_tokens", "wall_time_seconds")},
                         "evidence": {"path": str(Path(path).resolve()), "line": line,
                                      "task_id": task_id, "rollout_id": row.get("rollout_id", 0),
                                      "content_hash": hashlib.sha256(raw).hexdigest(),
                                      "task_source_hash": row.get("task_source_hash"),
                                      "verifier_id": (row.get("verification") or {}).get("verifier_id")}}
    return rows, duplicates


def compare_task_differences(before_path, after_path, output_dir, skill_use_path=None):
    before, before_duplicates = _collect(before_path)
    after, after_duplicates = _collect(after_path)
    groups = defaultdict(Counter)
    transitions = Counter()
    tasks = []
    source_mismatches = []
    for key in sorted(before.keys() & after.keys()):
        left, right = before[key], after[key]
        a, b = left["success"], right["success"]
        category = ("unknown" if a is None or b is None else "new_success" if not a and b else
                    "new_regression" if a and not b else "retained_success" if a else "retained_failure")
        groups[str(left["domain"])][category] += 1
        transitions[f"{left['error_type']} -> {right['error_type']}"] += 1
        old_source, new_source = left["evidence"]["task_source_hash"], right["evidence"]["task_source_hash"]
        if old_source is not None and new_source is not None and old_source != new_source:
            source_mismatches.append({"task_id": key[0], "rollout_id": key[1]})
        tasks.append({"task_id": key[0], "rollout_id": key[1], "transition": category,
                      "before": left, "after": right,
                      "cost_delta": {field: right["costs"][field] - left["costs"][field]
                                     if isinstance(left["costs"][field], (int, float))
                                     and isinstance(right["costs"][field], (int, float)) else None
                                     for field in left["costs"]}})
    counts = Counter(t["transition"] for t in tasks)
    output_dir = Path(output_dir)
    summary = {"status": "compared", "before_path": str(Path(before_path).resolve()),
               "after_path": str(Path(after_path).resolve()), "paired_tasks": len(tasks),
               "same_id_set": before.keys() == after.keys(), "counts": dict(counts),
               "domain_transitions": {key: dict(value) for key, value in groups.items()},
               "error_transitions": dict(transitions), "source_mismatches": source_mismatches,
               "duplicates": {"before": before_duplicates, "after": after_duplicates},
               "missing_in_after": [list(key) for key in sorted(before.keys() - after.keys())],
               "extra_in_after": [list(key) for key in sorted(after.keys() - before.keys())],
               "notes": ["Transitions use official success booleans, not positive partial reward.",
                         "Source/verifier/ID differences are reported for Meta to assess comparability.",
                         "Counts do not establish causal contribution of individual edits."]}
    # Keep the pre-intervention claim immutable; attach factual outcomes separately.
    if skill_use_path:
        usage = json.loads(Path(skill_use_path).read_text(encoding="utf-8"))
        summary["prediction_comparison"] = {
            "skill_use_path": str(Path(skill_use_path).resolve()),
            "used_skill_ids": usage.get("used_skill_ids", []),
            "prospective_prediction": usage.get("prediction", {}),
            "observed_task_transitions": dict(counts),
            "assessment": "Meta must compare the frozen prediction with task evidence; no automatic causal verdict."}
    details = output_dir / "paired_tasks.json"
    write_json(details, {"tasks": tasks})
    summary["paired_tasks_path"] = str(details.resolve())
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    return {**summary, "summary_path": str(summary_path.resolve())}
