"""Interpret one pinned G policy over a trusted, immutable JSON evidence envelope.

All model stages go through ``invoke_stage`` supplied by the Codex backend. This
module has no model provider, arbitrary code evaluator, shell, or file query tool.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sia.task_meta.meta_harness.policies import ALIASES, validate_policy
from sia.task_meta.storage import save_json
from sia.task_meta.meta_harness import five_stage


class PolicyBudgetExceeded(ValueError):
    pass


class CandidateRejected(ValueError):
    pass


class AnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analysis: str = Field(min_length=1, max_length=12000)
    hypotheses: list[str] = Field(default_factory=list, max_length=16)
    source_ids: list[str] = Field(default_factory=list, max_length=128)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _bounded_inline(payload, maximum_bytes=600000):
    """Keep full evidence in operation.json; native Codex input has a byte limit."""
    inline = copy.deepcopy(payload)
    feedback = inline.get('trusted_facts', {}).get('constraint_feedback')
    if feedback and len(_json(feedback).encode()) > 20000:
        inline['trusted_facts']['constraint_feedback'] = {
            'content_reference': {'file': 'meta_input/operation.json',
                'json_path': ['trusted_facts', 'constraint_feedback']},
            'attempts': [{k: a.get(k) for k in ('attempt_id', 'action', 'status', 'gain',
                'positive_gain', 'reason', 'parent_content_hash')} for a in feedback]}
    # Large analysis/repair candidates and accumulated evidence use the same bounded
    # interface; no evidence is deleted from the authoritative operation file.
    while len(_json(inline).encode()) > maximum_bytes:
        candidates = [(len(_json(v).encode()), k) for k,v in inline.items()
                      if not (isinstance(v, dict) and 'content_reference' in v)]
        size, key = max(candidates)
        if size < 1000:
            raise ValueError('Operation metadata exceeds native input capacity')
        inline[key] = {'content_reference': {'file': 'meta_input/operation.json', 'json_path': [key]},
                       'bytes': size, 'instruction': 'Read this declared field before relying on its evidence.'}
    return inline


def _plain(value):
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else copy.deepcopy(value)


_MISSING = object()


def lookup(value, path, default=None):
    current = value
    for key in path.split("."):
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and key.isdigit() and int(key) < len(current):
            current = current[int(key)]
        else:
            return default
    return current


def condition(expression, scope):
    """Evaluate only validated JSON comparisons, never a source expression."""
    if type(expression) is bool:
        return expression
    if "all" in expression:
        return all(condition(value, scope) for value in expression["all"])
    if "any" in expression:
        return any(condition(value, scope) for value in expression["any"])
    if "not" in expression:
        return not condition(expression["not"], scope)
    observed = lookup(scope, expression["path"], _MISSING)
    op = expression["op"]
    if op == "exists":
        return observed is not _MISSING
    expected = lookup(scope, expression["value_path"], _MISSING) if "value_path" in expression else expression["value"]
    if observed is _MISSING or expected is _MISSING:
        return False
    if op in {"eq", "ne"}:
        equal = type(observed) is type(expected) and observed == expected
        return equal if op == "eq" else not equal
    if op in {"gt", "gte", "lt", "lte"}:
        if type(observed) not in (int, float) or type(expected) not in (int, float):
            return False
        return {"gt": observed > expected, "gte": observed >= expected,
                "lt": observed < expected, "lte": observed <= expected}[op]
    if op in {"in", "contains"}:
        container, needle = (expected, observed) if op == "in" else (observed, expected)
        if isinstance(container, str):
            return isinstance(needle, str) and needle in container
        if isinstance(container, dict):
            return isinstance(needle, str) and needle in container
        if isinstance(container, list):
            return any(type(needle) is type(item) and needle == item for item in container)
        return False
    raise ValueError("Unknown validated condition")


def _scope(envelope, **extra):
    return {"facts": envelope.get("trusted_facts", {}), "trusted_facts": envelope.get("trusted_facts", {}),
            "task": envelope.get("task_state", {}), "task_state": envelope.get("task_state", {}),
            "decision": envelope.get("decision", {}), "latest_experience": envelope.get("latest_experience", {}), **extra}


def _source_records(rows, kind):
    result, seen = [], {}
    if not isinstance(rows, list) or len(rows) > 100000:
        raise ValueError("Evidence index must be a bounded list of controller records")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Evidence rows must be objects")
        if kind == "trajectory" and row.get("split", "evolve_train") != "evolve_train":
            raise ValueError("Only training trajectories may enter the Meta evidence index")
        if row.get("source_role") in {"independent_validation","final_test","evolution_feedback"} or row.get("purpose") == "report_only":
            raise ValueError("External evidence is forbidden for Meta")
        digest = _hash(row)
        identifier = row.get("source_id") or (row.get("experience_id") if kind == "experience" else None)
        identifier = str(identifier or f"{kind}:{index}:{digest[:16]}")
        if identifier in seen and seen[identifier] != digest:
            raise ValueError("Source ID refers to conflicting immutable records")
        if identifier in seen:
            raise ValueError("Source IDs must be unique in the operation envelope")
        seen[identifier] = digest
        enriched = copy.deepcopy(row)
        if kind == "trajectory":
            reward = row.get("terminal_reward", 0)
            enriched["success"] = type(reward) in (int, float) and reward > 0
        result.append({"id": identifier, "source_index": index, "source_hash": digest, "item": enriched})
    return result


def _clip(value, limit, field, truncations):
    if isinstance(value, str):
        if len(value) > limit:
            truncations.append({"field": field, "original_chars": len(value), "kept_range": [0, limit],
                                "omitted_range": [limit, len(value)], "source_hash": _hash(value)})
            return value[:limit]
        return value
    rendered = _json(value)
    if len(rendered) <= limit:
        return copy.deepcopy(value)
    truncations.append({"field": field, "original_chars": len(rendered), "kept_range": [0, limit],
                        "omitted_range": [limit, len(rendered)], "source_hash": _hash(value), "encoding": "canonical_json"})
    return {"canonical_json_excerpt": rendered[:limit], "complete": False}


def _projection(item, fields, limit, truncations):
    projected = {}
    for field in fields:
        value = lookup(item, field, _MISSING)
        if value is not _MISSING:
            projected[field] = _clip(value, limit, field, truncations)
    return projected


def _slice_text(content, offset, limit):
    start, stop = min(offset, len(content)), min(offset + limit, len(content))
    omitted = [[0, start]] if start else []
    if stop < len(content):
        omitted.append([stop, len(content)])
    return content[start:stop], {"original_chars": len(content), "kept_range": [start, stop],
                                 "omitted_ranges": omitted, "source_hash": _hash(content)}


def _rank(records, policy, envelope, operation):
    accepted, omitted = [], []
    for record in records:
        scope = _scope(envelope, item=record["item"], operation=operation)
        if not condition(policy["filter"], scope):
            omitted.append({"source_id": record["id"], "reason": "policy_filter"})
            continue
        record = copy.deepcopy(record)
        matches = [rule for rule in policy["score_rules"] if condition(rule["when"], scope)]
        record["relevance"] = sum(rule["weight"] for rule in matches)
        record["relevance_rules"] = [rule["id"] for rule in matches]
        accepted.append(record)
    # Stable sorting permits composable priority fields; no arbitrary sort code.
    for rule in reversed(policy["sort"]):
        def key(record, field=rule["field"]):
            value = lookup(record["item"], field)
            return (0, float(value)) if type(value) in (int, float) else (1, _json(value))
        accepted.sort(key=key, reverse=rule["direction"] == "desc")
    accepted.sort(key=lambda row: row["relevance"], reverse=True)
    if policy["group_by"]:
        groups = {}
        for row in accepted:
            group = _json([lookup(row["item"], field) for field in policy["group_by"]])
            groups.setdefault(group, []).append(row)
        # Round-robin groups in ranked order. Changing group keys changes which
        # examples survive a bounded max_items rather than only changing labels.
        accepted = []
        for index in range(max((len(group) for group in groups.values()), default=0)):
            accepted.extend(group[index] for group in groups.values() if index < len(group))
    omitted.extend({"source_id": row["id"], "reason": "max_items"} for row in accepted[policy["max_items"]:])
    return accepted[:policy["max_items"]], omitted


def _fragments(record, settings, envelope, operation):
    messages = record["item"].get("messages", [])
    if not isinstance(messages, list):
        messages = []
    anchors, selected, mandatory, truncations = [], set(), set(), []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if condition(settings["anchor"], _scope(envelope, item=record["item"], message=message, operation=operation)):
            anchors.append(index)
            selected.update(range(max(0, index - settings["before"]), min(len(messages), index + settings["after"] + 1)))
        is_tool = message.get("role") == "tool" or str(message.get("content", "")).startswith("Tool calling observation:")
        if is_tool and ("*" in settings["retain_tools"] or message.get("name") in settings["retain_tools"]):
            mandatory.add(index)
    if not selected and settings["fallback"] != "none":
        size = settings["max_messages"]
        selected.update(range(min(size, len(messages))) if settings["fallback"] == "head"
                        else range(max(0, len(messages) - size), len(messages)))
    selected = set(sorted(selected)[:settings["max_messages"]]) | mandatory
    fragments = []
    for index in sorted(selected):
        message = messages[index]
        value = copy.deepcopy(message) if index in mandatory else {
            key: _clip(item, settings["message_chars"], f"messages.{index}.{key}", truncations)
            for key, item in message.items()}
        fragments.append({"message_index": index, "mandatory_tool_observation": index in mandatory,
                          "source_hash": _hash(message), "message": value})
    return {"anchor_indices": anchors, "fragments": fragments, "truncations": truncations,
            "omitted_message_indices": sorted(set(range(len(messages))) - selected)}


def _fit(items, limit, preference):
    order = list(range(len(items)))
    if preference == "tail":
        order.reverse()
    elif preference == "head_and_tail":
        order = [index for pair in zip(range((len(items) + 1) // 2), range(len(items) - 1, len(items) // 2 - 1, -1)) for index in pair]
        order = list(dict.fromkeys(order))
    kept, omitted, used = [], [], 2
    for index in order:
        item = items[index]
        size = len(_json(item)) + 1
        if used + size > limit:
            if any(fragment.get("mandatory_tool_observation") for fragment in item.get("fragments", [])):
                raise PolicyBudgetExceeded("A required tool observation cannot fit the fixed evidence budget")
            omitted.append({"source_id": item["source_id"], "reason": "context_character_budget", "item_chars": size})
        else:
            kept.append(index)
            used += size
    return [items[index] for index in sorted(kept)], omitted


def _identity(envelope, policy, bundle_identity, kind, operation):
    return {"kind": kind, "operation": operation, "g": copy.deepcopy(bundle_identity),
            "policy_hash": _hash(policy), "source_hash": _hash(envelope)}


def build_evidence(envelope, policy, bundle_identity, *, operation="routing", context_policy=None, cache=None):
    started = time.perf_counter()
    selection = policy["evidence"]
    context_policy = context_policy or {"max_evidence_chars": selection["max_chars"], "selection": "head"}
    identity = _identity(envelope, policy, bundle_identity, "evidence", operation)
    identity["context_policy"] = copy.deepcopy(context_policy)
    key = _hash(identity)
    if cache is not None and key in cache:
        return copy.deepcopy(cache[key])
    records = _source_records(envelope.get("raw_trajectories", []), "trajectory")
    ranked, omitted = _rank(records, selection, envelope, operation)
    selected = []
    for record in ranked:
        fragments = _fragments(record, selection["fragments"], envelope, operation)
        projected = _projection(record["item"], selection["fields"], selection["field_chars"], fragments["truncations"])
        selected.append({"source_id": record["id"], "source_index": record["source_index"], "source_hash": record["source_hash"],
                         "projection": projected, "group": {field: lookup(record["item"], field) for field in selection["group_by"]},
                         "relevance": record["relevance"], "relevance_rules": record["relevance_rules"], **fragments})
    selected, clipped = _fit(selected, min(selection["max_chars"], context_policy["max_evidence_chars"]), context_policy["selection"])
    package = {"identity": identity, "cache_key": key, "source_count": len(records), "selected": selected,
               "omitted": omitted + clipped, "trusted_facts_hash": _hash(envelope.get("trusted_facts", {})),
               "raw_source_mutated": False}
    package["package_hash"] = _hash(package)
    package["cost"] = {"processing_seconds": time.perf_counter() - started, "selected_chars": len(_json(selected)), "model_calls": 0}
    if cache is not None:
        cache[key] = copy.deepcopy(package)
    return package


def build_experience_context(envelope, policy, bundle_identity, *, operation="routing", cache=None):
    started = time.perf_counter()
    selection = policy["experience"]
    identity = _identity(envelope, policy, bundle_identity, "experience", operation)
    key = _hash(identity)
    if cache is not None and key in cache:
        return copy.deepcopy(cache[key])
    records = _source_records(envelope.get("experiences", []), "experience")
    accepted, conditions, omitted = [], {}, []
    for record in records:
        scope = _scope(envelope, item=record["item"], operation=operation)
        verdict = {name: [{"rule_id": rule["id"], "condition": copy.deepcopy(rule["when"]),
                           "matched": condition(rule["when"], scope), "interpretation": rule["instruction"]}
                          for rule in selection[name]] for name in ("applicability", "failure")}
        conditions[record["id"]] = verdict
        applicable = not selection["require_applicable"] or any(value["matched"] for value in verdict["applicability"])
        failed = selection["exclude_failed"] and any(value["matched"] for value in verdict["failure"])
        if applicable and not failed:
            accepted.append(record)
        else:
            omitted.append({"source_id": record["id"], "reason": "applicability_or_failure_condition", "conditions": verdict})
    ranked, excluded = _rank(accepted, selection, envelope, operation)
    selected = []
    for record in ranked:
        truncated = []
        selected.append({"source_id": record["id"], "source_hash": record["source_hash"], "source_index": record["source_index"],
                         "summary": _projection(record["item"], selection["fields"], selection["field_chars"], truncated),
                         "conditions": conditions[record["id"]], "truncations": truncated,
                         "group": {field: lookup(record["item"], field) for field in selection["group_by"]},
                         "relevance": record["relevance"], "relevance_rules": record["relevance_rules"]})
    selected, clipped = _fit(selected, selection["max_chars"], "head")
    package = {"identity": identity, "cache_key": key, "source_count": len(records), "selected": selected,
               "omitted": omitted + excluded + clipped, "raw_ledger_mutated": False,
               "conditions_are_policy_predicates_not_proven_generalizations": True}
    package["package_hash"] = _hash(package)
    package["cost"] = {"processing_seconds": time.perf_counter() - started, "selected_chars": len(_json(selected)), "model_calls": 0}
    if cache is not None:
        cache[key] = copy.deepcopy(package)
    return package


def _prompt_package(package):
    # Wall-time costs and full omitted ID lists belong to the audit, not a prompt
    # key. Their removal also makes deterministic stage replay stable.
    return {key: copy.deepcopy(value) for key, value in package.items() if key not in {"cost", "omitted"}} | {
        "omitted_count": len(package.get("omitted", []))}


def execute(bundle, operation, operation_input, schema, invoke_stage, audit_dir, *,
            external_budget=None, validate_candidate=None, cache=None):
    """Run one version-pinned operation and return one contract-valid candidate."""
    started = time.monotonic()
    bundle.verify()
    files = bundle.read_files()
    policy = validate_policy(json.loads(files["evolution.json"]))
    from . import graph as meta_graph
    graph_mode = policy["schema_version"] == 2
    context_policy = json.loads(files["context.json"])
    operation = ALIASES.get(operation, operation)
    if operation not in policy["workflows"]:
        raise ValueError("Operation has no G workflow")
    if not isinstance(operation_input, dict) or not isinstance(operation_input.get("trusted_facts", {}), dict):
        raise ValueError("Operation requires the trusted structured evidence envelope")
    envelope = copy.deepcopy(operation_input)
    # Source eligibility is part of the fixed boundary, even when G omits an
    # evidence-selection step entirely.
    _source_records(envelope.get("raw_trajectories", []), "trajectory")
    _source_records(envelope.get("experiences", []), "experience")
    source_hash = _hash(envelope)
    source_snapshot_hash = _hash(operation_input)
    g = {"bundle_hash": bundle.hash, "version": bundle.version}
    if graph_mode:
        g.update(meta_graph.identity(policy))
    five_identity = five_stage.identity(files)
    if five_identity:
        g.update(five_identity)
    budget = {"max_invocations": 12, "wall_time_seconds": 300, **(external_budget or {})}
    if type(budget["max_invocations"]) is not int or not 1 <= budget["max_invocations"] <= 64:
        raise ValueError("Invalid external invocation budget")
    if type(budget["wall_time_seconds"]) not in (int, float) or not 0 < budget["wall_time_seconds"] <= 7200:
        raise ValueError("Invalid external wall-time budget")
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit = {"g": g, "operation": operation, "source_hash": source_hash, "policy_hash": _hash(policy),
             "candidate_id": _hash([g, operation, source_hash]), "candidate_revision": 0,
             "entrypoint": "sia.task_meta.meta_harness.runtime.execute", "events": [], "model_invocations": 0,
             "status": "running", "external_budget": budget, "raw_source_mutated": False,
             "decision_source": getattr(invoke_stage, "decision_source", "unknown")}
    state = {"operation": operation, "candidate": None, "candidate_valid": False, "candidate_delivery_valid": False, "checks": {},
             "last_check_passed": True, "analyses": [], "dependencies": [], "evidence": {}, "experience_context": {}}
    candidate_errors = []

    def save():
        save_json(audit_dir / "policy_runtime.json", audit)

    def guard():
        if time.monotonic() - started > budget["wall_time_seconds"]:
            raise PolicyBudgetExceeded("Meta operation wall-time budget exhausted")
        if _hash(operation_input) != source_snapshot_hash or (not graph_mode and _hash(envelope) != source_hash):
            raise ValueError("Immutable operation evidence was mutated")
        bundle.verify()
        if bundle.hash != g["bundle_hash"]:
            raise ValueError("Pinned G identity changed during operation")

    def review_source_ids():
        # One catalog for the prompt and validator; only controller-bound sources.
        allowed = {"current_G", *[x.get("experience_id") for x in envelope.get("experiences", [])]}
        allowed.update(x.get("decision", {}).get("decision_id") for x in envelope.get("experiences", []))
        allowed.update(x.get("source_id") for x in envelope.get("raw_trajectories", []))
        allowed.update(x["source_id"] for x in state["dependencies"])
        for pair in (envelope.get("outcome_review") or {}).get("pairs", []):
            allowed.update(row["source_id"] for row in (pair.get("before"), pair.get("after")) if row)
        allowed.update(pair["source_id"] for pair in (envelope.get("outcome_review") or {}).get("training_comparison",{}).get("pairs",[]))
        for attempt in (envelope.get('outcome_review') or {}).get('candidate_attempts', []):
            allowed.add(attempt['attempt_id'])
            allowed.add(attempt.get('decision', {}).get('decision_id'))
            outcome = attempt.get('paired_outcome', {})
            for pair in outcome.get('pairs', []):
                allowed.update(row['source_id'] for row in (pair.get('before'), pair.get('after')) if row)
            allowed.update(pair['source_id'] for pair in outcome.get('training_comparison', {}).get('pairs', []))
        allowed.discard(None)
        return allowed

    def candidate_check():
        errors = []
        if not state["candidate_delivery_valid"]:
            errors.append("Native candidate delivery has not passed; retained repair input is not an accepted candidate")
        if state["candidate"] is None:
            errors.extend(candidate_errors or ["No candidate has been proposed"])
        else:
            try:
                typed = schema.model_validate(state["candidate"])
                available = envelope.get("trusted_facts", {}).get("available_actions")
                action = state["candidate"].get("action")
                if (operation == "routing" and available is not None and action is not None
                        and (not isinstance(available, dict) or available.get(action, {}).get("available") is not True)):
                    errors.append("Candidate action is unavailable in the immutable controller facts")
                if five_identity:
                    library = json.loads(files["principles.json"])
                    active_ids = {p["principle_id"] for p in library["records"] if p["active"]}
                    if operation == "routing":
                        if typed.expectations is None or not set(typed.used_principle_ids + typed.expectations.principle_ids) <= active_ids:
                            errors.append("Routing needs expectations and only active principle references")
                        if {c.scope for c in typed.expectations.costs} != {"one_time_update", "per_rollout"}:
                            errors.append("Include one cost estimate per scope; unknown values are null")
                        forecast = typed.expectations.performance
                        observed = envelope.get("trusted_facts", {}).get("performance", {}).get(forecast.metric)
                        if forecast.baseline is not None and (type(observed) not in (int, float) or abs(observed - forecast.baseline) > 1e-9):
                            errors.append("Performance baseline must match the exact trusted metric value or be null")
                    if operation in {"meta_self_update", "final_consolidation"}:
                        outcome = envelope.get("outcome_review") or {}
                        expected = outcome.get("expectation", {})
                        if typed.five_stage is None or (typed.five_stage.expectation_id != expected.get("expectation_id")
                                or typed.five_stage.outcome_hash != outcome.get("outcome_hash")):
                            raise ValueError("Five-stage review must bind the frozen expectation and actual outcome; "
                                "copy these exact fields into five_stage: " + _json({
                                    "expectation_id": expected.get("expectation_id"),
                                    "outcome_hash": outcome.get("outcome_hash")}) +
                                ". An aggregate candidate round intentionally has expectation_id=null; "
                                "do not substitute an individual candidate's expectation or outcome hash.")
                        allowed = review_source_ids()
                        for finding in (typed.five_stage.implementation, typed.five_stage.activation, typed.five_stage.outcome):
                            unavailable = set(finding.evidence_ids) - allowed
                            if unavailable:
                                raise ValueError("Review cites an unavailable source: " + ", ".join(sorted(unavailable)))
                            if finding.status in {"met", "partially_met", "not_met"} and not finding.evidence_ids:
                                raise ValueError("A factual review conclusion requires references")
                        _, flags = five_stage.materialize(typed, files, allowed_evidence=allowed)
                        save_json(audit_dir / "five_stage_candidate.json", {"g": g, **flags,
                            "review": typed.five_stage.model_dump(mode="json"), "new_state_consumed": False})
                if validate_candidate:
                    verdict = validate_candidate(typed)
                    if verdict is False or (isinstance(verdict, dict) and verdict.get("passed") is False):
                        errors.extend(verdict.get("errors", ["Fixed interface checker rejected the candidate"]) if isinstance(verdict, dict)
                                      else ["Fixed interface checker rejected the candidate"])
            except (ValidationError, ValueError) as exc:
                errors.append(str(exc))
        state["candidate_valid"] = not errors
        state["checks"]["candidate_contract"] = {"passed": not errors, "errors": errors, "source": "fixed_schema_and_interface"}
        state["last_check_passed"] = not errors
        return errors

    def check(name):
        passed, details = True, {}
        if name == "source_integrity":
            guard()
            details = {"source_hash": source_hash, "trusted_facts_hash": _hash(envelope.get("trusted_facts", {}))}
        elif name in {"candidate_schema", "candidate_interface"}:
            errors = candidate_check()
            passed, details = not errors, {"errors": errors}
        elif name == "requested_targets":
            changes = envelope.get("decision", {}).get("requested_changes", [])
            targets = [change.get("target") for change in changes]
            passed = len(targets) == len(set(targets)) and all(isinstance(value, str) and value for value in targets)
            details = {"targets": targets, "decision_source_hash": _hash(envelope.get("decision", {}))}
        elif name == "available_action":
            action = (state["candidate"] or envelope.get("decision", {})).get("action")
            available = envelope.get("trusted_facts", {}).get("available_actions", {})
            record = available.get(action, {})
            passed = action is None or record.get("available") is True
            details = {"action": action, "availability": copy.deepcopy(record)}
        elif name == "experience_outcome":
            latest = envelope.get("latest_experience") or {}
            details = {field: copy.deepcopy(latest[field]) for field in ("experience_id", "requested_change", "actual_change",
                "observed_performance_delta", "performance_delta", "cost_before", "cost_after", "update_cost") if field in latest}
            details["comparison_kind"] = "recorded_observed_delta_not_isolated_causality"
        else:
            raise ValueError("Undeclared engineering check")
        verdict = {"passed": passed, "details": details, "source": "controller_check", "check": name}
        state["checks"][name] = verdict
        state["last_check_passed"] = passed
        return verdict

    def model_step(step, stage_id, *, repair=False):
        nonlocal candidate_errors
        if graph_mode and graph_event["iteration"] > 1:
            stage_id += "_visit_" + str(graph_event["iteration"])
        guard()
        if audit["model_invocations"] >= budget["max_invocations"]:
            raise PolicyBudgetExceeded("Meta operation model-invocation budget exhausted")
        mechanism = "self_update" if operation in {"meta_self_update", "final_consolidation"} else "diagnosis"
        scope = _scope(envelope, **state)
        matched = [rule for rule in policy[mechanism]["rules"] if condition(rule["when"], scope)]
        instruction = "\n".join([policy[mechanism]["instruction"], *[rule["instruction"] for rule in matched], step.get("instruction", "")])
        payload = {"operation": operation, "stage_id": stage_id, "stage_kind": step["kind"], "g": g,
                   "trusted_facts": envelope.get("trusted_facts", {}), "task_state": envelope.get("task_state", {}),
                   "decision": envelope.get("decision", {}), "request": envelope.get("instruction", ""),
                   "evidence": _prompt_package(state["evidence"]),
                   "experience_context": _prompt_package(state["experience_context"]),
                   "dependencies": state["dependencies"], "checks": state["checks"], "analyses": state["analyses"],
                   "candidate": state["candidate"] if repair else None,
                   "candidate_errors": candidate_errors if repair else [],
                   "candidate_id": audit["candidate_id"], "candidate_revision": audit["candidate_revision"],
                   "workflow_steps": [{"id": value["id"], "kind": value["kind"]} for value in policy["workflows"][operation]]}
        if five_identity:
            library = json.loads(files["principles.json"])
            payload["meta_memory"] = {"identity": five_identity, "views": five_stage.library_views(library),
                "active_principles": [p for p in library["records"] if p["active"]],
                "retired_principles": [{"principle_id": p["principle_id"], "active": False,
                    "counter_evidence": p["counter_evidence"]} for p in library["records"] if not p["active"]]}
            payload["protocol_phase"] = "before_task_update" if operation == "routing" else (
                "after_task_reevaluation" if operation == "meta_self_update" else operation)
            feedback_protocol = envelope.get("trusted_facts",{}).get("feedback_protocol",{})
            if feedback_protocol.get("purpose") == "evolution_train":
                payload["protocol_phase"] = feedback_protocol["phase"]
            payload["principle_contract"] = {
                "existing_principle_ids": sorted(p["principle_id"] for p in library["records"]),
                "active_principle_ids": sorted(p["principle_id"] for p in library["records"] if p["active"]),
                "id_contract": "G workflow/rule IDs are not principle IDs. compared_ids must be exact existing_principle_ids. REVISE/MERGE/RETIRE must target an exact active_principle_id; do not invent aliases. ADD alone creates a new principle ID. Copy the selected prior record from meta_input/principles.json and increment its revision when revising.",
                "store": "principles.json is Meta Memory, atomically committed inside G, never Task Artifacts",
                "maintenance": "Use five_stage.principle_operations; do not put principles.json or self_update_protocol.md in bundle_files. ADD starts revision=1, active=true, evidence_state=tentative. MERGE needs new independent evidence. REVISE increments revision. RETIRE has record=null and must remove the bound G rule.",
                "targets": "Each substantive G change needs five_stage.harness_bindings: active principle ID, exact file#/JSON/pointer (or instructions.md), purpose and observable next event. Register that target in the principle g_targets.",
                "evidence_ids": sorted(review_source_ids()),
                "decision": "Use exact trusted performance metric units; unknown baseline/delta/cost are null, not invented. Include exactly one one_time_update and one per_rollout cost. Final test results are unavailable.",
                "no_change": "Fast policy: status UPDATED and actual Meta content change are required. Grounded Memory-only updates are valid; leave harness and other Bundle files unchanged when only Memory changes. Record facts, counterexamples or unresolved evidence without inventing principles. NO_CHANGE and revision-only changes fail. Use exact listed evidence IDs."}
            memory_contract = envelope.get("trusted_facts",{}).get("memory_update_contract")
            if memory_contract:
                payload["principle_contract"].update(maintenance=memory_contract["instructions"],
                    id_contract="Only ADD new skill.<COMPONENT>.<id> followed by principle.<id>; preserve every existing ID and record.",
                    targets="Keep g_targets and harness_bindings empty; consume appended Memory in the next round.")
            outcome = envelope.get("outcome_review")
            payload["principle_contract"]["required_review_binding"] = {
                "expectation_id": ((outcome or {}).get("expectation") or {}).get("expectation_id"),
                "outcome_hash": (outcome or {}).get("outcome_hash"),
                "instruction": "Copy these exact values into five_stage. For a round with candidate_attempts, expectation_id=null is intentional and outcome_hash binds the whole round; individual candidates retain their own frozen expectations and hashes. Do not substitute one candidate's binding. For final_consolidation with no single outcome_review both values are null; cite individual experiences as historical evidence without inventing an aggregate intervention. If training_comparison is present, compare all fixed training-task summaries and their failures; keep training replay distinct from the held-out development probe and cite its registered train_pair IDs."}
            if outcome:
                payload["outcome_review"] = {k:copy.deepcopy(v) for k,v in outcome.items() if k != "pairs"}
                # Full traces for every attempted candidate stay in the same read-only
                # dependency; do not multiply them into the top-level prompt.
                def compact_candidate_trace(value):
                    if isinstance(value, list):
                        return [compact_candidate_trace(item) for item in value]
                    if not isinstance(value, dict):
                        return value
                    compact = {k: compact_candidate_trace(v) for k, v in value.items()
                               if k not in {'messages', 'events', 'tool_calls', 'transitions'}}
                    if value.get('source_hash') and value.get('source_id'):
                        compact['full_trace_dependency'] = 'meta_input/internal_dev_comparison.json'
                    return compact
                if 'candidate_attempts' in payload['outcome_review']:
                    payload['outcome_review']['candidate_attempts'] = compact_candidate_trace(payload['outcome_review']['candidate_attempts'])
                trace_fields = {"messages", "events", "tool_calls", "transitions"}
                payload["outcome_review"]["pairs"] = [{**pair, **{side: {
                    **{k:v for k,v in pair[side].items() if k not in trace_fields},
                    "trace_reference": {"file": "meta_input/internal_dev_comparison.json",
                        "json_path": ["pairs", index, side],
                        "available_fields": sorted(trace_fields & pair[side].keys()),
                        "source_hash": pair[side]["source_hash"]}}
                    if pair.get(side) else None for side in ("before", "after")}}
                    for index, pair in enumerate(outcome["pairs"])]
                payload["outcome_review"]["full_readonly_dev_trajectories"] = "meta_input/internal_dev_comparison.json"
        if graph_mode:
            payload["method2_contract"] = {"types": list(meta_graph.TYPES),
                "graph_identity": meta_graph.identity(policy), "metadata": policy["method2"],
                "fast": "Change local content only. Memory-only changes count. Preserve graph nodes, handlers, interfaces, guards, edges and loop bounds. Evidence uncertainty can be Memory content; do not invent conclusions.",
                "slow": "Use five_stage.structural_mismatches for recorded missing, redundant, or misrouted processing that remains after content review, and five_stage.slow_edit for at most one Insert/Prune/Rewire. Cite actual mismatch evidence and replay requirements; do not force a slow edit or invent persistence. Retain fast content. No Task side effects in graph loops.",
                # The audit's input_hash includes evidence-processing timings.
                # Keep it in the audit, not in a deterministically replayed request.
                "current_node": {k: v for k, v in graph_event.items() if k != "input_hash"}}
        stage_files = {"meta_input/operation.json": _json(payload),
                       **{"meta_input/G/" + name: content for name, content in files.items()}}
        # The fixed validator is an interface dependency, not an evolvable file.
        # Make its exact rules available when proposing a new G Bundle.
        if operation in {"meta_self_update", "final_consolidation"}:
            stage_files["meta_input/meta_policy_validator.py"] = Path(__file__).with_name("policies.py").read_text(encoding="utf-8")
            if graph_mode:
                stage_files["meta_input/meta_graph_contract.py"] = Path(__file__).with_name("graph.py").read_text(encoding="utf-8")
        if five_identity:
            stage_files["meta_input/self_update_protocol.md"] = files["self_update_protocol.md"]
            if operation in {"meta_self_update", "final_consolidation"}:
                stage_files["meta_input/principle_contract.py"] = Path(five_stage.__file__).read_text(encoding="utf-8")
            if envelope.get("outcome_review"):
                stage_files["meta_input/internal_dev_comparison.json"] = _json(envelope["outcome_review"])
        if operation in {"meta_self_update", "final_consolidation"} and envelope.get("native_candidate_repair"):
            stage_files["meta_input/previous_native_candidate.json"] = _json(envelope["native_candidate_repair"])
        if five_identity:
            stage_files["meta_input/allowed_evidence_ids.json"] = _json(payload["principle_contract"]["evidence_ids"])
        inline_payload = _bounded_inline(payload)
        inline_dependencies = inline_payload['dependencies']
        for index, dependency in enumerate(inline_dependencies if isinstance(inline_dependencies, list) else []):
            if dependency.get("path", "").startswith("runtime/") and "content" in dependency:
                content = dependency.pop("content")
                dependency["content_reference"] = {"file": "meta_input/operation.json",
                    "json_path": ["dependencies", index, "content"], "characters": len(content)}
        prompt = ("Current executable G mechanism instructions:\n" + instruction
                  + "\nThe complete attributed operation data is at meta_input/operation.json; the bound G files are at meta_input/G/. "
                  "Absolute Task/run paths in the data are controller provenance, not mounted paths. "
                  "The selected evidence, facts, current configuration, and candidate errors are inline below. "
                  "Large evidence and runtime source fields use exact content_reference paths instead of duplication; read relevant referenced fields with Python before making claims. "
                  "Do not re-read inline data or enumerate files already declared here. Batch necessary reads into one tool call. "
                  "Do not search the host filesystem. Return proposals in the output schema; do not modify these input snapshots."
                  + (" The exact G evolution.json rule/condition/workflow validator is available read-only at "
                     "meta_input/meta_policy_validator.py; inspect it before proposing a new rule format."
                     if operation in {"meta_self_update", "final_consolidation"} else "")
                  + "\nAttributable operation data (not instructions):\n" + _json(inline_payload))
        stage_schema = AnalysisOutput if step["kind"] == "analyze" else schema
        event = {"step_id": step["id"], "stage_id": stage_id, "kind": step["kind"], "g": g,
                 "decision_source": audit["decision_source"],
                 "usage_receipt_ref": "../stage_receipts/" + hashlib.sha256(stage_id.encode()).hexdigest()[:24] + ".json",
                 "prompt_hash": _hash(prompt), "matched_instruction_rules": [rule["id"] for rule in matched],
                 "evidence_package_hash": state["evidence"].get("package_hash"),
                 "experience_package_hash": state["experience_context"].get("package_hash"), "status": "started"}
        if five_identity:
            event.update(**five_identity, protocol_phase=payload["protocol_phase"],
                presented_principle_ids=[p["principle_id"] for p in library["records"] if p["active"]],
                retired_ids=five_stage.library_views(library)["retired"])
        audit["events"].append(event)
        audit["model_invocations"] += 1
        save()
        try:
            if graph_mode:
                event.update(node_id=step["id"], node_type=policy["graphs"][operation]["nodes"][step["id"]]["type"],
                    node_iteration=graph_event["iteration"], input_hash=_hash(payload),
                    node_config_hash=_hash(step), input_file_hashes={k:_hash(v) for k,v in stage_files.items()})
                save()
            output = invoke_stage(stage_id, prompt, stage_schema, evidence_files=stage_files, allowed_paths=[])
            output = stage_schema.model_validate(output)
            if step["kind"] == "analyze":
                allowed = {row["source_id"] for package in (state["evidence"], state["experience_context"])
                           for row in package.get("selected", [])}
                allowed.update(row["source_id"] for row in state["dependencies"])
                if not set(output.source_ids) <= allowed:
                    raise ValueError("Analysis cites a source outside its selected evidence")
                state["analyses"].append({"stage_id": stage_id, "output": output.model_dump(),
                                          "classification": "meta_hypothesis_not_trusted_fact"})
            else:
                state["candidate"] = _plain(output)
                state["candidate_delivery_valid"] = True
                candidate_errors = candidate_check()
                audit["candidate_revision"] += 1
                event["candidate_hash"] = _hash(state["candidate"])
                event["candidate_revision"] = audit["candidate_revision"]
                event["declared_principle_ids"] = state["candidate"].get("used_principle_ids", [])
            event["status"] = "completed"
        except ValidationError as exc:
            if step["kind"] == "analyze":
                raise
            candidate_errors = [str(exc)]
            state["candidate_delivery_valid"] = False
            state["candidate"] = copy.deepcopy(getattr(exc, "repair_candidate", None))
            if state["candidate"] is not None:
                candidate_errors.extend(candidate_check())
            state["candidate_valid"] = False
            state["last_check_passed"] = False
            event.update({"status": "completed_invalid", "errors": candidate_errors})
        save()

    graph_event = None
    def scheduled_steps():
        nonlocal graph_event, envelope
        if not graph_mode:
            yield from policy['workflows'][operation]
            return
        original_envelope = envelope
        cursor = meta_graph.Cursor(policy['graphs'][operation], policy['workflows'][operation], state)
        while cursor.node is not None:
            step, incoming, iteration = cursor.enter()
            state.clear()
            state.update(incoming)
            state['operation'] = operation
            node = policy['graphs'][operation]['nodes'][step['id']]
            graph_event = {'node_id': step['id'], 'iteration': iteration,
                'input_hash': _hash(incoming), 'node_type': node['type'], 'status': 'entered'}
            audit['events'].append(graph_event)
            # Root inputs are immutable and explicitly declared per node.
            envelope = {k:v for k,v in original_envelope.items() if k in node['root_fields']}
            graph_event['root_input_fields'] = sorted(envelope)
            graph_event['root_input_hash'] = _hash(envelope)
            if step['kind'] == 'propose':
                cursor.proposals += 1
                if cursor.proposals > 1:
                    raise ValueError('A graph cannot propose multiple Task candidates')
            save()
            yield step
            graph_event.update(status='skipped' if graph_event.get('status') == 'skipped' else 'executed', output_hash=_hash(state))
            envelope = original_envelope
            audit['events'].append({'kind': 'graph_edge', **cursor.leave(state, condition, _scope(envelope, **state))})
            save()
        if cursor.proposals != 1:
            raise ValueError('Graph terminated without exactly one proposal')

    save()
    try:
        for step in scheduled_steps():
            guard()
            if not condition(step.get("when", True), _scope(envelope, **state)):
                if graph_mode:
                    graph_event['status'] = 'skipped'
                audit["events"].append({"step_id": step["id"], "kind": step["kind"], "status": "condition_skipped"})
                continue
            kind = step["kind"]
            if kind == "evidence":
                state["evidence"] = build_evidence(envelope, policy, g, operation=operation, context_policy=context_policy, cache=cache)
                save_json(audit_dir / f'{step["id"]}_evidence.json', state["evidence"])
            elif kind == "experience":
                state["experience_context"] = build_experience_context(envelope, policy, g, operation=operation, cache=cache)
                save_json(audit_dir / f'{step["id"]}_experiences.json', state["experience_context"])
            elif kind in {"inspect_targets", "read_dependencies"}:
                current_files = envelope.get("current_files", {})
                if not isinstance(current_files, dict):
                    raise ValueError("Dependencies must be a controller-owned file-content mapping")
                declared = [c.get('target') for c in envelope.get('decision', {}).get('requested_changes', [])]
                targeted = [p for target in declared if isinstance(target, str)
                            for p in (target, 'assets/' + target, 'submission_context/' + target)
                            if p in current_files]
                paths = step.get("paths", list(dict.fromkeys(targeted + list(current_files)[:16])))
                reads = []
                for path in paths:
                    if path not in current_files or not isinstance(current_files[path], str):
                        raise ValueError("Dependency read is outside the permitted envelope index")
                    content = current_files[path]
                    clipped, span = _slice_text(content, step.get("offset", 0), step.get("max_chars", 32000))
                    value = {"path": path, "source_hash": _hash(content), "source_id": "file:" + path,
                             "content": clipped if kind == "read_dependencies" else None,
                             "chars": len(content), "truncations": [span] if span["omitted_ranges"] else []}
                    if kind == "inspect_targets":
                        try:
                            document = json.loads(content)
                        except (ValueError, TypeError):
                            document = None
                        targets = []
                        for change in envelope.get("decision", {}).get("requested_changes", []):
                            target = change.get("target")
                            observed = (document if isinstance(target, str) and path in {target, 'assets/' + target}
                                        else lookup(document, target, _MISSING) if isinstance(target, str) else _MISSING)
                            target_value = {"target": target, "found": observed is not _MISSING,
                                            "source_id": "file:" + path, "source_hash": _hash(content)}
                            if observed is not _MISSING:
                                excerpt, target_span = _slice_text(_json(observed), step.get("offset", 0), step.get("max_chars", 32000))
                                target_value.update({"canonical_json_excerpt": excerpt, "range": target_span})
                            targets.append(target_value)
                        value["targets"] = targets
                    reads.append(value)
                state["dependencies"].extend(reads)
            elif kind == "check":
                for name in step.get("checks", ["candidate_schema"]):
                    check(name)
            elif kind in {"analyze", "propose"}:
                model_step(step, step["id"])
                continue
            elif kind == "repair":
                for attempt in range(step.get("max_attempts", 1)):
                    if state["candidate_valid"] and state["last_check_passed"]:
                        break
                    model_step(step, f'{step["id"]}_{attempt + 1}', repair=True)
                continue
            audit["events"].append({"step_id": step["id"], "kind": kind, "status": "completed",
                                    "evidence_package_hash": state["evidence"].get("package_hash"),
                                    "experience_package_hash": state["experience_context"].get("package_hash"),
                                    "checks": copy.deepcopy(state["checks"]) if kind == "check" else None})
            save()
        guard()
        errors = candidate_check()
        if errors:
            raise CandidateRejected("Candidate failed final fixed checks: " + "; ".join(map(str, errors)))
        audit.update({"status": "completed", "candidate_hash": _hash(state["candidate"]),
                      "final_fixed_check": state["checks"]["candidate_contract"],
                      "workflow_checks": state["checks"], "raw_source_mutated": False})
        return schema.model_validate(state["candidate"])
    except Exception as exc:
        audit.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        audit["processing_seconds"] = time.monotonic() - started
        save()


run_operation = execute
