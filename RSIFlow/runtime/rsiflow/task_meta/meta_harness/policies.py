"""Bounded, composable JSON policies for the project-owned Meta Harness.

Predicates only traverse the controller's JSON envelope. No expressions, Python,
regular expressions, imports, filesystem lookups or shell execution are allowed.
"""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

OPERATIONS = {"routing", "harness_patch", "artifact_patch", "model_request", "meta_self_update",
              "final_consolidation", "compatibility_smoke"}
ALIASES = {"route": "routing", "learn": "meta_self_update", "model_prepare": "model_request"}
STEP_KINDS = {"evidence", "experience", "inspect_targets", "read_dependencies", "check", "analyze", "propose", "repair"}
CHECKS = {"source_integrity", "requested_targets", "available_action", "candidate_schema", "candidate_interface", "experience_outcome"}
COMPARATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "in", "contains", "exists"}
PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+){0,11}$")


def _object(value, keys, name, required=None):
    if not isinstance(value, dict) or set(value) - set(keys) or (set(required or keys) - set(value)):
        raise ValueError(f"Invalid {name} fields")


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _text(value, limit, name, *, blank=False):
    if not isinstance(value, str) or len(value) > limit or (not blank and not value.strip()):
        raise ValueError(f"Invalid bounded {name}")


def _path(value):
    if not isinstance(value, str) or not PATH.fullmatch(value) or any(part.startswith("__") for part in value.split(".")):
        raise ValueError("Only bounded JSON field paths are supported")


def validate_condition(value, *, depth=0, counter=None):
    counter = [0] if counter is None else counter
    counter[0] += 1
    if depth > 6 or counter[0] > 64:
        raise ValueError("Condition depth/node budget exceeded")
    if type(value) is bool:
        return
    if not isinstance(value, dict):
        raise ValueError("Condition must be a boolean or predicate object")
    logical = set(value) & {"all", "any", "not"}
    if logical:
        if len(value) != 1:
            raise ValueError("Logical condition must have exactly one operator")
        key = next(iter(logical))
        children = [value[key]] if key == "not" else value[key]
        if not isinstance(children, list) or not 1 <= len(children) <= 16:
            raise ValueError("Logical condition requires 1..16 children")
        for child in children:
            validate_condition(child, depth=depth + 1, counter=counter)
        return
    _object(value, {"path", "op", "value", "value_path"}, "predicate", {"path", "op"})
    _path(value["path"])
    if value["op"] not in COMPARATORS:
        raise ValueError("Unknown predicate operator")
    if value["op"] == "exists":
        if set(value) != {"path", "op"}:
            raise ValueError("exists does not take a value")
    elif ("value" in value) == ("value_path" in value):
        raise ValueError("Predicate needs exactly one literal value or JSON value_path")
    elif "value_path" in value:
        _path(value["value_path"])
    elif len(json.dumps(value["value"], allow_nan=False)) > 4096:
        raise ValueError("Predicate literal exceeds budget")


def _rules(value, *, score=False):
    if not isinstance(value, list) or len(value) > 16:
        raise ValueError("At most 16 conditional rules are allowed")
    ids = set()
    for rule in value:
        key = "weight" if score else "instruction"
        _object(rule, {"id", "when", key}, "rule")
        _text(rule["id"], 64, "rule id")
        if rule["id"] in ids:
            raise ValueError("Rule IDs must be unique")
        ids.add(rule["id"])
        validate_condition(rule["when"])
        if score:
            number = rule[key]
            if type(number) not in (int, float) or not math.isfinite(number) or not -100 <= number <= 100:
                raise ValueError("Relevance weights must be finite in [-100,100]")
        else:
            _text(rule[key], 4000, "rule instruction")


def _selection(value, *, experience=False):
    keys = {"filter", "score_rules", "sort", "group_by", "max_items", "fields", "field_chars", "max_chars"}
    keys |= {"applicability", "failure", "require_applicable", "exclude_failed"} if experience else {"fragments"}
    _object(value, keys, "experience selection" if experience else "evidence selection")
    validate_condition(value["filter"])
    _rules(value["score_rules"], score=True)
    if not isinstance(value["sort"], list) or len(value["sort"]) > 4:
        raise ValueError("At most 4 sort fields are allowed")
    for entry in value["sort"]:
        _object(entry, {"field", "direction"}, "sort")
        _path(entry["field"])
        if entry["direction"] not in {"asc", "desc"}:
            raise ValueError("Invalid sort direction")
    for name, limit in (("group_by", 3), ("fields", 24)):
        if not isinstance(value[name], list) or len(value[name]) > limit:
            raise ValueError(f"Invalid {name} list")
        for path in value[name]:
            _path(path)
    _integer(value["max_items"], 0, 128, "max_items")
    _integer(value["field_chars"], 32, 12000, "field_chars")
    _integer(value["max_chars"], 256, 120000, "max_chars")
    if experience:
        for name in ("require_applicable", "exclude_failed"):
            if type(value[name]) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("applicability", "failure"):
            _rules(value[name])
    else:
        fragment = value["fragments"]
        _object(fragment, {"anchor", "before", "after", "fallback", "max_messages", "message_chars", "retain_tools"}, "fragments")
        validate_condition(fragment["anchor"])
        for key in ("before", "after"):
            _integer(fragment[key], 0, 32, key)
        _integer(fragment["max_messages"], 1, 64, "max_messages")
        _integer(fragment["message_chars"], 32, 24000, "message_chars")
        if fragment["fallback"] not in {"head", "tail", "none"}:
            raise ValueError("Invalid fallback slice")
        if not isinstance(fragment["retain_tools"], list) or len(fragment["retain_tools"]) > 32:
            raise ValueError("Invalid retained tool list")
        for name in fragment["retain_tools"]:
            _text(name, 128, "tool name")


def validate_policy(value):
    """Return a detached policy; strict schema never executes input text."""
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False)) > 128000:
        raise ValueError("Evolution policy exceeds the immutable file budget")
    graph_mode = value.get("schema_version") == 2
    keys = {"schema_version", "evidence", "experience", "diagnosis", "self_update", "workflows"}
    _object(value, keys | ({"graphs", "method2"} if graph_mode else set()), "evolution policy")
    if type(value["schema_version"]) is not int or value["schema_version"] not in {1, 2}:
        raise ValueError("Unsupported evolution policy version")
    _selection(value["evidence"])
    _selection(value["experience"], experience=True)
    for key in ("diagnosis", "self_update"):
        _object(value[key], {"instruction", "rules"}, key)
        _text(value[key]["instruction"], 12000, key + " instruction")
        _rules(value[key]["rules"])
    if not isinstance(value["workflows"], dict) or set(value["workflows"]) != OPERATIONS:
        raise ValueError("Every declared operation requires an explicit workflow")
    for operation, steps in value["workflows"].items():
        if not isinstance(steps, list) or not 1 <= len(steps) <= 16:
            raise ValueError("Workflow requires 1..16 bounded steps")
        ids, proposed, repair_count = set(), False, 0
        for step in steps:
            _object(step, {"id", "kind", "when", "instruction", "checks", "paths", "max_attempts", "max_chars", "offset"}, "step", {"id", "kind"})
            _text(step["id"], 64, "step id")
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", step["id"]) or step["id"] in ids:
                raise ValueError("Step IDs must be unique bounded identifiers")
            ids.add(step["id"])
            kind = step["kind"]
            if kind not in STEP_KINDS:
                raise ValueError("Unknown workflow primitive; jumps/loops/subworkflows are unsupported")
            validate_condition(step.get("when", True))
            if "instruction" in step:
                if kind not in {"analyze", "propose", "repair"}:
                    raise ValueError("Only model stages accept a stage instruction")
                _text(step["instruction"], 12000, "step instruction")
            for key, low, high in (("max_chars", 1, 120000), ("offset", 0, 10000000)):
                if key in step:
                    if kind not in {"inspect_targets", "read_dependencies"}:
                        raise ValueError("Only dependency inspections accept character ranges")
                    _integer(step[key], low, high, key)
            if kind == "propose":
                if proposed or step.get("when", True) is not True:
                    raise ValueError("Workflow requires one unconditional candidate proposal")
                proposed = True
            if kind == "repair":
                if not proposed:
                    raise ValueError("Repair must follow the same candidate proposal")
                _integer(step.get("max_attempts", 1), 1, 2, "repair max_attempts")
                repair_count += step.get("max_attempts", 1)
                if repair_count > 2:
                    raise ValueError("At most two same-candidate repair invocations are allowed")
            elif "max_attempts" in step:
                raise ValueError("Only same-candidate repair has a retry count")
            if "checks" in step and (kind != "check" or not isinstance(step["checks"], list) or not step["checks"] or not set(step["checks"]) <= CHECKS):
                raise ValueError("Only declared engineering checks are allowed")
            if "paths" in step:
                if kind not in {"inspect_targets", "read_dependencies"} or not isinstance(step["paths"], list) or len(step["paths"]) > 16:
                    raise ValueError("Only bounded dependency reads accept paths")
                for path in step["paths"]:
                    if (not isinstance(path, str) or not path or len(path) > 256 or path.startswith("/")
                            or "\\" in path or ":" in path or any(p in {"", ".", ".."} for p in path.split("/"))):
                        raise ValueError("Dependency paths must be relative envelope keys")
        if not proposed:
            raise ValueError(f"{operation} has no candidate proposal")
    if graph_mode:
        from .graph import validate
        validate(value)
    return copy.deepcopy(value)


def default_policy():
    """Load the project's normalized Meta policy; there is no legacy synthesized fallback."""
    seed = Path(__file__).resolve().parents[3] / "meta_harness" / "seed" / "evolution.json"
    if not seed.is_file():
        raise FileNotFoundError("Normalized Meta seed policy is required: " + str(seed))
    return validate_policy(json.loads(seed.read_text(encoding="utf-8")))


validate_evolution = validate_policy
default_evolution = default_policy
