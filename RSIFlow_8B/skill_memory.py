"""Append-only Meta cases, conditional rules and incidents with bounded retrieval.

The ledger is authoritative. The index is a rebuildable view, including appended
rule revisions. Retrieval ranks evidence, never chooses a Task component.
"""
from __future__ import annotations

import copy
import fcntl
import json
import os
import re
import uuid
from collections import Counter
from pathlib import Path


COMPONENTS = ("HARNESS", "MODEL", "ARTIFACTS")
DEFAULT_CONTEXT_CHARS = 24000
DEFAULT_PER_CATEGORY = 1
DOMAIN_ALIASES = {"envscaler": "tool_use", "deepcoder_taco": "code",
                  "nq_open": "searchqa", "hotpotqa": "searchqa", "2wiki": "searchqa"}

# This is shared by initial/resumed prompts; tools remain available on demand.
SKILL_GUIDANCE = """Meta skill use:
Before routing, read the current failure_fingerprint_path and context_path. Call
retrieve_skills(path=skills_path, fingerprint_path=..., per_category=meta_skill_per_category,
max_chars=meta_skill_context_chars). It returns HARNESS/MODEL/ARTIFACTS support,
failure/counterexample and conditional-rule cards plus external incidents. These
are advisory matches, not component rankings. Inspect avoid_when, confidence,
counterevidence and unknowns; use read_skill for full records and original rollout
references when needed. Empty retrieval is valid. Preserve the 48-excerpt routing
policy. model_call_count=0 is an observed symptom, not by itself proof of an
infrastructure defect; keep those task failures in the full paired denominator.
Before building a candidate, save decision.json with relevant_skill_ids, actual
evidence refs, rationale and prospective prediction: expected error changes,
target domains and existing successful behaviours at risk. Use record_skill_use
(path=attempt_dir/skill_use.json, skills_path, decision_path, retrieval_path) to
freeze those claims. It records cited IDs, not proof that the model understood them.
After paired scores, call compare_task_differences(before_trajectories,
after_trajectories, output_dir=attempt_dir/task_differences, skill_use_path=...).
Read new_success/new_regression counts and task references, then compare actual
changes with the frozen prediction. Net domain counts alone cannot prove no
regression. Keep costs, implementation failure and outcome failure distinct.
Append cases (kind=case, id=skill.<COMPONENT>.<id>) with context, diagnosis,
mechanism, prediction, measured_outcome, used_skill_ids, skill_use_path,
attempt number, applicability/unknowns and evidence refs. Record new_success and
new_regression counts and an explicit prediction_assessment.met only when supported.
Append conditional rules (kind=rule, id=principle.<id>) only when supported; give
when, recommend, avoid_when, expected_signal, support and counterevidence.
Use incident for a demonstrated runtime/scorer fault, rather than a Task repair
rule. For reusable concrete methods call maintain_skills(path=skills_path,
operations=[...]): add a procedure only for an uncovered method (record with
component, when, steps, preserve and evidence_refs); supplement an existing target
with support/counterevidence for the SAME conditions and method; revise its
conditions/steps/preserve with a linked new record; merge duplicates into an
explicit canonical record; retire an obsolete target, optionally replaced_by.
Each operation has action, reason and evidence_refs; add/revise use record,
supplement uses target plus support/counterevidence, merge uses targets plus record
or canonical_id, retire uses target. Meta decides semantic coverage and maintenance,
not a similarity threshold. Inspect retrieved versions before creating near-duplicates.
Methods use kind=procedure; cases remain immutable observations. Supplementation
does not automatically strengthen confidence. Revision/merge preserve evidence
and history, but do not claim old responses tested the new method. Retired/merged/
superseded methods are omitted from normal retrieval and remain readable by ID or
include_inactive=true. Existing append_skills/rule_update remain compatible.
Preserve old ledger lines. One intervention is one support event; summaries or
later uses of its ID are not independent replications. Bundled edits do not
identify each edit's contribution, and different task batches do not establish
a paired improvement. Absence of domain-specific SFT examples limits direct
supervision; it does not prove cross-domain transfer impossible. Report-only
independent validation never enters this skill learning loop. Each attempt can
add several cases/rules/revisions as warranted, with no total-record quota.
skill_usage_report(path=skills_path) reports documented attempt outcomes and
declared use; missing predictions/measurements remain unknown.
"""


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _unique(values):
    seen, result = set(), []
    for value in values:
        key = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _tokens(value):
    text = json.dumps(value, ensure_ascii=False).lower()
    tokens = set(re.findall(r"[a-z0-9_]+", text))
    for word in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.update(word[i:i+2] for i in range(max(1, len(word)-1)))
    tokens.update(DOMAIN_ALIASES[t] for t in tuple(tokens) if t in DOMAIN_ALIASES)
    return tokens - {"null", "true", "false", "when", "recommend", "mechanism", "context", "title",
                     "statement", "diagnosis", "signature", "boundary", "expected_signal",
                     "domains", "domain", "error_types", "dominant_errors", "incident_signatures"}


def _kind(record):
    kind = record.get("kind")
    if kind in {"rule", "principle"}:
        return "rule"
    if kind == "incident":
        return "incident"
    if kind == "procedure":
        return "procedure"
    return "case"


def _outcome(record):
    value = record.get("measured_outcome", record.get("outcome", {}))
    if isinstance(value, dict):
        delta = value.get("delta")
        if isinstance(delta, (float, int)):
            return "support" if delta > 0 else "failure"
        value = value.get("verdict", value.get("status", "unknown"))
    if str(value).lower() in {"failure", "rejected", "negative", "zero_gain", "unexecutable"}:
        return "failure"
    if str(value).lower() in {"success", "accepted", "positive"}:
        return "support"
    return "unresolved"


class SkillMemory:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.index_path = self.path.with_name("skills_index.json")

    def _read(self):
        events, warnings = [], []
        if not self.path.exists():
            return events, warnings
        with self.path.open("rb") as stream:
            offset = 0
            for line, raw in enumerate(stream, 1):
                start, offset = offset, offset + len(raw)
                if not raw.strip():
                    continue
                try:
                    event = json.loads(raw)
                    if not isinstance(event, dict):
                        raise ValueError("record must be a JSON object")
                except (ValueError, UnicodeError) as exc:
                    warnings.append({"line": line, "error": str(exc)})
                    continue
                event = copy.deepcopy(event)
                event.setdefault("id", event.get("principle_id") or f"legacy.line_{line}")
                events.append((event, {"path": str(self.path), "line": line,
                                       "offset_bytes": start, "length_bytes": len(raw)}))
        return events, warnings

    def view(self):
        events, warnings = self._read()
        records, revisions = {}, []
        for event, source in events:
            if event.get("kind") in {"rule_update", "skill_maintenance"}:
                revisions.append((event, source))
                continue
            identifier = str(event["id"])
            if identifier in records:
                warnings.append({"id": identifier, "reason": "duplicate ID; first record retained"})
                continue
            records[identifier] = {**event, "kind": _kind(event), "source": source,
                                   "confidence": event.get("confidence", "legacy_observation"),
                                   "active": event.get("active", event.get("status") not in
                                                       {"superseded", "merged", "retired"}),
                                   "status": event.get("status", "hypothesis" if _kind(event) == "rule"
                                                        else "observed")}
            if "support" not in records[identifier] and "supports" in event:
                records[identifier]["support"] = _list(event["supports"])
        for event, source in revisions:
            if event.get("kind") == "skill_maintenance":
                operation = event.get("operation")
                targets = _list(event.get("targets", event.get("target")))
                for identifier in targets:
                    target = records.get(str(identifier))
                    if target is None:
                        warnings.append({"id": event["id"], "reason": "maintenance target absent"})
                        continue
                    if operation == "supplement":
                        for field in ("support", "counterevidence", "evidence_refs"):
                            target[field] = _unique(_list(target.get(field)) + _list(event.get(field)))
                    elif operation in {"revise", "merge", "retire"}:
                        replacement = event.get("replacement_id")
                        if operation in {"revise", "merge"} and replacement not in records:
                            warnings.append({"id": event["id"], "reason": "replacement absent; target kept active"})
                            continue
                        if identifier == replacement:
                            continue
                        target["active"] = False
                        target["lifecycle_status"] = {"revise": "superseded", "merge": "merged",
                                                      "retire": "retired"}[operation]
                        if replacement:
                            target["replaced_by"] = replacement
                        if operation == "merge":
                            canonical = records[replacement]
                            for field in ("support", "counterevidence", "evidence_refs"):
                                canonical[field] = _unique(_list(canonical.get(field)) + _list(target.get(field)))
                            canonical["merged_from"] = _unique(_list(canonical.get("merged_from")) + [identifier])
                    else:
                        warnings.append({"id": event["id"], "reason": "unknown maintenance operation"})
                        continue
                    target.setdefault("maintenance", []).append({"id": event["id"], "operation": operation,
                                                                  "reason": event.get("reason"), "source": source,
                                                                  "evidence_refs": event.get("evidence_refs", [])})
                if operation in {"revise", "merge"} and event.get("replacement_id") in records:
                    replacement = records[event["replacement_id"]]
                    replacement["evidence_refs"] = _unique(_list(replacement.get("evidence_refs"))
                                                            + _list(event.get("evidence_refs")))
                    replacement.setdefault("maintenance", []).append(
                        {"id": event["id"], "operation": operation, "reason": event.get("reason"), "source": source})
                continue
            target = records.get(str(event.get("target")))
            if target is None:
                warnings.append({"id": event["id"], "reason": "revision target absent"})
                continue
            if target["kind"] != "rule":
                warnings.append({"id": event["id"], "reason": "only rules can be revised; case facts retained"})
                continue
            operation = event.get("operation")
            status = event.get("status") or {"weaken": "hypothesis", "strengthen": "single_paired_support",
                                             "contradict": "contradicted", "supersede": "superseded"}.get(operation)
            if status:
                target["status"] = status
                if status == "superseded":
                    target["active"] = False
            for field in ("confidence", "superseded_by", "statement", "recommend", "when",
                          "avoid_when", "expected_signal", "unknowns"):
                if field in event:
                    target[field] = copy.deepcopy(event[field])
            for field in ("support", "counterevidence", "evidence_refs"):
                target[field] = _unique(_list(target.get(field)) + _list(event.get(field)))
            target.setdefault("revisions", []).append({"id": event["id"], "operation": operation,
                                                       "reason": event.get("reason"), "source": source})
        return records, events, warnings

    def rebuild_index(self):
        records, events, warnings = self.view()
        entries = []
        for record in records.values():
            entries.append({key: record.get(key) for key in
                            ("id", "kind", "component", "status", "confidence", "when", "avoid_when",
                             "support", "counterevidence", "source", "revisions", "active", "version",
                             "previous_version", "merged_from", "replaced_by", "lifecycle_status", "maintenance")} | {
                                "title": self._title(record), "outcome": _outcome(record)})
        value = {"ledger_path": str(self.path), "event_count": len(events),
                 "record_count": len(records), "counts": dict(Counter(r["kind"] for r in records.values())),
                 "active_record_count": sum(r.get("active", True) for r in records.values()),
                 "entries": entries, "warnings": warnings}
        write_json(self.index_path, value)
        return value

    def append(self, entries):
        if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
            raise ValueError("append_skills requires a list of JSON objects")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        appended, reused, warnings = [], [], []
        with self.path.open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            _, events, existing_warnings = self.view()
            known = {str(e["id"]): e for e, _ in events}
            warnings.extend(existing_warnings)
            for incoming in entries:
                event = copy.deepcopy(incoming)
                prefix = ("principle" if event.get("kind") in {"rule", "principle"} else
                          "incident" if event.get("kind") == "incident" else
                          "rule_update" if event.get("kind") == "rule_update" else
                          "maintenance" if event.get("kind") == "skill_maintenance" else
                          "skill." + str(event.get("component", "GENERAL")))
                event.setdefault("id", event.get("principle_id") or prefix + "." + uuid.uuid4().hex)
                identifier = str(event["id"])
                if identifier in known:
                    if event == known[identifier]:
                        reused.append(identifier)
                    else:
                        warnings.append({"id": identifier, "reason": "ID already exists; append a revision or use a fresh ID"})
                    continue
                if event.get("kind") == "rule_update" and not event.get("target"):
                    warnings.append({"id": identifier, "reason": "revision has no target; retained as unresolved event"})
                # Keep a pre-existing unfinished final line separate from the new event.
                stream.seek(0, os.SEEK_END)
                if stream.tell():
                    stream.seek(-1, os.SEEK_END)
                    if stream.read(1) != b"\n":
                        stream.write(b"\n")
                stream.write((json.dumps(event, ensure_ascii=False, default=str) + "\n").encode())
                stream.flush()
                known[identifier] = event
                appended.append(identifier)
            # Serialize the derived index under the same ledger lock.
            index = self.rebuild_index()
        return {"status": "appended", "path": str(self.path), "appended_count": len(appended),
                "appended_ids": appended, "reused_ids": reused, "index_path": str(self.index_path),
                "record_count": index["record_count"], "warnings": warnings,
                "size_bytes": self.path.stat().st_size}

    def maintain(self, operations):
        """Execute Meta-selected maintenance as append-only records and events."""
        operations = _list(operations)
        results, appended, reused, warnings = [], [], [], []
        for operation in operations:
            if not isinstance(operation, dict):
                warnings.append({"reason": "maintenance operation must be an object"})
                continue
            action = operation.get("action")
            if action not in {"add", "supplement", "revise", "merge", "retire"}:
                warnings.append({"action": action, "reason": "unknown maintenance action"})
                continue
            # Stable request identity makes a receipt-lost retry idempotent.
            token = uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(operation, sort_keys=True, ensure_ascii=False)).hex
            event_id = operation.get("event_id") or "maintenance." + token
            records, events, _ = self.view()
            if any(event["id"] == event_id for event, _ in events):
                reused.append(event_id)
                results.append({"action": action, "status": "reused", "event_id": event_id})
                continue
            targets = _unique(_list(operation.get("targets", operation.get("target"))))
            if action != "add" and (not targets or any(str(t) not in records for t in targets)):
                warnings.append({"action": action, "targets": targets, "reason": "target absent; no records removed"})
                continue
            entries = []
            event = {"id": event_id, "kind": "skill_maintenance", "operation": action,
                     "targets": targets, "reason": operation.get("reason"),
                     "evidence_refs": _list(operation.get("evidence_refs"))}
            if action in {"add", "revise", "merge"}:
                if action == "merge" and operation.get("canonical_id"):
                    record = records.get(str(operation["canonical_id"]))
                    if record is None or not record.get("active", True):
                        warnings.append({"action": action, "reason": "canonical skill absent or inactive"})
                        continue
                else:
                    fields = operation.get("record")
                    if not isinstance(fields, dict):
                        warnings.append({"action": action, "reason": "provide the new or revised skill record"})
                        continue
                    record = {}
                    if action == "revise":
                        previous = records[str(targets[0])]
                        # Never reinterpret the original measured response as a new version's result.
                        omitted = {"id", "source", "maintenance", "revisions", "active", "lifecycle_status",
                                   "replaced_by", "measured_outcome", "outcome", "prediction", "prediction_assessment",
                                   "used_skill_ids", "attempt", "round", "kind", "version", "support", "supports", "counterevidence"}
                        record = {key: copy.deepcopy(value) for key, value in previous.items() if key not in omitted}
                        record["prior_version_support"] = _list(previous.get("support"))
                        record["prior_version_counterevidence"] = _list(previous.get("counterevidence"))
                        record["previous_version"] = previous["id"]
                        record["version"] = int(previous.get("version", 1)) + 1
                        record["confidence"] = "unvalidated_revision"
                        record["status"] = "hypothesis"
                    record.update(copy.deepcopy(fields))
                    record.setdefault("kind", "procedure")
                    record.setdefault("version", 1)
                    record.setdefault("status", "hypothesis")
                    record.setdefault("confidence", "unvalidated_method")
                    record.setdefault("id", "skill." + str(record.get("component", "GENERAL")) + "." + token)
                    if action != "add" and str(record["id"]) in records:
                        warnings.append({"action": action, "reason": "new version ID already exists; use canonical_id for merge"})
                        continue
                    record["evidence_refs"] = _unique(_list(record.get("evidence_refs")) + event["evidence_refs"])
                    if action == "merge":
                        record["merged_from"] = targets
                    entries.append(record)
                if action == "add":
                    receipt = self.append(entries)
                    appended.extend(receipt["appended_ids"])
                    reused.extend(receipt["reused_ids"])
                    warnings.extend(receipt["warnings"])
                    results.append({"action": action, "record_id": record["id"], "status": receipt["status"]})
                    continue
                event["replacement_id"] = record["id"]
            elif action == "supplement":
                event["support"] = _list(operation.get("support"))
                event["counterevidence"] = _list(operation.get("counterevidence"))
            elif action == "retire" and operation.get("replaced_by"):
                event["replacement_id"] = operation["replaced_by"]
            entries.append(event)
            receipt = self.append(entries)
            appended.extend(receipt["appended_ids"])
            reused.extend(receipt["reused_ids"])
            warnings.extend(receipt["warnings"])
            results.append({"action": action, "event_id": event_id,
                            "replacement_id": event.get("replacement_id"), "status": receipt["status"]})
        index = self.rebuild_index()
        return {"status": "maintained", "path": str(self.path), "index_path": str(self.index_path),
                "appended_count": len(appended), "appended_ids": appended, "reused_ids": _unique(reused),
                "results": results, "warnings": warnings, "record_count": index["record_count"],
                "active_record_count": index["active_record_count"]}

    @staticmethod
    def _title(record):
        return str(record.get("title") or record.get("statement") or record.get("diagnosis")
                   or record.get("mechanism") or record["id"])[:320]

    @staticmethod
    def _match(record, fingerprint):
        context = record.get("context", {})
        query = _tokens(fingerprint)
        body = {k: record.get(k) for k in ("title", "statement", "mechanism", "diagnosis", "when",
                                         "recommend", "signature", "boundary", "expected_signal", "steps", "preserve")}
        body["context"] = context
        # Legacy cases do not have structured triggers; outcome-domain labels help
        # navigation but do not imply the case improved every mentioned domain.
        outcome = record.get("measured_outcome", {})
        body["outcome_domains"] = list(outcome.keys()) if isinstance(outcome, dict) else []
        overlap = query & _tokens(body)
        # Match explicit failure signatures more strongly than generic domain words.
        errors = _tokens({k: fingerprint.get(k) for k in ("dominant_errors", "error_types", "incident_signatures")})
        score = len(overlap) + 3 * len(overlap & errors)
        return score, sorted(overlap)

    def retrieve(self, fingerprint=None, *, max_chars=DEFAULT_CONTEXT_CHARS,
                 per_category=DEFAULT_PER_CATEGORY, components=None, include_inactive=False):
        fingerprint = fingerprint or {}
        records, _, warnings = self.view()
        self.rebuild_index()
        groups = {component: {key: [] for key in ("support", "failure", "rules", "skills", "unresolved")}
                  for component in (components or COMPONENTS)}
        candidates = []
        for record in records.values():
            if not include_inactive and not record.get("active", True):
                continue
            searchable = record
            if record["kind"] in {"rule", "procedure"} and not record.get("context"):
                support = _list(record.get("support", record.get("supports")))
                searchable = {**record, "context": [records[str(s)].get("context", {})
                                                    for s in support if str(s) in records]}
            score, matched = self._match(searchable, fingerprint)
            if fingerprint and not score:
                continue
            candidates.append((score, str(record["id"]), record, matched))
        candidates.sort(key=lambda row: (-row[0], row[1]))
        incidents, cards, omitted = [], {}, []
        for score, identifier, record, matched in candidates:
            kind = record["kind"]
            category = ("failure" if record.get("status") in {"contradicted", "superseded"} else
                        "skills" if kind == "procedure" else
                        "rules" if kind == "rule" else _outcome(record))
            components_for_record = _list(record.get("component"))
            if not components_for_record:
                supports = _list(record.get("support", record.get("supports")))
                components_for_record = [records[str(s)].get("component") for s in supports if str(s) in records]
            targets = [c for c in components_for_record if c in groups] or list(groups)
            buckets = [incidents] if kind == "incident" else [groups[c][category] for c in targets]
            buckets = [bucket for bucket in buckets if len(bucket) < max(1, int(per_category))]
            if not buckets:
                continue
            card = {"id": identifier, "kind": kind, "component": record.get("component"),
                    "title": self._title(record), "status": record.get("status"),
                    "confidence": record.get("confidence"), "relevance": score, "matched_terms": matched,
                    "source": record["source"], "detail": record}
            card.update({field: record.get(field) for field in
                         ("active", "version", "previous_version", "replaced_by", "lifecycle_status")})
            # Keep boundaries available even when the full detail does not fit.
            card.update({field: [str(v)[:180] for v in _list(record.get(field))[:4]]
                         for field in ("when", "avoid_when", "steps", "preserve", "support", "counterevidence",
                                       "prior_version_support", "prior_version_counterevidence", "unknowns")})
            card["retired"] = not record.get("active", True)
            cards[identifier] = card
            for bucket in buckets:
                bucket.append(identifier)
        result = {"status": "retrieved", "ledger_path": str(self.path), "index_path": str(self.index_path),
                  "record_count": len(records), "matches": groups, "incidents": incidents,
                  "records": cards, "omitted_detail_ids": omitted,
                  "notes": ["Relevance orders evidence; it is not expected gain or a component recommendation.",
                            "Historical summaries are observations, not independently replicated rules."],
                  "warnings": warnings, "max_chars": max(512, int(max_chars))}
        # Keep full details accessible through read_skill; never inline full trajectories.
        limit = result["max_chars"]
        size = lambda: len(json.dumps(result, ensure_ascii=False))
        for identifier in reversed(list(cards)):
            if size() <= limit:
                break
            cards[identifier].pop("detail", None)
            omitted.append(identifier)
        while cards and size() > limit:
            identifier = next(reversed(cards))
            cards.pop(identifier)
            if identifier not in omitted:
                omitted.append(identifier)
            for categories in groups.values():
                for bucket in categories.values():
                    if identifier in bucket:
                        bucket.remove(identifier)
            if identifier in incidents:
                incidents.remove(identifier)
        # Very small budgets may not even fit the three empty component buckets.
        if size() > limit:
            result = {"status": "budget_too_small", "ledger_path": str(self.path),
                      "index_path": str(self.index_path), "record_count": len(records),
                      "max_chars": limit, "records": {}, "notes": ["Use read_skill or a larger retrieval budget."]}
        return result

    def read_skill(self, identifiers, *, offset_chars=0, max_chars=65536):
        records, _, warnings = self.view()
        identifiers = _list(identifiers)
        selected = [records[str(i)] for i in identifiers if str(i) in records]
        text = json.dumps(selected, ensure_ascii=False, indent=2)
        offset = max(0, int(offset_chars))
        end = min(len(text), offset + max(1, int(max_chars)))
        return {"status": "read", "ids": identifiers, "content": text[offset:end],
                "missing_ids": [i for i in identifiers if str(i) not in records],
                "total_chars": len(text), "next_offset_chars": end if end < len(text) else None,
                "warnings": warnings}

    def usage_report(self):
        """Descriptive attempt/use statistics; never infer that retrieval caused gain."""
        records, _, warnings = self.view()
        seen, uses, first = set(), {}, Counter()
        attempts = []
        for record in records.values():
            if record["kind"] != "case":
                continue
            evidence = record.get("evidence") or {}
            pair = [evidence.get("parent_performance_path"), evidence.get("candidate_performance_path")]
            identity = record.get("intervention_id") or (
                json.dumps([record.get("component"), *pair]) if all(pair) else record["id"])
            if identity in seen:
                continue
            seen.add(identity)
            outcome = record.get("measured_outcome", {})
            outcome = outcome if isinstance(outcome, dict) else {}
            category = _outcome(record)
            if record.get("attempt") == 1 and category in {"support", "failure"}:
                first["measured"] += 1
                first["positive_gain"] += category == "support"
            assessment = record.get("prediction_assessment") or {}
            prediction_met = assessment.get("met") if isinstance(assessment, dict) else None
            fact = {"id": record["id"], "round": record.get("round"), "attempt": record.get("attempt"),
                    "delta": outcome.get("delta"), "new_success": outcome.get("new_success"),
                    "new_regression": outcome.get("new_regression"),
                    "net_new_success": outcome["new_success"] - outcome["new_regression"]
                    if isinstance(outcome.get("new_success"), int)
                    and isinstance(outcome.get("new_regression"), int) else None,
                    "prediction_met_declared_by_meta": prediction_met if isinstance(prediction_met, bool) else None}
            attempts.append(fact)
            for identifier in _unique(_list(record.get("used_skill_ids"))):
                uses.setdefault(str(identifier), []).append(fact)
        known_predictions = [fact for fact in attempts if fact["prediction_met_declared_by_meta"] is not None]
        return {"status": "reported", "unique_interventions": len(attempts),
                "first_candidate_measured": first["measured"],
                "first_candidate_positive_rate": first["positive_gain"] / first["measured"] if first["measured"] else None,
                "prediction_assessed": len(known_predictions),
                "prediction_hit_rate_declared_by_meta": sum(f["prediction_met_declared_by_meta"] for f in known_predictions)
                / len(known_predictions) if known_predictions else None,
                "attempts": attempts, "skill_uses": uses,
                "notes": ["Positive rate is observed strict gain, not a controller acceptance decision.",
                          "Repeated summaries of one known before/after pair count once.",
                          "Skill citations and prediction assessment are declared by Meta; correlation is not causality."],
                "warnings": warnings}
