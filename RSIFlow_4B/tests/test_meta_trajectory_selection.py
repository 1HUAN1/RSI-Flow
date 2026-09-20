"""Deterministic detailed-trajectory quotas for Meta routing."""

from collections import Counter

from sia.task_meta.evolution_protocol import DOMAINS, meta_routing_rows
from sia.task_meta.intervention_evidence import decision_sources


def row(domain, index, success, error=None):
    value = {
        "task_id": f"{domain}-{success}-{index}",
        "question_id": f"{domain}-{success}-{index}",
        "trajectory_id": f"trajectory-{domain}-{success}-{index}",
        "rollout_id": 0,
        "domain": domain,
        "purpose": "evolution_train",
        "collection_stage": "parent_pre_update",
        "verification": {"status": "completed", "success": success},
        "terminal_reward": int(success),
        "events": [{"kind": "start"}, {"kind": "finish"}],
        "tool_calls": [],
        "model_calls": [],
        "transport_calls": [],
        "model_answer": "ok" if success else "bad",
    }
    if error is not None:
        value["error_type"] = error
    return value


def test_exact_domain_outcome_and_failure_error_round_robin():
    rows = []
    for domain in DOMAINS:
        rows.extend(row(domain, index, True) for index in range(20))
        rows.extend(row(domain, index, False, "a") for index in range(20))
        rows.extend(row(domain, index + 20, False, "b") for index in range(20))
        rows.extend(row(domain, index + 40, False, "c") for index in range(20))
    selected = meta_routing_rows(rows)
    assert len(selected) == 48
    assert Counter(item["domain"] for item in selected) == Counter(dict.fromkeys(DOMAINS, 16))
    for domain in DOMAINS:
        domain_rows = [item for item in selected if item["domain"] == domain]
        assert sum(item["verification"]["success"] is True for item in domain_rows) == 8
        failures = Counter(item["error_type"] for item in domain_rows
                           if item["verification"]["success"] is not True)
        assert sorted(failures.values()) == [2, 3, 3]
    assert [item["task_id"] for item in selected] == [
        item["task_id"] for item in meta_routing_rows(list(reversed(rows)))
    ]


def test_outcome_shortage_backfills_within_domain():
    rows = []
    rows.extend(row("tool_use", index, True) for index in range(2))
    rows.extend(row("tool_use", index, False, "a") for index in range(20))
    rows.extend(row("tool_use", index + 20, False, "b") for index in range(20))
    rows.extend(row("code", index, True) for index in range(30))
    rows.extend(row("code", index, False, "compile") for index in range(3))
    rows.extend(row("searchqa", index, True) for index in range(8))
    rows.extend(row("searchqa", index, False, "wrong_answer") for index in range(8))
    selected = meta_routing_rows(rows)
    grouped = {domain: [item for item in selected if item["domain"] == domain] for domain in DOMAINS}
    assert {domain: len(values) for domain, values in grouped.items()} == dict.fromkeys(DOMAINS, 16)
    assert sum(item["verification"]["success"] is True for item in grouped["tool_use"]) == 2
    assert Counter(item["error_type"] for item in grouped["tool_use"]
                   if item["verification"]["success"] is not True) == {"a": 7, "b": 7}
    assert sum(item["verification"]["success"] is True for item in grouped["code"]) == 13
    assert sum(item["verification"]["success"] is not True for item in grouped["code"]) == 3


def test_short_domain_capacity_is_reassigned_and_decision_sources_uses_selector():
    rows = []
    rows.extend(row("tool_use", index, index < 2, "tool_error" if index >= 2 else None)
                for index in range(4))
    for domain in ("code", "searchqa"):
        rows.extend(row(domain, index, True) for index in range(30))
        rows.extend(row(domain, index, False, "a" if index % 2 else "b")
                    for index in range(30))
    selected = meta_routing_rows(rows)
    assert len(selected) == 48
    assert Counter(item["domain"] for item in selected) == {
        "tool_use": 4, "code": 22, "searchqa": 22,
    }
    sources = decision_sources(rows, [])
    detailed = [item for item in sources
                if item["evidence_detail"] == "bounded_representative_excerpt"]
    assert {item["task_id"] for item in detailed} == {item["task_id"] for item in selected}
    assert len(detailed) == 48
