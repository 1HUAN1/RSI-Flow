"""Five-round extension preserves old batches, live ledger state and Prompt freedom."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data_protocol
import evolution_protocol
from extend_rounds import assert_task_disjoint
import pytest
from experiment_progress import ExperimentProgress
from launch_meta import experiment_prompt


def test_five_round_prompt_has_no_three_round_goal(tmp_path):
    prompt = experiment_prompt(tmp_path / "config.json", {"rounds": 5}, tmp_path / "run")
    assert "B1-B5: in each round" in prompt
    assert "round_number=1..5 for B1..B5" in prompt
    assert "B1-B3" not in prompt
    assert "final B3" not in prompt
    assert "HARNESS" in prompt and "MODEL" in prompt and "ARTIFACTS" in prompt


def test_existing_ledger_reloads_extended_policy(tmp_path):
    progress = ExperimentProgress(tmp_path, rounds=3)
    assert progress.rebuild()["final_stage"] == "B3"
    (tmp_path / "workflow_policy.json").write_text(json.dumps({"rounds": 5}))
    result = progress.rebuild()
    assert result["final_stage"] == "B5"
    assert len(result["stages"]) == 6
    assert "B1-B5" in result["mainline_goal"]
    assert not result["finish_allowed"]


def test_quota_contract_accepts_five_180_rounds():
    config = {"rounds": 5, "allocated_tasks": 900, "training_tasks_per_pass": 180,
              "train_quotas_per_round": evolution_protocol.SMALL_TRAIN_QUOTAS}
    assert sum(evolution_protocol.training_quotas(config).values()) == 180


def test_stage_lookup_uses_frozen_release_rounds(tmp_path, monkeypatch):
    (tmp_path / "protocol.json").write_text(json.dumps({"rounds": 5}))
    marker = {"round_id": 5}
    monkeypatch.setattr(evolution_protocol, "manifest", lambda path, role, number: marker)
    assert evolution_protocol.stage_manifest(tmp_path, 5) is marker


def test_allocation_respects_configured_round_count():
    items = [{"task_id": f"task_{i}", "group_id": f"group_{i}",
              "content_hash": f"content_{i}", "source": "source", "strata": {},
              "kind": "train"} for i in range(20)]
    batches, _ = data_protocol.allocate(items, {
        "rounds": 5, "split_seed": 42, "train_quotas_per_round": {"source": 2},
        "grouping": {"max_environment_fraction_per_round": .1},
        "validation_limits": {},
    }, {})
    assert [b["round_id"] for b in batches[:5]] == [1, 2, 3, 4, 5]
    assert all(len(b["tasks"]) == 2 for b in batches[:5])
    assert len({t["task_id"] for b in batches[:5] for t in b["tasks"]}) == 10


def test_new_tasks_can_share_environment_but_not_task_identity():
    old = {"tasks": [{"task_id": "old", "content_hash": "old_prompt", "group_id": "env_1"}]}
    new = {"tasks": [{"task_id": "new", "content_hash": "new_prompt", "group_id": "env_1"}]}
    assert_task_disjoint([old, new])
    with pytest.raises(ValueError, match="Repeated task"):
        assert_task_disjoint([old, old])
