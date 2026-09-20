"""Characterization: lock the exact return contract of load_agent_execution.

Complements the looser assertions in test_orchestrator_helpers.py by pinning the
precise dict shapes the feedback path depends on.
"""

import json

from sia.orchestrator import load_agent_execution


def test_missing_returns_exact_error(tmp_path):
    data, is_multi = load_agent_execution(str(tmp_path))
    assert is_multi is False
    assert data == {"error": "Execution log not found"}


def test_empty_multi_folder_returns_exact_error(tmp_path):
    (tmp_path / "agent_execution").mkdir()
    data, is_multi = load_agent_execution(str(tmp_path))
    assert is_multi is True
    assert data == {"error": "Empty execution folder", "type": "multi-trajectory"}


def test_malformed_single_returns_partial_preview(tmp_path):
    (tmp_path / "agent_execution.json").write_text("{not valid json")
    data, is_multi = load_agent_execution(str(tmp_path))
    assert is_multi is False
    assert data["error"] == "Parse error"
    assert data["raw_preview"] == "{not valid json"
    assert data["file_size"] == len("{not valid json")
    assert "parse_error" in data


def test_multi_trajectory_shape(tmp_path):
    exec_dir = tmp_path / "agent_execution"
    exec_dir.mkdir()
    for i in range(3):
        (exec_dir / f"execution_q{i}.json").write_text(json.dumps([{"role": "user", "content": f"q{i}"}]))
    data, is_multi = load_agent_execution(str(tmp_path))
    assert is_multi is True
    assert data["type"] == "multi-trajectory"
    assert data["count"] == 3
    assert [t[0]["content"] for t in data["trajectories"]] == ["q0", "q1", "q2"]
