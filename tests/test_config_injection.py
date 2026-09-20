"""Regression tests for the config-injection fix.

Previously these call sites read Config.X (class attribute), silently dropping any
injected/env-overridden Config instance. These tests assert the injected instance
is now honored.
"""

import json
from unittest.mock import MagicMock, patch

from sia.config import Config
from sia.context_manager import ContextManager
from sia.orchestrator import load_agent_execution, run_evaluation


def test_load_agent_execution_honors_injected_max_size(tmp_path):
    (tmp_path / "agent_execution.json").write_text(json.dumps([{"role": "user", "content": "hi"}]))

    # Tiny limit via injected config → treated as too large.
    data, is_multi = load_agent_execution(str(tmp_path), config=Config(MAX_EXECUTION_LOG_SIZE=1))
    assert is_multi is False
    assert data["error"] == "File too large"

    # Default config loads the file normally.
    data2, _ = load_agent_execution(str(tmp_path))
    assert isinstance(data2, list)
    assert data2[0]["role"] == "user"


@patch("sia.orchestrator.subprocess.run")
def test_run_evaluation_honors_injected_timeout(mock_run, tmp_path):
    gen_dir = tmp_path / "gen_1"
    gen_dir.mkdir()
    pub = tmp_path / "task" / "data" / "public"
    pub.mkdir(parents=True)
    (pub / "evaluate.py").write_text("pass")
    (gen_dir / "results.json").write_text(json.dumps({"accuracy": 1.0}))
    mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

    run_evaluation(str(gen_dir), str(tmp_path / "task"), "/fake/venv", config=Config(EVAL_TIMEOUT=123))

    assert mock_run.call_args.kwargs["timeout"] == 123


def test_context_manager_stores_injected_config(tmp_path):
    cfg = Config(AGENT_CODE_PREVIEW_LIMIT=7, CONTEXT_SUMMARY_MAX_TURNS=2)
    cm = ContextManager(str(tmp_path), {"meta_model": "x", "agent_impl": "claude"}, config=cfg)
    assert cm.cfg.AGENT_CODE_PREVIEW_LIMIT == 7
    assert cm.cfg.CONTEXT_SUMMARY_MAX_TURNS == 2


def test_from_env_override_reaches_instance(monkeypatch):
    monkeypatch.setenv("SIA_MAX_TURNS", "99")
    assert Config.from_env().DEFAULT_MAX_TURNS == 99
