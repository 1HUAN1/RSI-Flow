"""Tests for Codex-led orchestration without controller policy decisions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_loop import (CodexCli, CodexReply, MetaDecisionLoop, parse_codex_command,
                       MAINLINE_REMINDER, render_receipt_for_codex)  # noqa: E402


def test_parser_accepts_one_fenced_or_embedded_command_only():
    command = {"action": "tool", "request": {"operation": "write_text", "path": "c.txt"}}
    encoded = json.dumps(command)
    assert parse_codex_command(encoded) == command
    assert parse_codex_command("```json\n" + encoded + "\n```") == command
    assert parse_codex_command("Here is the command: " + encoded) == command
    assert parse_codex_command(encoded + " and " + encoded) == command
    assert parse_codex_command(encoded + ' and {"action":"finish"}') is None
    assert parse_codex_command('\":\"tool\",\"request\":{}}') is None


def test_small_receipt_is_visible_but_large_one_is_paged(tmp_path):
    path = tmp_path / "receipt.json"
    small = render_receipt_for_codex({"status": "read", "content": "48 excerpts"}, path)
    assert "48 excerpts" in small
    large = render_receipt_for_codex({"status": "read", "path": "sources.json",
                                      "content": "x" * 300000}, path)
    assert "inline_omitted_due_to_size" in large
    assert "sources.json" in large
    assert "offset_chars" in large
    assert "x" * 1000 not in large


class FakeCodex:
    def __init__(self, messages):
        self.messages = list(messages)
        self.calls = []

    def turn(self, prompt, *, thread_id, workspace, turn_dir):
        self.calls.append((prompt, thread_id))
        turn_dir.mkdir(parents=True, exist_ok=True)
        events = turn_dir / "events.jsonl"
        events.write_text('{"type":"thread.started","thread_id":"one-thread"}\n')
        return CodexReply("one-thread", self.messages.pop(0), events)


class FakeTools:
    def __init__(self):
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        if request.get("operation") == "broken":
            raise RuntimeError("task tool failed")
        return {"ok": True, "fact": request}


def test_codex_chooses_tool_order_and_deployment(tmp_path):
    codex = FakeCodex([
        json.dumps({"action": "tool", "request": {"operation": "run_parent", "round": 0}}),
        json.dumps({"action": "tool", "request": {"operation": "compare_scores", "round": 0}}),
        json.dumps({"action": "tool", "request": {"operation": "deploy_candidate", "candidate": "c1"}}),
        json.dumps({"action": "finish", "summary": "three operations completed"}),
    ])
    tools = FakeTools()
    journal = tmp_path / "journal"
    result = MetaDecisionLoop(workspace=tmp_path / "workspace", journal=journal,
                              codex=codex, tools=tools).run("start")
    assert result["status"] == "complete"
    assert [r["operation"] for r in tools.requests] == [
        "run_parent", "compare_scores", "deploy_candidate"]
    assert [thread for _, thread in codex.calls] == [None, "one-thread", "one-thread", "one-thread"]
    assert "controller has not decided" in codex.calls[1][0].lower()
    assert '"operation":"run_parent"' in codex.calls[1][0]
    assert (journal / "turns/0002/receipt.json").is_file()


def test_tool_failure_is_returned_to_codex_not_experiment_abort(tmp_path):
    codex = FakeCodex([
        '{"action":"tool","request":{"operation":"broken"}}',
        '{"action":"tool","request":{"operation":"repair"}}',
        '{"action":"finish","summary":"recovered"}',
    ])
    journal = tmp_path / "journal"
    tools = FakeTools()
    result = MetaDecisionLoop(workspace=tmp_path / "workspace", journal=journal,
                              codex=codex, tools=tools).run("start")
    receipt = json.loads((journal / "turns/0000/receipt.json").read_text())
    assert receipt["ok"] is False
    assert receipt["error_type"] == "RuntimeError"
    assert result["summary"] == "recovered"
    assert tools.requests[-1]["operation"] == "repair"
    assert MAINLINE_REMINDER in codex.calls[0][0]
    assert MAINLINE_REMINDER in codex.calls[1][0]
    assert MAINLINE_REMINDER not in codex.calls[2][0]


def test_unstructured_reply_is_repaired_in_same_thread(tmp_path):
    codex = FakeCodex(["not JSON", '{"action":"finish","summary":"fixed"}'])
    tools = FakeTools()
    result = MetaDecisionLoop(workspace=tmp_path / "workspace", journal=tmp_path / "journal",
                              codex=codex, tools=tools).run("start")
    assert result["summary"] == "fixed"
    assert codex.calls[1][1] == "one-thread"
    assert tools.requests == []


def test_resuming_ambiguous_tool_does_not_repeat_it(tmp_path):
    journal = tmp_path / "journal"
    journal.mkdir()
    (journal / "state.json").write_text(json.dumps({
        "thread_id": "one-thread", "turn": 2, "next_prompt": None,
        "status": "tool_inflight", "summary": None,
        "last_mainline_anchor": {"current_stage": "B1", "next_milestone": "parent_rollout"}}))
    codex = FakeCodex(['{"action":"finish","summary":"inspected existing output"}'])
    tools = FakeTools()
    result = MetaDecisionLoop(workspace=tmp_path / "workspace", journal=journal,
                              codex=codex, tools=tools, status_provider=lambda:
                              "Current stage: B1\nNext required milestone: parent_rollout").run("unused")
    assert result["status"] == "complete"
    assert tools.requests == []
    assert "may have completed" in codex.calls[0][0]
    assert MAINLINE_REMINDER in codex.calls[0][0]
    assert "Next required milestone: parent_rollout" in codex.calls[0][0]


def test_codex_cli_starts_persistent_thread_and_resumes(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "codex_home"
    home.mkdir()
    calls = []

    def fake_run(argv, *, input, text, stdout, stderr, cwd, env, timeout, check):
        calls.append((argv, input, env["CODEX_HOME"], cwd))
        output = Path(argv[argv.index("-o") + 1])
        output.write_text('{"action":"finish","summary":"ok"}')
        stdout.write('{"type":"thread.started","thread_id":"thread-1"}\n')
        return SimpleNamespace(returncode=0, stderr="")

    cli = CodexCli(codex_home=home, model="DeepSeek-V4.1-Flash")
    with patch("meta_loop.subprocess.run", side_effect=fake_run):
        first = cli.turn("start", thread_id=None, workspace=workspace,
                         turn_dir=tmp_path / "turn0")
        second = cli.turn("continue", thread_id=first.thread_id, workspace=workspace,
                          turn_dir=tmp_path / "turn1")
    assert first.thread_id == second.thread_id == "thread-1"
    assert "--ephemeral" not in calls[0][0]
    assert calls[0][0][1] == "exec"
    assert calls[1][0][2] == "resume"
    assert calls[0][2] == str(home)


def test_mainline_repeats_only_when_stage_or_milestone_changes(tmp_path):
    codex = FakeCodex([
        '{"action":"tool","request":{"operation":"read_text"}}',
        '{"action":"tool","request":{"operation":"read_json"}}',
        '{"action":"tool","request":{"operation":"inspect"}}',
        '{"action":"finish","summary":"done"}',
    ])
    statuses = iter((
        "Current stage: B1 — evolve\nNext required milestone: parent_rollout — run\nAcquired evidence: one",
        "Current stage: B1 — evolve\nNext required milestone: parent_rollout — run\nAcquired evidence: two",
        "Current stage: B1 — evolve\nNext required milestone: trajectory_analysis — inspect",
        "Current stage: B2 — evolve\nNext required milestone: parent_rollout — run",
    ))
    recorded = []
    journal = tmp_path / "journal"
    result = MetaDecisionLoop(
        workspace=tmp_path / "workspace", journal=journal,
        codex=codex, tools=FakeTools(), status_provider=lambda: next(statuses),
        progress_recorder=lambda turn_dir: recorded.append(turn_dir.name),
    ).run("start")

    assert result["status"] == "complete"
    prompts = [prompt for prompt, _ in codex.calls]
    assert MAINLINE_REMINDER in prompts[0]
    assert MAINLINE_REMINDER not in prompts[1]
    assert MAINLINE_REMINDER in prompts[2]
    assert MAINLINE_REMINDER in prompts[3]
    assert "Acquired evidence: one" in prompts[0]
    assert "Acquired evidence: two" not in prompts[1]
    assert recorded == ["0000", "0001", "0002"]
    state = json.loads((journal / "state.json").read_text())
    assert state["last_mainline_anchor"] == {
        "current_stage": "B2", "next_milestone": "parent_rollout",
    }


def test_finish_guard_keeps_same_session_until_required_stages_exist(tmp_path):
    codex = FakeCodex([
        '{"action":"finish","summary":"too early"}',
        '{"action":"finish","summary":"all rounds complete"}',
    ])
    answers = iter(((False, "B3 validation is still incomplete."), (True, "")))
    calls = []

    def guard():
        calls.append(True)
        return next(answers)

    result = MetaDecisionLoop(
        workspace=tmp_path / "workspace", journal=tmp_path / "journal",
        codex=codex, tools=FakeTools(), finish_guard=guard,
    ).run("start")

    assert result["summary"] == "all rounds complete"
    assert len(calls) == 2
    assert "B3 validation is still incomplete" in codex.calls[1][0]
    assert codex.calls[1][1] == "one-thread"
