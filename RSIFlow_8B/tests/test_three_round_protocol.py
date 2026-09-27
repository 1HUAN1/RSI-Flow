"""A fake complete three-round conversation with Codex owning all decisions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_loop import CodexReply, MetaDecisionLoop  # noqa: E402


class ScriptedCodex:
    def __init__(self, commands):
        self.commands = list(commands)
        self.thread_ids = []

    def turn(self, prompt, *, thread_id, workspace, turn_dir):
        self.thread_ids.append(thread_id)
        turn_dir.mkdir(parents=True, exist_ok=True)
        events = turn_dir / "events.jsonl"
        events.write_text('{"type":"thread.started","thread_id":"meta-thread"}\n')
        return CodexReply("meta-thread", json.dumps(self.commands.pop(0)), events)


class FactualTools:
    def __init__(self):
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        if request["operation"] == "compare_scores":
            return {"status": "compared", "delta": request["mock_delta"]}
        return {"status": "completed", "operation": request["operation"]}


def test_three_round_sequence_is_codex_chosen_and_retained(tmp_path):
    commands = []

    def tool(operation, round_number=None, **args):
        request = {"operation": operation, **args}
        if round_number is not None:
            request["round_number"] = round_number
        commands.append({"action": "tool", "request": request})

    tool("bootstrap", config_path="config.json", state_path="active_task.json")
    tool("snapshot_task_meta", destination="snapshots/A0", active_task_state="active_task.json",
         skills_path="meta/skills.jsonl", context_path="meta/context.json")
    tool("prepare_validation_snapshot", destination="snapshots/A0/round_snapshot.json",
         source_round_path="snapshots/A0/snapshot_receipt.json", round=0)
    tool("evaluate", role="independent_validation")
    for round_number, component in enumerate(("harnessforge", "sft", "artifacts"), start=1):
        tool("run_parent", round_number)
        tool("prepare_meta_evidence", round_number)
        tool(component, round_number)
        tool("run_candidate", round_number)
        tool("compare_scores", round_number, mock_delta=0.0 if round_number == 2 else 0.1)
        if round_number != 2:
            tool("activate_task", round_number, source_state=f"candidate_{round_number}.json",
                 active_state="active_task.json")
        tool("append_skills", round_number, path="meta/skills.jsonl")
        tool("snapshot_task_meta", round_number, destination=f"snapshots/round_{round_number}",
             active_task_state="active_task.json", skills_path="meta/skills.jsonl")
        tool("prepare_validation_snapshot", round_number,
             source_round_path=f"round_{round_number}/selection.json",
             destination=f"snapshots/round_{round_number}/round_snapshot.json")
        tool("evaluate", round_number, role="independent_validation")
    commands.append({"action": "finish", "summary": "three rounds completed"})

    codex, tools = ScriptedCodex(commands), FactualTools()
    journal = tmp_path / "run/meta_session"
    final = MetaDecisionLoop(workspace=tmp_path / "workspace", journal=journal,
                             codex=codex, tools=tools).run("three-round contract")

    operations = [request["operation"] for request in tools.requests]
    assert final["status"] == "complete"
    assert operations[:4] == ["bootstrap", "snapshot_task_meta", "prepare_validation_snapshot", "evaluate"]
    assert operations.count("run_parent") == operations.count("run_candidate") == 3
    assert operations.count("prepare_meta_evidence") == 3
    assert operations.count("compare_scores") == 3
    assert operations.count("append_skills") == 3
    assert operations.count("snapshot_task_meta") == 4
    assert operations.count("prepare_validation_snapshot") == 4
    assert operations.count("evaluate") == 4
    assert [request["round_number"] for request in tools.requests
            if request["operation"] == "activate_task"] == [1, 3]
    assert codex.thread_ids[0] is None
    assert set(codex.thread_ids[1:]) == {"meta-thread"}
    assert json.loads((journal / "state.json").read_text())["summary"] == "three rounds completed"
