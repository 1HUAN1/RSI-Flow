"""Seed runtime contracts and replay against pinned upstream method bodies."""

import ast
import copy
import json
import tempfile
import textwrap
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import ClassVar

from sia.task_meta.seed import SeedHarnessUpdater, _render, load_seed, run_seed, seed_capabilities, validate_seed
from sia.task_meta.storage import artifact_manifest, digest
from sia.task_meta.types import (
    ArtifactState,
    DecisionConstraintError,
    MetaDecision,
    TaskAgentState,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "seed_harness/seed.json"
REFERENCE = ROOT / "seed_harness/reference"


class Environment:
    tools: ClassVar[list] = [{"name": "lookup", "description": "Find a fact.",
              "parameters": {"properties": {"query": {"type": "string"}}, "required": ["query"]}}]

    def __init__(self):
        self.calls = []

    def step(self, name, arguments):
        self.calls.append((name, arguments))
        return "Paris"


class ReplayModel:
    def __init__(self, actions=None, memory_items=None):
        self.actions = iter(actions or ['{"tools":[{"name":"lookup","arguments":{"query":"France"}}]}',
                                        '{"tools":[{"name":"final_answer","arguments":{"answer":"Paris"}}]}'])
        self.memory_items = memory_items or []
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        first = messages[0]["content"]
        if first.startswith("Create the shortest"):
            content = "Look up the capital, then finalize."
        elif first.startswith("You are analyzing"):
            content = json.dumps({"step_summary": "", "key_extracts": self.memory_items})
        elif first.startswith("Summarize progress"):
            content = "The capital is observed. Finalize."
        elif first.startswith("You need to produce"):
            content = '{"answer":"Paris"}'
        elif first.startswith("You are managing"):
            content = "[1]"
        else:
            content = next(self.actions)
        return {"message": {"role": "assistant", "content": content}, "usage": {"output_tokens": 5},
                "binding": {"model_ref": "registered-current-task"}}


class SeedRuntimeTests(unittest.TestCase):
    def test_seed_is_executable_and_freezes_empty_initial_artifacts(self):
        spec = load_seed(SEED)
        self.assertEqual(spec["initialization"]["artifacts"], [])
        self.assertEqual(len(spec["initialization"]["coldstart_rules"]["COLDSTART_STRATEGIC_MEMORIES"]), 5)
        environment = Environment()
        result = run_seed(spec, ReplayModel(), environment, "What is France's capital?")
        self.assertEqual(result["final_answer"], "Paris")
        self.assertIsNone(result["error_type"])
        self.assertEqual(environment.calls, [("lookup", {"query": "France"})])
        self.assertEqual([call["operation"] for call in result["model_calls"]],
                         ["planning", "memory_extract", "action", "memory_extract", "action"])
        self.assertTrue(all(call["binding"]["model_ref"] == "registered-current-task" for call in result["model_calls"]))
        self.assertEqual(result["sft_conversations"][2]["messages"][-1], result["model_calls"][2]["assistant"])

    def test_shortterm_memory_is_isolated_and_context_selection_is_bounded(self):
        spec = load_seed(SEED)
        spec["context"]["artifact_char_limit"] = 4
        result = run_seed(spec, ReplayModel(memory_items=["France has Paris as its capital."]), Environment(), "Question", "abcdefgh")
        self.assertEqual(result["artifact_chars_used"], 4)
        self.assertEqual(result["artifact_chars_omitted"], 4)
        self.assertEqual(result["notes"], ["France has Paris as its capital."])
        clean = run_seed(load_seed(SEED), ReplayModel(), Environment(), "Question")
        self.assertEqual(clean["notes"], [])
        self.assertFalse(any("France has Paris" in str(call) for call in clean["model_calls"]))

    def test_summary_and_budget_finalization_have_real_calls(self):
        spec = load_seed(SEED)
        spec["planning"]["summary_interval"] = 2
        result = run_seed(spec, ReplayModel(), Environment(), "Question")
        self.assertIn("summary", [call["operation"] for call in result["model_calls"]])
        spec["planning"]["max_steps"] = 1
        result = run_seed(spec, ReplayModel(), Environment(), "Question")
        self.assertEqual(result["model_calls"][-1]["operation"], "budget_finalization")
        self.assertEqual(result["final_answer"], "Paris")

    def test_invalid_action_gets_only_declared_repair_and_no_fake_score(self):
        result = run_seed(load_seed(SEED), ReplayModel(actions=["not JSON", "still not JSON"]), Environment(), "Question")
        self.assertEqual(result["error_type"], "parse_error")
        self.assertIsNone(result["final_answer"])
        self.assertEqual([call["operation"] for call in result["model_calls"]][-2:], ["action", "action_repair"])
        self.assertEqual(result["tool_calls"], [])

    def test_failed_model_call_is_infrastructure_and_budget_is_enforced(self):
        def fail(*args, **kwargs):
            raise ValueError("Unavailable serving response")

        result = run_seed(load_seed(SEED), fail, Environment(), "Question")
        self.assertTrue(result["infrastructure_failure"])
        spec = load_seed(SEED)
        spec["budget"]["max_model_calls"] = 1
        result = run_seed(spec, ReplayModel(), Environment(), "Question")
        self.assertEqual(result["error_type"], "budget_exhausted")
        self.assertEqual(len(result["model_calls"]), 1)

    def test_terminal_tool_stops_without_an_extra_model_answer(self):
        class TerminalEnvironment(Environment):
            def step(self, name, arguments):
                return {"terminal": True, "final_answer": "Paris"}

        result = run_seed(load_seed(SEED), ReplayModel(), TerminalEnvironment(), "Question")
        self.assertEqual(len(result["model_calls"]), 3)
        self.assertEqual(result["final_answer"], "Paris")

    def test_seed_rejects_code_or_unknown_configuration(self):
        spec = load_seed(SEED)
        spec["action"]["exec"] = "any source"
        with self.assertRaisesRegex(ValueError, "undeclared"):
            validate_seed(spec)
        spec = load_seed(SEED)
        spec["prompts"]["planning_initial"] = "{% execute code %}"
        with self.assertRaisesRegex(ValueError, "placeholders"):
            validate_seed(spec)


class SeedUpdaterTests(unittest.TestCase):
    def test_requested_h_change_reloads_next_generation_and_preserves_model_and_assets(self):
        class Client:
            def complete(self, prompt, schema, **kwargs):
                return schema(edits=[{"target": "prompts.planning_initial", "value": "Plan exactly one lookup."}], summary="Shorten plan")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = root / "seed.json"
            harness.write_bytes(SEED.read_bytes())
            assets = root / "assets"
            assets.mkdir()
            (assets / "note.txt").write_text("unverified note")
            old = TaskAgentState(0, "registered-current-task", str(harness), ArtifactState(str(assets), artifact_manifest(assets)))
            before = digest(harness)
            decision = MetaDecision(action="HARNESS", diagnosis="Plan too long", evidence=[], rationale="Bound plan",
                proposed_change="Shorten planning prompt", expected_effect="Less planning", target_components=["HARNESS"],
                requested_changes=[{"id": "h1", "component": "HARNESS", "operation": "replace_config",
                                    "target": "prompts.planning_initial", "instruction": "Shorten"}])
            context = SimpleNamespace(generation=1, directory=root / "gen_1", meta_state=None, observation=SimpleNamespace())
            successor, update = SeedHarnessUpdater(Client()).apply(old, decision, context)
            self.assertEqual(digest(harness), before)
            self.assertEqual(successor.model_ref, old.model_ref)
            self.assertEqual(successor.artifacts.manifest, old.artifacts.manifest)
            calls = []

            def stop_model(messages, **kwargs):
                calls.append(messages)
                raise RuntimeError("Stop after request inspection")

            run_seed(load_seed(successor.harness_path), stop_model, Environment(), "Question")
            self.assertEqual(calls[0][0]["content"], "Plan exactly one lookup.")
            self.assertEqual(update.applied_changes[0]["target"], "prompts.planning_initial")
            self.assertIn("prompts.planning_initial", seed_capabilities(successor.harness_path)["operations"][0]["targets"])

    def test_extra_target_or_protected_budget_is_rejected(self):
        class Client:
            def complete(self, prompt, schema, **kwargs):
                return schema(edits=[{"target": "budget.max_tokens", "value": 4096}], summary="Extra budget")

        decision = MetaDecision(action="HARNESS", diagnosis="test", evidence=[], rationale="test", proposed_change="test",
            expected_effect="test", target_components=["HARNESS"], requested_changes=[{"id": "h1", "component": "HARNESS",
                "operation": "replace_config", "target": "planning.enabled", "instruction": "disable"}])
        with self.assertRaisesRegex(DecisionConstraintError, "all and only"):
            SeedHarnessUpdater(Client()).apply(TaskAgentState(0, "task", str(SEED)), decision,
                SimpleNamespace(generation=1, directory=Path("unused"), meta_state=None, observation=SimpleNamespace()))


def native_method(relative, class_name, name, namespace):
    """Execute the actual pinned method body, replacing only external dependencies."""
    path = REFERENCE / relative
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = copy.deepcopy(next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == name))
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class NativeReferenceReplayTests(unittest.TestCase):
    def test_initial_planning_and_action_requests_match_native_methods_for_canonical_json(self):
        """This is method replay with fake model/tools, not full native deployment."""
        class MessageRole(str, Enum):  # noqa: UP042 -- reproduce the pinned native enum exactly
            USER = "user"
            ASSISTANT = "assistant"
            SYSTEM = "system"
            TOOL_RESPONSE = "tool-response"

        Role = MessageRole

        class Logger:
            def log(self, *args, **kwargs):
                pass

        class Record(SimpleNamespace):
            pass

        class ToolCallRecord(Record):
            def dict(self):
                return {"name": self.name, "arguments": self.arguments}

        class NativeError(Exception):
            def __init__(self, message, *args):
                super().__init__(message)

        namespace = {"MessageRole": Role, "PlanningStep": Record, "SummaryStep": Record, "ToolCall": ToolCallRecord,
            "_render_template": lambda template, variables: _render(template, **variables),
            "populate_template": lambda template, variables: _render(template, **variables),
            "Rule": lambda *args, **kwargs: None, "Text": lambda *args, **kwargs: None,
            "Panel": lambda *args, **kwargs: None, "LogLevel": SimpleNamespace(INFO=1), "textwrap": textwrap,
            "json": json, "json_repair": SimpleNamespace(loads=json.loads), "ThreadPoolExecutor": ThreadPoolExecutor,
            "logger": SimpleNamespace(warning=lambda *args: None), "YELLOW_HEX": "yellow", "AgentGenerationError": NativeError}
        spec = load_seed(SEED)
        prompt = "What is France's capital?"
        planning_inputs = []

        def native_plan_model(messages):
            planning_inputs.append(messages)
            return Record(content="Look up the capital, then finalize.", reasoning_content="")

        native_planner = Record(model=native_plan_model, tools={}, memory=Record(steps=[]), logger=Logger(),
            prompt_templates={"planning": {"initial_plan": spec["prompts"]["planning_initial"], "task_input": spec["prompts"]["planning_task"]}},
            append_memory_guidance=lambda messages: None)
        plan_fn = native_method("references/harnessforge/planning_module/provider.py", "PlanningProvider", "topology_initialize", namespace)
        plan_fn(native_planner, prompt)
        plan = native_planner.memory.steps[0]
        plan.to_messages = MethodType(native_method("hf_runtime/Agents/memory.py", "PlanningStep", "to_messages", {"Message": dict, "MessageRole": Role}), plan)
        task = Record(task=prompt)
        task.to_messages = MethodType(native_method("hf_runtime/Agents/memory.py", "TaskStep", "to_messages", {"Message": dict, "MessageRole": Role}), task)
        native_history = [{"role": Role.SYSTEM, "content": [{"type": "text", "text": spec["prompts"]["action_system"]}]}]
        native_history += task.to_messages() + plan.to_messages(summary_mode=False)
        initial_native_history = copy.deepcopy(native_history)
        action_inputs = []
        action_outputs = iter(['{"tools":[{"name":"lookup","arguments":{"query":"France"}}]}',
                               '{"tools":[{"name":"final_answer","arguments":{"answer":"Paris"}}]}'])

        def native_action_model(messages):
            action_inputs.append(messages)
            return Record(content=next(action_outputs))

        tool_schema = copy.deepcopy(Environment.tools)
        tool_schema.append({"name": "final_answer", "description": "Gives a clear, accurate final answer to the given task.",
            "parameters": {"properties": {"answer": {"type": "string", "description": "The clear, accurate final answer to the task"}}, "required": ["answer"]}})
        native_agent = Record(memory_provider=None, task=prompt, step_number=1, logger=Logger(), tools={item["name"]: item for item in tool_schema},
            execute_model=native_action_model, write_memory_to_messages=lambda: native_history,
            execute_tool_call=lambda name, arguments: "Paris", _terminal_tool_answer=lambda *args: None,
            reformulate_tool_fuctions=lambda tools: json.dumps(tool_schema, indent=2, ensure_ascii=False),
            prompt_templates={"step": {"pre_messages": spec["prompts"]["action_step"]}}, max_tool_calls_per_step=1)
        step_fn = native_method("hf_runtime/Agents/agents.py", "ToolCallingAgent", "step", namespace)
        native_step = Record(memory_guidance=None, error=None)
        self.assertIsNone(step_fn(native_agent, native_step))
        action_memory_fn = native_method("hf_runtime/Agents/memory.py", "ActionStep", "to_messages", {"Message": dict, "MessageRole": Role})
        native_history.extend(action_memory_fn(native_step))
        native_final = step_fn(native_agent, Record())
        ours = run_seed(spec, ReplayModel(), Environment(), prompt)

        memory_inputs = []

        def native_memory_model(messages):
            memory_inputs.append(messages)
            return Record(content='{"step_summary":"","key_extracts":[]}')

        memory_provider = Record(model=native_memory_model, shortterm_memory=[],
            task_context={"current_step": 2, "last_context": "", "agent_steps": []},
            logger=Record(warning=lambda *args: None, debug=lambda *args: None, error=lambda *args, **kwargs: None),
            _parse_json_response=json.loads)
        memory_ns = {"json": json}
        for name in ("_calculate_context_delta", "_build_prev_steps_summary", "_call_llm"):
            method = native_method("references/harnessforge/memory_module/provider.py", "MemoryProvider", name, memory_ns)
            setattr(memory_provider, name, MethodType(method, memory_provider))
        context_method = native_method("hf_runtime/Agents/agents.py", "ToolCallingAgent", "_format_current_context", namespace)
        extract_method = native_method("references/harnessforge/memory_module/provider.py", "MemoryProvider", "_auto_extract_shortterm", memory_ns)
        extract_method(memory_provider, Record(query=prompt, context=context_method(Record(write_memory_to_messages=lambda: initial_native_history)),
                                              status=Record(value="in")))

        def normalize(messages):
            return [{"role": "user" if message["role"].value == "tool-response" else message["role"].value,
                     "content": "".join(block["text"] for block in message["content"])} for message in messages]

        self.assertEqual(ours["model_calls"][0]["messages"], normalize(planning_inputs[0]))
        self.assertEqual(ours["model_calls"][1]["messages"], [
            {"role": message["role"], "content": "".join(block["text"] for block in message["content"])}
            for message in memory_inputs[0]])
        self.assertEqual(ours["model_calls"][2]["messages"], normalize(action_inputs[0]))
        self.assertEqual(ours["model_calls"][4]["messages"], normalize(action_inputs[1]))
        self.assertEqual(ours["final_context"][-3]["content"], native_history[-1]["content"][0]["text"])
        self.assertEqual(ours["final_answer"], native_final)

    def test_native_budget_counts_plan_and_summary_and_calls_finalizer(self):
        class Record(SimpleNamespace):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        class NativeError(Exception):
            def __init__(self, message, *args):
                super().__init__(message)

        namespace = {"time": time, "ActionStep": Record, "AgentError": NativeError, "AgentMaxStepsError": NativeError,
                     "LogLevel": SimpleNamespace(INFO=1), "handle_agent_output_types": lambda value: value}
        run = native_method("hf_runtime/Agents/agents.py", "ToolCallingAgent", "_run", namespace)
        events = []
        native = Record(max_steps=1, summary_interval=8, memory=Record(steps=[]), logger=Record(log_rule=lambda *a, **k: None),
            planning_step=lambda task: events.append("planning"), summary_step=lambda *a, **k: events.append("summary"),
            step=lambda memory_step: events.append("action"),
            provide_final_answer=lambda task: (events.append("budget_finalization"), "", "Paris"))
        self.assertEqual(list(run(native, "Question"))[-1], "Paris")
        spec = load_seed(SEED)
        spec["planning"]["max_steps"] = 1
        ours = run_seed(spec, ReplayModel(), Environment(), "Question")
        self.assertEqual([call["operation"] for call in ours["model_calls"] if not call["operation"].startswith("memory_")], events)


if __name__ == "__main__":
    unittest.main()
