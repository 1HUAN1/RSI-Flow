"""CPU-only coverage for native HarnessForge bundle execution."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sia.task_meta.data import DOMAINS
from sia.task_meta.environments import AdapterResult
from sia.task_meta.harnessforge_manifest import HarnessBundleManifest, save_manifest
from sia.task_meta.harnessforge_runtime import run_harnessforge
from sia.task_meta.pipeline_execution import MultiDomainExecutor
from sia.task_meta.storage import artifact_manifest
from sia.task_meta.types import ArtifactState, TaskAgentState


ROOT = Path(__file__).resolve().parents[1]
BASE_HARNESS = ROOT / "upstream" / "HarnessForge_4B" / "harness_factory" / "base_harness"


def native_manifest() -> HarnessBundleManifest:
    source = HarnessBundleManifest.from_directory(BASE_HARNESS, harness_name="native_fixture")
    files = dict(source.files)
    files["action_module/prompts/toolcalling_agent.yaml"] = files[
        "action_module/prompts/toolcalling_agent.yaml"
    ].replace(
        "You are a closed-set ReAct tool-using assistant.",
        "RUNTIME_ACTION_MARKER: use the materialized candidate Action provider.",
    )
    files["memory_module/provider.py"] = files["memory_module/provider.py"].replace(
        "You are analyzing the current step of task execution.",
        "RUNTIME_MEMORY_MARKER: analyze the current native Memory step.",
    )
    files["memory_module/provider.py"] = files["memory_module/provider.py"].replace(
        "from module_memory.base_memory import BaseMemoryProvider",
        """from Agents.tools import Tool
from module_memory.base_memory import BaseMemoryProvider


class RuntimeMemoryTool(Tool):
    name = "memory_hint"
    description = "A tool injected by the candidate Memory provider."
    inputs = {}
    output_type = "string"

    def forward(self):
        return "candidate-memory-tool"
""",
    ).replace(
        "            memories = []",
        """            memories = []
            if request.status == MemoryStatus.BEGIN:
                memories.append(MemoryItem(
                    id="fixture_api_memory",
                    content="Candidate API memory is available.",
                    metadata={"wrapped_tool": RuntimeMemoryTool(),
                              "skill_name": "memory_hint",
                              "description": RuntimeMemoryTool.description},
                    type=MemoryItemType.API,
                ))""",
        1,
    )
    files["builder.py"] = files["builder.py"].replace(
        'setattr(agent, "harness_name", HARNESS_NAME)',
        'setattr(agent, "harness_name", f"{HARNESS_NAME}:{context.bench_type}")',
    ).replace(
        "    return agent\n",
        """    class CompatibleAgent:
        def __init__(self, inner):
            self._inner = inner
            self.planning_system = inner.planning_system
            self.action_system = inner.action_system
            self.harness_name = inner.harness_name

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def run(self, task):
            return self._inner.run(task)

    return CompatibleAgent(agent)
""",
        1,
    )
    return HarnessBundleManifest(source.harness_name, files)


class FakeEnvironment:
    def __init__(self) -> None:
        self.calls = []
        self.evaluate_calls = []
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Return the deterministic fixture evidence.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Evidence query"},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    def step(self, name, arguments):
        self.calls.append((name, arguments))
        return {"evidence": "observed-answer"}

    def evaluate(self, final_answer):
        self.evaluate_calls.append(final_answer)
        success = final_answer == "observed-answer"
        return AdapterResult(
            float(success),
            {"exact_match": float(success)},
            {"status": "completed", "success": success, "verifier_id": "fixture"},
            None if success else "incorrect_answer",
        )

    def close(self):
        pass


class FakeModel:
    enable_thinking = False

    def __init__(self) -> None:
        self.requests = []

    def __call__(self, messages, *, tools, seed, max_tokens, temperature):
        self.requests.append(
            {
                "messages": messages,
                "tools": tools,
                "seed": seed,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        joined = "\n".join(str(message.get("content", "")) for message in messages)
        if "You are a memory guidance system" in joined:
            content = json.dumps(
                {"selected_indices": [1], "guidance": "Consider using the declared tool."}
            )
        elif "Extract reusable learnings from this successful execution" in joined:
            content = json.dumps(
                {"strategic": ["Use one evidence lookup before finalizing."],
                 "operational": ["Pass only schema-declared lookup arguments."]}
            )
        elif "RUNTIME_MEMORY_MARKER" in joined:
            content = json.dumps(
                {
                    "step_summary": "The native candidate Memory inspected current progress.",
                    "key_extracts": [],
                }
            )
        elif "Create the shortest executable plan" in joined:
            content = "Call lookup once, then finalize from its observation."
        elif "observed-answer" in joined:
            content = json.dumps(
                {
                    "think": "The environment observation supports the answer.",
                    "tools": [
                        {"name": "final_answer", "arguments": {"answer": "observed-answer"}}
                    ],
                }
            )
        else:
            content = json.dumps(
                {
                    "think": "Obtain the required evidence.",
                    "tools": [{"name": "lookup", "arguments": {"query": "fixture"}}],
                }
            )
        return {
            "message": {"role": "assistant", "content": content},
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            "binding": {"weights": "fixture"},
            "finish_reason": "stop",
        }


class NativeRuntimeTests(unittest.TestCase):
    def test_materialized_builder_planning_action_memory_and_agents_run(self):
        manifest = native_manifest()
        environment = FakeEnvironment()
        model = FakeModel()

        with tempfile.TemporaryDirectory() as temporary:
            memory_root = Path(temporary) / "replica_0"
            result = run_harnessforge(
                manifest,
                model,
                environment,
                '{"prompt":"Find the fixture answer"}',
                "artifact fixture",
                73,
                memory_storage_root=memory_root,
                bench_type="tool_use",
                max_model_calls=12,
                max_tool_calls=4,
                max_tokens=192,
                max_steps=4,
            )
            persisted_memory = json.loads(
                (memory_root / "lightweight_memory" / "longterm_memory.json").read_text(encoding="utf-8")
            )
            second_model = FakeModel()
            second_environment = FakeEnvironment()
            second_result = run_harnessforge(
                manifest,
                second_model,
                second_environment,
                '{"prompt":"Find another fixture answer"}',
                seed=173,
                memory_storage_root=memory_root,
                bench_type="tool_use",
                max_model_calls=12,
                max_tool_calls=4,
                max_tokens=192,
                max_steps=4,
            )
            selection_prompts = [
                "\n".join(str(message.get("content", "")) for message in request["messages"])
                for request in second_model.requests
                if any(
                    "You are a memory guidance system" in str(message.get("content", ""))
                    for message in request["messages"]
                )
            ]
            reused_memory_seen = any(
                "Use one evidence lookup before finalizing." in prompt
                for prompt in selection_prompts
            )

        self.assertIsNone(result["error"])
        self.assertEqual(result["final_answer"], "observed-answer")
        self.assertEqual(environment.calls, [("lookup", {"query": "fixture"})])
        self.assertEqual(environment.evaluate_calls, ["observed-answer"])
        self.assertEqual(result["tool_calls"][0]["observation"], {"evidence": "observed-answer"})
        operations = [call["operation"] for call in result["model_calls"]]
        self.assertIn("planning", operations)
        self.assertIn("memory_extract", operations)
        self.assertEqual(operations.count("action"), 2)
        self.assertEqual(result["memory_system"], "lightweight_memory")
        self.assertTrue(result["agent_contract"]["type"].endswith(".CompatibleAgent"))
        self.assertEqual(result["agent_contract"]["harness_name"], "base_harness:tool_use")
        action_context = result["action_context"]
        self.assertEqual(action_context["bench_type"], "tool_use")
        self.assertFalse(action_context["strict_bench_tools"])
        self.assertTrue(action_context["reasoning_permitted"])
        expected_helper_names = {
            "vector_tool": "vector_similarity_retrieve",
            "reasoning_tool": "reasoning",
            "process_tool": "Process",
            "end_process_tool": "EndProcess",
            "delete_memory_tool": "DeleteMemory",
            "expert_parallel_tool": "expert_parallel",
            "camv_tool": "camv",
            "executor_tool": "Executor",
            "refine_tool": "Refine",
        }
        self.assertEqual(
            {
                field_name: evidence["name"]
                for field_name, evidence in action_context["helpers"].items()
                if evidence is not None
            },
            expected_helper_names,
        )
        self.assertIsNone(action_context["helpers"]["web_tool"])
        self.assertIsNone(action_context["helpers"]["crawl_tool"])
        self.assertTrue(action_context["vector_memory_bound"])
        self.assertTrue(all(action_context["bound_agent_references"].values()))
        self.assertTrue(reused_memory_seen)
        self.assertEqual(second_result["final_answer"], "observed-answer")
        self.assertTrue(result["memory_receipt"]["success"])
        self.assertTrue(result["memory_receipt"]["is_correct"])
        self.assertGreater(result["memory_receipt"]["trajectory_steps"], 0)
        self.assertEqual(result["memory_receipt"]["serialization"], "exclusive_replica_scope")
        self.assertTrue(
            any(
                item["content"] == "Use one evidence lookup before finalizing."
                for item in persisted_memory["strategic"]
            )
        )
        self.assertTrue(
            any(
                "RUNTIME_ACTION_MARKER" in str(message.get("content", ""))
                for call in result["model_calls"]
                for message in call["messages"]
            )
        )
        self.assertTrue(
            any(
                "RUNTIME_MEMORY_MARKER" in str(message.get("content", ""))
                for call in result["model_calls"]
                for message in call["messages"]
            )
        )
        self.assertTrue(
            any(
                "memory_hint" in str(message.get("content", ""))
                for call in result["model_calls"]
                for message in call["messages"]
                if call["operation"] == "action"
            )
        )
        self.assertLess(len(result["sft_conversations"]), len(result["model_calls"]))
        self.assertTrue(result["model_calls"][-1]["post_evaluation"])
        self.assertNotIn("post_evaluation", result["model_calls"][-2])
        module_paths = result["native_module_paths"]
        self.assertIn(
            "/harness_factory/module_action/base_action.py",
            module_paths["module_action.base_action"].replace("\\", "/"),
        )
        self.assertIn(
            "/harness_factory/module_planning/base_planning.py",
            module_paths["module_planning.base_planning"].replace("\\", "/"),
        )
        self.assertIn(
            "/harness_factory/module_memory/base_memory.py",
            module_paths["module_memory.base_memory"].replace("\\", "/"),
        )
        self.assertEqual(result["messages"][-1]["role"], "assistant")
        self.assertEqual([request["seed"] for request in model.requests], list(range(73, 73 + len(model.requests))))
        self.assertTrue(all(request["tools"] is None for request in model.requests))
        self.assertEqual(result["bundle_sha256"], manifest.bundle_sha256)

    def test_builder_can_disable_memory(self):
        source = native_manifest()
        files = dict(source.files)
        files["builder.py"] = files["builder.py"].replace(
            'DEFAULT_MEMORY_SYSTEM = "lightweight_memory"',
            'DEFAULT_MEMORY_SYSTEM = "disabled"',
        )
        manifest = HarnessBundleManifest(source.harness_name, files)
        result = run_harnessforge(
            manifest,
            FakeModel(),
            FakeEnvironment(),
            "fixture",
            bench_type="toolhop",
            max_model_calls=6,
            max_tool_calls=2,
            max_tokens=96,
            max_steps=3,
        )
        self.assertEqual(result["final_answer"], "observed-answer")
        self.assertIsNone(result["memory_system"])
        self.assertIsNone(result["memory_receipt"])
        self.assertFalse(any(call["operation"] == "memory_extract"
                             for call in result["model_calls"]))
        self.assertTrue(result["action_context"]["strict_bench_tools"])
        self.assertFalse(result["action_context"]["reasoning_permitted"])
        self.assertEqual(result["action_context"]["bench_type"], "toolhop")

    def test_candidate_exception_is_scored_as_task_failure(self):
        source = native_manifest()
        files = dict(source.files)
        files["builder.py"] = files["builder.py"].replace(
            "def build_agent_from_context(context: ActionContext) -> ToolCallingAgent:\n",
            "def build_agent_from_context(context: ActionContext) -> ToolCallingAgent:\n"
            "    raise RuntimeError('candidate fixture failure')\n",
            1,
        )
        manifest = HarnessBundleManifest(source.harness_name, files)
        environment = FakeEnvironment()
        result = run_harnessforge(
            manifest, FakeModel(), environment, "fixture",
            max_model_calls=4, max_tool_calls=2, max_tokens=96, max_steps=2,
        )
        self.assertEqual(result["error_type"], "candidate_error")
        self.assertFalse(result["infrastructure_failure"])
        self.assertEqual(result["final_answer"], "")
        self.assertEqual(environment.evaluate_calls, [""])
        self.assertFalse(result["_evaluation"].verification["success"])

    def test_evaluator_exception_is_infrastructure_and_not_retried(self):
        class BrokenEvaluator(FakeEnvironment):
            def evaluate(self, final_answer):
                self.evaluate_calls.append(final_answer)
                raise OSError("fixture evaluator unavailable")

        environment = BrokenEvaluator()
        result = run_harnessforge(
            native_manifest(), FakeModel(), environment, "fixture",
            max_model_calls=8, max_tool_calls=2, max_tokens=96, max_steps=3,
        )
        self.assertTrue(result["infrastructure_failure"])
        self.assertEqual(result["error_type"], "infrastructure")
        self.assertEqual(len(environment.evaluate_calls), 1)
        self.assertIsNone(result["_evaluation"])

    def test_model_budget_is_enforced_before_extra_transport(self):
        model = FakeModel()
        result = run_harnessforge(
            native_manifest(),
            model,
            FakeEnvironment(),
            "fixture",
            seed=9,
            max_model_calls=1,
            max_tool_calls=1,
            max_tokens=64,
            max_steps=2,
        )
        self.assertEqual(result["error_type"], "budget_exhausted")
        self.assertFalse(result["infrastructure_failure"])
        self.assertEqual(len(result["model_calls"]), 1)
        self.assertEqual(len(model.requests), 1)


class PipelineManifestDispatchTests(unittest.TestCase):
    def test_execute_loads_manifest_and_preserves_it_for_batches(self):
        manifest = native_manifest()

        class EmptyStore:
            compact_rollouts = False

            def window(self, cursor, quotas):
                return [], dict.fromkeys(DOMAINS, 0)

            def probe(self):
                return []

            def coverage(self, cursor):
                return {"all_tasks_scheduled": True}

        class CapturingExecutor(MultiDomainExecutor):
            def __init__(self):
                super().__init__(
                    EmptyStore(),
                    adapter_factory=None,
                    model_factory=None,
                    quotas=dict.fromkeys(DOMAINS, 0),
                    expected_domains=DOMAINS,
                )
                self.specs = []
                self.memory_roots = []

            def _batch(self, state, spec, jobs, directory, assets, *, probe, sources):
                self.specs.append(spec)
                self.memory_roots.append(self._harnessforge_memory_root)
                return [
                    {
                        "task_id": f"fixture-{domain}",
                        "rollout_id": 0,
                        "domain": domain,
                        "verification": {"status": "completed", "success": False},
                        "terminal_reward": 0.0,
                        "metrics": {"native_partial_score": 0.0, "f1": 0.0},
                        "error_type": "incorrect_answer",
                        "notes": [],
                        "wall_time_seconds": 0.0,
                        "model_call_count": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "unknown_usage_calls": 0,
                        "usage_complete": True,
                    }
                    for domain in DOMAINS
                ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "run"
            manifest_path = save_manifest(root / "candidate.json", manifest)
            assets_a = root / "assets_a"
            assets_a.mkdir()
            (assets_a / "note.txt").write_text("first", encoding="utf-8")
            state = TaskAgentState(
                0, "fixture-model", str(manifest_path),
                artifacts=ArtifactState(str(assets_a), artifact_manifest(assets_a)),
            )
            executor = CapturingExecutor()
            result = executor.execute(state, run_root / "gen_a")

            model_updated = TaskAgentState(
                0, "fixture-model-v2", str(manifest_path),
                artifacts=ArtifactState(str(assets_a), artifact_manifest(assets_a)),
                checkpoint_manifest=[{"path": "weights-v2", "sha256": "fixture-v2"}],
            )
            executor.execute(model_updated, run_root / "gen_b")

            assets_b = root / "assets_b"
            assets_b.mkdir()
            (assets_b / "note.txt").write_text("second", encoding="utf-8")
            artifact_updated = TaskAgentState(
                0, "fixture-model-v2", str(manifest_path),
                artifacts=ArtifactState(str(assets_b), artifact_manifest(assets_b)),
                checkpoint_manifest=model_updated.checkpoint_manifest,
            )
            executor.execute(artifact_updated, run_root / "gen_c")

            alternate_files = dict(manifest.files)
            alternate_files["builder.py"] += "\n# isolated bundle identity\n"
            alternate = HarnessBundleManifest("alternate_fixture", alternate_files)
            alternate_path = save_manifest(root / "alternate.json", alternate)
            isolated_state = TaskAgentState(
                0, "fixture-model-v2", str(alternate_path),
                artifacts=ArtifactState(str(assets_b), artifact_manifest(assets_b)),
                checkpoint_manifest=model_updated.checkpoint_manifest,
            )
            executor.execute(isolated_state, run_root / "gen_d")

        self.assertEqual(len(executor.specs), 8)
        self.assertTrue(all(isinstance(spec, HarnessBundleManifest) for spec in executor.specs))
        self.assertEqual(
            {spec.bundle_sha256 for spec in executor.specs},
            {manifest.bundle_sha256, alternate.bundle_sha256},
        )
        self.assertEqual(len(result.trajectories), len(DOMAINS))
        self.assertEqual(len(set(executor.memory_roots[:6])), 1)
        self.assertEqual(Path(executor.memory_roots[0]).name, manifest.bundle_sha256)
        self.assertEqual(len(set(executor.memory_roots[6:])), 1)
        self.assertEqual(Path(executor.memory_roots[6]).name, alternate.bundle_sha256)
        self.assertNotEqual(executor.memory_roots[0], executor.memory_roots[6])


if __name__ == "__main__":
    unittest.main()
