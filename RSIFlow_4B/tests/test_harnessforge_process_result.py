"""Native HarnessForge evidence must survive candidate unload and process IPC."""

import json
import multiprocessing
import pickle
import sys
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor

from test_harnessforge_runtime import FakeEnvironment, FakeModel, native_manifest
from sia.task_meta.harnessforge_runtime import run_harnessforge


class NativeValueEnvironment(FakeEnvironment):
    def step(self, name, arguments):
        from Agents.agent_types import AgentText

        observation = super().step(name, arguments)
        return {"evidence": AgentText(observation["evidence"])}

    def evaluate(self, final_answer):
        if type(final_answer) is not str:
            raise TypeError("Evaluator received a dynamically loaded string subclass")
        return super().evaluate(final_answer)


def native_rollout():
    with tempfile.TemporaryDirectory() as temporary:
        result = run_harnessforge(
            native_manifest(), FakeModel(), NativeValueEnvironment(),
            '{"prompt":"Find the fixture answer"}', seed=73,
            memory_storage_root=temporary, bench_type="tool_use",
            max_model_calls=12, max_tool_calls=4, max_tokens=192, max_steps=4,
        )
    if "Agents.agent_types" in sys.modules:
        raise AssertionError("Candidate modules leaked after rollout")
    return result


class ProcessResultTests(unittest.TestCase):
    def assert_portable_success(self, result):
        self.assertIsNone(result["error"])
        self.assertIs(type(result["final_answer"]), str)
        self.assertEqual(result["final_answer"], "observed-answer")
        self.assertIs(type(result["tool_calls"][0]["observation"]["evidence"]), str)
        self.assertEqual(result["_evaluation"].reward, 1.0)
        self.assertTrue(result["sft_conversations"])
        restored = pickle.loads(pickle.dumps(result))
        self.assertEqual(restored, result)
        evidence = {k: v for k, v in result.items() if k != "_evaluation"}
        self.assertEqual(json.loads(json.dumps(evidence)), evidence)

    def test_result_after_native_module_cleanup(self):
        self.assert_portable_success(native_rollout())

    def test_actual_process_pool_return(self):
        with ProcessPoolExecutor(
            max_workers=2, mp_context=multiprocessing.get_context("fork")
        ) as pool:
            futures = [pool.submit(native_rollout) for _ in range(2)]
            for future in futures:
                self.assert_portable_success(future.result(timeout=90))


if __name__ == "__main__":
    unittest.main()
