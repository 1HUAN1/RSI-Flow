"""Real Harness + offline model: reporting never invents terminal scores."""
import json
import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from harness_api import HarnessAPI
from official_evaluation import OfficialTurn
from report_environment import ReportEnvironment, serializable_trajectory
from sia.task_meta.data import TaskRecord
from sia.task_meta.environments import SearchQAAdapter, TACOAdapter
from sia.task_meta.harnessforge_manifest import HarnessBundleManifest
from sia.task_meta.seed import run_seed

ROOT = Path(__file__).resolve().parents[1]


class OfflineModel:
    enable_thinking = False

    def __call__(self, messages, **kwargs):
        content = json.dumps({'think': 'Submit the native answer.', 'tools': [
            {'name': 'final_answer', 'arguments': {'answer': 'native-answer'}}]})
        return {'message': {'role': 'assistant', 'content': content},
                'usage': {'prompt_tokens': 2, 'completion_tokens': 3}, 'finish_reason': 'stop'}


class OfficialBoundaryTests(unittest.TestCase):
    def manifest(self):
        return HarnessBundleManifest.from_directory(
            ROOT/'upstream/HarnessForge_4B/harness_factory/base_harness', harness_name='official_fixture')

    def assert_pending(self, value):
        self.assertIsNone(value.reward)
        self.assertEqual(value.metrics, {})
        self.assertEqual(value.verification['status'], 'pending_official')
        self.assertIsNone(value.verification['success'])
        self.assertFalse(value.infrastructure_error)

    def test_official_turn_is_not_a_local_grade_or_tool_executor(self):
        environment = OfficialTurn()
        for answer in ('valid looking answer', '', 'malformed answer'):
            self.assert_pending(environment.evaluate(answer))
        with self.assertRaisesRegex(ValueError, 'official outer'):
            environment.step('tool', {})

    def test_real_harness_to_api_returns_answer_without_scoring(self):
        # Do NOT mock run_seed: that missed the original environment contract bug.
        with tempfile.TemporaryDirectory() as directory:
            api = HarnessAPI.__new__(HarnessAPI)
            api.spec = self.manifest()
            api.config = SimpleNamespace(model_call_limit=32, max_output_tokens=2048, seed=42)
            api.directory = Path(directory)
            api.frozen = {'state_hash': 'offline-frozen-task'}
            api.user_spec = {'user_model': 'not-the-task'}
            import threading
            api.lock = threading.Lock(); api.request_locks = {}
            api.clients = queue.Queue(); api.clients.put(OfflineModel())
            body = {'model': 'gpt-rsi-offline', 'messages': [{'role': 'user', 'content': 'fixture'}]}
            response = api.complete(body)
            self.assertEqual(response['choices'][0]['message']['content'], 'native-answer')
            self.assertEqual(api.complete(body), response)
            receipts = [json.loads(p.read_text()) for p in Path(directory).glob('*.json') if '.call_' not in p.name]
            self.assertEqual(receipts[0]['status'], 'completed')
            self.assertEqual(receipts[0]['evaluation_status'], 'pending_official')

    def test_native_memory_skips_unscored_result_and_trajectory_is_json(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_seed(self.manifest(), OfflineModel(), OfficialTurn(), 'fixture',
                              memory_storage_root=directory, max_model_calls=16, max_steps=2)
        self.assertFalse(result['infrastructure_failure'], result['error'])
        self.assertEqual(result['final_answer'], 'native-answer')
        self.assert_pending(result['_evaluation'])
        self.assertEqual(result['memory_receipt'], {
            'status': 'skipped', 'reason': 'pending_official_evaluation'})
        self.assertFalse(any(c.get('post_evaluation') for c in result['model_calls']))
        self.assertEqual(json.loads(json.dumps(serializable_trajectory(result)))['evaluation']['reward'], None)

    def test_five_code_search_adapters_keep_public_tools_without_gold(self):
        for identifier in ('livecodebench', 'humaneval_plus', 'mbpp_plus', 'hotpotqa_dev', '2wiki_dev'):
            with self.subTest(benchmark=identifier):
                search = identifier.endswith('_dev')
                delegate = (SearchQAAdapter(SimpleNamespace(max_results=3, sha256='corpus'))
                            if search else TACOAdapter(SimpleNamespace(validate=lambda: None)))
                task = TaskRecord('fixture', 'searchqa' if search else 'code', identifier,
                                  'report_eval', 'public question', {} if search else {'tests': {'fn_name': 'f'}})
                adapter = ReportEnvironment(delegate)
                public = adapter.reset(task, 'report_eval', 42)
                self.assertTrue(public)
                self.assertEqual(adapter.tools, delegate.tools)
                with patch.object(delegate, 'evaluate', side_effect=AssertionError('No training scorer')):
                    self.assert_pending(adapter.evaluate('answer without private gold'))
                adapter.close()

    def test_error_receipt_keeps_underlying_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            api = HarnessAPI.__new__(HarnessAPI)
            import threading
            api.lock = threading.Lock(); api.request_locks = {}; api.directory = Path(directory)
            api.frozen = {'state_hash': 'fixture'}; api.user_spec = {'user_model': 'other'}
            api.config = SimpleNamespace(seed=42); api.spec = {}
            api.clients = queue.Queue(); api.clients.put(OfflineModel())
            with patch('sia.task_meta.seed.run_seed', return_value={
                    'infrastructure_failure': True, 'error_type': 'infrastructure',
                    'error': 'AttributeError: example original cause'}):
                with self.assertRaisesRegex(RuntimeError, 'example original cause'):
                    api.complete({'model': 'task', 'messages': []})
            record = json.loads(next(Path(directory).glob('*.json')).read_text())
            self.assertEqual(record['status'], 'requires_audit')
            self.assertIn('example original cause', record['harness_failure']['error'])


if __name__ == '__main__':
    unittest.main()
