"""Budget failures remain evidence; 180-task rounds retain complete pairing."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_contracts  # Establish the project/runtime import paths.
from test_harnessforge_runtime import FakeEnvironment, native_manifest
from evolution_protocol import SMALL_TRAIN_QUOTAS, training_quotas, make_manifest, assert_disjoint
from prepare_subset_release import select_round
from test_training_protocol import task
from launch import make_pipeline
from sia.task_meta.harnessforge_runtime import _ModelAdapter, _classify_error, run_harnessforge
from sia.task_meta.task_client import TaskContextBudgetExceeded, TaskInfrastructureError
from sia.task_meta.pipeline_execution import aggregate
from sia.task_meta.environments import EnvScalerAdapter


class FailureAndQuotaTests(unittest.TestCase):
    def test_context_rejection_is_not_transport_failure_and_not_redispatched(self):
        dispatches = []
        def rejected(*args, **kwargs):
            dispatches.append(1)
            raise TaskContextBudgetExceeded('fixture context exhausted')
        model = _ModelAdapter(rejected, seed=42, max_tokens=64, temperature=0., max_model_calls=12)
        for _ in range(2):
            with self.assertRaises(TaskContextBudgetExceeded): model([{'role':'user','content':'q'}])
        self.assertEqual(len(dispatches), 1)
        self.assertEqual(model.calls[0]['status'], 'rejected')
        self.assertFalse(model.calls[0]['generation_dispatched'])
        self.assertFalse(_classify_error(TaskContextBudgetExceeded('fixture'))[2])
        self.assertTrue(_classify_error(TaskInfrastructureError('HTTP 503'))[2])

    def test_native_agent_context_failure_is_scorable_and_remains_in_denominator(self):
        def rejected(*args, **kwargs): raise TaskContextBudgetExceeded('fixture')
        with tempfile.TemporaryDirectory() as directory:
            result = run_harnessforge(native_manifest(), rejected, FakeEnvironment(),
                'fixture question', memory_storage_root=directory, bench_type='searchqa',
                max_model_calls=4, max_tool_calls=4, max_tokens=64, max_steps=2)
        self.assertFalse(result['infrastructure_failure'])
        self.assertEqual(result['error_type'], 'context_budget_exhausted')
        score = result['_evaluation']
        self.assertEqual(score.reward, 0.)
        self.assertEqual(score.verification['status'], 'completed')
        self.assertFalse(score.verification['success'])
        row = dict(domain='searchqa', terminal_reward=score.reward, verification=score.verification,
                   metrics=score.metrics, error_type=result['error_type'], infrastructure_error=False)
        scored = aggregate([row], require_domains=False)
        self.assertEqual(scored['domains']['searchqa']['denominator'], 1)
        with self.assertRaises(TaskInfrastructureError):
            aggregate([{**row,'infrastructure_error':True}], require_domains=False)

    def test_180_selection_disjoint_deterministic_and_pipeline_quotas(self):
        config = json.loads((test_contracts.BUNDLE/'configs/train_180.json').read_text())
        self.assertEqual(training_quotas(config), SMALL_TRAIN_QUOTAS)
        selected = []
        for r in (1,2,3):
            rows = [{**task(s, r*10000+i), 'role':'train_evolution', 'round_id':r}
                    for s,q in SMALL_TRAIN_QUOTAS.items() for i in range(q*2)]
            parent = make_manifest('train_evolution', r, rows)
            chosen = select_round(parent, SMALL_TRAIN_QUOTAS, 42)
            reverse = select_round({**parent,'tasks':list(reversed(rows))}, SMALL_TRAIN_QUOTAS, 42)
            self.assertEqual(chosen, reverse)
            self.assertEqual(len(chosen['tasks']), 180)
            selected.append(chosen)
        assert_disjoint(selected)
        pipeline = make_pipeline(config, test_contracts.BUNDLE/'runtime')
        self.assertEqual(pipeline['window_quotas'], dict.fromkeys(['tool_use','code','searchqa'],60))
        self.assertEqual(pipeline['round_protocol']['train_quotas_per_round'], SMALL_TRAIN_QUOTAS)

    def test_unknown_checker_failure_still_blocks(self):
        adapter = EnvScalerAdapter.__new__(EnvScalerAdapter)
        adapter.terminated = False
        adapter.task = SimpleNamespace(task_id='fixture:other', payload={'checklist_with_func':[]})
        adapter.verifier_id = 'fixture'
        adapter.worker = SimpleNamespace(request=lambda request: {'checks':[{'valid':False,'result':None,'error':'unexpected'}]})
        score = adapter.evaluate('answer')
        self.assertTrue(score.infrastructure_error)
        self.assertEqual(score.error_type, 'official_checker_failure')


if __name__ == '__main__': unittest.main()
