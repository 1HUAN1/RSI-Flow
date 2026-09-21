"""Memory requirements survive branch replacement and native proposal/repair prompts."""
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_append_memory as memory_test
from sia.task_meta.meta_harness.bundle import MetaHarnessStore
from sia.task_meta.meta_harness.runtime import execute
from sia.task_meta.recursive_feedback import MEMORY_ID_CONTRACT, validate_recursive_memory
from sia.task_meta.round_evolution import RoundProtocol, MEMORY_POLICY
from sia.task_meta.types import MetaHarnessUpdate

ROOT = Path(__file__).resolve().parents[1]


class MemoryPromptContract(unittest.TestCase):
    def setUp(self):
        self.fixture = memory_test.AppendMemory()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_contract_survives_both_effect_branches_and_next_round(self):
        for policy in (None, 'single_candidate_strict_positive_gain'):
            for round_id in (1, 2):
                with self.subTest(policy=policy, round_id=round_id):
                    p = object.__new__(RoundProtocol)
                    p.config = SimpleNamespace(round_protocol={
                        'memory_policy': MEMORY_POLICY, 'candidate_policy': policy})
                    p.root = self.fixture.root
                    p.store = SimpleNamespace(round_id=round_id, sequential_domains=False)
                    p.meta_stage = 'effect'
                    p.selected_component = 'MODEL'
                    p.selected_decision = 'decision1'
                    p.selected_experience = 'experience1'
                    p.loaded_memory = {'existing': 'preserved'}
                    p.recursive_summary = {}
                    p.boundary = None
                    p.attempts = [{'action': 'MODEL', 'decision': {'decision_id': 'decision1'}}]
                    envelope = {'trusted_facts': {}}
                    with patch('sia.task_meta.round_evolution.read', return_value={'pairs': []}):
                        p.enrich_effect(envelope)
                    instructions = envelope['trusted_facts']['memory_update_contract']['instructions']
                    self.assertTrue(instructions.endswith(MEMORY_ID_CONTRACT))
                    self.assertEqual(instructions.count(MEMORY_ID_CONTRACT), 1)
                    self.assertEqual(p.loaded_memory, {'existing': 'preserved'})

    def test_native_generation_and_repair_keep_contract_outside_clipped_payload(self):
        store = MetaHarnessStore(self.fixture.root / 'store')
        bundle = store.initialize(ROOT / 'runtime/meta_harness/seed', 'pinned', 'binary')
        envelope = {'trusted_facts': {'memory_update_contract': {'instructions': MEMORY_ID_CONTRACT}},
                    'raw_trajectories': [], 'experiences': []}
        original = copy.deepcopy(envelope)
        calls = []

        class RepairCaptured(Exception):
            pass

        def invoke(stage_id, prompt, schema, evidence_files=None, allowed_paths=None):
            calls.append((stage_id, prompt, json.loads(evidence_files['meta_input/operation.json'])))
            if len(calls) == 2:
                raise RepairCaptured()
            # Schema-valid summary with unbound evidence forces the normal repair path.
            return self.fixture.update

        with patch('sia.task_meta.meta_harness.runtime._bounded_inline', return_value={'dependencies': []}):
            with self.assertRaises(RepairCaptured):
                execute(bundle, 'meta_self_update', envelope, MetaHarnessUpdate, invoke,
                        self.fixture.root / 'audit')
        self.assertEqual(len(calls), 2)
        self.assertIn('repair', calls[1][0])
        self.assertTrue(calls[1][2]['candidate_errors'])
        for _, prompt, _ in calls:
            self.assertIn(MEMORY_ID_CONTRACT, prompt)
        self.assertEqual(envelope, original)


    def test_repair_receives_native_and_append_errors_together(self):
        store = MetaHarnessStore(self.fixture.root / 'store_all_errors')
        bundle = store.initialize(ROOT / 'runtime/meta_harness/seed', 'pinned', 'binary')
        update = self.fixture.update.model_copy(deep=True)
        update.five_stage.principle_operations[0].record.evidence_state = 'supported'
        update.five_stage.principle_operations[0].evidence_ids.append('attempt1')
        general = update.five_stage.principle_operations[1]
        general.principle_id = general.record.principle_id = 'missing_prefix'
        envelope = {'trusted_facts': {'memory_update_contract': {'instructions': MEMORY_ID_CONTRACT}},
                    'raw_trajectories': [], 'experiences': [],
                    'outcome_review': {'expectation': {}, 'outcome_hash': 'outcome1', 'pairs': []}}
        captured = []
        class Repaired(Exception):
            pass
        def invoke(stage_id, prompt, schema, evidence_files=None, allowed_paths=None):
            data = json.loads(evidence_files['meta_input/operation.json'])
            if data['candidate_errors']:
                captured.extend(data['candidate_errors'])
                raise Repaired()
            return update
        def domain_check(value):
            return validate_recursive_memory(value, self.fixture.meta,
                [{'action': 'MODEL', 'attempt_id': 'attempt1', 'outcome_class': 'zero_gain',
                  'decision': {'decision_id': 'decision1'}}], 'experience1', None, [])
        with patch('sia.task_meta.meta_harness.runtime.five_stage.materialize',
                   side_effect=ValueError('New principles start active/tentative at revision 1')):
            with self.assertRaises(Repaired):
                execute(bundle, 'meta_self_update', envelope, MetaHarnessUpdate, invoke,
                        self.fixture.root / 'audit_all_errors', validate_candidate=domain_check)
        errors = '\n'.join(captured)
        self.assertIn('active/tentative', errors)
        self.assertIn('missing_prefix', errors)
        self.assertIn('principle.<unique_id>', errors)
        self.assertIn('five_stage.principle_operations[1]', errors)

    def test_success_and_failure_append_without_overwriting_prior_records(self):
        from sia.task_meta.meta_harness.five_stage import materialize
        for outcome, tag in [('accepted_positive_gain', 'success'), ('zero_gain', 'failure'),
                             ('negative_gain', 'failure'), ('candidate_unavailable', 'failure')]:
            with self.subTest(outcome=outcome):
                update = self.fixture.update.model_copy(deep=True)
                skill = update.five_stage.principle_operations[0]
                skill.evidence_ids.append('attempt1')
                skill.record.library_tags = [tag]
                attempts = [dict(attempt_id='attempt1', action='MODEL', outcome_class=outcome,
                                 decision={'decision_id': 'decision1'})]
                validate_recursive_memory(update, self.fixture.meta, attempts, 'experience1', None, [])
                files, _ = materialize(update, self.fixture.files,
                                       allowed_evidence={'pair1', 'decision1', 'experience1', 'attempt1'})
                before = json.loads(self.fixture.files['principles.json'])['records']
                after = json.loads(files['principles.json'])['records']
                self.assertEqual(after[:len(before)], before)
                self.assertEqual(len(after), len(before) + 2)


if __name__ == '__main__':
    unittest.main()
