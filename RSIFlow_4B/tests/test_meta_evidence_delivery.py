"""Evidence stays complete; model input, persistent skills and outputs are bounded."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import test_contracts  # project/runtime import paths
from test_append_memory import principle
from sia.task_meta.meta_harness.evidence_delivery import (
    EvidenceDelivery, prepare_delivery, canonical, INLINE_SKILL_BYTES)
from sia.task_meta.meta_harness.five_stage import validate_library
from sia.task_meta.meta_backends.input_budget import (
    MetaInputBudget, inventory, attach_archives, archive_descriptors)
from sia.task_meta.meta_backends.local_execution import _stage_files, _return_workspace


class EvidenceDeliveryTests(unittest.TestCase):
    def test_pairs_lossless_deduplicated_and_hashes_unchanged(self):
        trace = {'task_id': 't', 'source_id': 'source', 'source_hash': 'original',
                 'messages': [{'content': '证据' * 10000}], 'success': False}
        pair = {'before': trace, 'after': trace, 'success_delta': 0}
        outcome = {'pairs': [pair], 'candidate_attempts': [{'paired_outcome': {'pairs': [pair]}}],
                   'outcome_hash': 'immutable-original-hash'}
        original = copy.deepcopy(outcome)
        files = {'meta_input/internal_dev_comparison.json': canonical(outcome)}
        value = prepare_delivery({'outcome_review': outcome}, files)
        self.assertEqual(outcome, original)
        self.assertEqual(value['outcome_review']['outcome_hash'], 'immutable-original-hash')
        refs = [p for p in files if p.startswith('meta_input/evidence/objects/')]
        self.assertEqual(len(refs), 1)
        self.assertEqual(json.loads(files[refs[0]]), trace['messages'])
        self.assertLess(len(files['meta_input/internal_dev_comparison.json']), 4000)

    def test_skill_view_bounded_library_can_exceed_old_record_and_file_limits(self):
        library = {'schema_version': 1, 'records': [principle(f'skill.MODEL.{n}') for n in range(300)], 'revisions': []}
        validate_library(library)
        delivery = EvidenceDelivery()
        view = delivery.memory(library, 'MODEL')
        self.assertLessEqual(view['inline_bytes'], INLINE_SKILL_BYTES)
        self.assertLess(len(view['selected_records']), 300)
        index = json.loads(delivery.files['meta_input/skills/index.json'])
        self.assertEqual(len(index), 300)
        for item, record in zip(index, library['records']):
            self.assertEqual(json.loads(delivery.files[item['content_reference']['file']]), record)
        with patch('sia.task_meta.meta_harness.evidence_delivery.SKILL_LIBRARY_BYTES', 100):
            with self.assertRaisesRegex(ValueError, 'no history was deleted'):
                validate_library(library)

    def test_pager_reads_full_context_in_bounded_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {}
            value = prepare_delivery({'messages': ['abc' * 10000]}, files)
            for name, content in files.items():
                p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(content)
            command = ['python3', 'meta_input/read_evidence.py', 'meta_input/operation.json',
                       '--path', '["messages"]', '--expand', '--max-chars', '1000']
            result = json.loads(subprocess.check_output(command, cwd=root))
            self.assertEqual(len(result['text']), 1000)
            self.assertFalse(result['complete'])
            self.assertEqual(result['next_offset'], 1000)
            self.assertEqual(result['sha256'], hashlib.sha256(canonical(['abc' * 10000]).encode()).hexdigest())


class InputBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.budget = MetaInputBudget(evidence_bytes=8192, evidence_file_bytes=4096,
                                      skill_bytes=4096, control_bytes=4096)

    def put(self, name, content):
        p = self.root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(content)
        return p

    def test_independent_budget_boundaries_and_actionable_errors(self):
        self.put('meta_input/evidence/x', b'x' * 4096)
        self.put('meta_input/skills/x', b'x' * 4096)
        self.put('.meta_candidate.json', b'x' * 100)
        totals, _ = inventory(self.root, self.budget, 100)
        self.assertEqual(totals['output'], 100)
        p = self.put('meta_input/evidence/y', b'x' * 4097)
        with self.assertRaisesRegex(ValueError, 'file budget exceeded.*bytes=4097; limit=4096'):
            inventory(self.root, self.budget, 100)
        p.write_bytes(b'x' * 4096)
        inventory(self.root, self.budget, 100)
        self.put('meta_input/evidence/z', b'x')
        with self.assertRaisesRegex(ValueError, 'total budget exceeded'):
            inventory(self.root, self.budget, 100)

    def test_large_evidence_does_not_consume_output_budget_on_return(self):
        call = self.root / 'call'; child = self.root / 'child'
        self.put('call/codex_home/config.toml', b'fixture')
        self.put('call/schema.json', b'{}')
        self.put('call/workspace/AGENTS.md', b'fixture')
        data = b'x' * 2048
        self.put('call/workspace/meta_input/evidence/x', data)
        self.put('child/meta_input/evidence/x', data)
        self.put('child/AGENTS.md', b'fixture')
        self.put('child/.meta_candidate.json', b'{}')
        _stage_files(call, self.budget, 100)
        _return_workspace(child, call, 100, self.budget)
        self.assertEqual((call / 'workspace/meta_input/evidence/x').read_bytes(), data)
        self.assertTrue((call / 'workspace.input').exists())

    def test_input_mutation_and_links_rejected(self):
        call = self.root / 'call'; child = self.root / 'child'
        self.put('call/workspace/meta_input/evidence/x', b'original')
        self.put('child/meta_input/evidence/x', b'changed')
        with self.assertRaisesRegex(ValueError, 'Read-only Meta input changed'):
            _return_workspace(child, call, 100, self.budget)
        self.put('target', b'data')
        (child / 'link').symlink_to(self.root / 'target')
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            inventory(child, self.budget, 100)

    def test_full_training_archives_are_hash_bound_and_validation_excluded(self):
        row = dict(task_id='t', rollout_id=0, source_role='train_evolution',
                   purpose='evolution_train', split='evolve_train', round_id=1,
                   collection_stage='child_post_update', model_calls=[{'messages':['FULL CONTEXT']}])
        for n in (0, 1):
            self.put(f'feedback/gen_{n}/train_trajectories.jsonl', (json.dumps(row)+'\n').encode())
        descriptors = archive_descriptors(self.root, self.root/'feedback', 0)
        work = self.root/'work'
        self.put('work/meta_input/trajectory_archives.json', json.dumps(descriptors).encode())
        attach_archives(work, self.root, self.budget)
        index = json.loads((work/'meta_input/archives/index.json').read_text())
        self.assertEqual(len(index), 2)
        self.assertEqual(len(list((work/'meta_input/archives').glob('*.json'))), 2)  # index + deduplicated row
        ref = index[0]['content_reference']
        self.assertEqual(json.loads((work/ref['file']).read_bytes()), row)
        row['source_role'] = 'independent_validation'
        self.put('feedback/gen_0/train_trajectories.jsonl', (json.dumps(row)+'\n').encode())
        with self.assertRaisesRegex(ValueError, 'Non-training'):
            attach_archives(work, self.root, self.budget)


if __name__ == '__main__':
    unittest.main()
