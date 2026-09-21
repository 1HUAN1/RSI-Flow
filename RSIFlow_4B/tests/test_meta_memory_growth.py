"""Cumulative memory growth across three rounds, without API calls."""
import copy
import json
from pathlib import Path
import unittest

import test_contracts
from test_append_memory import AppendMemory, principle
from sia.task_meta.meta_harness.bundle import validate_files
from sia.task_meta.meta_harness.evidence_delivery import canonical
from sia.task_meta.meta_harness.five_stage import materialize
from sia.task_meta.round_evolution import validate_memory_append
from sia.task_meta.types import MetaHarnessUpdate


class MemoryGrowth(unittest.TestCase):
    def test_three_round_append_preserves_snapshots_and_rejects_duplicate(self):
        fixture = AppendMemory(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        snapshots = [json.loads(fixture.files['principles.json'])]
        for number in (1, 2, 3):
            text = fixture.update.model_dump_json().replace('r1', f'r{number}').replace(
                'experience1', f'experience{number}').replace('decision1', f'decision{number}')
            update = MetaHarnessUpdate.model_validate_json(text)
            validate_memory_append(update, fixture.meta, 'MODEL', f'decision{number}', f'experience{number}')
            files, _ = materialize(update, fixture.files,
                allowed_evidence={'pair1', f'decision{number}', f'experience{number}'})
            library = json.loads(files['principles.json'])
            self.assertEqual(library['records'][:-2], snapshots[-1]['records'])
            snapshots.append(copy.deepcopy(library))
            for name, value in files.items():
                (fixture.root/name).write_text(value)
            fixture.files = files
            with self.assertRaises(ValueError):
                validate_memory_append(update, fixture.meta, 'MODEL', f'decision{number}', f'experience{number}')
        self.assertEqual([len(s['records']) for s in snapshots], [1, 3, 5, 7])

    def test_large_skill_file_passes_bundle_validation(self):
        root = Path(__file__).resolve().parents[1]/'runtime/meta_harness/seed'
        files = {p.name:p.read_text() for p in root.iterdir() if p.is_file()}
        library = {'schema_version':1, 'records':[principle(f'skill.MODEL.{n}') for n in range(300)], 'revisions':[]}
        files['principles.json'] = canonical(library)
        self.assertGreater(len(files['principles.json'].encode()), 128000)
        self.assertEqual(validate_files(files), 'meta-bundle-v3')


if __name__ == '__main__':
    unittest.main()
