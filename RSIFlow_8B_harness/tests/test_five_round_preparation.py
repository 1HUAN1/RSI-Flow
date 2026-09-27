"""Preparation indexes all five training packs; no API, model or real dataset mutation."""
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import data_protocol
from evolution_protocol import make_manifest


def test_prepare_protocol_counts_and_indexes_all_five_packs(tmp_path, monkeypatch):
    config = json.loads((PROJECT / 'configs/train_180_a0_v1.json').read_text())
    corpus = tmp_path / 'public.sqlite'
    corpus.touch()
    corpus.with_suffix('.sqlite.manifest.json').write_text('{}')
    config.update(retrieval_index=str(corpus), exposure_ledger=None)
    packs = []
    for number in range(1, 6):
        tasks = [{'task_id': f'B{number}-{index}', 'source': 'test', 'domain': 'code'}
                 for index in range(180)]
        packs.append(make_manifest('train_evolution', number, tasks))
    packs += [make_manifest('independent_validation', None, []), make_manifest('final_test', None, [])]
    monkeypatch.setattr(data_protocol, 'catalog', lambda *args: [])
    monkeypatch.setattr(data_protocol, 'scan', lambda *args: ([], {}))
    monkeypatch.setattr(data_protocol, 'allocate', lambda *args: (packs, []))
    monkeypatch.setattr(data_protocol, 'materialize', lambda value, directory: value)
    indexed = []
    monkeypatch.setattr(data_protocol, 'native_index', lambda output, rounds, config: indexed.extend(rounds))
    report = data_protocol.prepare_protocol(config, {'benchmark_ids': []},
                                            tmp_path / 'training', tmp_path / 'evaluation')
    assert report['allocated_tasks'] == 900
    assert [pack['round_id'] for pack in indexed] == [1, 2, 3, 4, 5]
    assert sum(len(pack['tasks']) for pack in indexed) == 900
    assert all((tmp_path / 'training' / f'B{number}' / 'manifest.json').is_file()
               for number in range(1, 6))
