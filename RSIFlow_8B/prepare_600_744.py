"""Repack the existing 3000/744 cohorts; never sample new tasks or copy scores."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from data_protocol import native_index
from evolution_protocol import (assert_disjoint, file_hash, fingerprint, freeze,
                                make_manifest, manifest, read, training_quotas)
from extend_rounds import assert_task_disjoint

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'configs/train_600_5round_744.json'


def split_tasks(tasks, rounds=5, seed=42):
    """Preserve all task identities and Code interface ratio; balance source strata.

    EnvScaler scenarios may reuse an environment across rounds, as in the existing
    skill-only protocol. Tasks/content never repeat. Non-Tool families stay disjoint.
    """
    groups = defaultdict(list)
    for task in tasks:
        interface = task.get('strata', {}).get('interface') if task['source'] == 'deepcoder_taco' else None
        groups[(task['source'], interface)].append(task)
    batches = [[] for _ in range(rounds)]
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda t: fingerprint([seed, key, t['task_id']]))
        for index, task in enumerate(rows):
            number = index % rounds
            batches[number].append({**task, 'round_id': number + 1})
    result = [make_manifest('train_evolution', i + 1, rows, split_seed=seed)
              for i, rows in enumerate(batches)]
    assert_task_disjoint(result)
    assert_disjoint([{**b, 'tasks': [t for t in b['tasks'] if t['source'] != 'envscaler']}
                     for b in result])
    return result


def repack(value, destination):
    """Copy frozen record bytes, including complete ACE conversations, unchanged."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    sources = defaultdict(list)
    for row in value['tasks']:
        sources[row['source']].append(row)
    tasks = []
    verified = {}
    for source, rows in sorted(sources.items()):
        target = destination / (source + '.jsonl')
        packed = []
        with target.open('xb') as output:
            for row in rows:
                path = Path(row['record_file'])
                if path not in verified:
                    verified[path] = file_hash(path)
                if verified[path] != row['record_sha256']:
                    raise ValueError(f'Frozen source changed: {path}')
                with path.open('rb') as stream:
                    stream.seek(row['record_offset'])
                    payload = stream.read(row['record_length'])
                json.loads(payload)
                offset = output.tell()
                output.write(payload)
                packed.append({**row, 'record_file': str(target.resolve()),
                               'record_offset': offset, 'record_length': len(payload)})
        digest = file_hash(target)
        tasks.extend({**t, 'record_sha256': digest} for t in packed)
    return make_manifest(value['role'], value['round_id'], tasks,
                         **{k: v for k, v in value.items()
                            if k not in {'role', 'round_id', 'tasks', 'manifest_hash', 'schema_version'}})


def prepare(config_path=DEFAULT):
    config = read(config_path)
    output = Path(config['frozen_data_dir'])
    validation_path = Path(config['validation_manifest'])
    training = manifest(config['source_training_manifest'], 'train_evolution', 1)
    validation = manifest(config['source_validation_manifest'], 'independent_validation', None)
    assert_disjoint([training, validation])
    batches = split_tasks(training['tasks'], config['rounds'], config['split_seed'])
    quotas = training_quotas(config)
    if any(Counter(t['source'] for t in b['tasks']) != quotas for b in batches):
        raise ValueError('Source cohort cannot satisfy the requested per-round quotas')
    if Counter(t['source'] for t in validation['tasks']) != config['validation_limits']:
        raise ValueError('Validation cohort does not match configured counts')
    if output.exists() or validation_path.parent.exists():
        ready = output / 'READY.json'
        if ready.is_file():
            receipt = read(ready)
            if (receipt['source_training_hash'] == training['manifest_hash']
                    and receipt['source_validation_hash'] == validation['manifest_hash']
                    and receipt['config_hash'] == fingerprint(config)):
                return {**receipt, 'status': 'already_prepared'}
        raise FileExistsError('Use a new release directory; existing releases are never overwritten')
    packed = []
    for batch in batches:
        folder = output / f"B{batch['round_id']}"
        result = repack(batch, folder)
        freeze(folder / 'manifest.json', result)
        packed.append(result)
    heldout = repack(validation, validation_path.parent)
    freeze(validation_path, heldout)
    freeze(output / 'protocol.json', config)
    native_index(output, packed, config, link_retrieval=True)
    receipt = {'status': 'prepared_not_executed', 'rounds': config['rounds'],
               'training_tasks': sum(len(b['tasks']) for b in packed),
               'tasks_per_round': [len(b['tasks']) for b in packed],
               'validation_tasks': len(heldout['tasks']),
               'validation_by_source': config['validation_limits'],
               'source_training_hash': training['manifest_hash'],
               'source_validation_hash': validation['manifest_hash'],
               'config_hash': fingerprint(config),
               'cross_round_tool_environment_reuse': True,
               'evaluation_schedule': config['evaluation_schedule']}
    freeze(output / 'READY.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config), indent=2, ensure_ascii=False))
