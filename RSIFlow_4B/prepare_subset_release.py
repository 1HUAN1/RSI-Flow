"""Freeze a smaller task allocation, never import rollout/cache/Memory state."""
import argparse
from collections import Counter
from pathlib import Path

from data_protocol import balanced_sample, materialize, native_index
from evolution_protocol import (assert_disjoint, file_hash, freeze, make_manifest,
                                manifest, read, training_quotas)


def select_round(parent, quotas, seed, environment_fraction=0.1):
    tasks = []
    for source, count in quotas.items():
        cap = max(1, int(count * environment_fraction)) if source == 'envscaler' else None
        tasks.extend(balanced_sample([t for t in parent['tasks'] if t['source'] == source],
                                     count, seed + parent['round_id'], cap=cap))
    return make_manifest('train_evolution', parent['round_id'], tasks,
                         split_seed=seed, feedback_scope='all_B_r',
                         allocation_parent_hash=parent['manifest_hash'],
                         selection_policy='source_strata_stable_hash_no_outcomes',
                         rollout_policy='fresh_no_trajectory_or_memory_import')


def prepare(config, parent_root):
    quotas = training_quotas(config)
    destination = Path(config['frozen_data_dir']).resolve()
    if destination.exists():
        raise FileExistsError('Never overwrite a frozen data release: ' + str(destination))
    parents = [manifest(Path(parent_root)/f'B{r}/manifest.json', 'train_evolution', r)
               for r in (1, 2, 3)]
    validation = manifest(Path(config['validation_vault'])/'validation/manifest.json',
                          'independent_validation', None)
    selected = [select_round(p, quotas, config['split_seed'],
                             config['grouping']['max_environment_fraction_per_round'])
                for p in parents]
    assert_disjoint([*selected, validation])
    verified = set()
    for value in selected:
        for row in value['tasks']:
            identity = (row['record_file'], row['record_sha256'])
            if identity not in verified:
                if file_hash(identity[0]) != identity[1]:
                    raise ValueError('Source task pack changed: ' + identity[0])
                verified.add(identity)
    destination.mkdir(parents=True)
    published = []
    for value in selected:
        source_view = {**value, 'tasks':[
            {**row, 'path':row['record_file'], 'offset':row['record_offset'],
             'length':row['record_length']} for row in value['tasks']]}
        folder = destination/f"B{value['round_id']}"
        packed = materialize(source_view, folder)
        freeze(folder/'manifest.json', packed)
        published.append(packed)
    assert_disjoint([*published, validation])
    native_index(destination, published, config)
    audit = dict(status='prepared_not_executed', allocated_tasks=sum(len(m['tasks']) for m in published),
                 rollout_imports=0, memory_imports=0, selection_uses_outcomes=False,
                 validation_manifest_hash=validation['manifest_hash'],
                 rounds=[dict(round_id=m['round_id'], tasks=len(m['tasks']),
                              by_source=dict(Counter(t['source'] for t in m['tasks'])),
                              manifest_hash=m['manifest_hash']) for m in published])
    freeze(destination/'preparation_audit.json', audit)
    freeze(destination/'protocol.json', config)
    return audit


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--parent-data-root', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(read(args.config), args.parent_data_root), ensure_ascii=False, indent=2))
