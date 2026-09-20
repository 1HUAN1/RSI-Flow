#!/usr/bin/env python3
"""Export the common Task start for RSI, SIA and HarnessForge comparisons."""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sia.task_meta.durable import value_hash
from sia.task_meta.pipeline import load_config, project_path
from sia.task_meta.seed import load_seed
from sia.task_meta.storage import digest, save_json


def export(config, destination):
    destination = Path(destination)
    data = project_path(config.data_dir)
    seed_path = project_path(config.seed_harness)
    seed = load_seed(seed_path)
    model = json.loads((data / 'model_identity.json').read_text())
    record = {'schema_version': 1, 'task_model': model, 'harness_sha256': digest(seed_path),
              'artifact_initial_manifest': [], 'artifact_initial_state': 'empty',
              'dataset_manifest_sha256': digest(data / 'tasks.sqlite'),
              'retrieval_manifest': json.loads((data / 'search.sqlite.manifest.json').read_text()),
              'budget': seed['budget'], 'random_seed': config.seed,
              'probe_per_domain': config.probe_per_domain, 'window_quotas': config.window_quotas,
              'rollouts_per_task': config.rollouts_per_task, 'task_enable_thinking': config.task_enable_thinking,
              'reference': seed['reference'], 'baseline_repositories': {'SIA': 'https://github.com/hexo-ai/sia',
                     'HarnessForge': 'https://github.com/mingju-c/HarnessForge'},
              'alignment_status': 'controlled_common_start_exported; baseline executions not yet performed',
              'adaptations': 'See seed_harness/README.md; native baseline algorithms remain separately identified'}
    record['initialization_hash'] = value_hash(record)
    if destination.exists():
        if json.loads((destination / 'initialization.json').read_text()) != record:
            raise ValueError('Initialization export is immutable; use a new destination for another protocol')
        if digest(destination / 'seed.json') != record['harness_sha256'] or any((destination / 'artifacts').iterdir()):
            raise ValueError('Exported seed or empty initial artifacts were modified')
        return record
    destination.mkdir(parents=True)
    shutil.copy2(seed_path, destination / 'seed.json')
    (destination / 'artifacts').mkdir()
    save_json(destination / 'initialization.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/multidomain-dev.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = export(load_config(args.config), args.output)
    print(json.dumps({'status': result['alignment_status'], 'initialization_hash': result['initialization_hash']}, indent=2))
