"""Opt-in current-Qwen Task smoke: one train and one registered probe per domain.

Uses the existing local model service; never starts serving, Meta calls or SFT.
Outputs are diagnostic smoke subsets, not official experiment measurements.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sia.task_meta.data import DOMAINS, ManifestStore
from sia.task_meta.meta_harness.bundle import reject_links
from sia.task_meta.pipeline import PROJECT, adapter_factory, load_config, model_identity, project_path
from sia.task_meta.pipeline_execution import MultiDomainExecutor
from sia.task_meta.storage import digest, save_json
from sia.task_meta.task_client import LocalTaskClient
from sia.task_meta.task_harness import harness_identity, load_harness
from sia.task_meta.types import TaskAgentState


class _SmokeStore:
    """Only bound existing manifest selection; never resplit or rewrite data."""

    def __init__(self, store):
        self.store = store
        selected = {}
        for task in store.probe():
            selected.setdefault(task.domain, task)
        if set(selected) != set(DOMAINS):
            raise ValueError('Smoke requires one existing registered probe in every domain')
        self._probe = [selected[domain] for domain in DOMAINS]

    def window(self, cursor, quotas):
        if any(cursor.values()) or quotas != dict.fromkeys(DOMAINS, 1):
            raise ValueError('Smoke accepts only the first single-item window per domain')
        records, following = self.store.window(cursor, quotas)
        if len(records) != len(DOMAINS) or {record.domain for record in records} != set(DOMAINS):
            raise ValueError('Smoke requires exactly one existing training item in every domain')
        return records, following

    def probe(self):
        return list(self._probe)

    def coverage(self, cursor):
        return self.store.coverage(cursor)


def run_smoke(config_path, output, *, enabled=False, limit_per_domain=1, max_model_calls=16):
    if not enabled:
        raise ValueError('Task GPU inference is disabled; pass --enable-inference explicitly')
    if type(limit_per_domain) is not int or limit_per_domain != 1:
        raise ValueError('Smoke allows exactly one training item and one probe per domain')
    if type(max_model_calls) is not int or not 1 <= max_model_calls <= 32:
        raise ValueError('Smoke model calls per rollout must be in [1, 32]')
    config = load_config(config_path)
    lexical_output = Path(output) if Path(output).is_absolute() else PROJECT / output
    reject_links(lexical_output)
    output = project_path(lexical_output)
    if output == PROJECT / 'runs' or not output.is_relative_to(PROJECT / 'runs'):
        raise ValueError('Smoke outputs must use a new directory under project/runs')
    if output.exists():
        raise FileExistsError('Smoke requires a new output directory; reconcile historical receipts separately')
    seed_path = project_path(config.seed_harness)
    if load_harness(seed_path)['schema_version'] != 2:
        raise ValueError('This smoke verifies the explicit five-part v2 Task Harness')
    output.mkdir(parents=True, exist_ok=False)
    store, index = None, None
    try:
        identity = model_identity(config.task_checkpoint)
        store = ManifestStore(project_path(config.data_dir) / 'tasks.sqlite')
        store.validate_sources()
        selected_store = _SmokeStore(store)
        initial = output / 'gen_0'
        initial.mkdir()
        shutil.copy2(seed_path, initial / 'seed.json')
        state = TaskAgentState(0, identity['path'], str(initial / 'seed.json'),
                               checkpoint_path=identity['path'], checkpoint_manifest=identity['weights'])
        calls_per_rollout = min(max_model_calls, config.model_call_limit)
        protocol = {'status': 'running', 'kind': 'task_harness_smoke_subset', 'official_experiment_result': False,
                    'config_sha256': digest(Path(config_path)), 'harness': harness_identity(initial / 'seed.json'),
                    'model_identity': identity, 'model_calls_per_rollout': calls_per_rollout,
                    'max_total_model_calls': 6 * calls_per_rollout, 'max_output_tokens': config.max_output_tokens,
                    'train_items_per_domain': 1, 'probe_items_per_domain': 1, 'rollouts_per_task': 1,
                    'probe_ids': [task.task_id for task in selected_store.probe()],
                    'meta_api_calls': 0, 'training_run': False, 'model_updates': 0,
                    'service_started': False, 'seed_budget_modified': False}
        save_json(output / 'smoke_protocol.json', protocol)
        factory, index = adapter_factory(config)
        executor = MultiDomainExecutor(selected_store, factory,
            lambda current: LocalTaskClient(current, config.task_base_url, timeout=config.task_timeout,
                                            enable_thinking=config.task_enable_thinking),
            quotas=dict.fromkeys(DOMAINS, 1), rollouts_per_task=1, probe_rollouts=1,
            seed=config.seed, model_call_limit=calls_per_rollout, max_output_tokens=config.max_output_tokens)
        result = executor.execute(state, initial)
        summary = {**protocol, 'status': 'completed', 'smoke_subset_performance': result.performance,
                   'cost': result.cost, 'train_rollouts': len(result.trajectories), 'probe_rollouts': 3,
                   'sft_executed': False}
        save_json(output / 'smoke_result.json', summary)
        return summary
    except Exception as exc:
        save_json(output / 'smoke_failure.json', {'status': 'failed_or_pending', 'error_type': type(exc).__name__,
            'reason': str(exc), 'meta_api_calls': 0, 'training_run': False,
            'automatic_retry_permitted': False, 'note': 'Keep receipts and reconcile incomplete inference before any retry.'})
        raise
    finally:
        if index is not None:
            index.close()
        if store is not None:
            store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Existing pipeline JSON configuration')
    parser.add_argument('--output', required=True, help='New output directory under project/runs')
    parser.add_argument('--enable-inference', action='store_true', help='Explicitly permit local Task GPU inference')
    parser.add_argument('--limit-per-domain', type=int, choices=[1], default=1)
    parser.add_argument('--max-model-calls', type=int, choices=range(1, 33), default=16,
                        help='Maximum requests per rollout; also bounded by the existing config')
    args = parser.parse_args(argv)
    result = run_smoke(args.config, args.output, enabled=args.enable_inference,
                       limit_per_domain=args.limit_per_domain, max_model_calls=args.max_model_calls)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
