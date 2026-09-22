"""Run the next frozen Task baseline while Meta learns, without exposing it to Meta.

The child owns a separate store/protocol and writes native replayable receipts.
The controller joins it before validation or another GPU phase, and only routes
after committing the preceding Meta update.
"""
import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .durable import load_task, task_hash
from .evolution_protocol import freeze
from .file_lock import exclusive_lock
from .gpu_phases import process_identity
from .storage import digest, save_json


class EarlyRollout:
    def __init__(self, config, root):
        self.config, self.root = config, Path(root)
        self.number = None

    def start(self, number, task):
        if number >= self.config.max_generations:
            return
        state = copy.deepcopy(task)
        state.generation = number
        directory = self.root / f'round_{number}' / 'early_rollout'
        directory.mkdir(parents=True, exist_ok=True)
        request = {'number': number, 'task': asdict(state), 'task_hash': task_hash(state),
                   'protocol_sha256': digest(self.root / 'protocol.json')}
        from .completed_rollout_reuse import completed_request_reusable
        if completed_request_reusable(self.root, self.root / f'round_{number}/before/gen_{number}', request['task_hash']):
            self.number = number
            print(f'[early-rollout] round={number + 1}: authorized completed baseline retained', flush=True)
            return
        freeze(directory / 'request.json', request)
        self.number = number
        with exclusive_lock(directory / 'launch.lock'):
            status_path = directory / 'status.json'
            if status_path.exists():
                status = json.loads(status_path.read_text())
                if status['status'] == 'completed':
                    return
                if status.get('process') and process_identity(status['process']['pid']) == status['process']:
                    return
            launcher = directory / 'launcher.json'
            if launcher.exists():
                owner = json.loads(launcher.read_text()).get('process')
                if owner and process_identity(owner['pid']) == owner:
                    return
            env = {k: v for k, v in os.environ.items()
                   if not any(word in k.upper() for word in ('TOKEN', 'SECRET', 'PASSWORD', 'API_KEY'))}
            env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
            with (directory / 'worker.log').open('ab', buffering=0) as log:
                child = subprocess.Popen([self.config.trainer_python, '-u', '-m',
                    'sia.task_meta.early_rollout', str(self.root), str(number)],
                    env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    start_new_session=True)
            save_json(directory / 'launcher.json', {'pid': child.pid, 'process': process_identity(child.pid)})
            print(f'[early-rollout] round={number + 1} pid={child.pid}; routing waits for committed Meta', flush=True)

    def wait(self):
        if self.number is None:
            return
        directory = self.root / f'round_{self.number}' / 'early_rollout'
        while True:
            status = json.loads((directory / 'status.json').read_text()) if (directory / 'status.json').exists() else {}
            if status.get('status') == 'completed':
                self.number = None
                return
            if status.get('status') == 'failed':
                raise RuntimeError('Early rollout failed; retained receipts allow restart: ' + str(directory))
            owner = status.get('process') or json.loads((directory / 'launcher.json').read_text())['process']
            if not owner or process_identity(owner['pid']) != owner:
                raise RuntimeError('Early rollout worker exited before completion: ' + str(directory))
            time.sleep(1)


def worker(root, number):
    from .pipeline import PipelineConfig, adapter_factory, project_path, source_identity
    from .pipeline_execution import MultiDomainExecutor
    from .task_client import LocalTaskClient
    from .durable import DurableExecutor, StageJournal
    from .round_evolution import RoundProtocol, RoundStore
    from .sequential_loop import write_evaluation
    from .gpu_phases import ensure_services
    root = Path(root)
    directory = root / f'round_{number}' / 'early_rollout'
    with exclusive_lock(directory / 'worker.lock'):
        request = json.loads((directory / 'request.json').read_text())
        if digest(root / 'protocol.json') != request['protocol_sha256']:
            raise ValueError('Early rollout protocol changed')
        task = load_task(request['task'])
        if task_hash(task) != request['task_hash']:
            raise ValueError('Early rollout frozen Task changed')
        status = {'status': 'running', 'process': process_identity(os.getpid()), 'round': number + 1}
        save_json(directory / 'status.json', status)
        store = index = None
        try:
            saved_protocol = json.loads((root / 'protocol.json').read_text())
            config = PipelineConfig.model_validate(saved_protocol['config'])
            if saved_protocol['controller'] != source_identity():
                raise ValueError('Early rollout source differs from the authorized controller')
            if not 0 < number < config.max_generations:
                raise ValueError('Early rollout must be a subsequent allocated training round')
            deployment = json.loads((root / f'round_{number-1}/deployment.json').read_text())
            deployed = load_task(deployment['task_after'])
            deployed.generation = number
            if task_hash(deployed) != task_hash(task):
                raise ValueError('Early rollout must use the actual deployed or retained Task')
            store = RoundStore(project_path(config.data_dir) / 'tasks.sqlite')
            store.select_round(number + 1)
            factory, index = adapter_factory(config)
            journal = StageJournal(directory)
            executor = DurableExecutor(MultiDomainExecutor(store, factory,
                lambda state, base_url=None: LocalTaskClient(state, base_url or config.task_base_url,
                    timeout=config.task_timeout, enable_thinking=config.task_enable_thinking),
                quotas=config.window_quotas, rollouts_per_task=config.rollouts_per_task,
                probe_rollouts=config.probe_rollouts, seed=config.seed,
                model_call_limit=config.model_call_limit, max_output_tokens=config.max_output_tokens,
                journal=journal, replicas=config.task_replicas), journal)
            executor.executor.artifact_evaluation = config.artifact_evaluation
            protocol = RoundProtocol(config, root, executor)
            baseline = root / f'round_{number}' / 'before' / f'gen_{number}'
            protocol.bind_execution(task, baseline, 'parent_pre_update')
            if config.gpu_execution == 'phased_four' and not (baseline / 'execution_receipt.json').exists():
                ensure_services(config, task.checkpoint_path or task.model_ref)
            result = executor.execute(task, baseline)
            write_evaluation(baseline, task, result)
            save_json(directory / 'status.json', {**status, 'status': 'completed',
                'execution_receipt_sha256': digest(baseline / 'execution_receipt.json')})
        except BaseException as exc:
            save_json(directory / 'status.json', {**status, 'status': 'failed',
                'error_type': type(exc).__name__, 'reason': str(exc)})
            raise
        finally:
            if store is not None:
                store.close()
            if index is not None:
                index.close()


if __name__ == '__main__':
    worker(sys.argv[1], int(sys.argv[2]))
