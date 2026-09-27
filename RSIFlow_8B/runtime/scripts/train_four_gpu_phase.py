"""One fixed four-rank SFT operation between four-GPU inference phases."""
import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sia.task_meta.file_lock import exclusive_lock
from sia.task_meta.gpu_phases import ROOT, start_services, stop_services
from sia.task_meta.pipeline import load_config
from sia.task_meta.resources import training_gpu_lease
from sia.task_meta.storage import save_json
from train_task_meta_sft import contained_path, read_positive_rows, resolve_training_base


def main():
    def interrupted(signum, frame):
        raise TimeoutError(f'Training phase interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request-dir', type=Path, required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.gpu_execution != 'phased_four': raise ValueError('Four-GPU authorization/config required')
    runs_root = Path(config.output_root) / 'runs'
    directory = contained_path(runs_root, args.request_dir)
    request = json.loads((directory / 'training_request.json').read_text())
    base = resolve_training_base(request)
    read_positive_rows(directory, request)  # Preserve positive-only filtering before any GPU side effect.
    receipt = directory / 'gpu_phase_receipt.json'
    if receipt.exists():
        raise RuntimeError('Phase receipt already exists; audit training effects before any recovery')
    state = {'status': 'prepared', 'gpus': [0, 1, 2, 3], 'started_at': time.time(),
             'checkpoint_before': str(base), 'training_retried': False}
    save_json(receipt, state)
    with exclusive_lock(Path(config.output_root) / 'locks/four_gpu_phase.lock'):
        stop_services(config)
        state.update(status='inference_stopped', stopped_at=time.time())
        save_json(receipt, state)
        trained = None
        failure = None
        try:
            with contextlib.ExitStack() as leases:
                for gpu in range(4): leases.enter_context(training_gpu_lease(gpu))
                command = [config.trainer_python, '-m', 'torch.distributed.run', '--standalone',
                           '--nnodes=1', '--nproc_per_node=4', '--max_restarts=0',
                           str(ROOT / 'scripts/train_task_meta_sft.py'), '--request-dir', str(directory),
                           '--runs-dir', str(runs_root), '--base-url', config.task_base_url, '--defer-serving']
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='2')
                # No candidate or provider credentials reach rank processes.
                environment = {k: v for k, v in environment.items()
                               if not any(w in k.upper() for w in ('KEY', 'TOKEN', 'PASSWORD', 'SECRET', 'AUTH'))}
                child = subprocess.Popen(command, cwd=directory, env=environment, start_new_session=True)
                state.update(status='training_dispatched', training_pid=child.pid, command=command,
                             training_started_at=time.time())
                save_json(receipt, state)
                try:
                    result = child.wait(timeout=config.training_timeout_seconds)
                except BaseException:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL); child.wait()
                    raise
                if result != 0: raise RuntimeError(f'Four-rank trainer exited with status {result}')
                trained = json.loads((directory / 'checkpoint_trained.json').read_text())
                state.update(status='training_complete', training_finished_at=time.time())
                save_json(receipt, state)
        except BaseException as exc:
            failure = exc
            state.update(status='training_failed_requires_audit', error_type=type(exc).__name__)
            save_json(receipt, state)
        # Failed training never publishes its candidate. Restore the last valid checkpoint.
        checkpoint = Path(trained['checkpoint_path']) if trained else base
        bindings = start_services(config, checkpoint)
        state.update(restored_checkpoint=str(checkpoint), inference_ready_at=time.time(),
                     replica_gpus=[r['gpu'] for r in bindings])
        if failure:
            save_json(receipt, state)
            raise failure
        metrics_path = Path(trained['training_metrics'])
        metrics = json.loads(metrics_path.read_text())
        if metrics['world_size'] != 4 or {r['rank'] for r in metrics['rank_records']} != set(range(4)):
            raise RuntimeError('Missing rank-specific training evidence')
        metrics['serving_status'] = 'ready'
        save_json(metrics_path, metrics)
        trained['training_summary'] += ' Four local DDP ranks; all four inference replicas verified.'
        save_json(directory / 'checkpoint.json', trained)
        state.update(status='completed', completed_at=time.time())
        save_json(receipt, state)


if __name__ == '__main__': main()
