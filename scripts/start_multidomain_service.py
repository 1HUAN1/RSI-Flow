"""Explicit, foreground local Qwen3 service with a GPU/port lease."""
import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sia.task_meta.file_lock import exclusive_lock
from sia.task_meta.pipeline import PROJECT, load_config
from sia.task_meta.storage import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--enable-gpu', action='store_true')
    args = parser.parse_args()
    if not args.enable_gpu:
        parser.error('GPU serving must be explicitly enabled with --enable-gpu')
    config = load_config(args.config)
    endpoint = urlsplit(config.task_base_url)
    if endpoint.hostname != '127.0.0.1' or not endpoint.port:
        raise ValueError('Dedicated loopback IPv4 port required')
    with socket.socket() as check:
        check.bind(('127.0.0.1', endpoint.port))
    memory = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, check=True)
    free = {int(line.split(',')[0]): int(line.split(',')[1]) for line in memory.stdout.splitlines()}
    if free[config.inference_gpu] < 20000:
        raise RuntimeError('Selected inference GPU lacks the declared 20 GB free budget; other processes are preserved')
    lease = PROJECT / 'local_baseline/resource_leases' / f'inference_gpu_{config.inference_gpu}.json'
    lease.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(lease.parent / f'gpu_{config.inference_gpu}.lock'), exclusive_lock(lease.parent / f'port_{endpoint.port}.lock'):
        save_json(lease, {'pid': os.getpid(), 'gpu': config.inference_gpu, 'port': endpoint.port,
                         'checkpoint': config.task_checkpoint, 'owner': 'rsiH_multidomain', 'status': 'running'})
        environment = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in ('KEY','TOKEN','PASSWORD','SECRET','AUTH'))}
        environment.update({'CUDA_VISIBLE_DEVICES': str(config.inference_gpu), 'TASK_META_BASE_MODEL': config.task_checkpoint,
                            'PYTHONPATH': str(PROJECT)})
        try:
            subprocess.run([config.trainer_python, '-m', 'uvicorn', 'sia.task_meta.serve_gpu:app',
                            '--host', '127.0.0.1', '--port', str(endpoint.port)], cwd=PROJECT, env=environment, check=True)
        finally:
            save_json(lease, {'pid': os.getpid(), 'gpu': config.inference_gpu, 'port': endpoint.port,
                             'owner': 'rsiH_multidomain', 'status': 'stopped'})


if __name__ == '__main__':
    main()
