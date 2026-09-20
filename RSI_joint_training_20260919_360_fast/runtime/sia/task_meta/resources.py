"""Exclusive local GPU ownership for this controller's own training jobs."""
import contextlib
import json
import os
import subprocess
from pathlib import Path

from .types import DecisionConstraintError


@contextlib.contextmanager
def training_gpu_lease(gpu, *, minimum_free_mb=40000):
    import fcntl
    root = Path(__file__).resolve().parents[2] / 'local_baseline/resource_leases'
    root.mkdir(parents=True, exist_ok=True)
    with (root / f'gpu_{gpu}.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DecisionConstraintError('MODEL temporarily unavailable: this project already owns the training GPU') from exc
        try:
            status = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=15, check=True)
            free = {int(line.split(',')[0]): int(line.split(',')[1]) for line in status.stdout.splitlines()}
            if free.get(gpu, 0) < minimum_free_mb:
                raise DecisionConstraintError(f'MODEL temporarily unavailable: GPU {gpu} needs {minimum_free_mb} MB free; preserve existing processes')
            lock.seek(0)
            lock.truncate()
            json.dump({'pid': os.getpid(), 'gpu': gpu, 'minimum_free_mb': minimum_free_mb}, lock)
            lock.flush()
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
