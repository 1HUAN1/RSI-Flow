"""Container-aware CPU capacity and read-only kernel resource counters."""
import math
import os
from pathlib import Path


def available_cpu_count(cgroup_root=Path('/sys/fs/cgroup')):
    cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    try:
        quota, period = (Path(cgroup_root) / 'cpu.max').read_text().split()
        if quota != 'max':
            cores = min(cores, max(1, math.floor(int(quota) / int(period))))
    except (OSError, ValueError, ZeroDivisionError):
        pass  # Non-cgroup hosts use the process affinity.
    return max(1, cores)


def resource_snapshot(cgroup_root=Path('/sys/fs/cgroup')):
    result = {'effective_cpu_count': available_cpu_count(cgroup_root)}
    for name in ('cpu.max', 'cpu.stat', 'memory.max', 'memory.current', 'memory.events',
                 'pids.max', 'pids.current', 'pids.events'):
        try:
            result[name] = (Path(cgroup_root) / name).read_text().strip()
        except OSError:
            result[name] = None
    return result
