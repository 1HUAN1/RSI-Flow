"""Fixed EnvScaler validation subset and independent GPU rollout workers."""
import json
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from urllib.request import Request, ProxyHandler, build_opener

from sia.task_meta.data import ManifestStore, DOMAINS


class EnvScalerValidationStore(ManifestStore):
    def __init__(self, path, train_limit, probe_limit, *, selection_seed=None, repeat=False):
        self.selection_seed, self.repeat = selection_seed, repeat
        super().__init__(path)
        self.parent_counts = super().counts()
        population = list(self.conn.execute(
            "SELECT * FROM tasks WHERE domain='tool_use' AND split='evolve_train' ORDER BY seq"))
        if train_limit > len(population):
            raise ValueError('Requested training subset exceeds registered training split')
        selected = (population[:train_limit] if selection_seed is None else
                    random.Random(selection_seed).sample(population, train_limit))
        self.train = [self._record(row) for row in selected]
        self.fixed_probe = [self._record(row) for row in self.conn.execute(
            "SELECT t.* FROM tasks t JOIN probe p USING(task_id) WHERE domain='tool_use' ORDER BY order_key,task_id LIMIT ?", (probe_limit,))]
        if len(self.train) != train_limit or len(self.fixed_probe) != probe_limit:
            raise ValueError('Declared validation subset exceeds the registered split/probe')
        if {t.task_id for t in self.train} & {t.task_id for t in self.fixed_probe}:
            raise ValueError('Validation train/probe overlap')

    def window(self, cursor, quotas):
        following = dict.fromkeys(DOMAINS, 0) | (cursor or {})
        if any(following[d] for d in DOMAINS if d != 'tool_use'):
            raise ValueError('Inactive validation domain cursor changed')
        start = following['tool_use']
        if self.repeat and quotas['tool_use'] != len(self.train):
            raise ValueError('Fixed subset requires one complete selected cohort per generation')
        rows = list(self.train) if self.repeat else self.train[start:start + quotas['tool_use']]
        following['tool_use'] = min(len(self.train), start + len(rows))
        return rows, following

    def probe(self):
        return list(self.fixed_probe)

    def counts(self):
        result = {d: dict(evolve_train=0, search_dev=0, probe=0) for d in DOMAINS}
        result['tool_use'] = dict(evolve_train=len(self.train), search_dev=len(self.fixed_probe), probe=len(self.fixed_probe))
        return result

    def scope_identity(self):
        return {'name': 'envscaler_validation', 'train_ids': [t.task_id for t in self.train],
                'probe_ids': [t.task_id for t in self.fixed_probe], 'parent_counts': self.parent_counts,
                'selection': ('seeded random sample without replacement from registered training split' if self.selection_seed is not None else 'first immutable training seq'),
                'selection_seed': self.selection_seed, 'repeat_fixed_subset': self.repeat,
                'probe_selection': 'unchanged first pre-registered probe order; no performance selection'}


def sync_replicas(state, replicas):
    records = []
    opener = build_opener(ProxyHandler({}))
    for replica in replicas:
        url = replica['base_url'].rstrip('/')
        with opener.open(url.removesuffix('/v1') + '/health', timeout=30) as response:
            health = json.load(response)
        if not health.get('ready') or health.get('visible_devices') != str(replica['gpu']):
            raise ValueError('Replica device/readiness does not match the fixed configuration')
        expected = {'checkpoint_path': state.checkpoint_path, 'weights': state.checkpoint_manifest}
        binding = health.get('bindings', {}).get(state.model_ref)
        if binding is None:
            request = Request(url + '/local/checkpoints', data=json.dumps({
                'model_ref': state.model_ref, 'checkpoint_path': state.checkpoint_path}).encode(),
                headers={'Content-Type': 'application/json'})
            with opener.open(request, timeout=600) as response:
                loaded = json.load(response)
            if loaded.get('ready') is not True or loaded.get('model_ref') != state.model_ref:
                raise ValueError('Replica failed to load current checkpoint')
            binding = loaded.get('binding')
        if binding != expected:
            raise ValueError('Replica checkpoint fingerprint mismatch')
        records.append({**replica, 'binding': binding})
    return records


_worker = None


def _initialize(executor, state, spec, directory, assets, probe, sources, counter, request_slots):
    global _worker
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    replica = index % len(executor.replicas)
    executor._replica_endpoint = executor.replicas[replica]['base_url']
    original_factory = executor.model_factory
    slot = request_slots[replica]
    def limited_factory(task_state, base_url=None):
        client = original_factory(task_state, base_url=base_url)
        def complete(*args, **kwargs):
            # Bound HTTP queueing; GPU generation and request seeds stay serialized.
            with slot:
                return client(*args, **kwargs)
        complete.enable_thinking = getattr(client, 'enable_thinking', False)
        return complete
    executor.model_factory = limited_factory
    _worker = (executor, state, spec, directory, assets, probe, sources)


def _rollout(job):
    executor, state, spec, directory, assets, probe, sources = _worker
    task, number = job
    return executor._one(state, spec, task, number, directory, assets,
                         probe=probe, artifact_sources=sources)


def bounded_results(pool, function, jobs, capacity):
    """Refill on any completion; preserve result order without a batch barrier."""
    if capacity < 1:
        raise ValueError('Positive scheduling capacity required')
    iterator = iter(enumerate(jobs))
    pending, results = {}, {}
    exhausted = False
    while pending or not exhausted:
        while not exhausted and len(pending) < capacity:
            try:
                index, job = next(iterator)
            except StopIteration:
                exhausted = True
                break
            pending[pool.submit(function, job)] = index
        if pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                results[pending.pop(future)] = future.result()
    return [results[i] for i in range(len(results))]


def worker_count(replicas):
    cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    return replicas * min(4, max(1, cores // (2 * replicas)))


def run_parallel(executor, state, spec, jobs, directory, assets, probe, sources):
    context = multiprocessing.get_context('fork')
    counter = context.Value('i', 0)
    workers = worker_count(len(executor.replicas))
    request_slots = [context.BoundedSemaphore(2) for _ in executor.replicas]
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
            initializer=_initialize, initargs=(executor, state, spec, directory, assets, probe, sources,
                                             counter, request_slots)) as pool:
        return bounded_results(pool, _rollout, jobs, 2 * workers)
