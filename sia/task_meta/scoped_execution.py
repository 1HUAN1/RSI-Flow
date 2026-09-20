"""Fixed EnvScaler validation subset and independent GPU rollout workers."""
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from urllib.request import Request, ProxyHandler, build_opener

from sia.task_meta.data import ManifestStore, DOMAINS


class EnvScalerValidationStore(ManifestStore):
    def __init__(self, path, train_limit, probe_limit):
        super().__init__(path)
        self.parent_counts = super().counts()
        self.train = [self._record(row) for row in self.conn.execute(
            "SELECT * FROM tasks WHERE domain='tool_use' AND split='evolve_train' ORDER BY seq LIMIT ?", (train_limit,))]
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
        rows = self.train[start:start + quotas['tool_use']]
        following['tool_use'] += len(rows)
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
                'selection': 'first immutable training seq and first pre-registered probe order; no performance selection'}


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


def _initialize(executor, state, spec, directory, assets, probe, sources, counter):
    global _worker
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    executor._replica_endpoint = executor.replicas[index]['base_url']
    _worker = (executor, state, spec, directory, assets, probe, sources)


def _rollout(job):
    executor, state, spec, directory, assets, probe, sources = _worker
    task, number = job
    return executor._one(state, spec, task, number, directory, assets,
                         probe=probe, artifact_sources=sources)


def run_parallel(executor, state, spec, jobs, directory, assets, probe, sources):
    # Separate processes preserve the existing sandbox pre-exec isolation path;
    # no threaded fork/preexec is introduced. Returned rows retain schedule order.
    context = multiprocessing.get_context('fork')
    counter = context.Value('i', 0)
    with ProcessPoolExecutor(max_workers=len(executor.replicas), mp_context=context,
            initializer=_initialize, initargs=(executor, state, spec, directory, assets, probe, sources, counter)) as pool:
        return list(pool.map(_rollout, jobs))
