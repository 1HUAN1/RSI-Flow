"""Explicit, per-run authorization to reuse a completed baseline after repair.

Never rewrites provenance or authorizes incomplete work. The native durable
executor still verifies the Task, result, checkpoint and every evidence file.
"""
from pathlib import Path
from .storage import digest
from .durable import value_hash
from .evolution_protocol import read, fingerprint


def _entry(root, directory):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    authorization = root/'recovery/completed_rollout_reuse.json'
    if not authorization.exists():
        return None
    relative = directory.relative_to(root).as_posix()
    proof = read(authorization)
    entry = proof['entries'].get(relative)
    if entry is None:
        return None
    from .pipeline import source_identity
    current = read(root/'protocol.json')
    if proof['controller'] != source_identity() or current['controller'] != proof['controller']:
        raise ValueError('Completed rollout reuse controller is not authorized')
    for name, expected in entry['files'].items():
        path = root/name
        if not path.resolve().is_relative_to(root) or path.is_symlink() or digest(path) != expected:
            raise ValueError('Completed rollout reuse evidence changed: '+name)
    old = read(root/entry['prior_protocol'])
    # Meta client source identity may change with the repaired controller, but
    # all experiment inputs, models, data and configuration must be identical.
    ignored = {'controller', 'hash', 'meta_identity'}
    if {k:v for k,v in old.items() if k not in ignored} != {k:v for k,v in current.items() if k not in ignored}:
        raise ValueError('Completed rollout reuse experiment inputs changed')
    scope = read(directory/'execution_scope.json')['scope']
    if scope['collection_stage'] != 'parent_pre_update':
        raise ValueError('Only an explicitly reviewed completed baseline may be reused')
    if scope['protocol_hash'] != digest(root/entry['prior_protocol']) or scope['implementation'] != fingerprint(old['controller']):
        raise ValueError('Completed baseline does not match archived implementation')
    receipt = read(directory/'execution_receipt.json')
    if receipt['input_hash'] != value_hash([entry['task_hash'], digest(directory/'execution_scope.json')]):
        raise ValueError('Completed baseline Task binding changed')
    status = read(root/entry['early_status'])
    request = read(root/entry['early_request'])
    if (status['status'] != 'completed' or status['execution_receipt_sha256'] != digest(directory/'execution_receipt.json')
            or request['task_hash'] != entry['task_hash'] or request['protocol_sha256'] != scope['protocol_hash']):
        raise ValueError('Early rollout is incomplete or inconsistent')
    return entry, scope


def completed_request_reusable(root, directory, task_hash):
    checked = _entry(root, directory)
    if checked is None:
        return False
    if checked[0]['task_hash'] != task_hash:
        raise ValueError('Completed rollout belongs to a different Task')
    return True


def authorized_scope(root, directory, proposed):
    path = Path(directory)/'execution_scope.json'
    if not path.exists() or read(path)['scope'] == proposed:
        return proposed
    checked = _entry(root, directory)
    if checked is None:
        return proposed  # Normal immutable freeze rejects the mismatch.
    old = checked[1]
    ignored = {'implementation', 'protocol_hash'}
    if {k:v for k,v in old.items() if k not in ignored} != {k:v for k,v in proposed.items() if k not in ignored}:
        raise ValueError('Completed baseline cannot be reused for different Task/manifest/seed')
    return old  # Retain original provenance, not a relabelled execution.
