import copy
from pathlib import Path
import pytest
from sia.task_meta.evolution_protocol import fingerprint
from sia.task_meta.storage import save_json, digest
from sia.task_meta.durable import value_hash
from sia.task_meta.completed_rollout_reuse import authorized_scope, completed_request_reusable


def fixture(root, monkeypatch):
    previous = {'controller': {'runtime': 'old'}, 'config': {'seed': 42}, 'hash': 'old'}
    current = {'controller': {'runtime': 'new'}, 'config': {'seed': 42}, 'hash': 'new'}
    monkeypatch.setattr('sia.task_meta.pipeline.source_identity', lambda: current['controller'])
    prior = 'recovery/revisions/eval/protocol.json'
    save_json(root/prior, previous); save_json(root/'protocol.json', current)
    directory = root/'round_1/before/gen_1'
    scope = {'collection_stage': 'parent_pre_update', 'seed': 42, 'task': 'same', 'manifest': 'same',
             'protocol_hash': digest(root/prior), 'implementation': fingerprint(previous['controller'])}
    save_json(directory/'execution_scope.json', {'scope': scope})
    save_json(directory/'execution_receipt.json', {'input_hash': value_hash(['taskhash', digest(directory/'execution_scope.json')])})
    request = 'round_1/early_rollout/request.json'; status = 'round_1/early_rollout/status.json'
    save_json(root/request, {'task_hash': 'taskhash', 'protocol_sha256': scope['protocol_hash']})
    save_json(root/status, {'status': 'completed', 'execution_receipt_sha256': digest(directory/'execution_receipt.json')})
    names = [prior, request, status, 'round_1/before/gen_1/execution_scope.json', 'round_1/before/gen_1/execution_receipt.json']
    proof = {'controller': current['controller'], 'entries': {'round_1/before/gen_1': {
        'prior_protocol': prior, 'early_request': request, 'early_status': status, 'task_hash': 'taskhash',
        'files': {name: digest(root/name) for name in names}}}}
    save_json(root/'recovery/completed_rollout_reuse.json', proof)
    proposed = {**scope, 'implementation': fingerprint(current['controller']), 'protocol_hash': digest(root/'protocol.json')}
    return directory, scope, proposed


def test_reuse_keeps_original_provenance(tmp_path, monkeypatch):
    directory, old, proposed = fixture(tmp_path, monkeypatch)
    before = (directory/'execution_scope.json').read_bytes()
    assert completed_request_reusable(tmp_path, directory, 'taskhash')
    assert authorized_scope(tmp_path, directory, proposed) == old
    assert (directory/'execution_scope.json').read_bytes() == before


@pytest.mark.parametrize('field', ['task', 'manifest', 'seed', 'collection_stage'])
def test_behavior_changes_rejected(tmp_path, monkeypatch, field):
    directory, _, proposed = fixture(tmp_path, monkeypatch)
    proposed[field] = 'different'
    with pytest.raises(ValueError): authorized_scope(tmp_path, directory, proposed)


def test_tampered_receipt_and_different_task_rejected(tmp_path, monkeypatch):
    directory, _, proposed = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError): completed_request_reusable(tmp_path, directory, 'other')
    save_json(directory/'execution_receipt.json', {})
    with pytest.raises(ValueError): authorized_scope(tmp_path, directory, proposed)


def test_no_implicit_migration(tmp_path):
    directory = tmp_path/'round_1/before/gen_1'
    save_json(directory/'execution_scope.json', {'scope': {'old': 1}})
    assert authorized_scope(tmp_path, directory, {'new': 1}) == {'new': 1}
    assert not completed_request_reusable(tmp_path, directory, 'any')
