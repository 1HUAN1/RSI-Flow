import hashlib
import json
from pathlib import Path

import pytest

from sia.task_meta.meta_backends import CodexOpenRouterBackend, MetaBackendConfig
from sia.task_meta.meta_harness import MetaHarnessStore
from sia.task_meta.types import MetaDecision, MetaHarnessUpdate


@pytest.mark.parametrize('failure', [None, 'wrong_hash', 'undeclared_file'])
def test_native_candidate_file_requires_hash_and_original_boundaries(tmp_path, failure):
    store = MetaHarnessStore(tmp_path / 'meta')
    bundle = store.initialize(Path(__file__).resolve().parents[1] / 'meta_harness/seed')
    backend = CodexOpenRouterBackend(MetaBackendConfig(g_output_delivery='candidate_file'), tmp_path / 'meta', store)
    backend.decision_source = 'test_override'
    backend.bind_context('offline_contract', 1, 'a' * 64)
    prepared = backend.prepare('Offline serialization contract only', MetaHarnessUpdate, operation='meta_self_update')
    work = prepared.directory / 'workspace'
    data = json.dumps({**prepared.request.identity(), 'result': {
        'harness': bundle.read_files()['instructions.md'], 'rationale': 'offline fixture',
        'changed_rules': [], 'status': 'NO_CHANGE', 'bundle_files': {}}}).encode()
    (work / '.meta_candidate.json').write_bytes(data)
    (work / '.meta_response.json').write_text(json.dumps({'request_id': prepared.request.request_id,
        'candidate_file': '.meta_candidate.json', 'candidate_sha256':
        '0' * 64 if failure == 'wrong_hash' else hashlib.sha256(data).hexdigest()}))
    if failure == 'undeclared_file':
        (work / 'unexpected.txt').write_text('out of scope')
    (prepared.directory / 'events.jsonl').write_text('\n'.join(json.dumps(x) for x in [
        {'type': 'item.completed', 'item': {'type': 'command_execution', 'status': 'completed', 'exit_code': 0}},
        {'type': 'turn.completed'}]))
    result = {'returncode': 0, 'transport': [{'returned_model': backend.config.model, 'completed': True}]}
    if failure:
        with pytest.raises(ValueError, match='hash mismatch|undeclared'):
            backend.collect(prepared, result)
    else:
        value = backend.collect(prepared, result)
        assert value.harness == bundle.read_files()['instructions.md']
        assert json.loads((prepared.directory / 'candidate_delivery.json').read_text())['candidate_sha256'] == hashlib.sha256(data).hexdigest()


def test_routing_strict_receipt_preserves_original_optional_and_dynamic_fields(tmp_path):
    store = MetaHarnessStore(tmp_path / 'meta')
    store.initialize(Path(__file__).resolve().parents[1] / 'meta_harness/seed')
    backend = CodexOpenRouterBackend(MetaBackendConfig(g_output_delivery='candidate_file_all'), tmp_path / 'meta', store)
    backend.decision_source = 'test_override'
    backend.bind_context('offline_schema_contract', 0, 'a' * 64)
    prepared = backend.prepare('Offline schema contract only', MetaDecision, operation='routing')
    receipt_schema = json.loads((prepared.directory / 'schema.json').read_text())
    assert set(receipt_schema['required']) == set(receipt_schema['properties']) == {'request_id', 'candidate_file', 'candidate_sha256'}
    assert receipt_schema['additionalProperties'] is False
    full = json.loads((prepared.directory / 'workspace/meta_input/result_schema.json').read_text())
    assert 'harness_part' not in full['$defs']['RequestedChange']['required']
    candidate = {'action': 'HARNESS', 'diagnosis': 'fixture', 'evidence': [], 'rationale': 'fixture',
                 'proposed_change': 'fixture', 'expected_effect': 'unverified', 'expected_cost': {'model_calls': 1}}
    data = json.dumps({**prepared.request.identity(), 'result': candidate}).encode()
    work = prepared.directory / 'workspace'
    (work / '.meta_candidate.json').write_bytes(data)
    (work / '.meta_response.json').write_text(json.dumps({'request_id': prepared.request.request_id,
        'candidate_file': '.meta_candidate.json', 'candidate_sha256': hashlib.sha256(data).hexdigest()}))
    (prepared.directory / 'events.jsonl').write_text('\n'.join(json.dumps(x) for x in [
        {'type': 'item.completed', 'item': {'type': 'command_execution', 'status': 'completed', 'exit_code': 0}},
        {'type': 'turn.completed'}]))
    result = {'returncode': 0, 'transport': [{'returned_model': backend.config.model, 'completed': True}]}
    output = backend.collect(prepared, result)
    assert output.expected_cost == {'model_calls': 1}
    assert output.requested_changes == []
