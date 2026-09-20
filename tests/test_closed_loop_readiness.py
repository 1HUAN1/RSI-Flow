"""Readiness probes never start paid inference or promote mock results."""
import io
import json
from pathlib import Path

import pytest

from scripts import check_closed_loop_readiness as readiness


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def test_window_plan_continues_larger_domains_and_counts_interventions():
    counts = {'tool_use': {'evolve_train': 3}, 'code': {'evolve_train': 0}, 'searchqa': {'evolve_train': 11}}
    plan = readiness.window_plan(counts, dict.fromkeys(counts, 2), 4, 2)
    assert plan['available_windows'] == 6 and plan['scheduled_generations'] == 4
    assert plan['maximum_interventions'] == 3
    assert plan['per_domain']['tool_use']['scheduled_unique_tasks'] == 3
    assert plan['per_domain']['code']['scheduled_rollouts'] == 0
    assert plan['per_domain']['searchqa']['remaining_tasks'] == 3
    assert 'unknown' in plan['training_updates']


class HealthOpener:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def open(self, request, *, timeout):
        self.calls.append((request.full_url, request.get_method(), timeout))
        return io.BytesIO(json.dumps(self.payload).encode())


def test_health_only_gets_local_health_and_requires_exact_registered_weights(tmp_path):
    checkpoint = str(tmp_path.resolve())
    weights = [{'path': 'model.safetensors', 'bytes': 10, 'sha256': 'a' * 64}]
    opener = HealthOpener({'ready': True, 'device': 'cuda:0', 'private': 'NOT_FOR_REPORT',
                           'bindings': {checkpoint: {'checkpoint_path': checkpoint, 'weights': weights}}})
    result = readiness.health_summary('http://127.0.0.1:8071/v1', checkpoint, weights, opener=opener)
    assert result['passed'] and opener.calls == [('http://127.0.0.1:8071/health', 'GET', 3)]
    assert 'NOT_FOR_REPORT' not in json.dumps(result)
    assert not readiness.health_summary('http://127.0.0.1:8071/v1', checkpoint, [], opener=opener)['passed']
    opener.payload['bindings'][checkpoint]['checkpoint_path'] = str(tmp_path / 'other')
    assert not readiness.health_summary('http://127.0.0.1:8071/v1', checkpoint, weights, opener=opener)['passed']


@pytest.mark.parametrize('url', ['https://openrouter.ai/api/v1', 'http://user:secret@127.0.0.1:8071/v1',
                                'http://127.0.0.1:8071/v1?key=secret'])
def test_health_refuses_external_or_credential_urls_without_request(url):
    opener = HealthOpener({})
    with pytest.raises(ValueError, match='loopback'):
        readiness.health_summary(url, '/checkpoint', [], opener=opener)
    assert not opener.calls


def test_report_hash_checks_do_not_parse_final_gold_or_invent_missing_adapters(tmp_path):
    entry = tmp_path / 'official.py'
    entry.write_text('# pinned official source', encoding='utf-8')
    data = tmp_path / 'final.bin'
    data.write_bytes(b'\x00 not JSON; final answers are never parsed by readiness')
    ids = write(tmp_path / 'ids.json', ['1', '2'])
    spec = {'repository': 'https://example.invalid/official', 'commit': 'a' * 40,
            'entrypoint': str(entry), 'entrypoint_sha256': readiness.file_hash(entry),
            'data_path': str(data), 'data_sha256': readiness.file_hash(data),
            'task_ids_path': str(ids), 'task_ids_sha256': readiness.file_hash(ids),
            'python_executable': readiness.sys.executable, 'protocol': 'official_lcb_native_comparator_ipc'}
    path = write(tmp_path / 'specs.json', {'evaluators': {'livecodebench': spec},
                                          'blocked': {'bfcl_v3': {'status': 'BLOCKED_REPORT_ADAPTER'}}})
    result = readiness.report_specs(path, tmp_path)
    assert len(result) == 7 and result['livecodebench']['ready']
    assert result['livecodebench']['final_answers_parsed'] is False
    assert not result['bfcl_v3']['ready'] and not result['bfcl_v3']['adapter_implemented']
    data.write_bytes(b'changed')
    changed = readiness.report_specs(path, tmp_path)['livecodebench']
    assert not changed['ready'] and not changed['provenance_verified']
    assert changed['checks']['task_ids_sha256']


def test_existing_output_is_rejected_before_any_probe(tmp_path, monkeypatch):
    output = tmp_path / 'existing.json'
    output.write_text('original', encoding='utf-8')
    def prohibited(*_args, **_kwargs):
        raise AssertionError('No probe may run when output already exists')
    monkeypatch.setattr(readiness, 'inspect_readiness', prohibited)
    with pytest.raises(FileExistsError):
        readiness.main(['--config', 'unused', '--output', str(output)])
    assert output.read_text(encoding='utf-8') == 'original'


def test_hashed_compatibility_report_cannot_promote_mock_call_records(tmp_path):
    from sia.task_meta.meta_backends.codex_openrouter import COMPATIBILITY_CHECKS
    meta = {'backend': 'codex_openrouter', 'provider': 'openrouter', 'model': 'deepseek/deepseek-v4-flash-0731',
            'codex_commit': 'a' * 40, 'codex_binary_sha256': 'b' * 64, 'model_catalog_sha256': 'c' * 64,
            'provider_order': [], 'allow_provider_fallback': False, 'compatibility_report': str(tmp_path / 'compat.json')}
    report = {'backend': meta['backend'], 'provider': meta['provider'], 'model': meta['model'],
              'codex_commit': meta['codex_commit'], 'binary_sha256': meta['codex_binary_sha256'],
              'catalog_sha256': meta['model_catalog_sha256'], 'provider_order': [], 'allow_provider_fallback': False,
              'status': 'API_SMOKE_PASSED', 'checks': dict.fromkeys(COMPATIBILITY_CHECKS, True), 'evidence': []}
    for number in range(2):
        root = tmp_path / f'call_{number}'
        root.mkdir()
        for name in ('request.json', 'events.jsonl', 'transport.json', 'collected.json', 'bundle_load.json'):
            path = write(root / name, {'decision_source': 'test_override', 'runtime_verified': False})
            report['evidence'].append({'path': str(path), 'sha256': readiness.file_hash(path)})
    write(Path(meta['compatibility_report']), report)
    with pytest.raises(ValueError, match='Mock or unverified'):
        readiness.compatibility_evidence(meta)


def test_independent_missing_prerequisites_and_credentials_never_promote_readiness(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    config = json.loads((project / 'configs/multidomain-dev.json').read_text(encoding='utf-8'))
    config['data_dir'] = str(tmp_path / 'missing_data')
    config['task_checkpoint'] = str(tmp_path / 'missing_checkpoint')
    for name in ('codex_source', 'codex_executable', 'provenance_file', 'model_catalog_json', 'compatibility_report'):
        config['meta'][name] = str(tmp_path / ('absent_' + name))
    path = write(tmp_path / 'config.json', config)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'sensitive-fixture-never-recorded')
    monkeypatch.setattr(readiness, '_command', lambda *_args, **_kwargs: '')
    monkeypatch.setattr(readiness, 'health_summary', lambda *_args, **_kwargs: {'passed': False, 'ready': False})
    report = readiness.inspect_readiness(path, project=project, report_path=write(tmp_path / 'specs.json', {}))
    assert not any(report[key] for key in ('CODE_COMPLETE', 'RUNTIME_READY', 'PILOT_VERIFIED', 'REPORT_READY'))
    assert report['search_dev_coverage_audit']['status'] == 'unknown'
    missing = {row['check'] for row in report['blockers']}
    assert {'task_checkpoint_metadata_and_files', 'task_service_health', 'meta_pinned_binary_version',
            'meta_api_compatibility_evidence', 'seven_official_report_specs'} <= missing
    assert 'sensitive-fixture-never-recorded' not in json.dumps(report)
    assert report['api_inference_calls'] == 0 and not report['gpu_training_started']
    assert not (tmp_path / 'missing_data').exists()
