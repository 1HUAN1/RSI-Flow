#!/usr/bin/env python3
"""Bounded, read-only readiness inspection. Never infer, train, install or repair.

Only one local GET /health and existing executable capability/version probes are
allowed. The sole persistent write is a newly created output JSON. Final datasets
are streamed for SHA256 only; their answers never enter this report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
BENCHMARKS = ('bfcl_v3', 'acebench', 'livecodebench', 'humaneval_plus', 'mbpp_plus', 'hotpotqa_dev', '2wiki_dev')
ADAPTERS = {'hotpotqa_dev': 'official_hotpot_fullwiki_v1', '2wiki_dev': 'official_2wiki_original_v1',
            'humaneval_plus': 'official_evalplus_native_inputs_ipc', 'mbpp_plus': 'official_evalplus_native_inputs_ipc',
            'livecodebench': 'official_lcb_native_comparator_ipc'}


def read_json(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return json.load(stream)


def file_hash(path, *, timeout=20):
    started, result = time.monotonic(), hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
            if time.monotonic() - started > timeout:
                raise TimeoutError('Read-only file hash exceeded its time limit')
    return result.hexdigest()


def _path(project, value):
    path = Path(value)
    return path if path.is_absolute() else project / path


def _command(args, *, timeout=15, cwd=None):
    # No credentials, provider config or shell interpolation reach child probes.
    env = {name: os.environ[name] for name in ('PATH', 'SystemRoot', 'WINDIR', 'TEMP', 'TMP') if name in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
    process = subprocess.run(args, cwd=cwd, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                             text=True, timeout=timeout, check=False)
    if process.returncode:
        # Failed commands can emit arbitrary environment-dependent diagnostics.
        # Report only bounded exit evidence, never credential-bearing output.
        diagnostic = process.stderr[:600] if Path(args[0]).name == 'bwrap' else ''
        raise RuntimeError(f'Probe exited with code {process.returncode}: {diagnostic}')
    return process.stdout.strip()


def _expect(condition, message):
    if not condition:
        raise ValueError(message)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise ValueError('Health redirects are disabled')


def health_summary(base_url, checkpoint, expected_weights, *, opener=None):
    parsed = urlsplit(base_url)
    _expect(parsed.scheme == 'http' and parsed.hostname in {'127.0.0.1', 'localhost', '::1'}
            and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment,
            'Only a credential-free loopback HTTP health endpoint is permitted')
    url = urlunsplit((parsed.scheme, parsed.netloc, '/health', '', ''))
    opener = opener or build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(Request(url, method='GET'), timeout=3) as response:
        raw = response.read(1024 * 1024 + 1)
    _expect(len(raw) <= 1024 * 1024, 'Health response exceeds bounded metadata size')
    value = json.loads(raw)
    checkpoint = str(Path(checkpoint).resolve())
    binding = value.get('bindings', {}).get(checkpoint, {})
    matched = bool(binding) and str(Path(binding.get('checkpoint_path', '')).resolve()) == checkpoint
    weights_matched = bool(expected_weights) and binding.get('weights') == expected_weights
    return {'url': url, 'ready': value.get('ready') is True, 'checkpoint_matches': matched,
            'registered_weights_match': weights_matched, 'device': value.get('device'),
            'passed': value.get('ready') is True and matched and weights_matched,
            'inference_performed': False}


def window_plan(counts, quotas, generations, rollouts):
    per_domain = {}
    available = max((math.ceil(counts[d]['evolve_train'] / quotas[d]) for d in quotas), default=0)
    scheduled = min(generations, available)
    for domain, quota in quotas.items():
        size = counts[domain]['evolve_train']
        used = min(size, scheduled * quota)
        per_domain[domain] = {'pool': size, 'quota': quota, 'exhausted_after_windows': math.ceil(size / quota),
                              'scheduled_unique_tasks': used, 'remaining_tasks': size - used,
                              'scheduled_rollouts': used * rollouts}
    return {'available_windows': available, 'configured_generations': generations, 'scheduled_generations': scheduled,
            'maximum_interventions': max(0, scheduled - 1), 'per_domain': per_domain,
            'rule': 'without replacement; exhausted domains contribute zero; later domains continue',
            'training_updates': 'unknown until legal MODEL choices and positive trajectories exist'}


def report_specs(path, project):
    document = read_json(path)
    result = {}
    for name in BENCHMARKS:
        spec = document.get('evaluators', {}).get(name)
        record = {'configured': spec is not None, 'adapter_implemented': name in ADAPTERS,
                  'provenance_verified': False, 'ready': False, 'checks': {}}
        result[name] = record
        if not spec:
            record['reason'] = document.get('blocked', {}).get(name, {}).get('status', 'NOT_CONFIGURED')
            continue
        expected_protocol = ADAPTERS.get(name)
        record['checks']['protocol'] = spec.get('protocol') == expected_protocol and expected_protocol is not None
        commit = spec.get('commit', '')
        record['checks']['pinned_commit'] = len(commit) == 40 and all(c in '0123456789abcdef' for c in commit)
        record['checks']['evaluator_python'] = _path(project, spec.get('python_executable', '')).is_file()
        for key in ('entrypoint', 'data', 'task_ids'):
            path_key = 'entrypoint' if key == 'entrypoint' else key + '_path'
            try:
                record['checks'][key + '_sha256'] = file_hash(_path(project, spec[path_key])) == spec[key + '_sha256']
            except (OSError, ValueError, KeyError, TimeoutError):
                record['checks'][key + '_sha256'] = False
        if record['checks'].get('task_ids_sha256'):
            try:
                ids = read_json(_path(project, spec['task_ids_path']))
                record['checks']['unique_ids'] = (isinstance(ids, list) and bool(ids)
                    and all(isinstance(value, str) for value in ids) and len(set(ids)) == len(ids))
                record['id_count'] = len(ids)
            except (OSError, TypeError, ValueError):
                record['checks']['unique_ids'] = False
        else:
            record['checks']['unique_ids'] = False
        metadata = spec.get('metadata', {})
        if metadata.get('native_source') or metadata.get('source_sha256'):
            try:
                record['checks']['native_source_sha256'] = file_hash(_path(project, metadata['native_source'])) == metadata['source_sha256']
            except (OSError, ValueError, KeyError, TimeoutError):
                record['checks']['native_source_sha256'] = False
        record['provenance_verified'] = all(value for key, value in record['checks'].items() if key not in {'protocol', 'evaluator_python'})
        record['ready'] = record['adapter_implemented'] and all(record['checks'].values())
        record['final_answers_parsed'] = False
    return result


def compatibility_evidence(meta):
    from sia.task_meta.meta_backends.codex_openrouter import COMPATIBILITY_CHECKS
    report = read_json(meta['compatibility_report'])
    mapping = {'backend': 'backend', 'provider': 'provider', 'model': 'model', 'codex_commit': 'codex_commit',
               'binary_sha256': 'codex_binary_sha256', 'catalog_sha256': 'model_catalog_sha256',
               'provider_order': 'provider_order', 'allow_provider_fallback': 'allow_provider_fallback'}
    _expect(report.get('status') == 'API_SMOKE_PASSED' and all(report.get(k) == meta.get(v) for k, v in mapping.items()),
            'No matching successful real API compatibility report')
    _expect(all(report.get('checks', {}).get(k) is True for k in COMPATIBILITY_CHECKS) and bool(report.get('evidence')),
            'Compatibility evidence incomplete')
    calls = {}
    for item in report['evidence']:
        path = Path(item['path'])
        _expect(file_hash(path) == item['sha256'], 'API compatibility evidence changed')
        calls.setdefault(path.parent, {})[path.name] = path
    _expect(len(calls) >= 2, 'Compatibility requires independent initial and reloaded G calls')
    requests = 0
    for files in calls.values():
        _expect({'request.json', 'events.jsonl', 'transport.json', 'collected.json', 'bundle_load.json'} <= files.keys(),
                'Compatibility call evidence is incomplete')
        collected, load = read_json(files['collected.json']), read_json(files['bundle_load.json'])
        request, transport = read_json(files['request.json']), read_json(files['transport.json'])
        _expect(collected.get('decision_source') == 'codex_openrouter' and load.get('runtime_verified') is True
                and load.get('decision_source') == 'codex_openrouter', 'Mock or unverified Codex evidence cannot qualify')
        _expect(request.get('operation') == 'compatibility_smoke' and request.get('model') == meta['model']
                and request.get('provider') == meta['provider']
                and request.get('request_id') == collected.get('request_id') == load.get('request_id'),
                'Call identity differs across compatibility artifacts')
        rows = transport.get('requests', [])
        _expect(bool(rows) and all(row.get('completed') is True and row.get('returned_model') == meta['model']
                                  and row.get('http_status') == 200 for row in rows),
                'No successful exact-provider transport evidence')
        requests += len(rows)
    return {'existing_report_sha256': file_hash(meta['compatibility_report']), 'verified_call_directories': len(calls),
            'recorded_transport_requests': requests, 'real_calls_made_by_this_check': 0}


def inspect_readiness(config_path, *, project=PROJECT, report_path=None, search_dev_audit=None):
    project = Path(project).resolve()
    checks = []
    def check(name, group, action):
        try:
            detail = action()
            passed = detail.get('passed', True) if isinstance(detail, dict) else detail is not False
            row = {'name': name, 'group': group, 'passed': passed, 'details': detail}
        except Exception as exc:
            reason = str(exc)[:1200]
            for key, value in os.environ.items():
                if len(value) >= 8 and key.upper().endswith(('_KEY', '_TOKEN', '_PASSWORD', '_SECRET')):
                    reason = reason.replace(value, '[REDACTED]')
            row = {'name': name, 'group': group, 'passed': False, 'error_type': type(exc).__name__, 'reason': reason}
        checks.append(row)
        return row

    raw = read_json(config_path)
    meta = raw.get('meta', {})
    data_dir = _path(project, raw.get('data_dir', 'data/pipeline_v2'))
    def code_imports():
        names = ('pipeline', 'seed', 'task_harness', 'meta', 'meta_harness.runtime', 'meta_backends.codex_openrouter',
                 'reporting', 'evalplus_isolated', 'lcb_isolated')
        code = ('import importlib,sys; sys.path.insert(0,sys.argv[1]); '
                '[importlib.import_module("sia.task_meta."+n) for n in sys.argv[2:]]')
        _command([sys.executable, '-B', '-c', code, str(project), *names], timeout=20)
        for path in (project / 'sia/task_meta').rglob('*.py'):
            compile(path.read_text(encoding='utf-8'), str(path), 'exec')
        return {'modules': list(names), 'loaded_model_weights': False}
    check('controller_imports_and_syntax', 'code', code_imports)
    def config_valid():
        from sia.task_meta.pipeline import PipelineConfig
        PipelineConfig.model_validate(raw).checked()
        return True
    check('pipeline_config', 'code', config_valid)
    def task_seed():
        from sia.task_meta.task_harness import harness_identity, load_harness
        path = _path(project, raw['seed_harness'])
        _expect(load_harness(path)['schema_version'] == 2, 'Current Task seed is not v2')
        return harness_identity(path)
    check('task_harness_v2', 'code', task_seed)
    def meta_seed():
        from sia.task_meta.meta_harness.bundle import validate_files
        root = _path(project, meta.get('harness_root', 'meta_harness')) / 'seed'
        files = {name: (root / name).read_text(encoding='utf-8') for name in ('instructions.md', 'context.json', 'workflow.json', 'evolution.json')}
        validate_files(files)
        return {'files': {name: file_hash(root / name) for name in files}, 'schema': 'meta-bundle-v2'}
    check('meta_harness_v2', 'code', meta_seed)
    expected_weights = []
    def model_files():
        checkpoint = Path(raw['task_checkpoint']).resolve()
        identity = read_json(data_dir / 'model_identity.json')
        _expect(identity['path'] == str(checkpoint) and identity.get('model_type') == 'qwen3', 'Registered model differs')
        weights = identity.get('weights', [])
        _expect(bool(weights), 'No registered checkpoint weights')
        for entry in weights:
            path = checkpoint / entry['path']
            _expect(path.is_file() and path.stat().st_size == entry['bytes'], 'Missing or size-mismatched model weight')
        for name, expected in identity.get('tokenizer_and_config', {}).items():
            _expect(file_hash(checkpoint / name) == expected, 'Tokenizer/config fingerprint changed')
        expected_weights.extend(weights)
        return {'registered_weights': len(weights), 'model_config_exists': (checkpoint / 'config.json').is_file(),
                'weights_content_rehashed': False, 'model_loaded': False}
    check('task_checkpoint_metadata_and_files', 'runtime', model_files)
    check('task_service_health', 'runtime', lambda: health_summary(raw['task_base_url'], raw['task_checkpoint'], expected_weights))
    def training_dependencies():
        python = raw['trainer_python']
        _expect(Path(python).is_file(), 'Configured trainer Python is absent')
        code = ('import importlib.util,json; names=["torch","transformers","peft","accelerate","datasets"]; '
                'print(json.dumps({n:importlib.util.find_spec(n) is not None for n in names}))')
        found = json.loads(_command([python, '-B', '-c', code], timeout=20))
        return {'passed': all(found.values()), 'dependencies': found, 'training_started': False}
    check('trainer_dependencies', 'runtime', training_dependencies)
    plan = {}
    def data_windows():
        from sia.task_meta.data import ManifestStore
        store = ManifestStore(data_dir / 'tasks.sqlite')
        try:
            counts = store.counts()
        finally:
            store.close()
        plan.update(window_plan(counts, raw['window_quotas'], raw['max_generations'], raw['rollouts_per_task']))
        metadata = read_json(data_dir / 'manifest.json')
        _expect(metadata['manifest_sha256'] == file_hash(data_dir / 'tasks.sqlite'), 'Prepared manifest changed')
        _expect(metadata['split_seed'] == raw['seed'] and metadata['probe_per_domain'] == raw['probe_per_domain']
                and metadata['search_dev_fraction'] == raw['search_dev_fraction'], 'Prepared data protocol differs')
        _expect(metadata['source_manifest_hash'] == file_hash(raw['source_manifest']), 'Source catalog changed')
        _expect(plan['scheduled_generations'] > 0, 'Training windows exhausted')
        return plan
    check('data_windows', 'runtime', data_windows)
    def search_index():
        metadata = read_json(data_dir / 'search.sqlite.manifest.json')
        _expect(metadata['sha256'] == file_hash(data_dir / 'search.sqlite'), 'Frozen search index changed')
        _expect(metadata['data_manifest_hash'] == file_hash(data_dir / 'tasks.sqlite'), 'Corpus belongs to another manifest')
        _expect(metadata.get('label_fields_indexed') is False, 'Unverified corpus field policy')
        return {'nq_coverage': metadata.get('nq_coverage', 'unknown'), 'search_dev_evidence_coverage': 'unknown',
                'corpus_sha256': metadata['sha256'], 'final_labels_read': False}
    check('frozen_search_corpus', 'runtime', search_index)
    check('meta_credentials_present', 'runtime', lambda: {'passed': bool(os.environ.get('OPENROUTER_API_KEY')),
                                                         'credential_value_recorded': False})
    check('configured_execution_mode', 'runtime', lambda: {'passed': raw.get('mode') in {'api_smoke', 'pilot', 'full'},
                                                         'mode': raw.get('mode'), 'dev_api_disabled': raw.get('mode') == 'dev'})
    def native_schema():
        import tomllib

        import jsonschema

        from sia.task_meta.meta_backends.codex_openrouter import render_codex_config
        from sia.task_meta.meta_backends.contracts import MetaBackendConfig
        parsed = MetaBackendConfig.model_validate(meta)
        _expect(parsed.model == 'deepseek/deepseek-v4-flash-0731' and parsed.base_url == 'https://openrouter.ai/api/v1',
                'Meta identity differs from the fixed protocol')
        schema = Path(meta['codex_source']) / 'codex-rs/core/config.schema.json'
        jsonschema.validate(tomllib.loads(render_codex_config(parsed, isolated=True)), read_json(schema))
        return {'schema_sha256': file_hash(schema), 'provider': parsed.provider, 'model': parsed.model}
    check('meta_native_config_schema', 'runtime', native_schema)
    def pinned_binary():
        binary = Path(meta['codex_executable'])
        _expect(file_hash(binary) == meta['codex_binary_sha256'], 'Codex binary hash mismatch')
        provenance = read_json(meta['provenance_file'])
        _expect(provenance.get('commit') == meta['codex_commit'] and provenance.get('binary_sha256') == meta['codex_binary_sha256'],
                'Runtime provenance differs')
        version = _command([str(binary), '--version'], timeout=10)
        _expect(version == 'codex-cli 0.153.4', 'Unexpected pinned CLI version')
        return {'version': version, 'sha256': meta['codex_binary_sha256'], 'provenance_matched': True}
    check('meta_pinned_binary_version', 'runtime', pinned_binary)
    def pinned_source():
        source = meta['codex_source']
        head = _command(['git', '-C', source, 'rev-parse', 'HEAD'])
        dirty = _command(['git', '-c', 'core.fsmonitor=false', '-C', source, 'status', '--porcelain', '--untracked-files=no'])
        _expect(head == meta['codex_commit'] and not dirty, 'Pinned source checkout differs')
        return {'commit': head, 'tracked_tree_clean': True}
    check('meta_pinned_source', 'runtime', pinned_source)
    def catalog():
        path = Path(meta['model_catalog_json'])
        _expect(file_hash(path) == meta['model_catalog_sha256'], 'Model catalog fingerprint differs')
        value = read_json(path)
        models = value.get('models', []) if isinstance(value, dict) else value
        _expect(any(row.get('slug') == meta['model'] and isinstance(row.get('context_window'), int)
                    and row['context_window'] > 0 for row in models), 'Exact model catalog entry absent')
        return {'sha256': meta['model_catalog_sha256'], 'exact_model_found': True}
    check('meta_model_catalog', 'runtime', catalog)
    check('meta_api_compatibility_evidence', 'runtime', lambda: compatibility_evidence(meta))
    def meta_isolation():
        candidate = project / 'third_party/bubblewrap/usr/bin/bwrap'
        binary = str(candidate) if candidate.is_file() else shutil.which('bwrap')
        _expect(sys.platform == 'linux' and bool(binary), 'Required existing Linux bubblewrap absent')
        _command([binary, '--unshare-all', '--die-with-parent', '--ro-bind', '/', '/', '/bin/true'], timeout=10)
        # Runtime currently resolves via PATH; do not silently claim the local
        # probe path makes the actual backend runnable.
        discoverable = shutil.which('bwrap')
        return {'passed': bool(discoverable) and Path(discoverable).resolve() == Path(binary).resolve(),
                'probe_succeeded': True, 'runtime_path_matches_probe': bool(discoverable) and Path(discoverable).resolve() == Path(binary).resolve()}
    check('meta_namespace_isolation', 'runtime', meta_isolation)
    def task_isolation():
        from sia.task_meta.sandbox import probe_isolation
        evidence = probe_isolation()
        return {'passed': evidence.get('available') is True, 'capability_probe': evidence, 'candidate_executed': False}
    check('task_sandbox_capability', 'runtime', task_isolation)
    reports = {}
    def final_specs():
        reports.update(report_specs(report_path or project / 'configs/report-evaluators.json', project))
        return {'passed': all(row['ready'] for row in reports.values()), 'benchmarks': reports}
    check('seven_official_report_specs', 'report', final_specs)
    audit = {'status': 'unknown', 'measured_by_this_script': False, 'final_answers_inspected': False}
    if search_dev_audit:
        path = Path(search_dev_audit)
        audit.update(path=str(path), exists=path.is_file())
        if path.is_file():
            audit.update(status='existing_audit_not_semantically_revalidated', sha256=file_hash(path))
    code_ready = all(row['passed'] for row in checks if row['group'] == 'code')
    runtime_ready = code_ready and all(row['passed'] for row in checks if row['group'] == 'runtime')
    return {'schema_version': 'closed-loop-readiness-v1', 'created_at': datetime.now(UTC).isoformat(),
            'config_sha256': file_hash(config_path), 'mode': raw.get('mode'),
            'CODE_COMPLETE': False, 'code_static_checks_passed': code_ready, 'test_evidence_status': 'not_run',
            'code_complete_reason': 'Import/schema checks alone cannot establish completion of the full acceptance suite.',
            'RUNTIME_READY': runtime_ready, 'PILOT_VERIFIED': False,
            'REPORT_READY': all(row['passed'] for row in checks if row['group'] == 'report'),
            'pilot_reason': 'No trusted real-run pilot evidence validator was executed; mock or self-asserted success cannot qualify.',
            'checks': checks, 'blockers': [{'check': row['name'], 'group': row['group'], 'error_type': row.get('error_type'),
                                          'reason': row.get('reason', 'Required predicate did not pass')}
                                          for row in checks if not row['passed']],
            'search_dev_coverage_audit': audit, 'window_plan': plan,
            'allowed_actions_performed': ['local_file_reads', 'local_health_GET', 'bounded_existing_capability_probes'],
            'api_inference_calls': 0, 'gpu_training_started': False, 'report_evaluations_executed': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report-specs')
    parser.add_argument('--search-dev-audit')
    args = parser.parse_args(argv)
    target = Path(args.output)
    if target.exists():
        raise FileExistsError('Readiness output must be a new file')
    report = inspect_readiness(args.config, report_path=args.report_specs, search_dev_audit=args.search_dev_audit)
    # Exclusive creation prevents overwriting even if another process raced us.
    with target.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({name: report[name] for name in ('CODE_COMPLETE', 'RUNTIME_READY', 'PILOT_VERIFIED', 'REPORT_READY')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
