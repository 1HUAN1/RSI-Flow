"""Frozen Task final submissions, without a Meta client or evolution feedback."""
from __future__ import annotations

import contextlib
import copy
import json
import time
from pathlib import Path

from .data import TaskRecord
from .durable import load_task, task_hash, value_hash
from .environments import SearchQAAdapter, TACOAdapter, extract_python
from .file_lock import exclusive_lock
from .reporting import BENCHMARK_IDS, OfficialEvaluatorSpec, ReportBlocked
from .retrieval import FrozenSearchIndex
from .sandbox import LinuxSandbox
from .harnessforge_manifest import load_manifest
from .seed import _BudgetExhausted, load_seed, run_seed
from .storage import artifact_manifest, checkpoint_manifest, digest, save_json
from .task_client import LocalTaskClient, TaskInfrastructureError
from .types import UpdatePending
from report_environment import ReportEnvironment, serializable_trajectory


def public_final_task(identifier, row):
    """Only native public prompt/interface fields cross the Task boundary."""
    if identifier in {'hotpotqa_dev', '2wiki_dev'}:
        return TaskRecord(str(row['_id']), 'searchqa', identifier, 'report_eval', row['question'], {})
    if identifier in {'humaneval_plus', 'mbpp_plus'}:
        return TaskRecord(str(row['task_id']), 'code', identifier, 'report_eval', row['prompt'],
                          {'tests': {'fn_name': row['entry_point']}})
    if identifier == 'livecodebench':
        metadata = row['metadata']
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        examples = row['public_test_cases']
        examples = json.loads(examples) if isinstance(examples, str) else examples
        return TaskRecord(str(row['question_id']), 'code', identifier, 'report_eval',
                          row['question_content'] + '\n' + (row.get('starter_code') or ''),
                          {'public_tests': examples, 'tests': {'fn_name': metadata.get('func_name')}})
    raise ReportBlocked(f'{identifier}: native multi-turn Tool adapter has not been validated')


def generate_report_predictions(config, frozen_path, specs_path, output_dir, *, enabled=False):
    if not enabled:
        raise ValueError('Final prediction generation requires explicit --enable-inference')
    with exclusive_lock(Path(output_dir) / '.prediction.lock'):
        return _generate_report_predictions(config, frozen_path, specs_path, output_dir)


def _generate_report_predictions(config, frozen_path, specs_path, output_dir):
    frozen = json.loads(Path(frozen_path).read_text())
    state = load_task(frozen['task_state'])
    if frozen.get('status') != 'frozen_for_report_eval' or task_hash(state) != frozen['state_hash']:
        raise ValueError('Final generation requires an unchanged frozen Task')
    if checkpoint_manifest(state.checkpoint_path) != frozen.get('checkpoint_files'):
        raise ValueError('Frozen checkpoint identity is absent or changed')
    seed_spec = load_seed(state.harness_path)
    fixed_harness_identity = load_manifest(state.harness_path).identity
    if frozen.get('task_harness_identity') != fixed_harness_identity:
        raise ValueError('Frozen HarnessForge runtime identity is absent or changed')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configured = json.loads(Path(specs_path).read_text())['evaluators']
    if set(configured) - set(BENCHMARK_IDS):
        raise ValueError('Unregistered final benchmark')
    artifacts = ''
    artifact_sources = []
    if state.artifacts.directory:
        if artifact_manifest(state.artifacts.directory) != state.artifacts.manifest:
            raise ValueError('Frozen assets were modified')
        artifact_sources = [{**r, 'content': (Path(state.artifacts.directory) / r['path']).read_text(encoding='utf-8')}
                            for r in state.artifacts.manifest]
        artifacts = '\n'.join(r['content'] for r in artifact_sources)[:12000]
    from .pipeline import project_path
    data = project_path(config.data_dir)
    corpus = json.loads((data / 'search.sqlite.manifest.json').read_text())
    protocol = {'frozen_hash': frozen['state_hash'], 'specs_sha256': digest(Path(specs_path)),
                'corpus_sha256': corpus['sha256'], 'max_model_calls': config.model_call_limit,
                'max_output_tokens': config.max_output_tokens, 'task_enable_thinking': config.task_enable_thinking,
                'task_seed': config.seed, 'final_submissions_per_task': 1, 'feedback_to_meta': False}
    protocol['task_harness_identity'] = fixed_harness_identity
    protocol_path = output_dir / 'prediction_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Final generation protocol changed on resume')
    save_json(protocol_path, protocol)
    index = FrozenSearchIndex(data / 'search.sqlite', expected_sha256=corpus['sha256'])
    client = LocalTaskClient(state, config.task_base_url, timeout=config.task_timeout,
                             enable_thinking=config.task_enable_thinking)
    summary = {}
    try:
        for identifier, config_spec in configured.items():
            spec = OfficialEvaluatorSpec(**{**config_spec, 'benchmark': identifier})
            expected = spec.validate()
            cache = output_dir / identifier
            cache.mkdir(exist_ok=True)
            source = Path(spec.data_path)
            rows = json.loads(source.read_text()) if source.suffix == '.json' else None
            with source.open() as stream:
                iterator = rows if rows is not None else (json.loads(line) for line in stream if line.strip())
                seen = set()
                for row in iterator:
                    task = public_final_task(identifier, row)
                    if task.task_id not in expected or task.task_id in seen:
                        raise ValueError('Final prediction source/ID manifest mismatch')
                    seen.add(task.task_id)
                    path = cache / (value_hash(task.task_id) + '.json')
                    call_directory = cache / (value_hash(task.task_id) + '.calls')
                    if path.exists():
                        saved = json.loads(path.read_text())
                        saved_hash = saved.pop('record_hash', None)
                        if saved_hash != value_hash(saved):
                            raise ValueError('Final cache contents changed or receipt is missing')
                        if saved['state_hash'] != frozen['state_hash'] or saved['task_id'] != task.task_id:
                            raise ValueError('Final cache identity mismatch')
                        continue
                    if call_directory.exists() and any(call_directory.iterdir()):
                        raise UpdatePending('Final Task call receipts exist without a completed submission; reconcile before retrying')
                    environment = ReportEnvironment(
                        SearchQAAdapter(index) if task.domain == 'searchqa' else TACOAdapter(LinuxSandbox()))
                    seed = int(value_hash([config.seed, task.task_id, 'report_eval'])[:8], 16) % (2**31)
                    calls = []
                    def model(messages, *, tools=None, seed, max_tokens, temperature, calls=calls,
                              call_directory=call_directory, task=task, identifier=identifier):
                        if max_tokens > config.max_output_tokens:
                            raise ValueError('Final Task budget exceeded')
                        if len(calls) >= config.model_call_limit:
                            raise _BudgetExhausted('Maximum controller Task model calls reached')
                        entry = {'call_id': len(calls), 'messages': copy.deepcopy(messages), 'tools': copy.deepcopy(tools),
                                 'seed': seed, 'max_tokens': max_tokens, 'temperature': temperature, 'status': 'started',
                                 'usage_complete': False, 'unknown_usage_calls': 1,
                                 'chat_template_kwargs': {'enable_thinking': config.task_enable_thinking}}
                        calls.append(entry)
                        receipt_path = call_directory / f'{entry["call_id"]:04d}.json'
                        receipt = {'binding': {'state_hash': frozen['state_hash'], 'task_id': task.task_id,
                                              'benchmark': identifier, 'protocol_sha256': value_hash(protocol)},
                                   'call_id': entry['call_id'], 'request_sha256': value_hash(entry),
                                   'request': copy.deepcopy(entry), 'status': 'dispatching'}
                        save_json(receipt_path, receipt)
                        try:
                            response = copy.deepcopy(client(messages, tools=tools, seed=seed, max_tokens=max_tokens, temperature=temperature))
                            response.setdefault('chat_template_kwargs', copy.deepcopy(entry['chat_template_kwargs']))
                            entry.update(status='completed', response=copy.deepcopy(response))
                            usage = response.get('usage') or {}
                            entry['usage_complete'] = all(type(usage.get(key)) is int and usage[key] >= 0
                                                          for key in ('prompt_tokens', 'completion_tokens'))
                            entry['unknown_usage_calls'] = int(not entry['usage_complete'])
                            save_json(receipt_path, {**receipt, 'status': 'completed', 'response': copy.deepcopy(response)})
                            return response
                        except Exception as exc:
                            entry.update(status='failed', error_type=type(exc).__name__)
                            save_json(receipt_path, {**receipt, 'status': 'failed', 'error_type': type(exc).__name__,
                                                     'response': copy.deepcopy(entry.get('response'))})
                            raise
                    started = time.monotonic()
                    try:
                        public = environment.reset(task, 'report_eval', seed)
                        result = run_seed(seed_spec, model, environment, json.dumps(public), artifacts, seed,
                                          artifact_sources=artifact_sources)
                        if result.get('infrastructure_failure'):
                            raise TaskInfrastructureError(result.get('error'))
                        answer = result.get('final_answer') or ''
                        if not isinstance(answer, str):
                            answer = json.dumps(answer)
                        if task.domain == 'code':
                            with contextlib.suppress(ValueError):
                                answer = extract_python(answer)
                        record = {'task_id': task.task_id, 'split': 'report_eval', 'state_hash': frozen['state_hash'],
                            'final_submission_count': 1, 'final_answer': answer, 'infrastructure_error': False,
                            'trajectory': serializable_trajectory(result), 'evaluation_status': 'pending_official',
                            'transport_calls': calls, 'wall_seconds': time.monotonic() - started}
                        record['record_hash'] = value_hash(record)
                        save_json(path, record)
                    finally:
                        environment.close()
                if seen != set(expected):
                    raise ReportBlocked('Final source is incomplete; denominator cannot be reduced')
            target = output_dir / (identifier + '.jsonl')
            with target.with_suffix('.jsonl.tmp').open('w') as stream:
                for task_id in expected:
                    stream.write(json.dumps(json.loads((cache / (value_hash(task_id) + '.json')).read_text())) + '\n')
            target.with_suffix('.jsonl.tmp').replace(target)
            summary[identifier] = {'status': 'submissions_completed_unscored', 'count': len(expected)}
            save_json(output_dir / 'prediction_status.json', summary)
    finally:
        index.close()
    return summary

