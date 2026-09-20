#!/usr/bin/env python3
"""Final-only BFCL-v3 evaluation through the frozen HarnessForge candidate.

No Meta client, training feedback, model selection, or provider credentials.
The committed final Task is frozen before observing any BFCL predictions.
"""
from __future__ import annotations

import argparse
import base64
import copy
import importlib.util
import io
import json
import multiprocessing
import re
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from sia.task_meta.durable import load_task, task_hash, value_hash
from sia.task_meta.file_lock import exclusive_lock
from sia.task_meta.harnessforge_manifest import load_manifest
from sia.task_meta.harnessforge_production import harnessforge_identity
from sia.task_meta.pipeline import PipelineConfig, source_identity
from sia.task_meta.sandbox import LinuxSandbox, SandboxLimits
from sia.task_meta.seed import load_seed, run_seed, _BudgetExhausted
from sia.task_meta.storage import artifact_manifest, checkpoint_manifest, digest, save_json
from sia.task_meta.task_client import LocalTaskClient

COMMIT = 'ea13468e4423454d0c213704fb87cf7cb3990433'
TASK_CONTEXT_LIMIT = 32768  # Verified active service configuration; never truncate inputs.
TOKENIZERS = {}
CLASS_DOCS = {'GorillaFileSystem': 'gorilla_file_system', 'MathAPI': 'math_api',
    'MessageAPI': 'message_api', 'TwitterAPI': 'posting_api', 'TicketAPI': 'ticket_api',
    'TradingBot': 'trading_bot', 'TravelAPI': 'travel_booking', 'VehicleControlAPI': 'vehicle_control'}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def receipt_binding_matches(record, path, binding, job):
    if record.get('binding') == binding:
        return True
    imported = job.get('imported_receipts', {})
    expected = {**binding, 'protocol_hash': job.get('imported_protocol_hash')}
    relative = path.relative_to(path.parents[1] if path.parent.name == 'calls' else path.parent).as_posix()
    return (record.get('binding') == expected and imported.get(relative) == digest(path))


def package_official(root):
    """Copy only trusted executable dependencies; no questions, answers or secrets."""
    prefixes = ('eval_checker/multi_turn_eval/', 'eval_checker/ast_eval/', 'constants/type_mappings.py')
    files = {p.relative_to(root.parent).as_posix(): p.read_bytes() for p in root.rglob('*.py')
             if p.relative_to(root).as_posix().startswith(prefixes)}
    for name in ('bfcl_eval', 'bfcl_eval/eval_checker', 'bfcl_eval/constants'):
        files.setdefault(name + '/__init__.py', b'')
    mpmath = Path(importlib.util.find_spec('mpmath').origin).parent
    files.update({p.relative_to(mpmath.parent).as_posix(): p.read_bytes() for p in mpmath.rglob('*.py')})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    return base64.b64encode(buffer.getvalue()).decode(), {n: __import__('hashlib').sha256(v).hexdigest() for n, v in files.items()}


def session(package, seed):
    sandbox = LinuxSandbox(limits=SandboxLimits(wall_seconds=45, cpu_seconds=30,
        memory_bytes=1024**3, output_bytes=8*1024**2, file_bytes=16*1024**2))
    return sandbox.session(Path(__file__).with_name('bfcl_report_worker.py').read_text(),
                           extra_files={'official.b64': package}, seed=seed)


def rpc(worker, payload):
    result = worker.request(payload)
    if result.get('infrastructure_error'):
        raise RuntimeError('BFCL worker: ' + result['infrastructure_error'] + ': ' + result.get('detail', ''))
    return result


def tool_schema(schema):
    """Translate BFCL interface type names into their JSON-Schema equivalents."""
    if not isinstance(schema, dict):
        return copy.deepcopy(schema)
    result = copy.deepcopy(schema)
    aliases = {'dict': 'object', 'float': 'number', 'tuple': 'array', 'int': 'integer',
               'str': 'string', 'bool': 'boolean', 'Array': 'array', 'ArrayList': 'array',
               'Boolean': 'boolean', 'HashMap': 'object', 'String': 'string', 'char': 'string',
               'double': 'number', 'long': 'integer'}
    if result.get('type') in ('any', ''):
        result.pop('type')
    elif isinstance(result.get('type'), str):
        result['type'] = aliases.get(result['type'], result['type'])
    for key in ('properties', '$defs', 'definitions'):
        if isinstance(result.get(key), dict):
            result[key] = {name: tool_schema(value) for name, value in result[key].items()}
    if 'items' in result:
        result['items'] = tool_schema(result['items'])
    return result


def native_functions(root, row):
    if 'function' in row:
        return copy.deepcopy(row['function'])
    functions = []
    for name in row['involved_classes']:
        functions.extend(read_rows(root / 'data/multi_turn_func_doc' / (CLASS_DOCS[name] + '.json')))
    return functions


class Environment:
    def __init__(self, row, functions, worker):
        self.row, self.worker = row, worker
        self.functions = {f['name']: f for f in functions}
        self.available = set(self.functions)
        for names in row.get('missed_function', {}).values():
            self.available.difference_update(names)
        self.calls = []
        self.tools = []

    def turn(self, number):
        self.available.update(self.row.get('missed_function', {}).get(str(number), []))
        self.tools = [{'type': 'function', 'function': {**f, 'parameters': tool_schema(f['parameters'])}}
                      for name, f in self.functions.items() if name in self.available]
        self.calls = []

    def step(self, name, arguments):
        if name not in self.available:
            return {'error': 'Tool not available on this turn', 'terminated': False}
        if self.worker is None:
            self.calls.append({name: copy.deepcopy(arguments)})
            return {'observation': 'Proposed call recorded. This API has no executable result in the single-turn benchmark.',
                    'terminated': False}
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name) or any(
                not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key) for key in arguments):
            return {'error': 'Invalid native function or argument identifier', 'terminated': False}
        call = name + '(' + ', '.join(key + '=' + repr(value) for key, value in arguments.items()) + ')'
        # Only JSON literals and declared names enter the official execution worker.
        self.calls.append([call])
        result = rpc(self.worker, {'operation': 'step', 'call': call})
        observation = '\n'.join(result['outputs'])
        return {'observation': observation, 'error': observation if observation.startswith('Error during execution:') else None,
                'terminated': False}


def case_job(job):
    config = PipelineConfig.model_validate(job['config'])
    frozen, row = job['frozen'], job['row']
    state = load_task(frozen['task_state'])
    directory = Path(job['output']) / 'cases' / row['id']
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(directory / '.case.lock'):
        binding = {'state_hash': frozen['state_hash'], 'protocol_hash': job['protocol_hash'], 'task_id': row['id']}
        path = directory / 'prediction.json'
        if path.exists():
            saved = json.loads(path.read_text())
            recorded = saved.pop('record_hash')
            if recorded != value_hash(saved) or not receipt_binding_matches(saved, path, binding, job):
                raise ValueError('Completed BFCL receipt identity changed')
            return row['id']
        client = LocalTaskClient(state, job['endpoint'], timeout=config.task_timeout,
                                 enable_thinking=config.task_enable_thinking)
        seed_spec = load_seed(state.harness_path)
        if load_manifest(state.harness_path).identity != frozen['task_harness_identity']:
            raise ValueError('Frozen Harness changed')
        seed = int(value_hash([config.seed, row['id'], 'report_eval'])[:8], 16) % (2**31)
        calls, infra_errors = [], []
        if state.checkpoint_path not in TOKENIZERS:
            from transformers import AutoTokenizer
            TOKENIZERS[state.checkpoint_path] = AutoTokenizer.from_pretrained(state.checkpoint_path, local_files_only=True)
        tokenizer = TOKENIZERS[state.checkpoint_path]

        def model(messages, *, tools=None, seed, max_tokens, temperature):
            if time.time() >= job['deadline'] or len(calls) >= config.model_call_limit:
                raise _BudgetExhausted('BFCL final evaluation call/deadline limit')
            if max_tokens > config.max_output_tokens:
                raise ValueError('Frozen Task output budget exceeded')
            request = {'messages': copy.deepcopy(messages), 'tools': tools, 'seed': seed,
                       'max_tokens': max_tokens, 'temperature': temperature}
            encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                return_tensors='pt', return_dict=True,
                tools=tools or None, enable_thinking=config.task_enable_thinking)
            prompt_tokens = encoded['input_ids'].shape[-1]
            if prompt_tokens + max_tokens > TASK_CONTEXT_LIMIT:
                save_json(directory/'context_budget_exhausted.json', {'binding': binding,
                    'request_hash': value_hash(request), 'prompt_tokens': prompt_tokens,
                    'reserved_output_tokens': max_tokens, 'context_limit': TASK_CONTEXT_LIMIT,
                    'dispatched': False, 'classification': 'task_resource_limit_not_transport_failure'})
                raise _BudgetExhausted('Frozen Task context budget exhausted; input was not truncated')
            call_path = directory / 'calls' / f'{len(calls):04d}.json'
            receipt = {'binding': binding, 'request': request, 'request_hash': value_hash(request), 'status': 'dispatching'}
            if call_path.exists():
                old = json.loads(call_path.read_text())
                if not receipt_binding_matches(old, call_path, binding, job) or old['request_hash'] != receipt['request_hash']:
                    raise ValueError('BFCL replay request changed')
                if old['status'] != 'completed':
                    infra_errors.append('pending_response_requires_audit')
                    raise RuntimeError('Unconfirmed Task receipt: audit before replay')
                calls.append(old)
                return copy.deepcopy(old['response'])
            save_json(call_path, receipt)
            try:
                response = client(**request)
                receipt.update(status='completed', response=response)
                save_json(call_path, receipt)
                calls.append(receipt)
                return copy.deepcopy(response)
            except Exception as error:
                receipt.update(status='failed_response_unconfirmed', error_type=type(error).__name__,
                               error_detail=str(error)[:1000])
                save_json(call_path, receipt)
                infra_errors.append(type(error).__name__)
                raise

        worker = session(job['package'], seed) if 'initial_config' in row else None
        environment = Environment(row, job['functions'], worker)
        history, turn_results, predictions = [], [], []
        artifacts = job['artifact_sources']
        try:
            if worker:
                rpc(worker, {'operation': 'reset', 'entry': {k: row[k] for k in ('id', 'initial_config', 'involved_classes')}, 'seed': seed})
            for turn, messages in enumerate(row['question']):
                if time.time() >= job['deadline']:
                    raise TimeoutError('Final evaluation global deadline')
                environment.turn(turn)
                visible = copy.deepcopy(messages)
                if str(turn) in row.get('missed_function', {}):
                    if visible:
                        raise ValueError('Official missing-function turn unexpectedly has a user message')
                    visible = [{'role': 'user', 'content': 'Additional functions are now available. Continue the previous request using the updated tool interfaces.'}]
                history.extend(visible)
                prompt = 'Public conversation, in chronological order:\n' + json.dumps(history, ensure_ascii=False)
                if worker is None:
                    prompt += '\nThis is a declarative function-call task: tools record proposals without executing an external API. Submit your proposed calls, or give the user an appropriate response when no call is warranted, then finish.'
                outcome = run_seed(seed_spec, model, environment, prompt, seed=seed + turn*1000,
                                   artifact_sources=artifacts)
                if infra_errors:
                    raise RuntimeError('Task transport failed: ' + ','.join(infra_errors))
                turn_results.append(outcome)
                predictions.append(copy.deepcopy(environment.calls))
                for tool in outcome['tool_calls']:
                    history.append({'role': 'assistant', 'content': json.dumps({'name': tool['name'], 'arguments': tool['arguments']}, ensure_ascii=False)})
                    history.append({'role': 'tool', 'content': json.dumps(tool.get('observation', {'error': tool.get('error_type')}), ensure_ascii=False)})
                history.append({'role': 'assistant', 'content': str(outcome.get('final_answer') or '')})
                save_json(directory / f'turn_{turn}.json', outcome)
                if outcome.get('final_answer') is None:
                    break
        finally:
            if worker:
                worker.close()
        usage = {'model_calls': len(calls), 'input_tokens': 0, 'output_tokens': 0, 'unknown_usage_calls': 0}
        for call in calls:
            u = call['response'].get('usage') or {}
            if all(type(u.get(k)) is int for k in ('prompt_tokens', 'completion_tokens')):
                usage['input_tokens'] += u['prompt_tokens']; usage['output_tokens'] += u['completion_tokens']
            else:
                usage['unknown_usage_calls'] += 1
        result = {'binding': binding, 'split': 'report_eval', 'feedback_to_meta': False,
                  'task_harness_identity': frozen['task_harness_identity'],
                  'prediction': predictions if worker else (predictions[0] if predictions else []),
                  'turns_completed': len(turn_results), 'turns_expected': len(row['question']),
                  'error_types': [x.get('error_type') for x in turn_results], 'usage': usage,
                  'api_cost_usd': None, 'generated_assets_exported_to_training': False}
        save_json(path, {**result, 'record_hash': value_hash(result)})
        return row['id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bfcl-root', type=Path, default=Path('/root/data/RSI_iclr2027/dataset/test/tool_use/bfcl_v3/bfcl_eval'))
    parser.add_argument('--manifest', type=Path, default=Path('/root/data/RSI_iclr2027/dataset/manifests/bfcl_v3_manifest.json'))
    parser.add_argument('--enable-inference', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--wait-for-completion', action='store_true')
    parser.add_argument('--workers', type=int, default=3, choices=(1, 2, 3))
    parser.add_argument('--max-wall-seconds', type=int, default=86400)
    args = parser.parse_args()
    if args.max_wall_seconds <= 0 or args.max_wall_seconds > 86400:
        raise ValueError('Final evaluation deadline must be within 24 hours')
    root = args.bfcl_root.resolve()
    manifest = json.loads(args.manifest.read_text())
    if manifest['revision'] != COMMIT or manifest['tag'] != 'v1.3':
        raise ValueError('Expected pinned BFCL v3 v1.3 release')
    checkout = Path(manifest['local_path'])
    actual_commit = subprocess.run(['git', '-C', str(checkout), 'rev-parse', 'HEAD'],
                                   check=True, capture_output=True, text=True).stdout.strip()
    if actual_commit != COMMIT:
        raise ValueError('Official evaluator checkout changed')
    original = checkout / 'berkeley-function-call-leaderboard/bfcl_eval'
    rows, sources = [], {}
    for category, count in manifest['category_counts'].items():
        path = root / 'data' / (category + '.json')
        if digest(path) != digest(original/'data'/path.name):
            raise ValueError('Official benchmark data differs from pinned checkout')
        data = read_rows(path)
        if len(data) != count or any(not r['id'].startswith(category.removeprefix('BFCL_v3_') + '_') for r in data):
            raise ValueError('Official category denominator mismatch')
        rows.extend(data); sources[str(path)] = digest(path)
    if len(rows) != 4441 or len({r['id'] for r in rows}) != 4441:
        raise ValueError('BFCL-v3 full release ID manifest mismatch')
    package, files = package_official(root)
    for name, expected in files.items():
        if name.startswith('bfcl_eval/'):
            source = original.parent/name
            if source.exists() and digest(source) != expected:
                raise ValueError('Official checker source mismatch: ' + name)
    # Import/permission readiness, no API, GPU request, benchmark answer or score.
    with session(package, 42) as probe:
        rpc(probe, {'operation': 'ready'})
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'adapter_readiness.json', {'status': 'dependencies_and_isolation_ready',
        'expected_cases': len(rows), 'official_commit': COMMIT, 'categories': manifest['category_counts'],
        'source_hashes': sources, 'code_hashes': files, 'model_calls': 0, 'scores': None})
    if args.prepare_only:
        print(json.dumps({'ready': True, 'expected': len(rows), 'model_calls': 0})); return
    if not args.enable_inference:
        raise ValueError('Actual final evaluation requires --enable-inference')
    config = PipelineConfig.model_validate(json.loads(args.config.read_text()))
    final_path = args.run / 'final_state.json'
    while args.wait_for_completion and not final_path.exists():
        time.sleep(30)
    final = json.loads(final_path.read_text())
    if final['status'] != 'completed' or final['generations_executed'] != config.max_generations:
        raise ValueError('Every configured Task/Meta round must finish before final BFCL evaluation')
    if final.get('task_update_policy') != 'single_candidate_strict_positive_gain_v1':
        raise ValueError('Final BFCL evaluation requires the strict single-candidate deployment policy')
    protocol = json.loads((args.run / 'protocol.json').read_text())
    if protocol['controller'] != source_identity():
        raise ValueError('Training runtime changed; freeze requires an audited engineering boundary')
    state = load_task(final['task_state'])
    frozen = {'status': 'frozen_for_report_eval', 'task_state': asdict(state), 'state_hash': task_hash(state),
        'checkpoint_files': checkpoint_manifest(state.checkpoint_path),
        'task_harness_identity': harnessforge_identity(state.harness_path),
        'protocol_hash': protocol['hash'], 'selection': {
            'rule': final['task_update_policy'], 'generation': state.generation,
            'rounds_completed': final['rounds_completed']},
        'feedback_to_meta': False}
    if (output / 'frozen_task.json').exists() and json.loads((output / 'frozen_task.json').read_text()) != frozen:
        raise ValueError('Final Task cannot be reselected on resume')
    from sia.task_meta.scoped_execution import sync_replicas
    replicas = sync_replicas(state, config.task_replicas)
    artifact_sources = []
    if state.artifacts.directory:
        if artifact_manifest(state.artifacts.directory) != state.artifacts.manifest:
            raise ValueError('Frozen artifact contents changed')
        artifact_sources = [{**r, 'content': (Path(state.artifacts.directory)/r['path']).read_text()} for r in state.artifacts.manifest]
    base = {'frozen': frozen, 'config': config.model_dump(), 'output': str(output), 'package': package,
            'artifact_sources': artifact_sources}
    protocol_record = {'frozen_state_hash': frozen['state_hash'], 'data_sources': sources, 'official_code': files,
        'adapter_sha256': digest(Path(__file__)), 'worker_sha256': digest(Path(__file__).with_name('bfcl_report_worker.py')),
        'model_call_limit_per_case': config.model_call_limit, 'max_tokens_per_call': config.max_output_tokens,
        'task_context_limit': TASK_CONTEXT_LIMIT, 'input_truncation': False,
        'expected_ids': [r['id'] for r in rows], 'single_submission_per_case': True, 'feedback_to_meta': False,
        'conversation_protocol': 'Each public turn uses unchanged run_seed/run_harness with chronological public history; native environment state persists across turns',
        'score_protocol': 'Pinned official AST/state/response checkers; per-category accuracy and unweighted sample micro accuracy, not leaderboard weighted overall'}
    with exclusive_lock(output / '.evaluation.lock'):
        imported = {}
        import_path = output / 'receipt_import.json'
        if import_path.exists():
            imported = json.loads(import_path.read_text())
            previous = imported['source_protocol']
            if ({k: v for k, v in previous.items() if k != 'adapter_sha256'} !=
                    {k: v for k, v in protocol_record.items() if k != 'adapter_sha256'}
                    or imported['target_adapter_sha256'] != protocol_record['adapter_sha256']
                    or imported['config_sha256'] != digest(args.config)
                    or imported['frozen_state_hash'] != frozen['state_hash']
                    or imported['clock'] != json.loads((output / 'clock.json').read_text())):
                raise ValueError('Audited receipt import changed Task, scoring, config or budget')
        p = output / 'prediction_protocol.json'
        if p.exists() and json.loads(p.read_text()) != protocol_record:
            raise ValueError('Final evaluation protocol changed on resume')
        save_json(p, protocol_record)
        clock_path = output / 'clock.json'
        if not clock_path.exists():
            save_json(clock_path, {'started': time.time(), 'deadline': time.time()+args.max_wall_seconds})
        base['deadline'] = json.loads(clock_path.read_text())['deadline']
        base['protocol_hash'] = value_hash(protocol_record)
        pending, cursor, completed = {}, 0, 0
        pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'))
        try:
            while cursor < len(rows) or pending:
                while cursor < len(rows) and len(pending) < args.workers:
                    row = rows[cursor]
                    endpoint = replicas[cursor % args.workers]['base_url']
                    job = {**base, 'row': row, 'functions': native_functions(root, row), 'endpoint': endpoint}
                    if imported:
                        job['imported_protocol_hash'] = value_hash(imported['source_protocol'])
                        job['imported_receipts'] = imported['cases'].get(row['id'], {})
                    pending[pool.submit(case_job, job)] = row['id']; cursor += 1
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    task_id = future.result(); pending.pop(future); completed += 1
                    save_json(output / 'progress.json', {'phase': 'predicting', 'completed': completed, 'expected': len(rows), 'last_task_id': task_id, 'time': time.time()})
        except Exception as error:
            save_json(output / 'failure.json', {'stage': 'predicting', 'error_type': type(error).__name__,
                'detail': str(error), 'completed': completed, 'pending_case_ids': list(pending.values()), 'time': time.time()})
            for future in pending:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        scores = {}
        usage = dict(model_calls=0, input_tokens=0, output_tokens=0, unknown_usage_calls=0)
        gold_cache = {}
        for row in rows:
            category = row['id'].rsplit('_', 1)[0]
            prediction = json.loads((output/'cases'/row['id']/'prediction.json').read_text())
            for name in usage:
                usage[name] += prediction['usage'][name]
            gold_path = root/'data/possible_answer'/('BFCL_v3_'+category+'.json')
            if category not in scores:
                scores[category] = {'correct': 0, 'completed': 0, 'expected': manifest['category_counts']['BFCL_v3_'+category]}
            ground_truth = []
            if 'relevance' not in category:
                if category not in gold_cache:
                    if digest(gold_path) != digest(original/'data/possible_answer'/gold_path.name):
                        raise ValueError('Official answer identity changed')
                    gold_cache[category] = {r['id']: r['ground_truth'] for r in read_rows(gold_path)}
                ground_truth = gold_cache[category][row['id']]
            with session(package, 42) as scorer:
                result = rpc(scorer, {'operation': 'score', 'entry': row, 'prediction': prediction['prediction'],
                                      'ground_truth': ground_truth, 'seed': 42})
            save_json(output/'cases'/row['id']/'official_score.json', result)
            scores[category]['correct'] += int(result['valid']); scores[category]['completed'] += 1
            save_json(output/'progress.json', {'phase': 'scoring', 'scored': sum(v['completed'] for v in scores.values()),
                'expected': len(rows), 'last_task_id': row['id'], 'time': time.time()})
        for value in scores.values():
            value['accuracy'] = value['correct']/value['expected']
        save_json(output/'result.json', {'status': 'completed', 'benchmark': 'BFCL-v3', 'official_commit': COMMIT,
            'expected': len(rows), 'completed': len(rows), 'per_category': scores,
            'sample_micro_accuracy': sum(v['correct'] for v in scores.values())/len(rows),
            'leaderboard_weighted_overall': None, 'frozen_state_hash': frozen['state_hash'],
            'selection': frozen['selection'], 'feedback_to_meta': False, 'usage': usage, 'api_cost_usd': None})
        print(json.dumps({'status': 'completed', 'expected': len(rows), 'output': str(output)}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if '--output' in sys.argv:
            destination = Path(sys.argv[sys.argv.index('--output') + 1])
            save_json(destination/'launch_failure.json', {'error_type': type(error).__name__,
                'detail': str(error), 'time': time.time(), 'automatic_retry': False})
        raise

